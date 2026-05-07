import torch
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import torch.nn as nn

# ---------------- SECA Attention ----------------
class SECAAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ---------------- Residual Block ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.GroupNorm(32, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn3 = nn.GroupNorm(32, out_channels)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels)
        )

    def forward(self, x):
        x1 = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += x1
        out = F.relu(out)
        return out

# ---------------- Attention SNN ----------------
class AttentionSNN(nn.Module):
    def __init__(self, num_steps, beta, spike_grad, layers_active, width_mult: float):
        super().__init__()
        assert len(layers_active) == 3, "layers_active must be a tuple of 3 booleans"

        base_c1, base_c2 = 64, 128
        c1, c2 = int(base_c1 * width_mult), int(base_c2 * width_mult)

        self.num_steps = num_steps

        # spike metrics
        self.total_spikes = 0
        self.total_possible_spikes = 0
        self.reset_spike_stats()

        # -------- Block 1 --------
        self.res1 = ResidualBlock(3, c1)
        self.attn1 = SECAAttention(c1) if layers_active[0] else nn.Identity()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------- Block 2 --------
        self.res2 = ResidualBlock(c1, c2)
        self.attn2 = SECAAttention(c2) if layers_active[1] else nn.Identity()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------- Block 3 --------
        self.res3 = ResidualBlock(c2, c2)
        self.attn3 = SECAAttention(c2) if layers_active[2] else nn.Identity()
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # -------- Classifier --------
        self.fc = nn.Linear(c2 * 4 * 4, 100)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    # ---------------- Forward Pass ----------------
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_rec = []

        # reset metrics
        self.reset_spike_stats()

        x = self.res1(x)
        x = self.attn1(x)
        x = self.pool1(x)

        # spike logging for debugging
        spike_log = {'lif1': [], 'lif2': [], 'lif3': [], 'lif_out': []}

        for _ in range(self.num_steps):
            # first LIF layer
            spk1, mem1 = self.lif1(x, mem1)
            self.total_spikes += spk1.sum().item()
            self.total_possible_spikes += spk1.numel()

            # residual + attention layers
            h = self.res2(spk1)
            h = self.attn2(h)
            h = self.pool2(h)
            spk2, mem2 = self.lif2(x, mem2)
            self.total_spikes += spk2.sum().item()
            self.total_possible_spikes += spk2.numel()

            h = self.res3(h)
            h = self.attn3(h)
            h = self.pool3(h)
            spk3, mem3 = self.lif3(x, mem3)
            self.total_spikes += spk3.sum().item()
            self.total_possible_spikes += spk3.numel()

            # output layer
            cur = self.fc(h.flatten(1))
            spk_out, mem_out = self.lif_out(cur, mem_out)
            self.total_spikes += spk_out.sum().item()
            self.total_possible_spikes += spk_out.numel()

            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

    # ---------------- Metric Functions ----------------
    def get_spike_sparsity(self):
        if self.total_possible_spikes == 0:
            return 0.0
        return 1.0 - (self.total_spikes / self.total_possible_spikes)

    def get_spike_rate(self):
        if self.total_possible_spikes == 0:
            return 0.0
        return self.total_spikes / self.total_possible_spikes

    def get_spike_stats(self):
        return {
            'total_spikes': self.total_spikes,
            'total_possible_spikes': self.total_possible_spikes,
            'spike_sparsity': self.get_spike_sparsity(),
            'spike_rate': self.get_spike_rate()
        }

    def reset_spike_stats(self):
        self.total_spikes = 0
        self.total_possible_spikes = 0

    def print_spike_stats(self):
        sparsity = self.get_spike_sparsity()
        rate = self.get_spike_rate()
        print(f"Total spikes: {self.total_spikes}")
        print(f"Total possible spikes: {self.total_possible_spikes}")
        print(f"Spike sparsity: {sparsity:.4f}")
        print(f"Spike rate: {rate:.4f}")

    # ---------------- Model Info ----------------
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'num_steps': self.num_steps,
            'total_params': total_params,
            'trainable_params': trainable_params
        }