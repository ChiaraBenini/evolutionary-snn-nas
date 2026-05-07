import os
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import SECAAttention, ResidualBlock, AttentionSNN
import numpy as np
import random
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
import time

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def init_torch():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_torch_ddp(model):
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend)

    torch.cuda.set_device(local_rank)
    model = model.to(f"cuda:{local_rank}")

    return DDP(model, device_ids=[local_rank])

def train(model, num_epochs, batch_size):
    data_path = r'./data/'

    # ---- MNIST ----
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    MNIST_train = datasets.MNIST(
        data_path, train=True, download=True, transform=mnist_transform
    )

    MNIST_test = datasets.MNIST(
        data_path, train=False, download=True, transform=mnist_transform
    )

    train_sampler = DistributedSampler(MNIST_train)
    test_sampler = DistributedSampler(MNIST_test)

    train_loader = DataLoader(
        MNIST_train,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True
    )

    test_loader = DataLoader(
        MNIST_test,
        batch_size=batch_size,
        sampler=test_sampler,
        drop_last=True
    )

    # Visualizing datasets
    for data, label in train_loader:
        print('Data Shape:', data.shape)  # Data Shape: torch.Size([128, 1, 28, 28])
        print('Label Shape:', label.shape)  # Label Shape: torch.Size([128])
        break

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    loss_fn = SF.ce_rate_loss()

    rtt = time.perf_counter()
    loss_hist = []
    test_acc_hist = []
    best_acc = 0.0

    for epoch in range(num_epochs):

        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(local_rank)
            targets = targets.to(local_rank)

            # forward pass
            model.train()

            spk_rec = model(data)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

        dist.barrier()
        with torch.no_grad():
            # Test set forward pass
            test_acc = batch_accuracy(test_loader, model)
            if best_acc <= test_acc:
                best_acc = test_acc

            # Only one process has to print (to avoid double lines in terminal)
            if (global_rank == 0):
                print(f"Epoch: {epoch}, Time: {time.perf_counter() - rtt} seconds, Loss: {loss_val.item()}, Test Acc: {test_acc * 100:.2f}%, Best ACC: {best_acc * 100:.2f}%")
                rtt = time.perf_counter()
                test_acc_hist.append(test_acc.item())
    
    # Again, avoid double printing
    if (global_rank == 0):
        print("--- Training complete ---")

# Test batch accuracy
def batch_accuracy(data_loader, model):
    with torch.no_grad():
        total = 0
        acc = 0
        model.eval()

        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(local_rank)
            targets = targets.to(local_rank)
            spk_rec = model(data)
            acc += SF.accuracy_rate(spk_rec, targets, num_classes=10) * spk_rec.size(0)
            total += spk_rec.size(0)

    return acc / total

local_rank = 0
global_rank = 0

if __name__ == "__main__":
    init_torch()
    
    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    #### Training parameters ####
    nb_classes = 10  # Change for different datasets
    num_epochs = 10
    batch_size = 512

    os.environ['CUDA_VISIBLE_DEVICES'] = "1,3" # Fill in the numbers of the GPUs you want to use
    #############################

    #### Model parameters ####
    num_steps = 8                    # Spiking simulation steps per forward pass, should be at least 6
    beta = 0.5                       # Membrane decay rate              
    surr = surrogate.atan(alpha=2.0) # Surrogate gradient type (Can also try: surrogate.fast_sigmoid(slope=25))
    ##########################

    model = AttentionSNN(num_steps, beta, surr)
    model = init_torch_ddp(model)

    train(model, num_epochs, batch_size)

    # Free GPU resources
    torch.distributed.destroy_process_group()