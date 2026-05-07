import time
import random
import numpy as np

import torch
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from snntorch import functional as SF
import snntorch.utils as utils
import time
import ssl

data_path = "./data/"
batch_size = 64

def init_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Training
def train(model, num_epochs, device, max_train_samples = 0, max_test_samples = 0):
    print(f"Device: {device}")
    model_dev = model.to(device)

    # CIFAR10
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, ), (0.48215827, ), (0.44653124, ))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, ), (0.48215827, ), (0.44653124, ))
    ])

    train_dataset = datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )

    # Determine train and test sample size
    if max_train_samples and max_train_samples < len(train_dataset):
        train_subset = Subset(train_dataset, range(max_train_samples))
    else:
        train_subset = train_dataset

    if max_test_samples and max_test_samples < len(test_dataset):
        test_subset = Subset(test_dataset, range(max_test_samples))
    else:
        test_subset = test_dataset

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = SF.ce_rate_loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    best_acc = 0.0
    train_loss_hist = []
    test_acc_hist = []

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model_dev.train()

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            utils.reset(model)

            spk_rec = model_dev(data)

            if isinstance(spk_rec, tuple):
                spk_rec = spk_rec[0]

            loss = loss_fn(spk_rec, targets)
            train_loss_hist.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update accuracy record
        test_acc = batch_accuracy(test_loader, model_dev, device)
        test_acc_hist.append(test_acc)
        best_acc = max(best_acc, test_acc)

        scheduler.step(test_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"Test Acc {test_acc * 100:.2f}% | "
            f"Best {best_acc * 100:.2f}% | "
            f"Time {time.perf_counter() - start_time:.1f}s"
        )
    
    results = {
        "evolution": {
            "final_accuracy": best_acc.item(),
            "training_history": {
                "train_loss": train_loss_hist,
                "test_accuracy": test_acc_hist,
            },
            "total_params": sum(p.numel() for p in model.parameters()),
            "active_layers": getattr(model, "active_layers", None),
            "spike_sparsity": getattr(model, "spike_sparsity", None),
        }
    }

    return results, best_acc


# Accuracy
def batch_accuracy(data_loader, model, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            utils.reset(model)

            spk_rec = model(data)

            if isinstance(spk_rec, tuple):
                spk_rec = spk_rec[0]

            acc = SF.accuracy_rate(spk_rec, targets, num_classes=100)

            correct += acc * data.size(0)
            total += data.size(0)

    return correct / total