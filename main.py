# ------------------ Standard Library ------------------
import os
import time
import json
import random
from datetime import datetime
from pathlib import Path

# ------------------ Core PyTorch ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ------------------ DDP / Multi-GPU ------------------
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ------------------ TorchVision ------------------
from torchvision import datasets, transforms

# ------------------ Optimizers & Schedulers ------------------
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------ snnTorch ------------------
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

# ------------------ Model Components ------------------
from model import (
    AttentionSNN,
    ResidualBlock,
    SECAAttention,
)

import train_snn
from ea import SNNGenotype, SimpleEA

# Specify GPU number
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Based on model of original authors
BASELINE_GENOTYPE = SNNGenotype(
    layers_active=(True, False, False),
    num_steps=12,      
    beta=0.90,         
    width_mult=1.0     
)

def ea_evolve(population_size, elite_fraction, mutation_prob):
    ea = SimpleEA(
        population_size,
        elite_fraction,
        mutation_prob
    )

    history = ea.evolve(generations=3, device=device)

    best_fitness, best_accuracy, best_genotype, best_score = history[-1]
    print(f"\nEvolved genotype: {best_genotype}")
    print(f"Active layers: {best_genotype.layers_active}")
    print(f"Fitness: {best_fitness:.3f}, Accuracy: {best_accuracy:.3f}")

    return best_genotype

if __name__ == "__main__":
    train_snn.init_torch()

    # Run EA
    ea_genotype: SNNGenotype = ea_evolve(6, 0.33, 0.1)

    # Define manual genotype
    MODEL_GENOTYPE = SNNGenotype(
    layers_active=(True, False, False),  
    num_steps=20,      
    beta=0.92,        
    width_mult=1.0     
    )

    model = AttentionSNN(
        num_steps=ea_genotype.num_steps,
        beta=ea_genotype.beta,
        spike_grad=surrogate.atan(2.0),
        layers_active=ea_genotype.layers_active,
        width_mult=ea_genotype.width_mult
    )

    # Run full training session for 25 epochs
    results = train_snn.train(
        model,
        num_epochs=25,
        device=device
    )

    # Print stats
    print(model.get_spike_stats())