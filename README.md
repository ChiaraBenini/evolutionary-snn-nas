# Evolutionary Neural Architecture Search for Attention-Enabled Spiking Neural Networks

Research project exploring evolutionary Neural Architecture Search (NAS) for optimizing attention-enabled Spiking Neural Networks (SNNs).

## Overview

This project investigates whether evolutionary optimization can improve the performance of attention-based SNN architectures.

We integrate:
- Spiking-Efficient Channel Attention (SECA)
- Convolutional Spiking Neural Networks
- Evolutionary Neural Architecture Search
- Distributed multi-GPU training

The evolutionary algorithm optimizes:
- Attention layer placement
- Simulation time steps
- Membrane decay parameters
- Channel width

## Features

- Attention-enabled SNN architecture
- Evolutionary NAS framework
- CIFAR-10 / CIFAR-100 experiments
- Multi-GPU distributed training
- Spike sparsity and energy-efficiency analysis

## Technologies

- Python
- PyTorch
- DistributedDataParallel (DDP)
- Spiking Neural Networks
- Evolutionary Algorithms
- Neural Architecture Search


## Key Results

- Evolutionary NAS discovered competitive SNN configurations
- Comparable performance to manually designed baselines
- Increased simulation steps improved stability
- Trade-offs observed between accuracy, sparsity, and training cost

## Research Focus

This work explores:
- Neuromorphic computing
- Efficient AI architectures
- Automated neural architecture design
- Attention mechanisms for SNNs

## Future Work

- Multi-objective NAS optimization
- Hardware-aware fitness functions
- Larger search spaces
- Advanced spiking attention modules
- Energy-aware optimization
