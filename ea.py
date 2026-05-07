# ea.py
import random
import copy
import torch
import time
import traceback
import json
from model import AttentionSNN

from snntorch import surrogate

import train_snn

"""Implements a simple, stable evolutionary algorithm for expensive models."""
"""Defines the legal values for each evolvable genotype field."""

SEARCH_SPACE = {
    # Binary masks for which layers are active (4 positions)
    # At least one layer must be True (enforced in genotype_utils)
    "layers_active": [
        (True, False, False),   # 1 layer active
        (False, True, False),    # layers 2 and 4
        (False, False, True),    # layers 2 and 4
        (False, True, True),    # layers 2 and 4
        (True, False, True),    # layers 1 and 3
        (True, True, False),    # 2 layers active
        (True, True, True),     # 3 layers active
        # ... add more combinations if needed
    ],

    # Temporal depth (8-20 range)
    "num_steps": [8, 10, 12, 14, 16, 18, 20],

    # LIF membrane decay
    "beta": [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95],

    # Channel width multiplier
    "width_mult": [0.5, 1.0, 1.5, 2.0]
}

# ============================================================
# Reproducibility seeding
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ============================================================


class SimpleEA:
    def __init__(
            self,
            population_size=8,
            elite_fraction=0.25,
            mutation_prob=0.1,
            seed=None
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_prob = mutation_prob

        if seed is not None:
            set_seed(seed)

    def init_population(self):
        # Creates an initial population of random genotypes
        return [random_genotype() for _ in range(self.population_size)]

    def mutate(self, genotype):
        # Mutate the genotype by changing one field
        g = copy.deepcopy(genotype)

        fields = ["layers_active", "num_steps", "beta", "width_mult"]
        field = random.choice(fields)

        if field == "layers_active":
            current = list(g.layers_active)

            idx = random.randint(0, 2)
            current[idx] = not current[idx]

            # Ensure at least one active layer
            if not any(current):
                current[random.randint(0, 2)] = True

            g.layers_active = tuple(current)

        else:
            current_value = getattr(g, field)
            possible_values = SEARCH_SPACE[field]

            if len(possible_values) > 1:
                new_values = [v for v in possible_values if v != current_value]
                new_value = random.choice(new_values)
            else:
                new_value = possible_values[0]

            setattr(g, field, new_value)

        return g

    def evolve(self, generations, device, verbose=True):
        """
        Core evolutionary loop.
        Returns: history list of (fitness, accuracy, genotype, metrics)
        """

        population = self.init_population()
        history = []

        if verbose:
            print("Starting evolution")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {generations}")
            print(f"Elite fraction: {self.elite_fraction}")
            print(f"Mutation probability: {self.mutation_prob}")

        for gen in range(generations):
            if verbose:
                print(f"\nGeneration {gen}")

            scored = []

            # Evaluation
            for g in population:
                fitness, accuracy, metrics = evaluate_genotype(g, device)
                scored.append((fitness, accuracy, g, metrics))

                if verbose:
                    print(f"acc={accuracy:.4f} | fitness={fitness:.4f} | layers={g.layers_active}/4 | steps={g.num_steps} | beta={g.beta} | width_mult={g.width_mult}")

            # Selection
            scored.sort(key=lambda x: x[0], reverse=True)
            history.append(scored[0])

            best_fitness, best_accuracy, best_genotype, best_metrics = scored[0]

            if verbose:
                print("Best individual")
                print(f"accuracy: {best_accuracy:.4f}")
                print(f"fitness: {best_fitness:.4f}")
                print(f"layers: {best_genotype.layers_active}")
                if 'spike_sparsity' in best_metrics:
                    print(f"sparsity: {best_metrics['spike_sparsity']:.4f}")

            # Elitism
            num_elites = max(1, int(self.elite_fraction * self.population_size))
            elites = scored[:num_elites]
            new_population = [g for _, _, g, _ in elites]

            # Reproduction
            while len(new_population) < self.population_size:
                parent = random.choice(elites)[2]

                if random.random() < self.mutation_prob:
                    child = self.mutate(parent)
                else:
                    child = copy.deepcopy(parent)

                new_population.append(child)

            population = new_population

        return history

from dataclasses import dataclass
from typing import Tuple

"""Defines the evolvable parameters: attention layers, temporal depth, width, and LIF decay."""

@dataclass
class SNNGenotype:
    """
    Genotype controlling the SNN architecture.
    - layers_active: binary mask for which residual+attention blocks are active
                     Order: [res1+attn1, res2+attn2, attn3, attn4] = 4 positions
    - num_steps: number of temporal simulation steps (T)
    - beta: LIF membrane decay rate
    - width_mult: channel width multiplier (e.g., 0.5, 1.0, 1.5)
    """
    layers_active: Tuple[bool, bool, bool, bool]  # 4 positions
    num_steps: int
    beta: float
    width_mult: float

# Utility functions for creating and mutating genotypes

def random_genotype() -> SNNGenotype:
    """Sample a random genotype from the search space."""
    return SNNGenotype(
        layers_active=random.choice(SEARCH_SPACE["layers_active"]),
        num_steps=random.choice(SEARCH_SPACE["num_steps"]),
        beta=random.choice(SEARCH_SPACE["beta"]),
        width_mult=random.choice(SEARCH_SPACE["width_mult"]),
    )

def calculate_fitness(accuracy, genotype):
    # Calculate fitness: ONLY accuracy, no penalties
    return accuracy

def validate_genotype(g: SNNGenotype):
    # Check layers_active
    assert len(g.layers_active) == 4, f"layers_active must have 4 elements, got {len(g.layers_active)}"
    assert any(g.layers_active), "At least one layer must be active"

    # Check num_steps
    assert 8 <= g.num_steps <= 20, f"num_steps must be between 8 and 20, got {g.num_steps}"

    # Check beta
    assert 0.85 <= g.beta <= 0.95, f"beta must be between 0.85 and 0.95, got {g.beta}"

    # Check width_mult
    assert g.width_mult in [0.5, 1.0, 1.5], f"width_mult must be 0.5, 1.0, or 1.5, got {g.width_mult}"


def evaluate_genotype(genotype, device, debug=False):
    """Builds and evaluates an AttentionSNN from a genotype.
    Returns: (fitness, accuracy, metrics_dict) tuple"""

    if debug:
        print(f"Evaluating genotype: {genotype}")
        print(f"Active layers: {sum(genotype.layers_active)}/4")

    # Build model from genotype
    try:
        model = AttentionSNN(
            num_steps=genotype.num_steps,
            beta=genotype.beta,
            spike_grad=surrogate.atan(2.0),
            layers_active=genotype.layers_active,
            width_mult=genotype.width_mult
        )
        if debug:
            print(f"Model created successfully")
    except Exception as e:
        print(f"ERROR: Model creation failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0, {}

    # Train and get accuracy
    start_time = time.time()

    if debug:
        print(f"Starting training...")

    try:
        # Use the patched trainer wrapper
        _, acc = train_snn.train(
            model=model,
            num_epochs=4,
            device=device,
            max_train_samples=10000,
            max_test_samples=2500
        )
        training_time = time.time() - start_time

        if debug:
            print(f"Training completed: {acc:.4f} accuracy, {training_time:.1f}s")

    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 0.0, 0.0, {}

    # Calculate fitness (just accuracy now)
    fitness = calculate_fitness(acc, genotype)

    # Create comprehensive metrics dictionary
    metrics = {
        'accuracy': acc,
        'fitness': fitness,
        'training_time': training_time,
        'epochs': 2,
        'active_layers': sum(genotype.layers_active),
        'num_steps': genotype.num_steps,
        'beta': genotype.beta,
        'width_mult': genotype.width_mult,
        'layers_active': genotype.layers_active,  # Keep as tuple
    }

    # Add spike statistics from model
    if hasattr(model, 'get_spike_stats'):
        try:
            spike_stats = model.get_spike_stats()
            metrics.update(spike_stats)
            if debug:
                sparsity = spike_stats.get('spike_sparsity', 0)
                print(f"Spike sparsity: {sparsity:.3f}")
        except Exception as e:
            if debug:
                print(f"Could not get spike stats: {e}")

    # Add model complexity metrics
    if hasattr(model, 'get_model_info'):
        try:
            model_info = model.get_model_info()
            metrics['total_params'] = model_info.get('total_params', 0)
            metrics['trainable_params'] = model_info.get('trainable_params', 0)
            if debug:
                print(f"Model params: {metrics['total_params']:,}")
        except Exception as e:
            if debug:
                print(f"Could not get model info: {e}")

    return fitness, acc, metrics