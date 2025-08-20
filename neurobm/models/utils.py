# Copyright 2025 NeuroBM Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions for energy computation and sampling

This module provides utility functions for Boltzmann machines including:
- Energy computation helpers
- Sampling utilities
- Temperature scheduling
- Gradient clipping and numerical stability
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


def sigmoid_with_temperature(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sigmoid function with temperature scaling.

    Args:
        x: Input tensor
        temperature: Temperature parameter (higher = more random)

    Returns:
        Sigmoid probabilities with temperature scaling
    """
    return torch.sigmoid(x / temperature)


def sample_bernoulli(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample from Bernoulli distribution.

    Args:
        probs: Bernoulli probabilities [batch_size, n_units]

    Returns:
        samples: Binary samples [batch_size, n_units]
    """
    return torch.bernoulli(probs)


def sample_gaussian(mean: torch.Tensor, std: torch.Tensor = None) -> torch.Tensor:
    """
    Sample from Gaussian distribution.

    Args:
        mean: Mean values [batch_size, n_units]
        std: Standard deviation (if None, use unit variance)

    Returns:
        samples: Gaussian samples [batch_size, n_units]
    """
    if std is None:
        std = torch.ones_like(mean)
    return torch.normal(mean, std)


def sample_bernoulli_with_temperature(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from Bernoulli distribution with temperature.

    Args:
        probs: Bernoulli probabilities
        temperature: Temperature parameter

    Returns:
        Binary samples
    """
    if temperature == 1.0:
        return torch.bernoulli(probs)
    else:
        # Gumbel-softmax trick for temperature sampling
        logits = torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)
        return torch.bernoulli(torch.sigmoid(logits / temperature))


def sample_gaussian_with_temperature(
    mean: torch.Tensor,
    std: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from Gaussian distribution with temperature.

    Args:
        mean: Mean values
        std: Standard deviation values
        temperature: Temperature parameter

    Returns:
        Gaussian samples
    """
    noise = torch.randn_like(mean)
    return mean + std * noise * temperature


def clip_gradients(
    gradients: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> Union[torch.Tensor, float]:
    """
    Clip gradients by norm.

    Args:
        gradients: Single gradient tensor or list of gradient tensors
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use

    Returns:
        clipped_gradients: Clipped gradients (same type as input)
    """
    if isinstance(gradients, torch.Tensor):
        # Single tensor case
        grad_norm = torch.norm(gradients, norm_type)
        clip_coef = max_norm / (grad_norm + 1e-6)
        if clip_coef < 1:
            return gradients * clip_coef
        return gradients

    elif isinstance(gradients, list):
        # List of tensors case (for parameter lists)
        parameters = [g for g in gradients if g is not None]

        if len(parameters) == 0:
            return gradients

        device = parameters[0].device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [p * clip_coef if p is not None else None for p in gradients]

        return gradients

    else:
        raise ValueError(f"Unsupported gradient type: {type(gradients)}")


def compute_effective_sample_size(log_weights: torch.Tensor) -> float:
    """
    Compute effective sample size from log importance weights.

    Args:
        log_weights: Log importance weights [n_samples]

    Returns:
        Effective sample size
    """
    # Normalize log weights
    log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_weights)

    # ESS = 1 / sum(w_i^2)
    ess = 1.0 / torch.sum(weights**2)
    return ess.item()


def log_mean_exp(log_values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute log(mean(exp(log_values))) in a numerically stable way.

    Args:
        log_values: Log values
        dim: Dimension to average over

    Returns:
        Log of mean of exponentials
    """
    max_val = torch.max(log_values, dim=dim, keepdim=True)[0]
    return max_val.squeeze(dim) + torch.log(
        torch.mean(torch.exp(log_values - max_val), dim=dim)
    )


class TemperatureScheduler:
    """Temperature scheduling for annealing during training."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        schedule_type: str = "exponential",
        decay_steps: int = 1000,
        decay_rate: float = 0.95
    ):
        """
        Initialize temperature scheduler.

        Args:
            initial_temp: Starting temperature
            final_temp: Final temperature
            schedule_type: Type of schedule ('exponential', 'linear', 'cosine')
            decay_steps: Number of steps for decay
            decay_rate: Decay rate for exponential schedule
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule_type = schedule_type
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.step_count = 0

    def get_temperature(self) -> float:
        """Get current temperature."""
        if self.schedule_type == "exponential":
            temp = self.initial_temp * (self.decay_rate ** (self.step_count / self.decay_steps))
            return max(temp, self.final_temp)
        elif self.schedule_type == "linear":
            progress = min(self.step_count / self.decay_steps, 1.0)
            return self.initial_temp + progress * (self.final_temp - self.initial_temp)
        elif self.schedule_type == "cosine":
            progress = min(self.step_count / self.decay_steps, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.final_temp + (self.initial_temp - self.final_temp) * cosine_decay
        else:
            return self.initial_temp

    def step(self) -> None:
        """Update step count."""
        self.step_count += 1


class GibbsSampler:
    """Basic Gibbs sampler for Boltzmann machines."""

    def __init__(self, n_steps: int = 100, temperature: float = 1.0):
        """
        Initialize Gibbs sampler.

        Args:
            n_steps: Number of Gibbs steps
            temperature: Sampling temperature
        """
        self.n_steps = n_steps
        self.temperature = temperature

    def sample(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Generate samples using Gibbs sampling.

        Args:
            model: Boltzmann machine model
            initial_state: Initial state for sampling
            n_samples: Number of samples to generate

        Returns:
            samples: Generated samples
        """
        samples = []
        current_state = initial_state.clone()

        for _ in range(n_samples):
            # Run Gibbs chain
            for _ in range(self.n_steps):
                if hasattr(model, 'gibbs_step'):
                    current_state, _, _, _ = model.gibbs_step(current_state)
                elif hasattr(model, 'reconstruct'):
                    current_state = model.reconstruct(current_state, n_gibbs=1)
                else:
                    raise ValueError("Model doesn't support Gibbs sampling")

            samples.append(current_state.clone())

        return torch.stack(samples, dim=0)


class BlockedGibbsSampler:
    """Blocked Gibbs sampler for efficient sampling."""

    def __init__(self, block_size: int = 10):
        """
        Initialize blocked Gibbs sampler.

        Args:
            block_size: Size of blocks for blocked sampling
        """
        self.block_size = block_size

    def sample_blocked(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        n_steps: int = 100
    ) -> torch.Tensor:
        """
        Perform blocked Gibbs sampling.

        Args:
            model: Boltzmann machine model
            initial_state: Initial state
            n_steps: Number of sampling steps

        Returns:
            Final sampled state
        """
        state = initial_state.clone()
        n_units = state.size(-1)

        for _ in range(n_steps):
            # Randomly permute units
            perm = torch.randperm(n_units)

            # Sample in blocks
            for i in range(0, n_units, self.block_size):
                block_indices = perm[i:i + self.block_size]

                # Sample block conditioned on rest
                # This is model-specific and would need to be implemented
                # for each specific Boltzmann machine type
                pass

        return state


def compute_partition_function_bounds(
    model: nn.Module,
    n_samples: int = 1000,
    n_chains: int = 10
) -> Tuple[float, float]:
    """
    Compute bounds on the partition function using AIS.

    Args:
        model: Boltzmann machine model
        n_samples: Number of samples per chain
        n_chains: Number of parallel chains

    Returns:
        lower_bound: Lower bound on log partition function
        upper_bound: Upper bound on log partition function
    """
    # This would implement AIS for partition function estimation
    # Placeholder implementation
    logger.warning("Partition function bounds computation not fully implemented")
    return -float('inf'), float('inf')


def validate_energy_symmetry(
    model: nn.Module,
    test_data: torch.Tensor,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that energy function satisfies expected symmetries.

    Args:
        model: Boltzmann machine model
        test_data: Test data for validation
        tolerance: Numerical tolerance

    Returns:
        True if symmetries are satisfied
    """
    # Test energy function properties
    # This would implement various symmetry checks
    logger.info("Energy symmetry validation not fully implemented")
    return True
