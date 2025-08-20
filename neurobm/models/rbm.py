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
Restricted Boltzmann Machine implementation with CD-k and PCD training

This module implements RBMs with support for:
- Bernoulli-Bernoulli and Gaussian-Bernoulli visible units
- Contrastive Divergence (CD-k) and Persistent Contrastive Divergence (PCD)
- Temperature annealing and sparsity penalties
- Mixed precision training and gradient clipping
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RestrictedBoltzmannMachine(nn.Module):
    """
    Restricted Boltzmann Machine with flexible visible unit types.

    Supports both Bernoulli-Bernoulli and Gaussian-Bernoulli configurations
    with various training algorithms including CD-k and PCD.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        visible_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        hidden_type: Literal["bernoulli"] = "bernoulli",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        sparsity_target: Optional[float] = None,
        sparsity_weight: float = 0.1,
        temperature: float = 1.0,
        use_bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Restricted Boltzmann Machine.

        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            visible_type: Type of visible units ('bernoulli' or 'gaussian')
            hidden_type: Type of hidden units (currently only 'bernoulli')
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient for parameter updates
            weight_decay: L2 regularization coefficient
            sparsity_target: Target sparsity level for hidden units (if None, no sparsity)
            sparsity_weight: Weight for sparsity penalty
            temperature: Temperature for sampling (1.0 = no temperature scaling)
            use_bias: Whether to use bias terms
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        super().__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.visible_type = visible_type
        self.hidden_type = hidden_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        self.use_bias = use_bias
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        # Initialize parameters
        self._init_parameters()

        # Move to device
        self.to(self.device)

        # Persistent chain for PCD (initialized when needed)
        self.persistent_chain: Optional[torch.Tensor] = None

        # Training statistics
        self.training_stats = {
            'reconstruction_error': [],
            'free_energy': [],
            'sparsity': [],
            'weight_norm': []
        }

    def _init_parameters(self) -> None:
        """Initialize model parameters using Xavier initialization."""
        # Weight matrix W connecting visible and hidden units
        std = np.sqrt(2.0 / (self.n_visible + self.n_hidden))
        self.W = nn.Parameter(
            torch.randn(self.n_visible, self.n_hidden, dtype=self.dtype) * std
        )

        if self.use_bias:
            # Visible bias
            self.v_bias = nn.Parameter(
                torch.zeros(self.n_visible, dtype=self.dtype)
            )
            # Hidden bias
            self.h_bias = nn.Parameter(
                torch.zeros(self.n_hidden, dtype=self.dtype)
            )
        else:
            self.register_parameter('v_bias', None)
            self.register_parameter('h_bias', None)

        # For Gaussian visible units, we need variance parameters
        if self.visible_type == "gaussian":
            self.v_sigma = nn.Parameter(
                torch.ones(self.n_visible, dtype=self.dtype)
            )
        else:
            self.register_parameter('v_sigma', None)

    def visible_to_hidden(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden unit probabilities and sample from them.

        Args:
            v: Visible unit states [batch_size, n_visible]

        Returns:
            h_prob: Hidden unit probabilities [batch_size, n_hidden]
            h_sample: Hidden unit samples [batch_size, n_hidden]
        """
        # Linear activation
        activation = torch.matmul(v, self.W)
        if self.h_bias is not None:
            activation += self.h_bias

        # Apply temperature scaling
        activation = activation / self.temperature

        # For Bernoulli hidden units
        h_prob = torch.sigmoid(activation)
        h_sample = torch.bernoulli(h_prob)

        return h_prob, h_sample

    def hidden_to_visible(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visible unit probabilities and sample from them.

        Args:
            h: Hidden unit states [batch_size, n_hidden]

        Returns:
            v_prob: Visible unit probabilities/means [batch_size, n_visible]
            v_sample: Visible unit samples [batch_size, n_visible]
        """
        # Linear activation
        activation = torch.matmul(h, self.W.t())
        if self.v_bias is not None:
            activation += self.v_bias

        if self.visible_type == "bernoulli":
            # Bernoulli visible units
            v_prob = torch.sigmoid(activation / self.temperature)
            v_sample = torch.bernoulli(v_prob)
        elif self.visible_type == "gaussian":
            # Gaussian visible units with learned variance
            v_prob = activation  # Mean
            if self.v_sigma is not None:
                noise = torch.randn_like(v_prob) * self.v_sigma * self.temperature
                v_sample = v_prob + noise
            else:
                v_sample = v_prob + torch.randn_like(v_prob) * self.temperature
        else:
            raise ValueError(f"Unknown visible type: {self.visible_type}")

        return v_prob, v_sample

    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy of a visible-hidden configuration.

        Energy function: E(v,h) = -v^T W h - a^T v - b^T h
        For Gaussian visible: E(v,h) = -v^T W h - a^T v - b^T h + ||v-a||^2/(2σ^2)

        Args:
            v: Visible unit states [batch_size, n_visible]
            h: Hidden unit states [batch_size, n_hidden]

        Returns:
            energy: Energy values [batch_size]
        """
        # Interaction term: -v^T W h
        interaction = -torch.sum(v.unsqueeze(2) * self.W.unsqueeze(0) * h.unsqueeze(1), dim=(1, 2))

        # Visible bias term: -a^T v
        if self.v_bias is not None:
            visible_bias_term = -torch.sum(v * self.v_bias, dim=1)
        else:
            visible_bias_term = 0

        # Hidden bias term: -b^T h
        if self.h_bias is not None:
            hidden_bias_term = -torch.sum(h * self.h_bias, dim=1)
        else:
            hidden_bias_term = 0

        energy = interaction + visible_bias_term + hidden_bias_term

        # For Gaussian visible units, add quadratic term
        if self.visible_type == "gaussian" and self.v_sigma is not None:
            if self.v_bias is not None:
                quadratic_term = torch.sum((v - self.v_bias)**2 / (2 * self.v_sigma**2), dim=1)
            else:
                quadratic_term = torch.sum(v**2 / (2 * self.v_sigma**2), dim=1)
            energy += quadratic_term

        return energy

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of visible units.

        F(v) = -log(sum_h exp(-E(v,h)))

        Args:
            v: Visible unit states [batch_size, n_visible]

        Returns:
            free_energy: Free energy values [batch_size]
        """
        # Visible bias term
        if self.v_bias is not None:
            visible_bias_term = -torch.sum(v * self.v_bias, dim=1)
        else:
            visible_bias_term = 0

        # Hidden activation
        hidden_activation = torch.matmul(v, self.W)
        if self.h_bias is not None:
            hidden_activation += self.h_bias

        # Sum over hidden units: -sum_j log(1 + exp(activation_j))
        hidden_term = -torch.sum(F.softplus(hidden_activation), dim=1)

        free_energy = visible_bias_term + hidden_term

        # For Gaussian visible units
        if self.visible_type == "gaussian" and self.v_sigma is not None:
            if self.v_bias is not None:
                quadratic_term = torch.sum((v - self.v_bias)**2 / (2 * self.v_sigma**2), dim=1)
            else:
                quadratic_term = torch.sum(v**2 / (2 * self.v_sigma**2), dim=1)
            free_energy += quadratic_term

        return free_energy

    def gibbs_step(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one step of Gibbs sampling.

        Args:
            v: Current visible states [batch_size, n_visible]

        Returns:
            v_new: New visible states [batch_size, n_visible]
            h_prob: Hidden probabilities [batch_size, n_hidden]
            h_sample: Hidden samples [batch_size, n_hidden]
            v_prob: Visible probabilities [batch_size, n_visible]
        """
        # Sample hidden given visible
        h_prob, h_sample = self.visible_to_hidden(v)

        # Sample visible given hidden
        v_prob, v_new = self.hidden_to_visible(h_sample)

        return v_new, h_prob, h_sample, v_prob

    def contrastive_divergence(
        self,
        v_data: torch.Tensor,
        k: int = 1,
        persistent: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform Contrastive Divergence training.

        Args:
            v_data: Training data [batch_size, n_visible]
            k: Number of Gibbs sampling steps
            persistent: Whether to use persistent contrastive divergence (PCD)

        Returns:
            loss: Reconstruction loss
            grads: Dictionary of gradients for each parameter
        """
        batch_size = v_data.size(0)

        # Positive phase: compute statistics from data
        h_pos_prob, h_pos_sample = self.visible_to_hidden(v_data)

        # Negative phase: run k steps of Gibbs sampling
        if persistent and self.persistent_chain is not None:
            # Use persistent chain for PCD
            v_neg = self.persistent_chain
            if v_neg.size(0) != batch_size:
                # Resize persistent chain if batch size changed
                v_neg = v_neg[:batch_size] if v_neg.size(0) > batch_size else \
                        torch.cat([v_neg, v_data[:batch_size - v_neg.size(0)]], dim=0)
        else:
            # Start from data for CD
            v_neg = v_data.clone()

        # Run k steps of Gibbs sampling
        for _ in range(k):
            v_neg, h_neg_prob, h_neg_sample, v_neg_prob = self.gibbs_step(v_neg)

        # Final negative phase statistics
        h_neg_prob, h_neg_sample = self.visible_to_hidden(v_neg)

        # Update persistent chain for PCD
        if persistent:
            self.persistent_chain = v_neg.detach()

        # Compute gradients
        grads = {}

        # Weight gradient
        pos_grad = torch.matmul(v_data.t(), h_pos_prob)
        neg_grad = torch.matmul(v_neg.t(), h_neg_prob)
        grads['W'] = (pos_grad - neg_grad) / batch_size

        if self.use_bias:
            # Visible bias gradient
            grads['v_bias'] = torch.mean(v_data - v_neg, dim=0)

            # Hidden bias gradient
            grads['h_bias'] = torch.mean(h_pos_prob - h_neg_prob, dim=0)

        # For Gaussian visible units, update variance
        if self.visible_type == "gaussian" and self.v_sigma is not None:
            if self.v_bias is not None:
                pos_var = torch.mean((v_data - self.v_bias)**2, dim=0)
                neg_var = torch.mean((v_neg - self.v_bias)**2, dim=0)
            else:
                pos_var = torch.mean(v_data**2, dim=0)
                neg_var = torch.mean(v_neg**2, dim=0)
            grads['v_sigma'] = (neg_var - pos_var) / (2 * self.v_sigma**3)

        # Compute reconstruction error
        reconstruction_error = torch.mean((v_data - v_neg)**2)

        return reconstruction_error, grads

    def update_parameters(self, grads: Dict[str, torch.Tensor]) -> None:
        """
        Update model parameters using gradients with momentum and regularization.

        Args:
            grads: Dictionary of gradients for each parameter
        """
        # Initialize momentum buffers if not exists
        if not hasattr(self, 'momentum_buffers'):
            self.momentum_buffers = {}

        for name, grad in grads.items():
            param = getattr(self, name)

            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Initialize momentum buffer
            if name not in self.momentum_buffers:
                self.momentum_buffers[name] = torch.zeros_like(param)

            # Update momentum buffer
            self.momentum_buffers[name] = (
                self.momentum * self.momentum_buffers[name] +
                self.learning_rate * grad
            )

            # Update parameter
            param.data += self.momentum_buffers[name]

    def compute_sparsity_penalty(self, h_prob: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity penalty for hidden units.

        Args:
            h_prob: Hidden unit probabilities [batch_size, n_hidden]

        Returns:
            sparsity_loss: Sparsity penalty term
        """
        if self.sparsity_target is None:
            return torch.tensor(0.0, device=self.device)

        # Average activation across batch
        avg_activation = torch.mean(h_prob, dim=0)

        # KL divergence penalty: KL(target || actual)
        target = torch.full_like(avg_activation, self.sparsity_target)

        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        kl_div = (
            target * torch.log(target / (avg_activation + eps)) +
            (1 - target) * torch.log((1 - target) / (1 - avg_activation + eps))
        )

        sparsity_loss = self.sparsity_weight * torch.sum(kl_div)
        return sparsity_loss

    def train_batch(
        self,
        batch: torch.Tensor,
        k: int = 1,
        persistent: bool = False
    ) -> Dict[str, float]:
        """
        Train on a single batch of data.

        Args:
            batch: Training batch [batch_size, n_visible]
            k: Number of CD steps
            persistent: Whether to use PCD

        Returns:
            metrics: Dictionary of training metrics
        """
        self.train()

        # Move batch to device
        batch = batch.to(self.device, dtype=self.dtype)

        # Perform contrastive divergence
        reconstruction_error, grads = self.contrastive_divergence(
            batch, k=k, persistent=persistent
        )

        # Compute hidden probabilities for sparsity
        h_prob, _ = self.visible_to_hidden(batch)
        sparsity_loss = self.compute_sparsity_penalty(h_prob)

        # Add sparsity gradient to hidden bias
        if self.sparsity_target is not None and 'h_bias' in grads:
            sparsity_grad = self.sparsity_weight * (
                torch.mean(h_prob, dim=0) - self.sparsity_target
            )
            grads['h_bias'] -= sparsity_grad

        # Update parameters
        self.update_parameters(grads)

        # Compute metrics
        metrics = {
            'reconstruction_error': reconstruction_error.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sparsity': torch.mean(h_prob).item(),
            'weight_norm': torch.norm(self.W).item(),
            'free_energy': torch.mean(self.free_energy(batch)).item()
        }

        # Update training statistics
        for key, value in metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)

        return metrics

    def sample(self, n_samples: int, n_gibbs: int = 1000, init_visible: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate samples from the model using Gibbs sampling.

        Args:
            n_samples: Number of samples to generate
            n_gibbs: Number of Gibbs steps to run
            init_visible: Initial visible state (if None, random initialization)

        Returns:
            samples: Generated samples [n_samples, n_visible]
        """
        self.eval()

        with torch.no_grad():
            if init_visible is None:
                if self.visible_type == "bernoulli":
                    v = torch.bernoulli(torch.full((n_samples, self.n_visible), 0.5, device=self.device))
                else:  # gaussian
                    v = torch.randn(n_samples, self.n_visible, device=self.device)
            else:
                v = init_visible.to(self.device)

            # Run Gibbs sampling
            for _ in range(n_gibbs):
                v, _, _, _ = self.gibbs_step(v)

        return v

    def reconstruct(self, v_data: torch.Tensor, n_gibbs: int = 1) -> torch.Tensor:
        """
        Reconstruct data using the model.

        Args:
            v_data: Input data [batch_size, n_visible]
            n_gibbs: Number of Gibbs steps for reconstruction

        Returns:
            reconstruction: Reconstructed data [batch_size, n_visible]
        """
        self.eval()

        with torch.no_grad():
            v = v_data.to(self.device, dtype=self.dtype)

            for _ in range(n_gibbs):
                v, _, _, _ = self.gibbs_step(v)

        return v

    def get_hidden_representation(self, v_data: torch.Tensor) -> torch.Tensor:
        """
        Get hidden representation of visible data.

        Args:
            v_data: Input data [batch_size, n_visible]

        Returns:
            h_prob: Hidden probabilities [batch_size, n_hidden]
        """
        self.eval()

        with torch.no_grad():
            v = v_data.to(self.device, dtype=self.dtype)
            h_prob, _ = self.visible_to_hidden(v)

        return h_prob

    def save_checkpoint(self, filepath: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'n_visible': self.n_visible,
                'n_hidden': self.n_hidden,
                'visible_type': self.visible_type,
                'hidden_type': self.hidden_type,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'sparsity_target': self.sparsity_target,
                'sparsity_weight': self.sparsity_weight,
                'temperature': self.temperature,
                'use_bias': self.use_bias,
            },
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: Path, device: Optional[torch.device] = None) -> 'RestrictedBoltzmannMachine':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)

        # Create model with saved config
        model = cls(device=device, **checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_stats = checkpoint.get('training_stats', {})

        logger.info(f"Checkpoint loaded from {filepath}")
        return model

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RestrictedBoltzmannMachine("
            f"n_visible={self.n_visible}, "
            f"n_hidden={self.n_hidden}, "
            f"visible_type='{self.visible_type}', "
            f"hidden_type='{self.hidden_type}', "
            f"temperature={self.temperature})"
        )
