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
Conditional RBM for temporal sequence modeling.

This module implements Conditional Restricted Boltzmann Machines (CRBMs) for
modeling temporal sequences. CRBMs extend standard RBMs by conditioning the
hidden units on previous visible states, making them suitable for sequence
modeling tasks.

Key features:
- Temporal dependencies through autoregressive connections
- Support for variable-length sequences
- Contrastive divergence training adapted for sequences
- Prediction and generation capabilities
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

from .utils import sigmoid_with_temperature, sample_bernoulli, clip_gradients

logger = logging.getLogger(__name__)


class ConditionalRBM(nn.Module):
    """
    Conditional Restricted Boltzmann Machine for temporal sequence modeling.

    The CRBM extends the standard RBM by adding directed connections from
    previous visible states to current hidden states, enabling the model
    to capture temporal dependencies in sequential data.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        n_history: int = 3,
        visible_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        temperature: float = 1.0,
        use_bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Conditional RBM.

        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            n_history: Number of previous time steps to condition on
            visible_type: Type of visible units ("bernoulli" or "gaussian")
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient for parameter updates
            weight_decay: L2 regularization coefficient
            temperature: Temperature for sampling
            use_bias: Whether to use bias terms
            device: Device to run computations on
            dtype: Data type for parameters
        """
        super().__init__()

        if n_visible <= 0 or n_hidden <= 0:
            raise ValueError("Number of units must be positive")
        if n_history <= 0:
            raise ValueError("History length must be positive")
        if visible_type not in ["bernoulli", "gaussian"]:
            raise ValueError("visible_type must be 'bernoulli' or 'gaussian'")

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_history = n_history
        self.visible_type = visible_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.use_bias = use_bias
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        # Initialize parameters
        self._init_parameters()

        # Move to device
        self.to(self.device)

        # Training statistics
        self.training_stats = {
            'reconstruction_error': [],
            'prediction_error': [],
            'free_energy': []
        }

    def _init_parameters(self) -> None:
        """Initialize model parameters."""
        # Standard RBM weights (visible to hidden)
        std_w = np.sqrt(2.0 / (self.n_visible + self.n_hidden))
        self.W = nn.Parameter(
            torch.randn(self.n_visible, self.n_hidden, dtype=self.dtype) * std_w
        )

        # Autoregressive weights (history to hidden)
        std_a = np.sqrt(2.0 / (self.n_visible * self.n_history + self.n_hidden))
        self.A = nn.Parameter(
            torch.randn(self.n_visible * self.n_history, self.n_hidden, dtype=self.dtype) * std_a
        )

        # Bias terms
        if self.use_bias:
            self.v_bias = nn.Parameter(torch.zeros(self.n_visible, dtype=self.dtype))
            self.h_bias = nn.Parameter(torch.zeros(self.n_hidden, dtype=self.dtype))
        else:
            self.register_parameter('v_bias', None)
            self.register_parameter('h_bias', None)

        # Momentum buffers
        self.register_buffer('W_momentum', torch.zeros_like(self.W))
        self.register_buffer('A_momentum', torch.zeros_like(self.A))
        if self.use_bias:
            self.register_buffer('v_bias_momentum', torch.zeros_like(self.v_bias))
            self.register_buffer('h_bias_momentum', torch.zeros_like(self.h_bias))

    def _prepare_history(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Prepare history context from sequences.

        Args:
            sequences: Input sequences [batch_size, seq_len, n_visible]

        Returns:
            history: Flattened history context [batch_size, seq_len-n_history, n_visible*n_history]
        """
        batch_size, seq_len, n_visible = sequences.shape

        if seq_len <= self.n_history:
            raise ValueError(f"Sequence length {seq_len} must be > history length {self.n_history}")

        # Create sliding windows of history
        history_list = []
        for t in range(self.n_history, seq_len):
            # Get history from t-n_history to t-1
            hist = sequences[:, t-self.n_history:t, :].reshape(batch_size, -1)
            history_list.append(hist)

        return torch.stack(history_list, dim=1)  # [batch_size, seq_len-n_history, n_visible*n_history]

    def visible_to_hidden(
        self,
        v_data: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden probabilities and samples given visible data.

        Args:
            v_data: Visible data [batch_size, n_visible] or [batch_size, seq_len, n_visible]
            history: History context [batch_size, n_visible*n_history] or None

        Returns:
            h_prob: Hidden probabilities [batch_size, n_hidden]
            h_sample: Hidden samples [batch_size, n_hidden]
        """
        # Handle sequence input
        if v_data.dim() == 3:
            batch_size, seq_len, n_visible = v_data.shape
            v_data = v_data.reshape(-1, n_visible)
            if history is not None:
                history = history.reshape(-1, self.n_visible * self.n_history)

        # Compute hidden activation
        h_activation = torch.matmul(v_data, self.W)

        # Add autoregressive contribution
        if history is not None:
            h_activation += torch.matmul(history, self.A)

        # Add bias
        if self.use_bias:
            h_activation += self.h_bias

        # Compute probabilities and samples
        h_prob = sigmoid_with_temperature(h_activation, self.temperature)
        h_sample = sample_bernoulli(h_prob)

        return h_prob, h_sample

    def hidden_to_visible(self, h_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visible probabilities and samples given hidden data.

        Args:
            h_data: Hidden data [batch_size, n_hidden]

        Returns:
            v_prob: Visible probabilities [batch_size, n_visible]
            v_sample: Visible samples [batch_size, n_visible]
        """
        # Compute visible activation
        v_activation = torch.matmul(h_data, self.W.t())

        # Add bias
        if self.use_bias:
            v_activation += self.v_bias

        # Compute probabilities and samples based on visible type
        if self.visible_type == "bernoulli":
            v_prob = sigmoid_with_temperature(v_activation, self.temperature)
            v_sample = sample_bernoulli(v_prob)
        else:  # gaussian
            v_prob = v_activation  # Mean of Gaussian
            v_sample = torch.normal(v_prob, std=1.0)

        return v_prob, v_sample

    def energy(
        self,
        v_data: torch.Tensor,
        h_data: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy of visible-hidden configuration.

        Args:
            v_data: Visible data [batch_size, n_visible]
            h_data: Hidden data [batch_size, n_hidden]
            history: History context [batch_size, n_visible*n_history]

        Returns:
            energy: Energy values [batch_size]
        """
        # Visible-hidden interaction
        vh_term = torch.sum(v_data * torch.matmul(h_data, self.W.t()), dim=1)

        # History-hidden interaction
        ah_term = 0.0
        if history is not None:
            ah_term = torch.sum(history * torch.matmul(h_data, self.A.t()), dim=1)

        # Bias terms
        v_bias_term = 0.0
        h_bias_term = 0.0
        if self.use_bias:
            v_bias_term = torch.sum(v_data * self.v_bias, dim=1)
            h_bias_term = torch.sum(h_data * self.h_bias, dim=1)

        energy = -(vh_term + ah_term + v_bias_term + h_bias_term)
        return energy

    def free_energy(self, v_data: torch.Tensor, history: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute free energy of visible configuration.

        Args:
            v_data: Visible data [batch_size, n_visible]
            history: History context [batch_size, n_visible*n_history]

        Returns:
            free_energy: Free energy values [batch_size]
        """
        # Visible bias term
        v_bias_term = 0.0
        if self.use_bias:
            v_bias_term = torch.sum(v_data * self.v_bias, dim=1)

        # Hidden activation
        h_activation = torch.matmul(v_data, self.W)
        if history is not None:
            h_activation += torch.matmul(history, self.A)
        if self.use_bias:
            h_activation += self.h_bias

        # Log-sum-exp of hidden activations
        hidden_term = torch.sum(F.softplus(h_activation), dim=1)

        free_energy = -(v_bias_term + hidden_term)
        return free_energy

    def gibbs_step(
        self,
        v_data: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one step of Gibbs sampling.

        Args:
            v_data: Current visible state [batch_size, n_visible]
            history: History context [batch_size, n_visible*n_history]

        Returns:
            v_new: New visible state [batch_size, n_visible]
            h_prob: Hidden probabilities [batch_size, n_hidden]
            h_sample: Hidden samples [batch_size, n_hidden]
            v_prob: Visible probabilities [batch_size, n_visible]
        """
        # Sample hidden given visible and history
        h_prob, h_sample = self.visible_to_hidden(v_data, history)

        # Sample visible given hidden
        v_prob, v_new = self.hidden_to_visible(h_sample)

        return v_new, h_prob, h_sample, v_prob

    def contrastive_divergence_sequence(
        self,
        sequences: torch.Tensor,
        k: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform contrastive divergence on sequences.

        Args:
            sequences: Input sequences [batch_size, seq_len, n_visible]
            k: Number of CD steps

        Returns:
            loss: Reconstruction loss
            grads: Dictionary of gradients
        """
        batch_size, seq_len, n_visible = sequences.shape

        # Prepare history and targets
        history = self._prepare_history(sequences)  # [batch_size, seq_len-n_history, n_visible*n_history]
        targets = sequences[:, self.n_history:, :]  # [batch_size, seq_len-n_history, n_visible]

        # Flatten for processing
        history_flat = history.reshape(-1, self.n_visible * self.n_history)
        targets_flat = targets.reshape(-1, self.n_visible)

        # Positive phase
        h_pos_prob, h_pos_sample = self.visible_to_hidden(targets_flat, history_flat)

        # Negative phase - start from targets
        v_neg = targets_flat.clone()
        for _ in range(k):
            v_neg, h_neg_prob, h_neg_sample, v_neg_prob = self.gibbs_step(v_neg, history_flat)

        # Final negative phase statistics
        h_neg_prob, h_neg_sample = self.visible_to_hidden(v_neg, history_flat)

        # Compute gradients
        grads = {}
        n_samples = targets_flat.size(0)

        # Standard RBM weight gradient
        pos_grad_w = torch.matmul(targets_flat.t(), h_pos_prob)
        neg_grad_w = torch.matmul(v_neg.t(), h_neg_prob)
        grads['W'] = (pos_grad_w - neg_grad_w) / n_samples

        # Autoregressive weight gradient
        pos_grad_a = torch.matmul(history_flat.t(), h_pos_prob)
        neg_grad_a = torch.matmul(history_flat.t(), h_neg_prob)
        grads['A'] = (pos_grad_a - neg_grad_a) / n_samples

        if self.use_bias:
            # Bias gradients
            grads['v_bias'] = torch.mean(targets_flat - v_neg, dim=0)
            grads['h_bias'] = torch.mean(h_pos_prob - h_neg_prob, dim=0)

        # Compute reconstruction error
        reconstruction_error = F.mse_loss(v_neg, targets_flat)

        return reconstruction_error, grads

    def update_parameters(self, grads: Dict[str, torch.Tensor]) -> None:
        """
        Update model parameters using gradients with momentum.

        Args:
            grads: Dictionary of gradients for each parameter
        """
        with torch.no_grad():
            # Update W with momentum
            if 'W' in grads:
                grad_w = grads['W'] + self.weight_decay * self.W
                grad_w = clip_gradients(grad_w, max_norm=5.0)
                self.W_momentum = self.momentum * self.W_momentum + self.learning_rate * grad_w
                self.W += self.W_momentum

            # Update A with momentum
            if 'A' in grads:
                grad_a = grads['A'] + self.weight_decay * self.A
                grad_a = clip_gradients(grad_a, max_norm=5.0)
                self.A_momentum = self.momentum * self.A_momentum + self.learning_rate * grad_a
                self.A += self.A_momentum

            # Update biases
            if self.use_bias:
                if 'v_bias' in grads:
                    grad_vb = clip_gradients(grads['v_bias'], max_norm=5.0)
                    self.v_bias_momentum = self.momentum * self.v_bias_momentum + self.learning_rate * grad_vb
                    self.v_bias += self.v_bias_momentum

                if 'h_bias' in grads:
                    grad_hb = clip_gradients(grads['h_bias'], max_norm=5.0)
                    self.h_bias_momentum = self.momentum * self.h_bias_momentum + self.learning_rate * grad_hb
                    self.h_bias += self.h_bias_momentum

    def train_batch(
        self,
        sequences: torch.Tensor,
        k: int = 1
    ) -> Dict[str, float]:
        """
        Train on a batch of sequences.

        Args:
            sequences: Training sequences [batch_size, seq_len, n_visible]
            k: Number of CD steps

        Returns:
            metrics: Dictionary of training metrics
        """
        self.train()

        # Move to device
        sequences = sequences.to(self.device, dtype=self.dtype)

        # Perform contrastive divergence
        reconstruction_error, grads = self.contrastive_divergence_sequence(sequences, k=k)

        # Update parameters
        self.update_parameters(grads)

        # Compute metrics
        history = self._prepare_history(sequences)
        targets = sequences[:, self.n_history:, :]
        free_energy = torch.mean(self.free_energy(
            targets.reshape(-1, self.n_visible),
            history.reshape(-1, self.n_visible * self.n_history)
        ))

        metrics = {
            'reconstruction_error': reconstruction_error.item(),
            'free_energy': free_energy.item(),
            'weight_norm_w': torch.norm(self.W).item(),
            'weight_norm_a': torch.norm(self.A).item()
        }

        # Update training statistics
        self.training_stats['reconstruction_error'].append(reconstruction_error.item())
        self.training_stats['free_energy'].append(free_energy.item())

        return metrics

    def predict_next(
        self,
        history_sequence: torch.Tensor,
        n_steps: int = 1,
        n_gibbs: int = 10
    ) -> torch.Tensor:
        """
        Predict next steps in a sequence.

        Args:
            history_sequence: History sequence [batch_size, seq_len, n_visible]
            n_steps: Number of steps to predict
            n_gibbs: Number of Gibbs steps for each prediction

        Returns:
            predictions: Predicted sequence [batch_size, n_steps, n_visible]
        """
        self.eval()

        with torch.no_grad():
            batch_size, seq_len, n_visible = history_sequence.shape

            if seq_len < self.n_history:
                raise ValueError(f"History sequence length {seq_len} must be >= {self.n_history}")

            # Initialize with history
            current_sequence = history_sequence.clone()
            predictions = []

            for step in range(n_steps):
                # Get recent history
                recent_history = current_sequence[:, -self.n_history:, :].reshape(
                    batch_size, -1
                )

                # Start with random visible state
                v_current = torch.rand(batch_size, n_visible, device=self.device, dtype=self.dtype)

                # Run Gibbs sampling to get prediction
                for _ in range(n_gibbs):
                    v_current, _, _, _ = self.gibbs_step(v_current, recent_history)

                predictions.append(v_current.unsqueeze(1))

                # Update current sequence with prediction
                current_sequence = torch.cat([current_sequence, v_current.unsqueeze(1)], dim=1)

            return torch.cat(predictions, dim=1)

    def generate_sequence(
        self,
        length: int,
        batch_size: int = 1,
        n_gibbs: int = 10,
        initial_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate sequences from the model.

        Args:
            length: Length of sequences to generate
            batch_size: Number of sequences to generate
            n_gibbs: Number of Gibbs steps per time step
            initial_sequence: Initial sequence [batch_size, n_history, n_visible]

        Returns:
            sequences: Generated sequences [batch_size, length, n_visible]
        """
        self.eval()

        with torch.no_grad():
            # Initialize sequence
            if initial_sequence is not None:
                if initial_sequence.size(1) != self.n_history:
                    raise ValueError(f"Initial sequence must have length {self.n_history}")
                current_sequence = initial_sequence.clone()
            else:
                # Random initialization
                current_sequence = torch.rand(
                    batch_size, self.n_history, self.n_visible,
                    device=self.device, dtype=self.dtype
                )

            # Generate remaining steps
            remaining_steps = length - self.n_history
            if remaining_steps > 0:
                predictions = self.predict_next(current_sequence, remaining_steps, n_gibbs)
                current_sequence = torch.cat([current_sequence, predictions], dim=1)

            return current_sequence

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'n_visible': self.n_visible,
                'n_hidden': self.n_hidden,
                'n_history': self.n_history,
                'visible_type': self.visible_type,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'temperature': self.temperature,
                'use_bias': self.use_bias
            },
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, filepath)
        logger.info(f"CRBM checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path], device: Optional[torch.device] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)

        # Create model with saved config
        model = cls(device=device, **checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_stats = checkpoint['training_stats']

        logger.info(f"CRBM loaded from {filepath}")
        return model

    def __repr__(self) -> str:
        """String representation."""
        return (f"ConditionalRBM(n_visible={self.n_visible}, n_hidden={self.n_hidden}, "
                f"n_history={self.n_history}, visible_type='{self.visible_type}')")
