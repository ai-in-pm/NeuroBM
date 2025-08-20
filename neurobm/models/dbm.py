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
Deep Boltzmann Machine with layer-wise pretraining and mean-field inference

This module implements DBMs following Salakhutdinov & Hinton (2009) with:
- Layer-wise pretraining using RBMs
- Mean-field variational inference
- Joint fine-tuning with wake-sleep algorithm
- Support for 2-3 layer architectures
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

from .rbm import RestrictedBoltzmannMachine
from .utils import TemperatureScheduler, clip_gradients

logger = logging.getLogger(__name__)


class DeepBoltzmannMachine(nn.Module):
    """
    Deep Boltzmann Machine with layer-wise pretraining.

    Implements a multi-layer Boltzmann machine with undirected connections
    between adjacent layers. Supports 2-3 layer architectures with
    layer-wise pretraining followed by joint fine-tuning.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        visible_type: Literal["bernoulli", "gaussian"] = "bernoulli",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        temperature: float = 1.0,
        mean_field_steps: int = 10,
        mean_field_tolerance: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Deep Boltzmann Machine.

        Args:
            layer_sizes: List of layer sizes [n_visible, n_hidden1, n_hidden2, ...]
            visible_type: Type of visible units ('bernoulli' or 'gaussian')
            learning_rate: Learning rate for joint training
            momentum: Momentum coefficient
            weight_decay: L2 regularization coefficient
            temperature: Temperature for sampling
            mean_field_steps: Maximum steps for mean-field inference
            mean_field_tolerance: Convergence tolerance for mean-field
            device: Device to place tensors on
            dtype: Data type for tensors
        """
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("DBM requires at least 2 layers")
        if len(layer_sizes) > 4:
            raise ValueError("DBM supports at most 4 layers")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.visible_type = visible_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.mean_field_steps = mean_field_steps
        self.mean_field_tolerance = mean_field_tolerance
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        # Initialize parameters
        self._init_parameters()

        # Move to device
        self.to(self.device)

        # Pretraining RBMs (created during pretraining)
        self.pretrain_rbms: List[RestrictedBoltzmannMachine] = []

        # Training statistics
        self.training_stats = {
            'reconstruction_error': [],
            'free_energy': [],
            'mean_field_iterations': []
        }

    def _init_parameters(self) -> None:
        """Initialize DBM parameters."""
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        # Initialize weights between adjacent layers
        for i in range(self.n_layers - 1):
            n_lower = self.layer_sizes[i]
            n_upper = self.layer_sizes[i + 1]

            # Xavier initialization
            std = np.sqrt(2.0 / (n_lower + n_upper))
            weight = nn.Parameter(
                torch.randn(n_lower, n_upper, dtype=self.dtype) * std
            )
            self.weights.append(weight)

        # Initialize biases for each layer
        for i, size in enumerate(self.layer_sizes):
            bias = nn.Parameter(torch.zeros(size, dtype=self.dtype))
            self.biases.append(bias)

        # For Gaussian visible units
        if self.visible_type == "gaussian":
            self.v_sigma = nn.Parameter(
                torch.ones(self.layer_sizes[0], dtype=self.dtype)
            )
        else:
            self.register_parameter('v_sigma', None)

    def energy(self, states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute energy of a configuration across all layers.

        Args:
            states: List of layer states [batch_size, layer_size]

        Returns:
            energy: Energy values [batch_size]
        """
        batch_size = states[0].size(0)
        energy = torch.zeros(batch_size, device=self.device, dtype=self.dtype)

        # Interaction terms between adjacent layers
        for i in range(self.n_layers - 1):
            interaction = -torch.sum(
                states[i].unsqueeze(2) * self.weights[i].unsqueeze(0) * states[i + 1].unsqueeze(1),
                dim=(1, 2)
            )
            energy += interaction

        # Bias terms
        for i, state in enumerate(states):
            bias_term = -torch.sum(state * self.biases[i], dim=1)
            energy += bias_term

        # For Gaussian visible units, add quadratic term
        if self.visible_type == "gaussian" and self.v_sigma is not None:
            quadratic_term = torch.sum(
                (states[0] - self.biases[0])**2 / (2 * self.v_sigma**2),
                dim=1
            )
            energy += quadratic_term

        return energy

    def mean_field_inference(
        self,
        v_data: torch.Tensor,
        init_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Perform mean-field variational inference.

        Args:
            v_data: Visible data [batch_size, n_visible]
            init_states: Initial hidden states (if None, random initialization)

        Returns:
            mean_field_states: List of mean-field probabilities for each layer
            n_iterations: Number of iterations until convergence
        """
        batch_size = v_data.size(0)

        # Ensure no gradients are computed during inference
        with torch.no_grad():
            # Initialize hidden states
            if init_states is None:
                states = [v_data.detach()]  # Visible layer is clamped
                for i in range(1, self.n_layers):
                    # Initialize with sigmoid of bias
                    init_prob = torch.sigmoid(self.biases[i].unsqueeze(0).expand(batch_size, -1))
                    states.append(init_prob)
            else:
                states = [v_data.detach()] + [s.detach() for s in init_states[1:]]

            # Mean-field updates
            for iteration in range(self.mean_field_steps):
                old_states = [s.clone() for s in states]

                # Update each hidden layer
                for layer in range(1, self.n_layers):
                    # Compute input from adjacent layers
                    total_input = self.biases[layer].unsqueeze(0)

                    # Input from layer below
                    if layer > 0:
                        total_input = total_input + torch.matmul(states[layer - 1], self.weights[layer - 1])

                    # Input from layer above
                    if layer < self.n_layers - 1:
                        total_input = total_input + torch.matmul(states[layer + 1], self.weights[layer].t())

                    # Update with sigmoid
                    states[layer] = torch.sigmoid(total_input / self.temperature)

                # Check convergence
                converged = True
                for layer in range(1, self.n_layers):
                    diff = torch.max(torch.abs(states[layer] - old_states[layer]))
                    if diff > self.mean_field_tolerance:
                        converged = False
                        break

                if converged:
                    return states, iteration + 1

            return states, self.mean_field_steps

    def pretrain_layer(
        self,
        layer_idx: int,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        k_steps: int = 1,
        verbose: bool = True
    ) -> RestrictedBoltzmannMachine:
        """
        Pretrain a single layer using RBM.

        Args:
            layer_idx: Index of layer to pretrain (0 = first hidden layer)
            data_loader: Data loader for training
            epochs: Number of training epochs
            k_steps: Number of CD steps
            verbose: Whether to print progress

        Returns:
            Trained RBM for this layer
        """
        if layer_idx >= self.n_layers - 1:
            raise ValueError(f"Layer index {layer_idx} out of range")

        # Determine visible and hidden sizes for this RBM
        if layer_idx == 0:
            n_visible = self.layer_sizes[0]
            visible_type = self.visible_type
        else:
            n_visible = self.layer_sizes[layer_idx]
            visible_type = "bernoulli"  # Hidden layers are always Bernoulli

        n_hidden = self.layer_sizes[layer_idx + 1]

        # Create RBM for this layer
        rbm = RestrictedBoltzmannMachine(
            n_visible=n_visible,
            n_hidden=n_hidden,
            visible_type=visible_type,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            temperature=self.temperature,
            device=self.device,
            dtype=self.dtype
        )

        # Prepare data for this layer
        if layer_idx == 0:
            # First layer trains on raw data
            train_data = data_loader
        else:
            # Higher layers train on representations from lower layers
            train_data = self._get_layer_representations(data_loader, layer_idx)

        # Train RBM
        if verbose:
            logger.info(f"Pretraining layer {layer_idx + 1}/{self.n_layers - 1}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_data:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Extract data from (data, labels) tuple

                metrics = rbm.train_batch(batch, k=k_steps)
                epoch_loss += metrics['reconstruction_error']
                n_batches += 1

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Store pretrained RBM
        if len(self.pretrain_rbms) <= layer_idx:
            self.pretrain_rbms.extend([None] * (layer_idx + 1 - len(self.pretrain_rbms)))
        self.pretrain_rbms[layer_idx] = rbm

        # Initialize DBM weights from RBM
        self._initialize_from_rbm(layer_idx, rbm)

        return rbm

    def _get_layer_representations(
        self,
        data_loader: torch.utils.data.DataLoader,
        target_layer: int
    ) -> List[torch.Tensor]:
        """Get representations at a specific layer using pretrained RBMs."""
        representations = []

        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            # Forward through pretrained RBMs up to target layer
            current_repr = batch.to(self.device, dtype=self.dtype)

            for layer_idx in range(target_layer):
                if layer_idx < len(self.pretrain_rbms) and self.pretrain_rbms[layer_idx] is not None:
                    current_repr = self.pretrain_rbms[layer_idx].get_hidden_representation(current_repr)
                else:
                    raise ValueError(f"Layer {layer_idx} not pretrained yet")

            representations.append(current_repr.cpu())

        return representations

    def _initialize_from_rbm(self, layer_idx: int, rbm: RestrictedBoltzmannMachine) -> None:
        """Initialize DBM weights from pretrained RBM."""
        # Copy weights (with factor of 2 adjustment for DBM as per Salakhutdinov & Hinton)
        if layer_idx == 0 or layer_idx == self.n_layers - 2:
            # First and last layers get full weights
            self.weights[layer_idx].data.copy_(rbm.W.data)
        else:
            # Middle layers get half weights (due to double counting in DBM)
            self.weights[layer_idx].data.copy_(rbm.W.data * 0.5)

        # Copy biases
        if rbm.v_bias is not None:
            self.biases[layer_idx].data.copy_(rbm.v_bias.data)
        if rbm.h_bias is not None:
            self.biases[layer_idx + 1].data.copy_(rbm.h_bias.data)

        # Copy variance for Gaussian visible units
        if layer_idx == 0 and self.visible_type == "gaussian" and rbm.v_sigma is not None:
            self.v_sigma.data.copy_(rbm.v_sigma.data)

    def pretrain_all_layers(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs_per_layer: Union[int, List[int]] = 100,
        k_steps: int = 1,
        verbose: bool = True
    ) -> List[RestrictedBoltzmannMachine]:
        """
        Pretrain all layers sequentially.

        Args:
            data_loader: Training data loader
            epochs_per_layer: Epochs for each layer (int or list)
            k_steps: CD steps for pretraining
            verbose: Whether to print progress

        Returns:
            List of pretrained RBMs
        """
        if isinstance(epochs_per_layer, int):
            epochs_per_layer = [epochs_per_layer] * (self.n_layers - 1)
        elif len(epochs_per_layer) != self.n_layers - 1:
            raise ValueError("epochs_per_layer must match number of hidden layers")

        rbms = []
        for layer_idx in range(self.n_layers - 1):
            rbm = self.pretrain_layer(
                layer_idx=layer_idx,
                data_loader=data_loader,
                epochs=epochs_per_layer[layer_idx],
                k_steps=k_steps,
                verbose=verbose
            )
            rbms.append(rbm)

        if verbose:
            logger.info("Layer-wise pretraining completed")

        return rbms

    def joint_train_batch(
        self,
        batch: torch.Tensor,
        n_gibbs: int = 5,
        learning_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Joint training using wake-sleep algorithm.

        Args:
            batch: Training batch [batch_size, n_visible]
            n_gibbs: Number of Gibbs steps for negative phase
            learning_rate: Learning rate (if None, use default)

        Returns:
            metrics: Training metrics
        """
        self.train()
        lr = learning_rate or self.learning_rate

        batch = batch.to(self.device, dtype=self.dtype)
        batch_size = batch.size(0)

        # Positive phase: mean-field inference on data
        pos_states, mf_iterations = self.mean_field_inference(batch)

        # Negative phase: Gibbs sampling
        neg_states = self._gibbs_sampling(batch_size, n_gibbs)

        # Compute gradients
        grads = self._compute_joint_gradients(pos_states, neg_states)

        # Update parameters
        self._update_parameters(grads, lr)

        # Compute metrics
        reconstruction = self._reconstruct_from_states(pos_states)
        reconstruction_error = torch.mean((batch - reconstruction)**2)

        metrics = {
            'reconstruction_error': reconstruction_error.item(),
            'mean_field_iterations': mf_iterations,
            'free_energy': torch.mean(self._compute_free_energy(batch)).item()
        }

        # Update training statistics
        for key, value in metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)

        return metrics

    def _gibbs_sampling(self, batch_size: int, n_steps: int) -> List[torch.Tensor]:
        """Perform Gibbs sampling to generate negative samples."""
        # Initialize random states
        states = []
        for i, size in enumerate(self.layer_sizes):
            if i == 0 and self.visible_type == "gaussian":
                state = torch.randn(batch_size, size, device=self.device, dtype=self.dtype)
            else:
                state = torch.bernoulli(
                    torch.full((batch_size, size), 0.5, device=self.device, dtype=self.dtype)
                )
            states.append(state)

        # Run Gibbs sampling
        for _ in range(n_steps):
            # Sample each layer conditioned on its neighbors
            for layer in range(self.n_layers):
                if layer == 0:
                    # Visible layer
                    total_input = self.biases[0].unsqueeze(0)
                    if self.n_layers > 1:
                        total_input += torch.matmul(states[1], self.weights[0].t())

                    if self.visible_type == "bernoulli":
                        probs = torch.sigmoid(total_input / self.temperature)
                        states[0] = torch.bernoulli(probs)
                    else:  # gaussian
                        if self.v_sigma is not None:
                            noise = torch.randn_like(total_input) * self.v_sigma * self.temperature
                            states[0] = total_input + noise
                        else:
                            states[0] = total_input + torch.randn_like(total_input) * self.temperature

                elif layer == self.n_layers - 1:
                    # Top layer
                    total_input = self.biases[layer].unsqueeze(0)
                    total_input += torch.matmul(states[layer - 1], self.weights[layer - 1])

                    probs = torch.sigmoid(total_input / self.temperature)
                    states[layer] = torch.bernoulli(probs)

                else:
                    # Middle layers
                    total_input = self.biases[layer].unsqueeze(0)
                    total_input += torch.matmul(states[layer - 1], self.weights[layer - 1])
                    total_input += torch.matmul(states[layer + 1], self.weights[layer].t())

                    probs = torch.sigmoid(total_input / self.temperature)
                    states[layer] = torch.bernoulli(probs)

        return states

    def _compute_joint_gradients(
        self,
        pos_states: List[torch.Tensor],
        neg_states: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for joint training."""
        grads = {}
        batch_size = pos_states[0].size(0)

        # Weight gradients
        for i in range(self.n_layers - 1):
            pos_grad = torch.matmul(pos_states[i].t(), pos_states[i + 1])
            neg_grad = torch.matmul(neg_states[i].t(), neg_states[i + 1])
            grads[f'weights_{i}'] = (pos_grad - neg_grad) / batch_size

        # Bias gradients
        for i in range(self.n_layers):
            pos_bias = torch.mean(pos_states[i], dim=0)
            neg_bias = torch.mean(neg_states[i], dim=0)
            grads[f'biases_{i}'] = pos_bias - neg_bias

        # Variance gradient for Gaussian visible units
        if self.visible_type == "gaussian" and self.v_sigma is not None:
            pos_var = torch.mean((pos_states[0] - self.biases[0])**2, dim=0)
            neg_var = torch.mean((neg_states[0] - self.biases[0])**2, dim=0)
            grads['v_sigma'] = (neg_var - pos_var) / (2 * self.v_sigma**3)

        return grads

    def _update_parameters(self, grads: Dict[str, torch.Tensor], lr: float) -> None:
        """Update parameters using gradients."""
        # Initialize momentum buffers if not exists
        if not hasattr(self, 'momentum_buffers'):
            self.momentum_buffers = {}

        # Update weights
        for i in range(self.n_layers - 1):
            grad_key = f'weights_{i}'
            if grad_key in grads:
                grad = grads[grad_key]

                # Add weight decay
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * self.weights[i]

                # Momentum update
                if grad_key not in self.momentum_buffers:
                    self.momentum_buffers[grad_key] = torch.zeros_like(self.weights[i])

                self.momentum_buffers[grad_key] = (
                    self.momentum * self.momentum_buffers[grad_key] + lr * grad
                )
                self.weights[i].data += self.momentum_buffers[grad_key]

        # Update biases
        for i in range(self.n_layers):
            grad_key = f'biases_{i}'
            if grad_key in grads:
                grad = grads[grad_key]

                if grad_key not in self.momentum_buffers:
                    self.momentum_buffers[grad_key] = torch.zeros_like(self.biases[i])

                self.momentum_buffers[grad_key] = (
                    self.momentum * self.momentum_buffers[grad_key] + lr * grad
                )
                self.biases[i].data += self.momentum_buffers[grad_key]

        # Update variance
        if 'v_sigma' in grads and self.v_sigma is not None:
            grad = grads['v_sigma']

            if 'v_sigma' not in self.momentum_buffers:
                self.momentum_buffers['v_sigma'] = torch.zeros_like(self.v_sigma)

            self.momentum_buffers['v_sigma'] = (
                self.momentum * self.momentum_buffers['v_sigma'] + lr * grad
            )
            self.v_sigma.data += self.momentum_buffers['v_sigma']

    def _compute_free_energy(self, v_data: torch.Tensor) -> torch.Tensor:
        """Compute free energy using mean-field approximation."""
        states, _ = self.mean_field_inference(v_data)

        # Approximate free energy using mean-field states
        energy = self.energy(states)

        # Entropy terms (approximate)
        entropy = 0.0
        for i in range(1, self.n_layers):  # Skip visible layer
            h = states[i]
            entropy += torch.sum(
                -h * torch.log(h + 1e-8) - (1 - h) * torch.log(1 - h + 1e-8),
                dim=1
            )

        return energy - entropy

    def _reconstruct_from_states(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct visible data from hidden states."""
        # Use mean-field states to compute reconstruction
        if self.n_layers > 1:
            activation = torch.matmul(states[1], self.weights[0].t()) + self.biases[0]
            if self.visible_type == "bernoulli":
                return torch.sigmoid(activation)
            else:  # gaussian
                return activation
        else:
            return states[0]

    def sample(
        self,
        n_samples: int,
        n_gibbs: int = 1000,
        return_all_layers: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples from the model.

        Args:
            n_samples: Number of samples to generate
            n_gibbs: Number of Gibbs steps
            return_all_layers: Whether to return all layer states

        Returns:
            samples: Generated visible samples or all layer states
        """
        self.eval()

        with torch.no_grad():
            states = self._gibbs_sampling(n_samples, n_gibbs)

        if return_all_layers:
            return states
        else:
            return states[0]  # Return only visible layer

    def reconstruct(
        self,
        v_data: torch.Tensor,
        use_mean_field: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct data using the model.

        Args:
            v_data: Input data [batch_size, n_visible]
            use_mean_field: Whether to use mean-field inference

        Returns:
            reconstruction: Reconstructed data
        """
        self.eval()

        with torch.no_grad():
            if use_mean_field:
                states, _ = self.mean_field_inference(v_data)
                return self._reconstruct_from_states(states)
            else:
                # Use Gibbs sampling for reconstruction
                batch_size = v_data.size(0)
                states = [v_data.to(self.device, dtype=self.dtype)]

                # Initialize hidden layers randomly
                for i in range(1, self.n_layers):
                    state = torch.bernoulli(
                        torch.full((batch_size, self.layer_sizes[i]), 0.5,
                                 device=self.device, dtype=self.dtype)
                    )
                    states.append(state)

                # Run a few Gibbs steps
                states = self._gibbs_sampling_with_clamped_visible(states, n_steps=10)
                return self._reconstruct_from_states(states)

    def _gibbs_sampling_with_clamped_visible(
        self,
        states: List[torch.Tensor],
        n_steps: int
    ) -> List[torch.Tensor]:
        """Gibbs sampling with visible layer clamped."""
        for _ in range(n_steps):
            # Sample hidden layers only (visible is clamped)
            for layer in range(1, self.n_layers):
                if layer == self.n_layers - 1:
                    # Top layer
                    total_input = self.biases[layer].unsqueeze(0)
                    total_input += torch.matmul(states[layer - 1], self.weights[layer - 1])
                else:
                    # Middle layers
                    total_input = self.biases[layer].unsqueeze(0)
                    total_input += torch.matmul(states[layer - 1], self.weights[layer - 1])
                    total_input += torch.matmul(states[layer + 1], self.weights[layer].t())

                probs = torch.sigmoid(total_input / self.temperature)
                states[layer] = torch.bernoulli(probs)

        return states

    def get_hidden_representations(
        self,
        v_data: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get hidden representations using mean-field inference.

        Args:
            v_data: Input data [batch_size, n_visible]
            layer_idx: Specific layer to return (if None, return all hidden layers)

        Returns:
            representations: Hidden layer representations
        """
        self.eval()

        with torch.no_grad():
            states, _ = self.mean_field_inference(v_data)

            if layer_idx is not None:
                if layer_idx < 1 or layer_idx >= self.n_layers:
                    raise ValueError(f"Layer index {layer_idx} out of range")
                return states[layer_idx]
            else:
                return states[1:]  # Return all hidden layers

    def save_checkpoint(self, filepath: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'layer_sizes': self.layer_sizes,
                'visible_type': self.visible_type,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'temperature': self.temperature,
                'mean_field_steps': self.mean_field_steps,
                'mean_field_tolerance': self.mean_field_tolerance,
            },
            'training_stats': self.training_stats,
            'pretrain_rbms': [rbm.state_dict() if rbm is not None else None
                            for rbm in self.pretrain_rbms]
        }
        torch.save(checkpoint, filepath)
        logger.info(f"DBM checkpoint saved to {filepath}")

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Path,
        device: Optional[torch.device] = None
    ) -> 'DeepBoltzmannMachine':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)

        # Create model with saved config
        model = cls(device=device, **checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_stats = checkpoint.get('training_stats', {})

        # Restore pretrained RBMs if available
        pretrain_rbm_states = checkpoint.get('pretrain_rbms', [])
        model.pretrain_rbms = []

        for i, rbm_state in enumerate(pretrain_rbm_states):
            if rbm_state is not None:
                # Reconstruct RBM
                if i == 0:
                    visible_type = model.visible_type
                else:
                    visible_type = "bernoulli"

                rbm = RestrictedBoltzmannMachine(
                    n_visible=model.layer_sizes[i],
                    n_hidden=model.layer_sizes[i + 1],
                    visible_type=visible_type,
                    device=device
                )
                rbm.load_state_dict(rbm_state)
                model.pretrain_rbms.append(rbm)
            else:
                model.pretrain_rbms.append(None)

        logger.info(f"DBM checkpoint loaded from {filepath}")
        return model

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DeepBoltzmannMachine("
            f"layer_sizes={self.layer_sizes}, "
            f"visible_type='{self.visible_type}', "
            f"temperature={self.temperature})"
        )
