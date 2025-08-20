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
Latent space traversal and interpolation tools

This module provides tools for exploring latent spaces in Boltzmann machines:
- Latent space traversal and interpolation
- Counterfactual analysis
- Feature manipulation and editing
- Latent space visualization
- Semantic direction discovery
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

from ..models.rbm import RestrictedBoltzmannMachine
from ..models.dbm import DeepBoltzmannMachine

logger = logging.getLogger(__name__)


class LatentTraverser:
    """
    Latent space traversal and interpolation for Boltzmann machines.

    Provides tools to explore and manipulate latent representations
    for interpretability and counterfactual analysis.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine],
        device: Optional[torch.device] = None
    ):
        """
        Initialize latent traverser.

        Args:
            model: Boltzmann machine model
            device: Device for computations
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        # Cache for computed directions and traversals
        self._direction_cache = {}
        self._traversal_cache = {}

    def linear_interpolation(
        self,
        start_data: torch.Tensor,
        end_data: torch.Tensor,
        n_steps: int = 10,
        interpolation_space: str = "latent"
    ) -> torch.Tensor:
        """
        Perform linear interpolation between two data points.

        Args:
            start_data: Starting data point [n_features] or [1, n_features]
            end_data: Ending data point [n_features] or [1, n_features]
            n_steps: Number of interpolation steps
            interpolation_space: Space for interpolation ('latent', 'visible')

        Returns:
            interpolated_sequence: Interpolated sequence [n_steps, n_features]
        """
        start_data = start_data.to(self.device)
        end_data = end_data.to(self.device)

        # Ensure batch dimension
        if start_data.dim() == 1:
            start_data = start_data.unsqueeze(0)
        if end_data.dim() == 1:
            end_data = end_data.unsqueeze(0)

        if interpolation_space == "latent":
            # Interpolate in latent space
            start_latent = self._encode_to_latent(start_data)
            end_latent = self._encode_to_latent(end_data)

            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, n_steps, device=self.device)
            interpolated_latents = []

            for alpha in alphas:
                interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
                interpolated_latents.append(interpolated_latent)

            interpolated_latents = torch.cat(interpolated_latents, dim=0)

            # Decode back to visible space
            interpolated_sequence = self._decode_from_latent(interpolated_latents)

        elif interpolation_space == "visible":
            # Direct interpolation in visible space
            alphas = torch.linspace(0, 1, n_steps, device=self.device)
            interpolated_sequence = []

            for alpha in alphas:
                interpolated_point = (1 - alpha) * start_data + alpha * end_data
                interpolated_sequence.append(interpolated_point)

            interpolated_sequence = torch.cat(interpolated_sequence, dim=0)

        else:
            raise ValueError(f"Unknown interpolation space: {interpolation_space}")

        return interpolated_sequence

    def spherical_interpolation(
        self,
        start_data: torch.Tensor,
        end_data: torch.Tensor,
        n_steps: int = 10
    ) -> torch.Tensor:
        """
        Perform spherical interpolation (SLERP) in latent space.

        Args:
            start_data: Starting data point
            end_data: Ending data point
            n_steps: Number of interpolation steps

        Returns:
            interpolated_sequence: Spherically interpolated sequence
        """
        start_data = start_data.to(self.device)
        end_data = end_data.to(self.device)

        # Ensure batch dimension
        if start_data.dim() == 1:
            start_data = start_data.unsqueeze(0)
        if end_data.dim() == 1:
            end_data = end_data.unsqueeze(0)

        # Encode to latent space
        start_latent = self._encode_to_latent(start_data)
        end_latent = self._encode_to_latent(end_data)

        # Normalize vectors
        start_norm = F.normalize(start_latent, p=2, dim=1)
        end_norm = F.normalize(end_latent, p=2, dim=1)

        # Compute angle between vectors
        dot_product = torch.sum(start_norm * end_norm, dim=1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
        omega = torch.acos(dot_product)

        # SLERP interpolation
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        interpolated_latents = []

        for alpha in alphas:
            if torch.abs(omega) < 1e-6:
                # Vectors are nearly parallel, use linear interpolation
                interpolated_latent = (1 - alpha) * start_latent + alpha * end_latent
            else:
                # Spherical interpolation
                sin_omega = torch.sin(omega)
                interpolated_latent = (
                    torch.sin((1 - alpha) * omega) / sin_omega * start_latent +
                    torch.sin(alpha * omega) / sin_omega * end_latent
                )

            interpolated_latents.append(interpolated_latent)

        interpolated_latents = torch.cat(interpolated_latents, dim=0)

        # Decode back to visible space
        interpolated_sequence = self._decode_from_latent(interpolated_latents)

        return interpolated_sequence

    def latent_traversal(
        self,
        base_data: torch.Tensor,
        direction: torch.Tensor,
        n_steps: int = 10,
        step_size: float = 1.0,
        bidirectional: bool = True
    ) -> torch.Tensor:
        """
        Traverse latent space in a specific direction.

        Args:
            base_data: Base data point to start traversal from
            direction: Direction vector in latent space
            n_steps: Number of steps in each direction
            step_size: Size of each step
            bidirectional: Whether to traverse in both directions

        Returns:
            traversal_sequence: Sequence of traversed points
        """
        base_data = base_data.to(self.device)
        direction = direction.to(self.device)

        # Ensure batch dimension
        if base_data.dim() == 1:
            base_data = base_data.unsqueeze(0)

        # Encode base data to latent space
        base_latent = self._encode_to_latent(base_data)

        # Normalize direction
        direction_norm = F.normalize(direction, p=2, dim=-1)

        # Generate traversal steps
        if bidirectional:
            steps = torch.linspace(-n_steps * step_size, n_steps * step_size, 2 * n_steps + 1, device=self.device)
        else:
            steps = torch.linspace(0, n_steps * step_size, n_steps + 1, device=self.device)

        traversed_latents = []
        for step in steps:
            traversed_latent = base_latent + step * direction_norm
            traversed_latents.append(traversed_latent)

        traversed_latents = torch.cat(traversed_latents, dim=0)

        # Decode back to visible space
        traversal_sequence = self._decode_from_latent(traversed_latents)

        return traversal_sequence

    def discover_semantic_directions(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        method: str = "mean_difference",
        n_directions: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Discover semantic directions in latent space.

        Args:
            data: Input data [n_samples, n_features]
            labels: Labels for discovering directions [n_samples]
            method: Method for direction discovery ('mean_difference', 'pca', 'linear_classifier')
            n_directions: Number of directions to discover

        Returns:
            directions: Dictionary mapping direction names to direction vectors
        """
        cache_key = f"{method}_{n_directions}_{data.shape}_{torch.sum(labels).item()}"
        if cache_key in self._direction_cache:
            return self._direction_cache[cache_key]

        data = data.to(self.device)
        labels = labels.to(self.device)

        # Encode data to latent space
        latent_data = self._encode_to_latent(data)

        directions = {}

        if method == "mean_difference":
            directions = self._discover_mean_difference_directions(latent_data, labels)
        elif method == "pca":
            directions = self._discover_pca_directions(latent_data, n_directions)
        elif method == "linear_classifier":
            directions = self._discover_classifier_directions(latent_data, labels)
        else:
            raise ValueError(f"Unknown direction discovery method: {method}")

        self._direction_cache[cache_key] = directions
        return directions

    def _encode_to_latent(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data to latent space."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(self.model, RestrictedBoltzmannMachine):
                latent_probs, _ = self.model.visible_to_hidden(data)
                return latent_probs
            elif isinstance(self.model, DeepBoltzmannMachine):
                # Use mean-field inference
                states, _ = self.model.mean_field_inference(data)
                # Return first hidden layer
                return states[1] if len(states) > 1 else states[0]
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _decode_from_latent(self, latent_data: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to visible space."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(self.model, RestrictedBoltzmannMachine):
                visible_probs, _ = self.model.hidden_to_visible(latent_data)
                return visible_probs
            elif isinstance(self.model, DeepBoltzmannMachine):
                # For DBM, this is more complex - we need to reconstruct through all layers
                # For simplicity, we'll use the reconstruction method
                # This is an approximation since we're only providing one layer
                batch_size = latent_data.size(0)

                # Create dummy states for other layers
                states = [torch.zeros(batch_size, self.model.layer_sizes[0], device=self.device)]
                states.append(latent_data)

                # Add dummy states for higher layers if they exist
                for i in range(2, len(self.model.layer_sizes)):
                    dummy_state = torch.zeros(batch_size, self.model.layer_sizes[i], device=self.device)
                    states.append(dummy_state)

                # Reconstruct visible layer
                return self.model._reconstruct_from_states(states)
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _discover_mean_difference_directions(
        self,
        latent_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Discover directions using mean differences between classes."""
        directions = {}
        unique_labels = torch.unique(labels)

        # Compute mean for each class
        class_means = {}
        for label in unique_labels:
            mask = labels == label
            class_mean = torch.mean(latent_data[mask], dim=0)
            class_means[int(label.item())] = class_mean

        # Compute pairwise differences
        label_list = list(class_means.keys())
        for i, label1 in enumerate(label_list):
            for j, label2 in enumerate(label_list[i+1:], i+1):
                direction = class_means[label2] - class_means[label1]
                direction_name = f"class_{label1}_to_{label2}"
                directions[direction_name] = F.normalize(direction, p=2, dim=0)

        return directions

    def _discover_pca_directions(
        self,
        latent_data: torch.Tensor,
        n_directions: int
    ) -> Dict[str, torch.Tensor]:
        """Discover directions using PCA."""
        try:
            # Center the data
            centered_data = latent_data - torch.mean(latent_data, dim=0)

            # Compute covariance matrix
            cov_matrix = torch.matmul(centered_data.t(), centered_data) / (latent_data.size(0) - 1)

            # Compute eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)

            # Sort by eigenvalues (descending)
            sorted_indices = torch.argsort(eigenvals, descending=True)

            directions = {}
            for i in range(min(n_directions, len(sorted_indices))):
                idx = sorted_indices[i]
                direction = eigenvecs[:, idx]
                variance_explained = eigenvals[idx] / torch.sum(eigenvals)
                directions[f"pca_component_{i}"] = {
                    'direction': direction,
                    'variance_explained': float(variance_explained)
                }

            return directions

        except Exception as e:
            logger.warning(f"PCA direction discovery failed: {e}")
            return {}

    def _discover_classifier_directions(
        self,
        latent_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Discover directions using linear classifier weights."""
        try:
            from sklearn.linear_model import LogisticRegression

            latent_np = latent_data.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Train linear classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(latent_np, labels_np)

            directions = {}

            if len(np.unique(labels_np)) == 2:
                # Binary classification
                direction = torch.tensor(clf.coef_[0], dtype=torch.float32, device=self.device)
                directions['classifier_direction'] = F.normalize(direction, p=2, dim=0)
            else:
                # Multi-class classification
                for i, coef in enumerate(clf.coef_):
                    direction = torch.tensor(coef, dtype=torch.float32, device=self.device)
                    directions[f'classifier_class_{i}'] = F.normalize(direction, p=2, dim=0)

            return directions

        except ImportError:
            logger.warning("Scikit-learn not available for classifier direction discovery")
            return {}
        except Exception as e:
            logger.warning(f"Classifier direction discovery failed: {e}")
            return {}

    def counterfactual_analysis(
        self,
        original_data: torch.Tensor,
        target_change: Dict[str, float],
        feature_names: Optional[List[str]] = None,
        max_iterations: int = 100,
        learning_rate: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual examples by optimizing latent representations.

        Args:
            original_data: Original data point [n_features] or [1, n_features]
            target_change: Dictionary specifying desired changes {feature_name: target_value}
            feature_names: Names of features
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization

        Returns:
            counterfactual_results: Dictionary with counterfactual analysis results
        """
        original_data = original_data.to(self.device)

        # Ensure batch dimension
        if original_data.dim() == 1:
            original_data = original_data.unsqueeze(0)

        # Encode to latent space
        original_latent = self._encode_to_latent(original_data)

        # Initialize optimizable latent representation
        counterfactual_latent = original_latent.clone().detach().requires_grad_(True)

        # Set up optimizer
        optimizer = torch.optim.Adam([counterfactual_latent], lr=learning_rate)

        # Convert target changes to tensor indices
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(original_data.size(1))]

        target_indices = []
        target_values = []
        for feature_name, target_value in target_change.items():
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                target_indices.append(idx)
                target_values.append(target_value)

        target_indices = torch.tensor(target_indices, device=self.device)
        target_values = torch.tensor(target_values, device=self.device)

        # Optimization loop
        losses = []
        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Decode current latent representation
            reconstructed = self._decode_from_latent(counterfactual_latent)

            # Compute loss
            # 1. Target feature loss
            target_loss = F.mse_loss(
                reconstructed[0, target_indices],
                target_values
            )

            # 2. Regularization loss (stay close to original)
            reg_loss = F.mse_loss(counterfactual_latent, original_latent.detach())

            # 3. Reconstruction quality loss
            recon_loss = F.mse_loss(reconstructed, original_data)

            # Combined loss
            total_loss = target_loss + 0.1 * reg_loss + 0.01 * recon_loss

            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

            # Early stopping if converged
            if iteration > 10 and abs(losses[-1] - losses[-10]) < 1e-6:
                break

        # Generate final counterfactual
        with torch.no_grad():
            counterfactual_data = self._decode_from_latent(counterfactual_latent)

            # Compute changes
            changes = counterfactual_data - original_data

            results = {
                'original_data': original_data,
                'counterfactual_data': counterfactual_data,
                'changes': changes,
                'latent_change': counterfactual_latent - original_latent,
                'optimization_losses': losses,
                'target_achieved': self._check_target_achievement(
                    counterfactual_data, target_change, feature_names
                )
            }

        return results

    def _check_target_achievement(
        self,
        counterfactual_data: torch.Tensor,
        target_change: Dict[str, float],
        feature_names: List[str],
        tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """Check if target changes were achieved."""
        achievement = {}

        for feature_name, target_value in target_change.items():
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                actual_value = counterfactual_data[0, idx].item()
                achieved = abs(actual_value - target_value) <= tolerance
                achievement[feature_name] = achieved

        return achievement

    def batch_traversal(
        self,
        data_batch: torch.Tensor,
        direction: torch.Tensor,
        n_steps: int = 5,
        step_size: float = 1.0
    ) -> torch.Tensor:
        """
        Perform traversal for a batch of data points.

        Args:
            data_batch: Batch of data points [batch_size, n_features]
            direction: Direction vector for traversal
            n_steps: Number of steps in each direction
            step_size: Size of each step

        Returns:
            batch_traversals: Traversed sequences [batch_size, n_steps, n_features]
        """
        data_batch = data_batch.to(self.device)
        direction = direction.to(self.device)

        batch_traversals = []

        for i in range(data_batch.size(0)):
            traversal = self.latent_traversal(
                base_data=data_batch[i],
                direction=direction,
                n_steps=n_steps,
                step_size=step_size,
                bidirectional=False
            )
            batch_traversals.append(traversal.unsqueeze(0))

        return torch.cat(batch_traversals, dim=0)

    def visualize_traversal(
        self,
        traversal_sequence: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        max_features: int = 10
    ) -> None:
        """
        Visualize latent traversal results.

        Args:
            traversal_sequence: Sequence from traversal [n_steps, n_features]
            feature_names: Names of features
            save_path: Path to save visualization
            max_features: Maximum number of features to visualize
        """
        try:
            import matplotlib.pyplot as plt

            traversal_np = traversal_sequence.cpu().numpy()
            n_steps, n_features = traversal_np.shape

            # Limit number of features for visualization
            n_features_plot = min(n_features, max_features)

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot feature trajectories
            for i in range(n_features_plot):
                feature_name = feature_names[i] if feature_names else f"Feature {i}"
                axes[0].plot(traversal_np[:, i], label=feature_name, alpha=0.7)

            axes[0].set_title('Feature Trajectories During Traversal')
            axes[0].set_xlabel('Traversal Step')
            axes[0].set_ylabel('Feature Value')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)

            # Plot feature changes from start
            start_values = traversal_np[0, :]
            changes = traversal_np - start_values

            for i in range(n_features_plot):
                feature_name = feature_names[i] if feature_names else f"Feature {i}"
                axes[1].plot(changes[:, i], label=feature_name, alpha=0.7)

            axes[1].set_title('Feature Changes from Starting Point')
            axes[1].set_xlabel('Traversal Step')
            axes[1].set_ylabel('Change in Feature Value')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Traversal visualization saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for visualization")

    def generate_traversal_report(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive traversal analysis report.

        Args:
            data: Input data for analysis
            labels: Optional labels for direction discovery
            feature_names: Names of features
            save_path: Path to save report

        Returns:
            report: Comprehensive traversal analysis report
        """
        logger.info("Generating latent traversal analysis report...")

        report = {
            'model_type': type(self.model).__name__,
            'data_shape': list(data.shape),
            'latent_space_analysis': {}
        }

        # Discover semantic directions if labels are provided
        if labels is not None:
            directions = self.discover_semantic_directions(data, labels)
            report['semantic_directions'] = {
                'n_directions': len(directions),
                'direction_names': list(directions.keys())
            }

            # Perform sample traversals
            if directions:
                sample_idx = 0
                sample_data = data[sample_idx:sample_idx+1]

                traversal_results = {}
                for direction_name, direction_vector in directions.items():
                    if isinstance(direction_vector, dict):
                        direction_vector = direction_vector['direction']

                    traversal = self.latent_traversal(
                        base_data=sample_data,
                        direction=direction_vector,
                        n_steps=5,
                        bidirectional=True
                    )

                    # Compute traversal statistics
                    traversal_stats = {
                        'sequence_length': traversal.size(0),
                        'feature_variance': torch.var(traversal, dim=0).mean().item(),
                        'total_change': torch.norm(traversal[-1] - traversal[0]).item()
                    }

                    traversal_results[direction_name] = traversal_stats

                report['sample_traversals'] = traversal_results

        # Analyze latent space properties
        latent_representations = self._encode_to_latent(data)
        report['latent_space_analysis'] = {
            'latent_dimensionality': latent_representations.size(1),
            'latent_statistics': {
                'mean': torch.mean(latent_representations, dim=0).cpu().numpy().tolist(),
                'std': torch.std(latent_representations, dim=0).cpu().numpy().tolist(),
                'sparsity': float(torch.mean((latent_representations < 0.1).float())),
            }
        }

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Traversal analysis report saved to {save_path}")

        return report

    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._direction_cache.clear()
        self._traversal_cache.clear()
        logger.info("Latent traversal cache cleared")
