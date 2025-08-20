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
Saliency analysis and weight visualization tools

This module provides interpretability tools for Boltzmann machines:
- Weight saliency analysis
- Feature importance computation
- Gradient-based attribution methods
- Weight visualization and analysis
- Connection strength analysis
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


class SaliencyAnalyzer:
    """
    Saliency analysis for Boltzmann machines.

    Provides various methods to analyze feature importance,
    weight significance, and model interpretability.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine],
        device: Optional[torch.device] = None
    ):
        """
        Initialize saliency analyzer.

        Args:
            model: Boltzmann machine model to analyze
            device: Device for computations
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        # Cache for computed saliencies
        self._weight_saliency_cache = {}
        self._feature_importance_cache = {}

    def compute_weight_saliency(
        self,
        method: str = "magnitude",
        layer_idx: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Compute saliency of model weights.

        Args:
            method: Saliency method ('magnitude', 'variance', 'gradient')
            layer_idx: Layer index for DBM (if None, return all layers)

        Returns:
            saliency: Weight saliency values
        """
        cache_key = f"{method}_{layer_idx}"
        if cache_key in self._weight_saliency_cache:
            return self._weight_saliency_cache[cache_key]

        if isinstance(self.model, RestrictedBoltzmannMachine):
            saliency = self._compute_rbm_weight_saliency(method)
        elif isinstance(self.model, DeepBoltzmannMachine):
            saliency = self._compute_dbm_weight_saliency(method, layer_idx)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        self._weight_saliency_cache[cache_key] = saliency
        return saliency

    def _compute_rbm_weight_saliency(self, method: str) -> torch.Tensor:
        """Compute weight saliency for RBM."""
        weights = self.model.W

        if method == "magnitude":
            return torch.abs(weights)
        elif method == "variance":
            # Use weight magnitude as proxy for variance contribution
            return weights ** 2
        elif method == "gradient":
            # Requires gradient computation - placeholder for now
            logger.warning("Gradient-based saliency requires training data")
            return torch.abs(weights)
        else:
            raise ValueError(f"Unknown saliency method: {method}")

    def _compute_dbm_weight_saliency(
        self,
        method: str,
        layer_idx: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute weight saliency for DBM."""
        if layer_idx is not None:
            if layer_idx >= len(self.model.weights):
                raise ValueError(f"Layer index {layer_idx} out of range")
            weights = self.model.weights[layer_idx]

            if method == "magnitude":
                return torch.abs(weights)
            elif method == "variance":
                return weights ** 2
            else:
                return torch.abs(weights)
        else:
            # Return saliency for all layers
            saliencies = []
            for i in range(len(self.model.weights)):
                layer_saliency = self._compute_dbm_weight_saliency(method, i)
                saliencies.append(layer_saliency)
            return saliencies

    def compute_feature_importance(
        self,
        data: torch.Tensor,
        method: str = "reconstruction_error",
        n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute feature importance scores.

        Args:
            data: Input data [n_samples, n_features]
            method: Importance method ('reconstruction_error', 'energy_gradient', 'permutation')
            n_samples: Number of samples to use (if None, use all)

        Returns:
            importance: Feature importance scores [n_features]
        """
        cache_key = f"{method}_{n_samples}"
        if cache_key in self._feature_importance_cache:
            return self._feature_importance_cache[cache_key]

        data = data.to(self.device)
        if n_samples is not None and n_samples < data.size(0):
            indices = torch.randperm(data.size(0))[:n_samples]
            data = data[indices]

        if method == "reconstruction_error":
            importance = self._compute_reconstruction_importance(data)
        elif method == "energy_gradient":
            importance = self._compute_energy_gradient_importance(data)
        elif method == "permutation":
            importance = self._compute_permutation_importance(data)
        else:
            raise ValueError(f"Unknown importance method: {method}")

        self._feature_importance_cache[cache_key] = importance
        return importance

    def _compute_reconstruction_importance(self, data: torch.Tensor) -> torch.Tensor:
        """Compute importance based on reconstruction error."""
        self.model.eval()

        with torch.no_grad():
            # Baseline reconstruction error
            if isinstance(self.model, RestrictedBoltzmannMachine):
                reconstructed = self.model.reconstruct(data, n_gibbs=1)
            else:  # DBM
                reconstructed = self.model.reconstruct(data)

            baseline_error = torch.mean((data - reconstructed) ** 2, dim=0)

            # Compute importance by masking each feature
            importance_scores = torch.zeros(data.size(1), device=self.device)

            for feature_idx in range(data.size(1)):
                # Mask feature
                masked_data = data.clone()
                masked_data[:, feature_idx] = 0  # or use mean value

                # Reconstruct masked data
                if isinstance(self.model, RestrictedBoltzmannMachine):
                    masked_reconstructed = self.model.reconstruct(masked_data, n_gibbs=1)
                else:
                    masked_reconstructed = self.model.reconstruct(masked_data)

                # Compute reconstruction error
                masked_error = torch.mean((masked_data - masked_reconstructed) ** 2, dim=0)

                # Importance is the change in reconstruction error
                importance_scores[feature_idx] = torch.mean(masked_error - baseline_error)

        return torch.abs(importance_scores)

    def _compute_energy_gradient_importance(self, data: torch.Tensor) -> torch.Tensor:
        """Compute importance based on energy gradients."""
        data.requires_grad_(True)

        if isinstance(self.model, RestrictedBoltzmannMachine):
            # For RBM, use free energy
            energy = self.model.free_energy(data)
        else:
            # For DBM, use mean-field approximation
            energy = self.model._compute_free_energy(data)

        # Compute gradients
        total_energy = torch.sum(energy)
        gradients = torch.autograd.grad(total_energy, data, create_graph=False)[0]

        # Importance is the magnitude of gradients
        importance = torch.mean(torch.abs(gradients), dim=0)

        data.requires_grad_(False)
        return importance

    def _compute_permutation_importance(self, data: torch.Tensor) -> torch.Tensor:
        """Compute permutation-based feature importance."""
        self.model.eval()

        with torch.no_grad():
            # Baseline performance (reconstruction error)
            if isinstance(self.model, RestrictedBoltzmannMachine):
                baseline_reconstructed = self.model.reconstruct(data, n_gibbs=1)
            else:
                baseline_reconstructed = self.model.reconstruct(data)

            baseline_error = torch.mean((data - baseline_reconstructed) ** 2)

            importance_scores = torch.zeros(data.size(1), device=self.device)

            for feature_idx in range(data.size(1)):
                # Permute feature values
                permuted_data = data.clone()
                perm_indices = torch.randperm(data.size(0))
                permuted_data[:, feature_idx] = data[perm_indices, feature_idx]

                # Compute reconstruction error with permuted feature
                if isinstance(self.model, RestrictedBoltzmannMachine):
                    permuted_reconstructed = self.model.reconstruct(permuted_data, n_gibbs=1)
                else:
                    permuted_reconstructed = self.model.reconstruct(permuted_data)

                permuted_error = torch.mean((permuted_data - permuted_reconstructed) ** 2)

                # Importance is the increase in error
                importance_scores[feature_idx] = permuted_error - baseline_error

        return torch.abs(importance_scores)

    def analyze_weight_patterns(
        self,
        layer_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze patterns in model weights.

        Args:
            layer_idx: Layer index for DBM analysis

        Returns:
            analysis: Dictionary of weight pattern analysis
        """
        if isinstance(self.model, RestrictedBoltzmannMachine):
            weights = self.model.W
            analysis = self._analyze_weight_matrix(weights, "RBM")
        elif isinstance(self.model, DeepBoltzmannMachine):
            if layer_idx is not None:
                weights = self.model.weights[layer_idx]
                analysis = self._analyze_weight_matrix(weights, f"DBM_layer_{layer_idx}")
            else:
                # Analyze all layers
                analysis = {}
                for i, weight_matrix in enumerate(self.model.weights):
                    layer_analysis = self._analyze_weight_matrix(weight_matrix, f"layer_{i}")
                    analysis[f"layer_{i}"] = layer_analysis
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        return analysis

    def _analyze_weight_matrix(self, weights: torch.Tensor, name: str) -> Dict[str, Any]:
        """Analyze a single weight matrix."""
        weights_np = weights.detach().cpu().numpy()

        analysis = {
            'name': name,
            'shape': weights.shape,
            'mean': float(torch.mean(weights)),
            'std': float(torch.std(weights)),
            'min': float(torch.min(weights)),
            'max': float(torch.max(weights)),
            'sparsity': float(torch.sum(torch.abs(weights) < 1e-6) / weights.numel()),
            'frobenius_norm': float(torch.norm(weights, p='fro')),
            'spectral_norm': float(torch.norm(weights, p=2)),
            'rank_estimate': self._estimate_rank(weights),
            'weight_distribution': {
                'positive_ratio': float(torch.sum(weights > 0) / weights.numel()),
                'negative_ratio': float(torch.sum(weights < 0) / weights.numel()),
                'zero_ratio': float(torch.sum(torch.abs(weights) < 1e-6) / weights.numel())
            }
        }

        return analysis

    def _estimate_rank(self, weights: torch.Tensor, threshold: float = 1e-6) -> int:
        """Estimate effective rank of weight matrix."""
        try:
            _, s, _ = torch.svd(weights)
            rank = torch.sum(s > threshold).item()
            return rank
        except:
            return min(weights.shape)

    def compute_connection_strength(
        self,
        normalize: bool = True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Compute connection strength between units.

        Args:
            normalize: Whether to normalize by layer size

        Returns:
            connection_strength: Connection strength values
        """
        if isinstance(self.model, RestrictedBoltzmannMachine):
            weights = self.model.W
            strength = torch.sum(torch.abs(weights), dim=1)  # Sum over hidden units

            if normalize:
                strength = strength / weights.size(1)

            return strength

        elif isinstance(self.model, DeepBoltzmannMachine):
            strengths = []

            for i, weights in enumerate(self.model.weights):
                # Incoming connections strength
                incoming_strength = torch.sum(torch.abs(weights), dim=0)
                # Outgoing connections strength
                outgoing_strength = torch.sum(torch.abs(weights), dim=1)

                if normalize:
                    incoming_strength = incoming_strength / weights.size(0)
                    outgoing_strength = outgoing_strength / weights.size(1)

                strengths.append({
                    'incoming': incoming_strength,
                    'outgoing': outgoing_strength,
                    'layer': i
                })

            return strengths
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def visualize_weights(
        self,
        layer_idx: Optional[int] = None,
        max_units: int = 50,
        save_path: Optional[Path] = None
    ) -> None:
        """
        Visualize model weights.

        Args:
            layer_idx: Layer index for DBM (if None, visualize all)
            max_units: Maximum number of units to visualize
            save_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if isinstance(self.model, RestrictedBoltzmannMachine):
                self._visualize_rbm_weights(max_units, save_path)
            elif isinstance(self.model, DeepBoltzmannMachine):
                self._visualize_dbm_weights(layer_idx, max_units, save_path)
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualization")

    def _visualize_rbm_weights(self, max_units: int, save_path: Optional[Path]) -> None:
        """Visualize RBM weights."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        weights = self.model.W.detach().cpu().numpy()

        # Limit number of units for visualization
        n_visible = min(weights.shape[0], max_units)
        n_hidden = min(weights.shape[1], max_units)
        weights_subset = weights[:n_visible, :n_hidden]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Weight matrix heatmap
        sns.heatmap(weights_subset, ax=axes[0, 0], cmap='RdBu_r', center=0)
        axes[0, 0].set_title('Weight Matrix')
        axes[0, 0].set_xlabel('Hidden Units')
        axes[0, 0].set_ylabel('Visible Units')

        # Weight distribution
        axes[0, 1].hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Weight Distribution')
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')

        # Connection strength
        connection_strength = self.compute_connection_strength()
        axes[1, 0].bar(range(len(connection_strength)), connection_strength.cpu().numpy())
        axes[1, 0].set_title('Connection Strength per Visible Unit')
        axes[1, 0].set_xlabel('Visible Unit Index')
        axes[1, 0].set_ylabel('Connection Strength')

        # Weight magnitude by layer
        weight_magnitudes = torch.mean(torch.abs(self.model.W), dim=0).cpu().numpy()
        axes[1, 1].bar(range(len(weight_magnitudes)), weight_magnitudes)
        axes[1, 1].set_title('Average Weight Magnitude per Hidden Unit')
        axes[1, 1].set_xlabel('Hidden Unit Index')
        axes[1, 1].set_ylabel('Average Weight Magnitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weight visualization saved to {save_path}")
        else:
            plt.show()

    def _visualize_dbm_weights(
        self,
        layer_idx: Optional[int],
        max_units: int,
        save_path: Optional[Path]
    ) -> None:
        """Visualize DBM weights."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if layer_idx is not None:
            # Visualize specific layer
            weights = self.model.weights[layer_idx].detach().cpu().numpy()

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Weight matrix heatmap
            n_lower = min(weights.shape[0], max_units)
            n_upper = min(weights.shape[1], max_units)
            weights_subset = weights[:n_lower, :n_upper]

            sns.heatmap(weights_subset, ax=axes[0, 0], cmap='RdBu_r', center=0)
            axes[0, 0].set_title(f'Layer {layer_idx} Weight Matrix')

            # Weight distribution
            axes[0, 1].hist(weights.flatten(), bins=50, alpha=0.7)
            axes[0, 1].set_title(f'Layer {layer_idx} Weight Distribution')

            plt.tight_layout()
        else:
            # Visualize all layers
            n_layers = len(self.model.weights)
            fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 10))

            if n_layers == 1:
                axes = axes.reshape(2, 1)

            for i, weights in enumerate(self.model.weights):
                weights_np = weights.detach().cpu().numpy()

                # Weight matrix heatmap
                n_lower = min(weights_np.shape[0], max_units)
                n_upper = min(weights_np.shape[1], max_units)
                weights_subset = weights_np[:n_lower, :n_upper]

                sns.heatmap(weights_subset, ax=axes[0, i], cmap='RdBu_r', center=0)
                axes[0, i].set_title(f'Layer {i} Weights')

                # Weight distribution
                axes[1, i].hist(weights_np.flatten(), bins=30, alpha=0.7)
                axes[1, i].set_title(f'Layer {i} Distribution')

            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"DBM weight visualization saved to {save_path}")
        else:
            plt.show()

    def generate_saliency_report(
        self,
        data: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive saliency analysis report.

        Args:
            data: Input data for analysis
            feature_names: Names of features
            save_path: Path to save report

        Returns:
            report: Comprehensive saliency report
        """
        logger.info("Generating saliency analysis report...")

        # Compute various saliency measures
        weight_saliency = self.compute_weight_saliency(method="magnitude")
        feature_importance = self.compute_feature_importance(data, method="reconstruction_error")
        weight_patterns = self.analyze_weight_patterns()
        connection_strength = self.compute_connection_strength()

        # Create report
        report = {
            'model_type': type(self.model).__name__,
            'model_parameters': {
                'n_visible': getattr(self.model, 'n_visible', None) or self.model.layer_sizes[0],
                'n_hidden': getattr(self.model, 'n_hidden', None) or self.model.layer_sizes[1:],
            },
            'weight_saliency': {
                'mean': float(torch.mean(weight_saliency)) if isinstance(weight_saliency, torch.Tensor) else None,
                'std': float(torch.std(weight_saliency)) if isinstance(weight_saliency, torch.Tensor) else None,
                'max': float(torch.max(weight_saliency)) if isinstance(weight_saliency, torch.Tensor) else None,
            },
            'feature_importance': {
                'scores': feature_importance.cpu().numpy().tolist(),
                'ranking': torch.argsort(feature_importance, descending=True).cpu().numpy().tolist(),
                'top_features': self._get_top_features(feature_importance, feature_names, top_k=5)
            },
            'weight_patterns': weight_patterns,
            'connection_analysis': {
                'mean_strength': float(torch.mean(connection_strength)) if isinstance(connection_strength, torch.Tensor) else None,
                'max_strength': float(torch.max(connection_strength)) if isinstance(connection_strength, torch.Tensor) else None,
            },
            'data_statistics': {
                'n_samples': data.size(0),
                'n_features': data.size(1),
                'data_mean': torch.mean(data, dim=0).cpu().numpy().tolist(),
                'data_std': torch.std(data, dim=0).cpu().numpy().tolist(),
            }
        }

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saliency report saved to {save_path}")

        return report

    def _get_top_features(
        self,
        importance_scores: torch.Tensor,
        feature_names: Optional[List[str]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top k most important features."""
        top_indices = torch.argsort(importance_scores, descending=True)[:top_k]

        top_features = []
        for i, idx in enumerate(top_indices):
            feature_info = {
                'rank': i + 1,
                'index': int(idx),
                'importance_score': float(importance_scores[idx]),
                'name': feature_names[idx] if feature_names else f"feature_{idx}"
            }
            top_features.append(feature_info)

        return top_features

    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._weight_saliency_cache.clear()
        self._feature_importance_cache.clear()
        logger.info("Saliency analysis cache cleared")
