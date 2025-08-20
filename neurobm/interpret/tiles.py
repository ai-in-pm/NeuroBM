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
Filter tiling and receptive field visualization.

This module provides tools for visualizing learned filters and receptive fields
in Boltzmann machines:
- Weight matrix visualization and tiling
- Receptive field analysis
- Filter pattern detection
- Feature map visualization
- Weight distribution analysis
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import logging
from pathlib import Path

from ..models.rbm import RestrictedBoltzmannMachine
from ..models.dbm import DeepBoltzmannMachine
from ..models.crbm import ConditionalRBM

logger = logging.getLogger(__name__)


class FilterVisualizer:
    """
    Comprehensive filter and receptive field visualization for Boltzmann machines.

    Provides tools for visualizing weight matrices, analyzing filter patterns,
    and understanding learned representations.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine, ConditionalRBM],
        device: Optional[torch.device] = None
    ):
        """
        Initialize filter visualizer.

        Args:
            model: Boltzmann machine model
            device: Device for computation
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        # Cache for computed visualizations
        self._visualization_cache = {}

    def visualize_weight_matrix(
        self,
        layer_idx: int = 0,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'RdBu_r',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize weight matrix as a heatmap.

        Args:
            layer_idx: Layer index (for DBM)
            figsize: Figure size
            cmap: Colormap for visualization
            save_path: Path to save figure

        Returns:
            figure: Matplotlib figure
        """
        # Get weight matrix
        if isinstance(self.model, RestrictedBoltzmannMachine):
            weights = self.model.W.detach().cpu().numpy()
            title = "RBM Weight Matrix"
        elif isinstance(self.model, ConditionalRBM):
            if layer_idx == 0:
                weights = self.model.W.detach().cpu().numpy()
                title = "CRBM Standard Weights"
            else:
                weights = self.model.A.detach().cpu().numpy()
                title = "CRBM Autoregressive Weights"
        elif isinstance(self.model, DeepBoltzmannMachine):
            if layer_idx < len(self.model.weights):
                weights = self.model.weights[layer_idx].detach().cpu().numpy()
                title = f"DBM Layer {layer_idx} Weights"
            else:
                raise ValueError(f"Layer index {layer_idx} out of range")
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(weights.T, cmap=cmap, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Value', rotation=270, labelpad=20)

        # Set labels and title
        ax.set_xlabel('Visible Units')
        ax.set_ylabel('Hidden Units')
        ax.set_title(title)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_filter_tiles(
        self,
        n_filters: Optional[int] = None,
        tile_shape: Optional[Tuple[int, int]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create tiled visualization of learned filters.

        Args:
            n_filters: Number of filters to visualize (if None, use all)
            tile_shape: Shape to reshape filters into (for 2D visualization)
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            figure: Matplotlib figure
        """
        # Get weights
        if isinstance(self.model, RestrictedBoltzmannMachine):
            weights = self.model.W.detach().cpu().numpy()  # [n_visible, n_hidden]
        elif isinstance(self.model, ConditionalRBM):
            weights = self.model.W.detach().cpu().numpy()
        elif isinstance(self.model, DeepBoltzmannMachine):
            weights = self.model.weights[0].detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        n_visible, n_hidden = weights.shape

        # Determine number of filters to show
        if n_filters is None:
            n_filters = min(n_hidden, 64)  # Limit for visualization
        else:
            n_filters = min(n_filters, n_hidden)

        # Determine grid layout
        grid_size = int(np.ceil(np.sqrt(n_filters)))

        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        if grid_size == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # Plot each filter
        for i in range(n_filters):
            row = i // grid_size
            col = i % grid_size

            filter_weights = weights[:, i]

            # Reshape if tile_shape is provided
            if tile_shape is not None:
                if np.prod(tile_shape) == len(filter_weights):
                    filter_weights = filter_weights.reshape(tile_shape)
                else:
                    logger.warning(f"Cannot reshape filter {i}: size mismatch")

            # Plot filter
            if filter_weights.ndim == 2:
                im = axes[row, col].imshow(filter_weights, cmap='RdBu_r')
            else:
                axes[row, col].plot(filter_weights)

            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')

        # Hide unused subplots
        for i in range(n_filters, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].set_visible(False)

        plt.suptitle('Learned Filters', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def analyze_receptive_fields(
        self,
        data: torch.Tensor,
        hidden_unit_indices: Optional[List[int]] = None,
        method: str = 'gradient'
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze receptive fields of hidden units.

        Args:
            data: Input data for analysis
            hidden_unit_indices: Specific hidden units to analyze
            method: Method for receptive field analysis ('gradient', 'activation')

        Returns:
            receptive_fields: Dictionary of receptive field information
        """
        data = data.to(self.device)
        self.model.eval()

        if isinstance(self.model, RestrictedBoltzmannMachine):
            n_hidden = self.model.n_hidden
        elif isinstance(self.model, ConditionalRBM):
            n_hidden = self.model.n_hidden
        elif isinstance(self.model, DeepBoltzmannMachine):
            n_hidden = self.model.layer_sizes[1]  # First hidden layer
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        if hidden_unit_indices is None:
            hidden_unit_indices = list(range(min(n_hidden, 20)))  # Limit for efficiency

        receptive_fields = {}

        for unit_idx in hidden_unit_indices:
            if method == 'gradient':
                rf = self._compute_gradient_receptive_field(data, unit_idx)
            elif method == 'activation':
                rf = self._compute_activation_receptive_field(data, unit_idx)
            else:
                raise ValueError(f"Unknown method: {method}")

            receptive_fields[f'unit_{unit_idx}'] = rf

        return receptive_fields

    def _compute_gradient_receptive_field(
        self,
        data: torch.Tensor,
        unit_idx: int
    ) -> torch.Tensor:
        """Compute receptive field using gradient analysis."""
        data.requires_grad_(True)

        # Forward pass to get hidden activations
        if isinstance(self.model, RestrictedBoltzmannMachine):
            h_prob, _ = self.model.visible_to_hidden(data)
        elif isinstance(self.model, ConditionalRBM):
            h_prob, _ = self.model.visible_to_hidden(data)
        elif isinstance(self.model, DeepBoltzmannMachine):
            states, _ = self.model.mean_field_inference(data)
            h_prob = states[1]  # First hidden layer
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        # Compute gradient of hidden unit activation w.r.t. input
        unit_activation = h_prob[:, unit_idx].sum()
        unit_activation.backward()

        receptive_field = data.grad.abs().mean(dim=0)

        return receptive_field.detach()

    def _compute_activation_receptive_field(
        self,
        data: torch.Tensor,
        unit_idx: int
    ) -> torch.Tensor:
        """Compute receptive field using activation correlation."""
        with torch.no_grad():
            # Get hidden activations
            if isinstance(self.model, RestrictedBoltzmannMachine):
                h_prob, _ = self.model.visible_to_hidden(data)
            elif isinstance(self.model, ConditionalRBM):
                h_prob, _ = self.model.visible_to_hidden(data)
            elif isinstance(self.model, DeepBoltzmannMachine):
                states, _ = self.model.mean_field_inference(data)
                h_prob = states[1]
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")

            unit_activations = h_prob[:, unit_idx]

            # Compute correlation between unit activation and input features
            correlations = torch.zeros(data.size(1), device=self.device)

            for feature_idx in range(data.size(1)):
                feature_values = data[:, feature_idx]
                correlation = torch.corrcoef(torch.stack([unit_activations, feature_values]))[0, 1]
                correlations[feature_idx] = correlation.abs()

            return correlations

    def visualize_receptive_fields(
        self,
        receptive_fields: Dict[str, torch.Tensor],
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize computed receptive fields.

        Args:
            receptive_fields: Dictionary of receptive field data
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            figure: Matplotlib figure
        """
        n_units = len(receptive_fields)
        grid_size = int(np.ceil(np.sqrt(n_units)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        if grid_size == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for i, (unit_name, rf_data) in enumerate(receptive_fields.items()):
            row = i // grid_size
            col = i % grid_size

            rf_np = rf_data.cpu().numpy()

            # Plot receptive field
            axes[row, col].bar(range(len(rf_np)), rf_np)
            axes[row, col].set_title(f'{unit_name}')
            axes[row, col].set_xlabel('Input Feature')
            axes[row, col].set_ylabel('Receptive Field Strength')

        # Hide unused subplots
        for i in range(n_units, grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].set_visible(False)

        plt.suptitle('Receptive Fields', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def analyze_weight_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze statistical properties of weight distributions.

        Returns:
            weight_stats: Dictionary of weight statistics
        """
        weight_stats = {}

        if isinstance(self.model, RestrictedBoltzmannMachine):
            weights = self.model.W.detach().cpu().numpy()
            weight_stats['main_weights'] = self._compute_weight_statistics(weights)

            if self.model.use_bias:
                v_bias = self.model.v_bias.detach().cpu().numpy()
                h_bias = self.model.h_bias.detach().cpu().numpy()
                weight_stats['visible_bias'] = self._compute_weight_statistics(v_bias)
                weight_stats['hidden_bias'] = self._compute_weight_statistics(h_bias)

        elif isinstance(self.model, ConditionalRBM):
            w_weights = self.model.W.detach().cpu().numpy()
            a_weights = self.model.A.detach().cpu().numpy()

            weight_stats['standard_weights'] = self._compute_weight_statistics(w_weights)
            weight_stats['autoregressive_weights'] = self._compute_weight_statistics(a_weights)

            if self.model.use_bias:
                v_bias = self.model.v_bias.detach().cpu().numpy()
                h_bias = self.model.h_bias.detach().cpu().numpy()
                weight_stats['visible_bias'] = self._compute_weight_statistics(v_bias)
                weight_stats['hidden_bias'] = self._compute_weight_statistics(h_bias)

        elif isinstance(self.model, DeepBoltzmannMachine):
            for i, weight_matrix in enumerate(self.model.weights):
                weights = weight_matrix.detach().cpu().numpy()
                weight_stats[f'layer_{i}_weights'] = self._compute_weight_statistics(weights)

        return weight_stats

    def _compute_weight_statistics(self, weights: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties of weight array."""
        return {
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'median': float(np.median(weights)),
            'skewness': float(stats.skew(weights.flatten())),
            'kurtosis': float(stats.kurtosis(weights.flatten())),
            'sparsity': float(np.mean(np.abs(weights) < 1e-6)),
            'l1_norm': float(np.sum(np.abs(weights))),
            'l2_norm': float(np.sqrt(np.sum(weights**2)))
        }

    def visualize_weight_distributions(
        self,
        weight_stats: Optional[Dict[str, Dict[str, float]]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize weight distribution statistics.

        Args:
            weight_stats: Weight statistics (if None, compute them)
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            figure: Matplotlib figure
        """
        if weight_stats is None:
            weight_stats = self.analyze_weight_distributions()

        # Create subplots for different statistics
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)

        # Extract data for plotting
        weight_names = list(weight_stats.keys())
        metrics = ['mean', 'std', 'sparsity', 'l1_norm', 'l2_norm', 'skewness']

        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[i // 3, i % 3])

            values = [weight_stats[name][metric] for name in weight_names]

            bars = ax.bar(range(len(weight_names)), values)
            ax.set_title(f'Weight {metric.title()}')
            ax.set_xticks(range(len(weight_names)))
            ax.set_xticklabels(weight_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Color bars based on values
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        plt.suptitle('Weight Distribution Analysis', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_comprehensive_visualization(
        self,
        data: torch.Tensor,
        save_dir: Optional[Path] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualization suite.

        Args:
            data: Input data for analysis
            save_dir: Directory to save visualizations
            feature_names: Names of input features

        Returns:
            figures: Dictionary of generated figures
        """
        logger.info("Creating comprehensive filter visualization suite...")

        figures = {}

        # Weight matrix visualization
        try:
            fig_weights = self.visualize_weight_matrix()
            figures['weight_matrix'] = fig_weights
            if save_dir:
                fig_weights.savefig(save_dir / 'weight_matrix.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Failed to create weight matrix visualization: {e}")

        # Filter tiles
        try:
            fig_tiles = self.create_filter_tiles()
            figures['filter_tiles'] = fig_tiles
            if save_dir:
                fig_tiles.savefig(save_dir / 'filter_tiles.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Failed to create filter tiles: {e}")

        # Receptive fields
        try:
            receptive_fields = self.analyze_receptive_fields(data)
            fig_rf = self.visualize_receptive_fields(receptive_fields)
            figures['receptive_fields'] = fig_rf
            if save_dir:
                fig_rf.savefig(save_dir / 'receptive_fields.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Failed to create receptive field visualization: {e}")

        # Weight distributions
        try:
            weight_stats = self.analyze_weight_distributions()
            fig_dist = self.visualize_weight_distributions(weight_stats)
            figures['weight_distributions'] = fig_dist
            if save_dir:
                fig_dist.savefig(save_dir / 'weight_distributions.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.warning(f"Failed to create weight distribution visualization: {e}")

        return figures


def create_model_architecture_diagram(
    model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine, ConditionalRBM],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a diagram showing the model architecture.

    Args:
        model: Boltzmann machine model
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(model, RestrictedBoltzmannMachine):
        # Draw RBM architecture
        n_visible = model.n_visible
        n_hidden = model.n_hidden

        # Visible layer
        visible_y = 0.2
        visible_x = np.linspace(0.1, 0.9, n_visible)

        # Hidden layer
        hidden_y = 0.8
        hidden_x = np.linspace(0.1, 0.9, n_hidden)

        # Draw connections
        for i, vx in enumerate(visible_x):
            for j, hx in enumerate(hidden_x):
                ax.plot([vx, hx], [visible_y, hidden_y], 'k-', alpha=0.3, linewidth=0.5)

        # Draw units
        for vx in visible_x:
            circle = plt.Circle((vx, visible_y), 0.02, color='lightblue', ec='black')
            ax.add_patch(circle)

        for hx in hidden_x:
            circle = plt.Circle((hx, hidden_y), 0.02, color='lightcoral', ec='black')
            ax.add_patch(circle)

        ax.text(0.5, 0.1, f'Visible Layer ({n_visible} units)', ha='center', fontsize=12)
        ax.text(0.5, 0.9, f'Hidden Layer ({n_hidden} units)', ha='center', fontsize=12)
        ax.set_title('Restricted Boltzmann Machine Architecture', fontsize=14)

    elif isinstance(model, ConditionalRBM):
        # Draw CRBM architecture with history connections
        n_visible = model.n_visible
        n_hidden = model.n_hidden
        n_history = model.n_history

        # Current visible layer
        visible_y = 0.2
        visible_x = np.linspace(0.3, 0.7, n_visible)

        # History layers
        history_y = 0.1
        history_x = np.linspace(0.1, 0.9, n_visible * n_history)

        # Hidden layer
        hidden_y = 0.8
        hidden_x = np.linspace(0.3, 0.7, n_hidden)

        # Draw connections
        # Standard RBM connections
        for vx in visible_x:
            for hx in hidden_x:
                ax.plot([vx, hx], [visible_y, hidden_y], 'k-', alpha=0.3, linewidth=0.5)

        # Autoregressive connections
        for hx_idx in history_x:
            for hx in hidden_x:
                ax.plot([hx_idx, hx], [history_y, hidden_y], 'r-', alpha=0.2, linewidth=0.3)

        # Draw units
        for vx in visible_x:
            circle = plt.Circle((vx, visible_y), 0.015, color='lightblue', ec='black')
            ax.add_patch(circle)

        for hx_idx in history_x:
            circle = plt.Circle((hx_idx, history_y), 0.01, color='lightgreen', ec='black')
            ax.add_patch(circle)

        for hx in hidden_x:
            circle = plt.Circle((hx, hidden_y), 0.015, color='lightcoral', ec='black')
            ax.add_patch(circle)

        ax.text(0.5, 0.05, f'History ({n_history} time steps)', ha='center', fontsize=10)
        ax.text(0.5, 0.15, f'Current Visible ({n_visible} units)', ha='center', fontsize=10)
        ax.text(0.5, 0.9, f'Hidden Layer ({n_hidden} units)', ha='center', fontsize=10)
        ax.set_title('Conditional RBM Architecture', fontsize=14)

    elif isinstance(model, DeepBoltzmannMachine):
        # Draw DBM architecture
        layer_sizes = model.layer_sizes
        n_layers = len(layer_sizes)

        layer_y = np.linspace(0.1, 0.9, n_layers)

        for layer_idx, (y_pos, layer_size) in enumerate(zip(layer_y, layer_sizes)):
            layer_x = np.linspace(0.1, 0.9, layer_size)

            # Draw units
            color = 'lightblue' if layer_idx == 0 else 'lightcoral'
            for x_pos in layer_x:
                circle = plt.Circle((x_pos, y_pos), 0.015, color=color, ec='black')
                ax.add_patch(circle)

            # Draw connections to next layer
            if layer_idx < n_layers - 1:
                next_layer_x = np.linspace(0.1, 0.9, layer_sizes[layer_idx + 1])
                next_y = layer_y[layer_idx + 1]

                for x1 in layer_x:
                    for x2 in next_layer_x:
                        ax.plot([x1, x2], [y_pos, next_y], 'k-', alpha=0.3, linewidth=0.3)

            # Add layer labels
            layer_name = 'Visible' if layer_idx == 0 else f'Hidden {layer_idx}'
            ax.text(0.95, y_pos, f'{layer_name} ({layer_size})', ha='left', va='center', fontsize=10)

        ax.set_title('Deep Boltzmann Machine Architecture', fontsize=14)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
