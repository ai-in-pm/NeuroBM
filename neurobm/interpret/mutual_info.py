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
Mutual information analysis between latent and visible units

This module provides mutual information analysis tools for Boltzmann machines:
- Mutual information estimation between visible and hidden units
- Information-theoretic analysis of representations
- Entropy computation and analysis
- Information flow analysis
- Representation quality assessment
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score

from ..models.rbm import RestrictedBoltzmannMachine
from ..models.dbm import DeepBoltzmannMachine

logger = logging.getLogger(__name__)


class MutualInformationAnalyzer:
    """
    Mutual information analysis for Boltzmann machines.

    Provides tools to analyze information flow and representation
    quality in Boltzmann machine models.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine],
        device: Optional[torch.device] = None
    ):
        """
        Initialize mutual information analyzer.

        Args:
            model: Boltzmann machine model to analyze
            device: Device for computations
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        # Cache for computed MI values
        self._mi_cache = {}

    def compute_mutual_information(
        self,
        visible_data: torch.Tensor,
        hidden_data: Optional[torch.Tensor] = None,
        method: str = "histogram",
        bins: int = 50,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute mutual information between visible and hidden units.

        Args:
            visible_data: Visible unit data [n_samples, n_visible]
            hidden_data: Hidden unit data [n_samples, n_hidden] (if None, compute from model)
            method: MI estimation method ('histogram', 'ksg', 'sklearn')
            bins: Number of bins for histogram method
            normalize: Whether to normalize MI values

        Returns:
            mi_matrix: Mutual information matrix [n_visible, n_hidden]
        """
        cache_key = f"{method}_{bins}_{normalize}_{visible_data.shape}"
        if cache_key in self._mi_cache:
            return self._mi_cache[cache_key]

        visible_data = visible_data.to(self.device)

        # Get hidden representations if not provided
        if hidden_data is None:
            hidden_data = self._get_hidden_representations(visible_data)
        else:
            hidden_data = hidden_data.to(self.device)

        # Convert to numpy for MI computation
        visible_np = visible_data.cpu().numpy()
        hidden_np = hidden_data.cpu().numpy()

        # Compute MI matrix
        n_visible = visible_np.shape[1]
        n_hidden = hidden_np.shape[1]
        mi_matrix = np.zeros((n_visible, n_hidden))

        for i in range(n_visible):
            for j in range(n_hidden):
                if method == "histogram":
                    mi_value = self._compute_mi_histogram(
                        visible_np[:, i], hidden_np[:, j], bins
                    )
                elif method == "ksg":
                    mi_value = self._compute_mi_ksg(
                        visible_np[:, i], hidden_np[:, j]
                    )
                elif method == "sklearn":
                    mi_value = self._compute_mi_sklearn(
                        visible_np[:, i], hidden_np[:, j]
                    )
                else:
                    raise ValueError(f"Unknown MI method: {method}")

                mi_matrix[i, j] = mi_value

        # Normalize if requested
        if normalize:
            # Normalize by joint entropy
            mi_matrix = self._normalize_mi_matrix(mi_matrix, visible_np, hidden_np)

        mi_tensor = torch.tensor(mi_matrix, dtype=torch.float32, device=self.device)
        self._mi_cache[cache_key] = mi_tensor

        return mi_tensor

    def _get_hidden_representations(self, visible_data: torch.Tensor) -> torch.Tensor:
        """Get hidden representations from visible data."""
        self.model.eval()

        with torch.no_grad():
            if isinstance(self.model, RestrictedBoltzmannMachine):
                hidden_probs, _ = self.model.visible_to_hidden(visible_data)
                return hidden_probs
            elif isinstance(self.model, DeepBoltzmannMachine):
                # Use mean-field inference to get all hidden layers
                states, _ = self.model.mean_field_inference(visible_data)
                # Return first hidden layer for simplicity
                return states[1] if len(states) > 1 else states[0]
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _compute_mi_histogram(self, x: np.ndarray, y: np.ndarray, bins: int) -> float:
        """Compute MI using histogram-based estimation."""
        try:
            # Create 2D histogram
            hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

            # Add small epsilon to avoid log(0)
            hist_2d = hist_2d + 1e-10

            # Normalize to get probabilities
            p_xy = hist_2d / np.sum(hist_2d)

            # Marginal probabilities
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)

            # Compute MI
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 1e-10:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

            return max(0.0, mi)  # MI should be non-negative

        except Exception as e:
            logger.warning(f"MI histogram computation failed: {e}")
            return 0.0

    def _compute_mi_ksg(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using KSG estimator (simplified version)."""
        try:
            # This is a simplified implementation
            # For production use, consider using dedicated libraries like NPEET

            # Discretize continuous variables
            n_bins = min(50, int(np.sqrt(len(x))))
            x_discrete = np.digitize(x, np.histogram(x, bins=n_bins)[1])
            y_discrete = np.digitize(y, np.histogram(y, bins=n_bins)[1])

            # Use sklearn's mutual_info_score
            mi = mutual_info_score(x_discrete, y_discrete)
            return max(0.0, mi)

        except Exception as e:
            logger.warning(f"MI KSG computation failed: {e}")
            return 0.0

    def _compute_mi_sklearn(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using sklearn estimators."""
        try:
            # Reshape for sklearn
            x_reshaped = x.reshape(-1, 1)

            # Use mutual_info_regression for continuous variables
            mi = mutual_info_regression(x_reshaped, y, random_state=42)[0]
            return max(0.0, mi)

        except Exception as e:
            logger.warning(f"MI sklearn computation failed: {e}")
            return 0.0

    def _normalize_mi_matrix(
        self,
        mi_matrix: np.ndarray,
        visible_data: np.ndarray,
        hidden_data: np.ndarray
    ) -> np.ndarray:
        """Normalize MI matrix by marginal entropies."""
        normalized_mi = np.zeros_like(mi_matrix)

        # Compute marginal entropies
        visible_entropies = np.array([
            self._compute_entropy(visible_data[:, i]) for i in range(visible_data.shape[1])
        ])
        hidden_entropies = np.array([
            self._compute_entropy(hidden_data[:, j]) for j in range(hidden_data.shape[1])
        ])

        # Normalize by geometric mean of marginal entropies
        for i in range(mi_matrix.shape[0]):
            for j in range(mi_matrix.shape[1]):
                normalizer = np.sqrt(visible_entropies[i] * hidden_entropies[j])
                if normalizer > 1e-10:
                    normalized_mi[i, j] = mi_matrix[i, j] / normalizer
                else:
                    normalized_mi[i, j] = 0.0

        return normalized_mi

    def _compute_entropy(self, data: np.ndarray, bins: int = 50) -> float:
        """Compute entropy of data."""
        try:
            hist, _ = np.histogram(data, bins=bins)
            hist = hist + 1e-10  # Add epsilon
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log(probs))
            return entropy
        except:
            return 0.0

    def compute_information_flow(
        self,
        data: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute information flow through the network layers.

        Args:
            data: Input data [n_samples, n_features]
            layer_indices: Specific layers to analyze (for DBM)

        Returns:
            flow_analysis: Dictionary of information flow metrics
        """
        data = data.to(self.device)

        if isinstance(self.model, RestrictedBoltzmannMachine):
            return self._compute_rbm_information_flow(data)
        elif isinstance(self.model, DeepBoltzmannMachine):
            return self._compute_dbm_information_flow(data, layer_indices)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _compute_rbm_information_flow(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute information flow for RBM."""
        self.model.eval()

        with torch.no_grad():
            # Get hidden representations
            hidden_probs, _ = self.model.visible_to_hidden(data)

            # Compute MI between visible and hidden
            mi_matrix = self.compute_mutual_information(data, hidden_probs)

            # Aggregate metrics
            total_mi = torch.sum(mi_matrix)
            avg_mi_per_visible = torch.mean(mi_matrix, dim=1)
            avg_mi_per_hidden = torch.mean(mi_matrix, dim=0)
            max_mi_per_visible = torch.max(mi_matrix, dim=1)[0]
            max_mi_per_hidden = torch.max(mi_matrix, dim=0)[0]

            return {
                'mi_matrix': mi_matrix,
                'total_mutual_information': total_mi,
                'avg_mi_per_visible': avg_mi_per_visible,
                'avg_mi_per_hidden': avg_mi_per_hidden,
                'max_mi_per_visible': max_mi_per_visible,
                'max_mi_per_hidden': max_mi_per_hidden,
                'information_compression_ratio': torch.sum(avg_mi_per_hidden) / torch.sum(avg_mi_per_visible)
            }

    def _compute_dbm_information_flow(
        self,
        data: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute information flow for DBM."""
        self.model.eval()

        with torch.no_grad():
            # Get representations at all layers
            states, _ = self.model.mean_field_inference(data)

            if layer_indices is None:
                layer_indices = list(range(len(states) - 1))

            flow_analysis = {}

            # Analyze flow between adjacent layers
            for i in layer_indices:
                if i + 1 < len(states):
                    lower_layer = states[i]
                    upper_layer = states[i + 1]

                    # Compute MI between layers
                    mi_matrix = self.compute_mutual_information(lower_layer, upper_layer)

                    layer_analysis = {
                        'mi_matrix': mi_matrix,
                        'total_mi': torch.sum(mi_matrix),
                        'avg_mi_per_lower': torch.mean(mi_matrix, dim=1),
                        'avg_mi_per_upper': torch.mean(mi_matrix, dim=0),
                        'compression_ratio': mi_matrix.size(1) / mi_matrix.size(0)
                    }

                    flow_analysis[f'layer_{i}_to_{i+1}'] = layer_analysis

            # Overall network analysis
            if len(states) >= 2:
                input_output_mi = self.compute_mutual_information(states[0], states[-1])
                flow_analysis['input_output_mi'] = {
                    'mi_matrix': input_output_mi,
                    'total_mi': torch.sum(input_output_mi)
                }

            return flow_analysis

    def analyze_representation_quality(
        self,
        data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        regime_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Analyze quality of learned representations.

        Args:
            data: Input data [n_samples, n_features]
            labels: Optional class labels [n_samples]
            regime_labels: Optional regime labels [n_samples]

        Returns:
            quality_analysis: Dictionary of representation quality metrics
        """
        data = data.to(self.device)

        # Get hidden representations
        hidden_repr = self._get_hidden_representations(data)

        analysis = {
            'representation_statistics': self._compute_representation_statistics(hidden_repr),
            'information_content': self._compute_information_content(hidden_repr),
        }

        # Add supervised analysis if labels are provided
        if labels is not None:
            analysis['supervised_quality'] = self._compute_supervised_quality(
                hidden_repr, labels
            )

        if regime_labels is not None:
            analysis['regime_discrimination'] = self._compute_regime_discrimination(
                hidden_repr, regime_labels
            )

        return analysis

    def _compute_representation_statistics(self, hidden_repr: torch.Tensor) -> Dict[str, float]:
        """Compute basic statistics of hidden representations."""
        hidden_np = hidden_repr.cpu().numpy()

        return {
            'mean_activation': float(np.mean(hidden_np)),
            'std_activation': float(np.std(hidden_np)),
            'sparsity': float(np.mean(hidden_np < 0.1)),  # Fraction of near-zero activations
            'saturation': float(np.mean(hidden_np > 0.9)),  # Fraction of saturated units
            'effective_dimensionality': self._compute_effective_dimensionality(hidden_repr),
            'activation_entropy': float(np.mean([
                self._compute_entropy(hidden_np[:, i]) for i in range(hidden_np.shape[1])
            ]))
        }

    def _compute_information_content(self, hidden_repr: torch.Tensor) -> Dict[str, float]:
        """Compute information-theoretic measures of representations."""
        hidden_np = hidden_repr.cpu().numpy()

        # Compute pairwise MI between hidden units
        n_hidden = hidden_np.shape[1]
        if n_hidden > 1:
            mi_values = []
            for i in range(min(n_hidden, 10)):  # Limit computation for efficiency
                for j in range(i + 1, min(n_hidden, 10)):
                    mi = self._compute_mi_histogram(hidden_np[:, i], hidden_np[:, j], bins=20)
                    mi_values.append(mi)

            avg_pairwise_mi = float(np.mean(mi_values)) if mi_values else 0.0
        else:
            avg_pairwise_mi = 0.0

        return {
            'average_pairwise_mi': avg_pairwise_mi,
            'total_entropy': float(np.sum([
                self._compute_entropy(hidden_np[:, i]) for i in range(hidden_np.shape[1])
            ])),
            'redundancy_estimate': avg_pairwise_mi * n_hidden * (n_hidden - 1) / 2
        }

    def _compute_effective_dimensionality(self, hidden_repr: torch.Tensor) -> float:
        """Compute effective dimensionality using participation ratio."""
        try:
            # Compute covariance matrix
            centered = hidden_repr - torch.mean(hidden_repr, dim=0)
            cov_matrix = torch.matmul(centered.t(), centered) / (hidden_repr.size(0) - 1)

            # Compute eigenvalues
            eigenvals = torch.linalg.eigvals(cov_matrix).real
            eigenvals = torch.clamp(eigenvals, min=0)  # Ensure non-negative

            # Participation ratio
            if torch.sum(eigenvals) > 1e-10:
                participation_ratio = (torch.sum(eigenvals) ** 2) / torch.sum(eigenvals ** 2)
                return float(participation_ratio)
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Effective dimensionality computation failed: {e}")
            return float(hidden_repr.size(1))  # Fallback to actual dimensionality

    def _compute_supervised_quality(
        self,
        hidden_repr: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute supervised quality metrics."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import cross_val_score

            hidden_np = hidden_repr.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Train simple classifier on representations
            clf = LogisticRegression(random_state=42, max_iter=1000)

            # Cross-validation scores
            cv_scores = cross_val_score(clf, hidden_np, labels_np, cv=5)

            # Fit and predict for additional metrics
            clf.fit(hidden_np, labels_np)
            predictions = clf.predict(hidden_np)

            return {
                'classification_accuracy': float(np.mean(cv_scores)),
                'classification_std': float(np.std(cv_scores)),
                'train_accuracy': float(accuracy_score(labels_np, predictions)),
                'f1_score': float(f1_score(labels_np, predictions, average='weighted'))
            }

        except ImportError:
            logger.warning("Scikit-learn not available for supervised quality analysis")
            return {}
        except Exception as e:
            logger.warning(f"Supervised quality computation failed: {e}")
            return {}

    def _compute_regime_discrimination(
        self,
        hidden_repr: torch.Tensor,
        regime_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute how well representations discriminate between regimes."""
        try:
            # Compute MI between hidden units and regime labels
            hidden_np = hidden_repr.cpu().numpy()
            regime_np = regime_labels.cpu().numpy()

            mi_scores = []
            for i in range(hidden_np.shape[1]):
                mi = self._compute_mi_sklearn(hidden_np[:, i], regime_np)
                mi_scores.append(mi)

            return {
                'avg_regime_mi': float(np.mean(mi_scores)),
                'max_regime_mi': float(np.max(mi_scores)),
                'regime_discriminative_units': int(np.sum(np.array(mi_scores) > 0.1))
            }

        except Exception as e:
            logger.warning(f"Regime discrimination computation failed: {e}")
            return {}

    def visualize_mutual_information(
        self,
        mi_matrix: torch.Tensor,
        visible_names: Optional[List[str]] = None,
        hidden_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize mutual information matrix.

        Args:
            mi_matrix: Mutual information matrix [n_visible, n_hidden]
            visible_names: Names for visible units
            hidden_names: Names for hidden units
            save_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            mi_np = mi_matrix.cpu().numpy()

            plt.figure(figsize=(12, 8))

            # Create heatmap
            sns.heatmap(
                mi_np,
                xticklabels=hidden_names or [f'H{i}' for i in range(mi_np.shape[1])],
                yticklabels=visible_names or [f'V{i}' for i in range(mi_np.shape[0])],
                cmap='viridis',
                cbar_kws={'label': 'Mutual Information'}
            )

            plt.title('Mutual Information between Visible and Hidden Units')
            plt.xlabel('Hidden Units')
            plt.ylabel('Visible Units')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"MI visualization saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualization")

    def generate_mi_report(
        self,
        data: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive mutual information analysis report.

        Args:
            data: Input data for analysis
            feature_names: Names of input features
            save_path: Path to save report

        Returns:
            report: Comprehensive MI analysis report
        """
        logger.info("Generating mutual information analysis report...")

        # Compute MI matrix
        mi_matrix = self.compute_mutual_information(data)

        # Compute information flow
        flow_analysis = self.compute_information_flow(data)

        # Analyze representation quality
        quality_analysis = self.analyze_representation_quality(data)

        # Create report
        report = {
            'model_type': type(self.model).__name__,
            'data_shape': list(data.shape),
            'mutual_information_analysis': {
                'mi_matrix_shape': list(mi_matrix.shape),
                'total_mi': float(torch.sum(mi_matrix)),
                'average_mi': float(torch.mean(mi_matrix)),
                'max_mi': float(torch.max(mi_matrix)),
                'min_mi': float(torch.min(mi_matrix)),
            },
            'information_flow': {
                k: {
                    'total_mi': float(v['total_mi']) if isinstance(v.get('total_mi'), torch.Tensor) else v.get('total_mi'),
                    'compression_ratio': float(v.get('compression_ratio', 0))
                } for k, v in flow_analysis.items() if isinstance(v, dict)
            },
            'representation_quality': quality_analysis,
            'feature_analysis': self._analyze_feature_mi(mi_matrix, feature_names)
        }

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"MI analysis report saved to {save_path}")

        return report

    def _analyze_feature_mi(
        self,
        mi_matrix: torch.Tensor,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Analyze MI patterns for individual features."""
        # Sum MI across hidden units for each visible unit
        feature_mi_totals = torch.sum(mi_matrix, dim=1)

        # Get top features by MI
        top_indices = torch.argsort(feature_mi_totals, descending=True)

        feature_analysis = {
            'total_mi_per_feature': feature_mi_totals.cpu().numpy().tolist(),
            'feature_ranking': top_indices.cpu().numpy().tolist(),
            'top_features': []
        }

        # Add feature names if available
        for i, idx in enumerate(top_indices[:5]):  # Top 5 features
            feature_info = {
                'rank': i + 1,
                'index': int(idx),
                'total_mi': float(feature_mi_totals[idx]),
                'name': feature_names[idx] if feature_names else f"feature_{idx}"
            }
            feature_analysis['top_features'].append(feature_info)

        return feature_analysis

    def clear_cache(self) -> None:
        """Clear cached MI computations."""
        self._mi_cache.clear()
        logger.info("Mutual information analysis cache cleared")
