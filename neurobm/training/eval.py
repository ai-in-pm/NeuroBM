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
Evaluation metrics and validation procedures.

This module provides comprehensive evaluation tools for Boltzmann machines:
- Reconstruction quality metrics
- Likelihood estimation and bounds
- Model comparison utilities
- Statistical significance testing
- Visualization helpers for evaluation results
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.rbm import RestrictedBoltzmannMachine
from ..models.dbm import DeepBoltzmannMachine
from ..models.crbm import ConditionalRBM
from .ais import AnnealedImportanceSampling

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation suite for Boltzmann machines.

    Provides methods for evaluating model quality, comparing models,
    and generating evaluation reports.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine, ConditionalRBM],
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Boltzmann machine model to evaluate
            device: Device for computation
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        # Initialize AIS for likelihood estimation
        self.ais = AnnealedImportanceSampling(model, device=self.device)

        # Evaluation history
        self.evaluation_history = {
            'reconstruction_metrics': [],
            'likelihood_estimates': [],
            'generation_quality': []
        }

    def reconstruction_metrics(
        self,
        test_data: torch.Tensor,
        n_gibbs: int = 1
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.

        Args:
            test_data: Test dataset [n_samples, n_features]
            n_gibbs: Number of Gibbs steps for reconstruction

        Returns:
            metrics: Dictionary of reconstruction metrics
        """
        self.model.eval()

        with torch.no_grad():
            test_data = test_data.to(self.device)

            # Get reconstructions
            if isinstance(self.model, ConditionalRBM):
                # For CRBM, we need sequences
                if test_data.dim() == 2:
                    # Convert to sequence format
                    seq_len = min(10, test_data.size(0))
                    test_sequences = test_data[:seq_len].unsqueeze(0)
                    reconstructions = self.model.predict_next(
                        test_sequences[:, :-1, :],
                        n_steps=1,
                        n_gibbs=n_gibbs
                    ).squeeze(1)
                    targets = test_sequences[:, -1:, :].squeeze(1)
                else:
                    reconstructions = self.model.predict_next(
                        test_data[:, :-1, :],
                        n_steps=1,
                        n_gibbs=n_gibbs
                    ).squeeze(1)
                    targets = test_data[:, -1, :]
            else:
                # For RBM/DBM
                reconstructions = self.model.reconstruct(test_data, n_gibbs=n_gibbs)
                targets = test_data

            # Compute metrics
            mse = F.mse_loss(reconstructions, targets).item()
            mae = F.l1_loss(reconstructions, targets).item()

            # Binary cross-entropy for binary data
            if self.model.visible_type == "bernoulli":
                bce = F.binary_cross_entropy(
                    torch.sigmoid(reconstructions),
                    targets,
                    reduction='mean'
                ).item()
            else:
                bce = None

            # Correlation coefficient
            recon_flat = reconstructions.flatten().cpu().numpy()
            target_flat = targets.flatten().cpu().numpy()
            correlation = np.corrcoef(recon_flat, target_flat)[0, 1]

            # Structural similarity (simplified)
            ssim = self._compute_ssim(reconstructions, targets)

            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'correlation': correlation,
                'ssim': ssim
            }

            if bce is not None:
                metrics['bce'] = bce

            # Store in history
            self.evaluation_history['reconstruction_metrics'].append(metrics)

            return metrics

    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute simplified structural similarity index.

        Args:
            x: First tensor
            y: Second tensor

        Returns:
            SSIM value
        """
        # Simplified SSIM computation
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)

        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

        ssim = numerator / denominator
        return ssim.item()

    def likelihood_estimation(
        self,
        test_data: torch.Tensor,
        n_particles: int = 100,
        n_runs: int = 5
    ) -> Dict[str, float]:
        """
        Estimate log-likelihood using AIS.

        Args:
            test_data: Test dataset
            n_particles: Number of AIS particles
            n_runs: Number of independent runs for variance estimation

        Returns:
            likelihood_stats: Dictionary of likelihood statistics
        """
        if isinstance(self.model, ConditionalRBM):
            logger.warning("Likelihood estimation not implemented for CRBM")
            return {'log_likelihood': float('nan'), 'std': float('nan')}

        self.model.eval()
        test_data = test_data.to(self.device)

        # Run multiple AIS estimates
        log_likelihoods = []

        for run in range(n_runs):
            try:
                # Estimate partition function
                log_z_estimate, log_z_std = self.ais.estimate_partition_function(n_particles)

                # Compute log-likelihood for test data
                with torch.no_grad():
                    free_energies = self.model.free_energy(test_data)
                    log_likelihood = -torch.mean(free_energies).item() - log_z_estimate

                log_likelihoods.append(log_likelihood)

            except Exception as e:
                logger.warning(f"AIS run {run} failed: {e}")
                continue

        if not log_likelihoods:
            return {'log_likelihood': float('nan'), 'std': float('nan')}

        # Compute statistics
        mean_ll = np.mean(log_likelihoods)
        std_ll = np.std(log_likelihoods)

        stats = {
            'log_likelihood': mean_ll,
            'std': std_ll,
            'n_runs': len(log_likelihoods),
            'all_estimates': log_likelihoods
        }

        # Store in history
        self.evaluation_history['likelihood_estimates'].append(stats)

        return stats

    def generation_quality(
        self,
        reference_data: torch.Tensor,
        n_samples: int = 1000,
        n_gibbs: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate quality of generated samples.

        Args:
            reference_data: Reference dataset for comparison
            n_samples: Number of samples to generate
            n_gibbs: Number of Gibbs steps for generation

        Returns:
            quality_metrics: Dictionary of generation quality metrics
        """
        self.model.eval()
        reference_data = reference_data.to(self.device)

        with torch.no_grad():
            if isinstance(self.model, ConditionalRBM):
                # Generate sequences
                seq_len = reference_data.size(1) if reference_data.dim() == 3 else 10
                generated_samples = self.model.generate_sequence(
                    length=seq_len,
                    batch_size=n_samples,
                    n_gibbs=n_gibbs
                )
                # Flatten for comparison
                generated_samples = generated_samples.reshape(-1, generated_samples.size(-1))
                reference_flat = reference_data.reshape(-1, reference_data.size(-1))
            else:
                # Generate samples for RBM/DBM
                if hasattr(self.model, 'sample'):
                    generated_samples = self.model.sample(n_samples, n_gibbs)
                else:
                    # Fallback: start from random and run Gibbs
                    samples = torch.rand(n_samples, self.model.n_visible, device=self.device)
                    for _ in range(n_gibbs):
                        samples = self.model.reconstruct(samples, n_gibbs=1)
                    generated_samples = samples

                reference_flat = reference_data

        # Convert to numpy for analysis
        gen_np = generated_samples.cpu().numpy()
        ref_np = reference_flat.cpu().numpy()

        # Statistical tests
        metrics = {}

        # Feature-wise statistics
        gen_mean = np.mean(gen_np, axis=0)
        ref_mean = np.mean(ref_np, axis=0)
        gen_std = np.std(gen_np, axis=0)
        ref_std = np.std(ref_np, axis=0)

        # Mean and variance differences
        metrics['mean_diff'] = np.mean(np.abs(gen_mean - ref_mean))
        metrics['std_diff'] = np.mean(np.abs(gen_std - ref_std))

        # Kolmogorov-Smirnov test for each feature
        ks_statistics = []
        ks_pvalues = []

        for i in range(min(gen_np.shape[1], ref_np.shape[1])):
            try:
                ks_stat, ks_pval = stats.ks_2samp(gen_np[:, i], ref_np[:, i])
                ks_statistics.append(ks_stat)
                ks_pvalues.append(ks_pval)
            except:
                continue

        if ks_statistics:
            metrics['ks_statistic'] = np.mean(ks_statistics)
            metrics['ks_pvalue'] = np.mean(ks_pvalues)

        # Wasserstein distance (simplified)
        try:
            from scipy.stats import wasserstein_distance
            wd_distances = []
            for i in range(min(10, gen_np.shape[1])):  # Limit to first 10 features
                wd = wasserstein_distance(gen_np[:, i], ref_np[:, i])
                wd_distances.append(wd)
            metrics['wasserstein_distance'] = np.mean(wd_distances)
        except ImportError:
            logger.warning("scipy.stats.wasserstein_distance not available")

        # Store in history
        self.evaluation_history['generation_quality'].append(metrics)

        return metrics

    def model_comparison(
        self,
        other_model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine, ConditionalRBM],
        test_data: torch.Tensor,
        metrics: List[str] = ['reconstruction', 'likelihood']
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare this model with another model.

        Args:
            other_model: Other model to compare against
            test_data: Test dataset
            metrics: List of metrics to compare

        Returns:
            comparison: Dictionary of comparison results
        """
        comparison = {}

        # Evaluate current model
        if 'reconstruction' in metrics:
            current_recon = self.reconstruction_metrics(test_data)

            # Evaluate other model
            other_evaluator = ModelEvaluator(other_model, self.device)
            other_recon = other_evaluator.reconstruction_metrics(test_data)

            comparison['reconstruction'] = {
                'current': current_recon,
                'other': other_recon,
                'improvement': {
                    key: current_recon[key] - other_recon[key]
                    for key in current_recon.keys()
                    if key in other_recon
                }
            }

        if 'likelihood' in metrics:
            current_ll = self.likelihood_estimation(test_data)

            other_evaluator = ModelEvaluator(other_model, self.device)
            other_ll = other_evaluator.likelihood_estimation(test_data)

            comparison['likelihood'] = {
                'current': current_ll,
                'other': other_ll,
                'improvement': current_ll['log_likelihood'] - other_ll['log_likelihood']
            }

        return comparison

    def comprehensive_evaluation(
        self,
        test_data: torch.Tensor,
        reference_data: Optional[torch.Tensor] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation and generate report.

        Args:
            test_data: Test dataset
            reference_data: Reference data for generation quality (optional)
            save_dir: Directory to save evaluation results

        Returns:
            evaluation_report: Complete evaluation results
        """
        logger.info("Running comprehensive model evaluation...")

        report = {
            'model_info': {
                'type': type(self.model).__name__,
                'n_visible': getattr(self.model, 'n_visible', None),
                'n_hidden': getattr(self.model, 'n_hidden', None),
                'visible_type': getattr(self.model, 'visible_type', None)
            }
        }

        # Reconstruction metrics
        logger.info("Computing reconstruction metrics...")
        report['reconstruction'] = self.reconstruction_metrics(test_data)

        # Likelihood estimation
        logger.info("Estimating likelihood...")
        report['likelihood'] = self.likelihood_estimation(test_data)

        # Generation quality
        if reference_data is not None:
            logger.info("Evaluating generation quality...")
            report['generation'] = self.generation_quality(reference_data)

        # Save results
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save numerical results
            import json
            with open(save_dir / 'evaluation_report.json', 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_report = self._convert_for_json(report)
                json.dump(json_report, f, indent=2)

            # Generate plots
            self._generate_evaluation_plots(report, save_dir)

        return report

    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _generate_evaluation_plots(self, report: Dict[str, Any], save_dir: Path):
        """Generate evaluation plots."""
        try:
            # Reconstruction metrics plot
            if 'reconstruction' in report:
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics = report['reconstruction']
                metric_names = [k for k in metrics.keys() if isinstance(metrics[k], (int, float))]
                metric_values = [metrics[k] for k in metric_names]

                ax.bar(metric_names, metric_values)
                ax.set_title('Reconstruction Metrics')
                ax.set_ylabel('Value')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(save_dir / 'reconstruction_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Training history plots if available
            if self.evaluation_history['reconstruction_metrics']:
                self._plot_evaluation_history(save_dir)

        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")

    def _plot_evaluation_history(self, save_dir: Path):
        """Plot evaluation history over time."""
        history = self.evaluation_history['reconstruction_metrics']
        if not history:
            return

        # Extract metrics over time
        metrics_over_time = {}
        for metric_name in history[0].keys():
            if isinstance(history[0][metric_name], (int, float)):
                metrics_over_time[metric_name] = [h[metric_name] for h in history]

        if not metrics_over_time:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, (metric_name, values) in enumerate(metrics_over_time.items()):
            if i >= len(axes):
                break

            axes[i].plot(values)
            axes[i].set_title(f'{metric_name.upper()} over evaluations')
            axes[i].set_xlabel('Evaluation')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True)

        # Hide unused subplots
        for i in range(len(metrics_over_time), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_dir / 'evaluation_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def compute_model_complexity(
    model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine, ConditionalRBM]
) -> Dict[str, int]:
    """
    Compute model complexity metrics.

    Args:
        model: Boltzmann machine model

    Returns:
        complexity_metrics: Dictionary of complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

    # Model-specific metrics
    if hasattr(model, 'n_visible'):
        metrics['n_visible'] = model.n_visible
    if hasattr(model, 'n_hidden'):
        metrics['n_hidden'] = model.n_hidden
    if hasattr(model, 'layer_sizes'):
        metrics['layer_sizes'] = model.layer_sizes
        metrics['n_layers'] = len(model.layer_sizes)

    return metrics


def statistical_significance_test(
    model1_scores: List[float],
    model2_scores: List[float],
    test: str = 'ttest'
) -> Dict[str, float]:
    """
    Test statistical significance of difference between model performances.

    Args:
        model1_scores: Performance scores for model 1
        model2_scores: Performance scores for model 2
        test: Statistical test to use ('ttest', 'wilcoxon', 'mannwhitney')

    Returns:
        test_results: Dictionary with test statistics and p-values
    """
    if test == 'ttest':
        statistic, pvalue = stats.ttest_ind(model1_scores, model2_scores)
    elif test == 'wilcoxon':
        statistic, pvalue = stats.wilcoxon(model1_scores, model2_scores)
    elif test == 'mannwhitney':
        statistic, pvalue = stats.mannwhitneyu(model1_scores, model2_scores)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': pvalue < 0.05,
        'test_type': test
    }
