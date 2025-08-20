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
Annealed Importance Sampling for likelihood estimation

This module implements AIS for estimating the partition function and
computing likelihood bounds for Boltzmann machines. Based on:
- Salakhutdinov & Murray (2008)
- Neal (2001) annealed importance sampling

Features:
- Geometric and linear annealing schedules
- Effective sample size diagnostics
- Variance estimation and confidence intervals
- Support for both RBM and DBM models
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
import math

from ..models.rbm import RestrictedBoltzmannMachine
from ..models.dbm import DeepBoltzmannMachine
from ..models.utils import compute_effective_sample_size, log_mean_exp

logger = logging.getLogger(__name__)


class AnnealedImportanceSampling:
    """
    Annealed Importance Sampling for Boltzmann machine likelihood estimation.

    Estimates the partition function Z and computes log-likelihood bounds
    using importance sampling with annealed intermediate distributions.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine],
        n_particles: int = 100,
        n_intermediate: int = 1000,
        schedule: str = 'geometric',
        device: Optional[torch.device] = None
    ):
        """
        Initialize AIS estimator.

        Args:
            model: Boltzmann machine model
            n_particles: Number of AIS particles/chains
            n_intermediate: Number of intermediate distributions
            schedule: Annealing schedule ('geometric', 'linear', 'sigmoid')
            device: Device for computation
        """
        self.model = model
        self.n_particles = n_particles
        self.n_intermediate = n_intermediate
        self.schedule = schedule
        self.device = device or torch.device('cpu')

        # Move model to device
        self.model.to(self.device)

        # Generate annealing schedule
        self.betas = self._generate_schedule()

        # Results storage
        self.log_weights = None
        self.log_z_estimate = None
        self.log_z_variance = None
        self.effective_sample_size = None

    def _generate_schedule(self) -> torch.Tensor:
        """Generate annealing schedule β_k ∈ [0, 1]."""
        if self.schedule == 'geometric':
            # Geometric spacing: β_k = (k/K)^2 for better performance
            betas = torch.linspace(0, 1, self.n_intermediate + 1, device=self.device) ** 2
        elif self.schedule == 'linear':
            # Linear spacing
            betas = torch.linspace(0, 1, self.n_intermediate + 1, device=self.device)
        elif self.schedule == 'sigmoid':
            # Sigmoid spacing for smoother transitions
            x = torch.linspace(-6, 6, self.n_intermediate + 1, device=self.device)
            betas = torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return betas

    def estimate_log_partition_function(
        self,
        verbose: bool = True,
        return_diagnostics: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Estimate log partition function using AIS.

        Args:
            verbose: Whether to show progress
            return_diagnostics: Whether to return diagnostic information

        Returns:
            log_z_estimate: Estimated log partition function
            diagnostics: Diagnostic information (if requested)
        """
        logger.info(f"Starting AIS with {self.n_particles} particles, {self.n_intermediate} steps")

        # Initialize particles from base distribution (uniform for RBM)
        particles = self._initialize_particles()

        # Initialize log weights
        log_weights = torch.zeros(self.n_particles, device=self.device)

        # AIS chain
        progress_bar = tqdm(
            range(len(self.betas) - 1),
            desc="AIS Progress",
            disable=not verbose
        )

        for k in progress_bar:
            beta_k = self.betas[k]
            beta_k1 = self.betas[k + 1]

            # Update log weights: w_k = w_{k-1} + (β_{k+1} - β_k) * [E_0(x) - E_1(x)]
            energy_diff = self._compute_energy_difference(particles, beta_k, beta_k1)
            log_weights += energy_diff

            # Transition step: sample from intermediate distribution
            particles = self._transition_step(particles, beta_k1)

            # Update progress
            if k % 100 == 0:
                current_ess = compute_effective_sample_size(log_weights)
                progress_bar.set_postfix({'ESS': f'{current_ess:.1f}'})

        # Store results
        self.log_weights = log_weights

        # Compute log partition function estimate
        self.log_z_estimate = log_mean_exp(log_weights).item()

        # Compute variance estimate
        normalized_weights = F.softmax(log_weights, dim=0)
        self.log_z_variance = torch.var(log_weights).item()

        # Compute effective sample size
        self.effective_sample_size = compute_effective_sample_size(log_weights)

        logger.info(f"AIS completed: log Z = {self.log_z_estimate:.4f} ± {np.sqrt(self.log_z_variance):.4f}")
        logger.info(f"Effective sample size: {self.effective_sample_size:.1f}/{self.n_particles}")

        if return_diagnostics:
            diagnostics = {
                'log_weights': log_weights.cpu().numpy(),
                'log_z_variance': self.log_z_variance,
                'effective_sample_size': self.effective_sample_size,
                'ess_ratio': self.effective_sample_size / self.n_particles,
                'schedule': self.betas.cpu().numpy()
            }
            return self.log_z_estimate, diagnostics
        else:
            return self.log_z_estimate

    def _initialize_particles(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Initialize particles from base distribution."""
        if isinstance(self.model, RestrictedBoltzmannMachine):
            # For RBM, initialize visible units uniformly
            if self.model.visible_type == "bernoulli":
                particles = torch.bernoulli(
                    torch.full((self.n_particles, self.model.n_visible), 0.5, device=self.device)
                )
            else:  # gaussian
                particles = torch.randn(self.n_particles, self.model.n_visible, device=self.device)
            return particles

        elif isinstance(self.model, DeepBoltzmannMachine):
            # For DBM, initialize all layers
            particles = []
            for i, size in enumerate(self.model.layer_sizes):
                if i == 0 and self.model.visible_type == "gaussian":
                    layer_particles = torch.randn(self.n_particles, size, device=self.device)
                else:
                    layer_particles = torch.bernoulli(
                        torch.full((self.n_particles, size), 0.5, device=self.device)
                    )
                particles.append(layer_particles)
            return particles
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _compute_energy_difference(
        self,
        particles: Union[torch.Tensor, List[torch.Tensor]],
        beta_k: float,
        beta_k1: float
    ) -> torch.Tensor:
        """Compute energy difference for weight update."""
        if isinstance(self.model, RestrictedBoltzmannMachine):
            # For RBM: E_0 is uniform (zero), E_1 is model energy
            # We need to compute energy with hidden units marginalized
            free_energy = self.model.free_energy(particles)
            return -(beta_k1 - beta_k) * free_energy

        elif isinstance(self.model, DeepBoltzmannMachine):
            # For DBM: use full energy
            energy = self.model.energy(particles)
            return -(beta_k1 - beta_k) * energy
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _transition_step(
        self,
        particles: Union[torch.Tensor, List[torch.Tensor]],
        beta: float
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform transition step using tempered Gibbs sampling."""
        # Temporarily set model temperature
        original_temp = self.model.temperature
        self.model.temperature = 1.0 / beta if beta > 0 else float('inf')

        try:
            if isinstance(self.model, RestrictedBoltzmannMachine):
                # Single Gibbs step for RBM
                new_particles, _, _, _ = self.model.gibbs_step(particles)
                return new_particles

            elif isinstance(self.model, DeepBoltzmannMachine):
                # Multiple Gibbs steps for DBM
                new_particles = self.model._gibbs_sampling_with_clamped_visible(particles, n_steps=1)
                return new_particles
            else:
                return particles

        finally:
            # Restore original temperature
            self.model.temperature = original_temp

    def estimate_log_likelihood(
        self,
        data: torch.Tensor,
        use_cached_z: bool = True
    ) -> torch.Tensor:
        """
        Estimate log-likelihood of data.

        Args:
            data: Data samples [batch_size, n_visible]
            use_cached_z: Whether to use cached partition function estimate

        Returns:
            log_likelihood: Log-likelihood estimates [batch_size]
        """
        if not use_cached_z or self.log_z_estimate is None:
            self.estimate_log_partition_function(verbose=False)

        data = data.to(self.device)

        if isinstance(self.model, RestrictedBoltzmannMachine):
            # For RBM: log p(v) = -F(v) - log Z
            free_energy = self.model.free_energy(data)
            log_likelihood = -free_energy - self.log_z_estimate

        elif isinstance(self.model, DeepBoltzmannMachine):
            # For DBM: approximate using mean-field
            # This is an approximation since exact likelihood is intractable
            approx_free_energy = self.model._compute_free_energy(data)
            log_likelihood = -approx_free_energy - self.log_z_estimate
            logger.warning("DBM likelihood is approximate (mean-field)")
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        return log_likelihood

    def compute_confidence_interval(
        self,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for log partition function estimate.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        if self.log_z_estimate is None or self.log_z_variance is None:
            raise ValueError("Must run AIS estimation first")

        # Assume normal distribution for log Z estimate
        std_error = np.sqrt(self.log_z_variance / self.n_particles)
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645

        margin = z_score * std_error
        lower_bound = self.log_z_estimate - margin
        upper_bound = self.log_z_estimate + margin

        return lower_bound, upper_bound

    def run_multiple_chains(
        self,
        n_runs: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run multiple independent AIS chains for better estimates.

        Args:
            n_runs: Number of independent runs
            verbose: Whether to show progress

        Returns:
            results: Dictionary with aggregated results
        """
        logger.info(f"Running {n_runs} independent AIS chains")

        estimates = []
        variances = []
        ess_values = []

        for run in range(n_runs):
            if verbose:
                logger.info(f"AIS run {run + 1}/{n_runs}")

            # Reset state for new run
            self.log_weights = None
            self.log_z_estimate = None

            # Run AIS
            log_z, diagnostics = self.estimate_log_partition_function(
                verbose=False, return_diagnostics=True
            )

            estimates.append(log_z)
            variances.append(diagnostics['log_z_variance'])
            ess_values.append(diagnostics['effective_sample_size'])

        # Aggregate results
        estimates = np.array(estimates)
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        mean_variance = np.mean(variances)
        mean_ess = np.mean(ess_values)

        results = {
            'mean_log_z': mean_estimate,
            'std_log_z': std_estimate,
            'individual_estimates': estimates,
            'mean_variance': mean_variance,
            'mean_ess': mean_ess,
            'ess_ratio': mean_ess / self.n_particles,
            'n_runs': n_runs
        }

        logger.info(f"Multi-chain AIS: log Z = {mean_estimate:.4f} ± {std_estimate:.4f}")
        logger.info(f"Average ESS: {mean_ess:.1f}/{self.n_particles}")

        # Update stored estimates with best values
        self.log_z_estimate = mean_estimate
        self.log_z_variance = std_estimate ** 2
        self.effective_sample_size = mean_ess

        return results

    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """
        Plot AIS diagnostics.

        Args:
            save_path: Path to save plot (if None, display only)
        """
        try:
            import matplotlib.pyplot as plt

            if self.log_weights is None:
                raise ValueError("Must run AIS estimation first")

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Log weights histogram
            axes[0, 0].hist(self.log_weights.cpu().numpy(), bins=50, alpha=0.7)
            axes[0, 0].set_title('Log Weights Distribution')
            axes[0, 0].set_xlabel('Log Weight')
            axes[0, 0].set_ylabel('Frequency')

            # Annealing schedule
            axes[0, 1].plot(self.betas.cpu().numpy())
            axes[0, 1].set_title('Annealing Schedule')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('β')

            # Effective sample size over time (if available)
            axes[1, 0].text(0.5, 0.5, f'ESS: {self.effective_sample_size:.1f}\n'
                                     f'ESS Ratio: {self.effective_sample_size/self.n_particles:.3f}',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Effective Sample Size')

            # Log Z estimate with confidence interval
            if self.log_z_variance is not None:
                lower, upper = self.compute_confidence_interval()
                axes[1, 1].errorbar([0], [self.log_z_estimate],
                                   yerr=[[self.log_z_estimate - lower], [upper - self.log_z_estimate]],
                                   fmt='o', capsize=5)
                axes[1, 1].set_title('Log Z Estimate')
                axes[1, 1].set_ylabel('Log Z')
                axes[1, 1].set_xlim(-0.5, 0.5)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"AIS diagnostics saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available, cannot plot diagnostics")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of AIS results."""
        if self.log_z_estimate is None:
            return {"status": "not_run"}

        summary = {
            "log_z_estimate": self.log_z_estimate,
            "log_z_variance": self.log_z_variance,
            "effective_sample_size": self.effective_sample_size,
            "ess_ratio": self.effective_sample_size / self.n_particles,
            "n_particles": self.n_particles,
            "n_intermediate": self.n_intermediate,
            "schedule": self.schedule,
            "status": "completed"
        }

        if self.log_z_variance is not None:
            lower, upper = self.compute_confidence_interval()
            summary["confidence_interval_95"] = (lower, upper)

        return summary


# Utility functions for AIS analysis
def compare_ais_schedules(
    model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine],
    schedules: List[str] = ['geometric', 'linear', 'sigmoid'],
    n_particles: int = 100,
    n_intermediate: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different AIS annealing schedules.

    Args:
        model: Boltzmann machine model
        schedules: List of schedule types to compare
        n_particles: Number of particles for each run
        n_intermediate: Number of intermediate steps

    Returns:
        results: Dictionary with results for each schedule
    """
    results = {}

    for schedule in schedules:
        logger.info(f"Testing {schedule} schedule")

        ais = AnnealedImportanceSampling(
            model=model,
            n_particles=n_particles,
            n_intermediate=n_intermediate,
            schedule=schedule
        )

        log_z, diagnostics = ais.estimate_log_partition_function(
            verbose=False, return_diagnostics=True
        )

        results[schedule] = {
            'log_z': log_z,
            'ess': diagnostics['effective_sample_size'],
            'ess_ratio': diagnostics['ess_ratio'],
            'variance': diagnostics['log_z_variance']
        }

    return results
