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
Comprehensive sampling script for trained NeuroBM models.

This script provides sampling capabilities for all model types:
- Generate new samples from trained models
- Conditional sampling with partial observations
- Temporal sequence generation for CRBMs
- Interpolation between samples
- Visualization of generated samples

Usage:
    # Basic sampling
    python scripts/sample.py --checkpoint=runs/base/final_model.pth --n=100

    # High temperature sampling
    python scripts/sample.py --checkpoint=runs/ptsd/final_model.pth --n=50 --temperature=2.0

    # Conditional sampling
    python scripts/sample.py --checkpoint=runs/base/final_model.pth --conditional --evidence="attention_span=0.8"

    # Sequence generation for CRBM
    python scripts/sample.py --checkpoint=runs/ptsd_pm/final_model.pth --sequence_length=20

    # Interpolation between samples
    python scripts/sample.py --checkpoint=runs/base/final_model.pth --interpolate --n_steps=10
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import yaml

from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.models.dbm import DeepBoltzmannMachine
from neurobm.models.crbm import ConditionalRBM
from neurobm.data.schema import get_schema

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sample from trained NeuroBM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # Sampling configuration
    parser.add_argument('--n', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--n_gibbs', type=int, default=100,
                       help='Number of Gibbs steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for sampling')

    # Conditional sampling
    parser.add_argument('--conditional', action='store_true',
                       help='Enable conditional sampling')
    parser.add_argument('--evidence', type=str,
                       help='Evidence for conditional sampling (e.g., "feature1=0.5,feature2=0.8")')
    parser.add_argument('--mask_prob', type=float, default=0.5,
                       help='Probability of masking features for conditional sampling')

    # Sequence generation (for CRBM)
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Length of sequences to generate (for CRBM)')
    parser.add_argument('--initial_sequence', type=str,
                       help='Path to initial sequence file (for CRBM)')

    # Interpolation
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate interpolations between samples')
    parser.add_argument('--n_steps', type=int, default=10,
                       help='Number of interpolation steps')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Output directory for samples')
    parser.add_argument('--save_format', type=str, choices=['csv', 'npy', 'both'], default='csv',
                       help='Format to save samples')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--regime', type=str,
                       help='Regime name for feature interpretation')

    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    logger.info(f"Using device: {device}")
    return device


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine model type from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_type = config.get('model_type', 'rbm')
    else:
        # Try to infer from state dict keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'A' in state_dict:
            model_type = 'crbm'
        elif any('weights' in key for key in state_dict.keys()):
            model_type = 'dbm'
        else:
            model_type = 'rbm'

    # Create model
    if model_type == 'rbm':
        model = RestrictedBoltzmannMachine.load_checkpoint(checkpoint_path, device)
    elif model_type == 'dbm':
        model = DeepBoltzmannMachine.load_checkpoint(checkpoint_path, device)
    elif model_type == 'crbm':
        model = ConditionalRBM.load_checkpoint(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    logger.info(f"Loaded {model_type.upper()} model from {checkpoint_path}")
    return model


def parse_evidence(evidence_str: str, feature_names: List[str]) -> Dict[str, float]:
    """Parse evidence string into feature values."""
    evidence = {}

    for item in evidence_str.split(','):
        if '=' in item:
            feature, value = item.strip().split('=')
            if feature in feature_names:
                evidence[feature] = float(value)
            else:
                logger.warning(f"Unknown feature in evidence: {feature}")

    return evidence


def generate_samples(
    model,
    n_samples: int,
    temperature: float = 1.0,
    n_gibbs: int = 100,
    batch_size: int = 32,
    device: torch.device = None
) -> torch.Tensor:
    """Generate samples from model."""
    if device is None:
        device = next(model.parameters()).device

    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)

            if isinstance(model, ConditionalRBM):
                # For CRBM, generate sequences
                samples = model.generate_sequence(
                    length=10,  # Default sequence length
                    batch_size=current_batch_size,
                    n_gibbs=n_gibbs
                )
                # Flatten sequences for consistency
                samples = samples.reshape(-1, samples.size(-1))
            else:
                # For RBM/DBM, generate samples
                if hasattr(model, 'sample'):
                    samples = model.sample(current_batch_size, n_gibbs)
                else:
                    # Fallback: start from random and run Gibbs
                    samples = torch.rand(current_batch_size, model.n_visible, device=device)
                    for _ in range(n_gibbs):
                        samples = model.reconstruct(samples, n_gibbs=1)

            all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)[:n_samples]


def conditional_sampling(
    model,
    evidence: Dict[str, float],
    feature_names: List[str],
    n_samples: int,
    n_gibbs: int = 100,
    device: torch.device = None
) -> torch.Tensor:
    """Generate conditional samples given evidence."""
    if device is None:
        device = next(model.parameters()).device

    # Create evidence mask and values
    evidence_mask = torch.zeros(len(feature_names), dtype=torch.bool, device=device)
    evidence_values = torch.zeros(len(feature_names), device=device)

    for feature, value in evidence.items():
        if feature in feature_names:
            idx = feature_names.index(feature)
            evidence_mask[idx] = True
            evidence_values[idx] = value

    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            # Start with random sample
            sample = torch.rand(len(feature_names), device=device)

            # Set evidence
            sample[evidence_mask] = evidence_values[evidence_mask]

            # Run Gibbs sampling while preserving evidence
            for _ in range(n_gibbs):
                if hasattr(model, 'gibbs_step'):
                    sample, _, _, _ = model.gibbs_step(sample.unsqueeze(0))
                    sample = sample.squeeze(0)
                else:
                    sample = model.reconstruct(sample.unsqueeze(0), n_gibbs=1).squeeze(0)

                # Restore evidence
                sample[evidence_mask] = evidence_values[evidence_mask]

            samples.append(sample.cpu())

    return torch.stack(samples)


def interpolate_samples(
    model,
    n_steps: int = 10,
    n_gibbs: int = 100,
    device: torch.device = None
) -> torch.Tensor:
    """Generate interpolations between two random samples."""
    if device is None:
        device = next(model.parameters()).device

    # Generate two endpoint samples
    sample1 = generate_samples(model, 1, n_gibbs=n_gibbs, device=device)
    sample2 = generate_samples(model, 1, n_gibbs=n_gibbs, device=device)

    # Create interpolation
    alphas = torch.linspace(0, 1, n_steps, device=device)
    interpolations = []

    for alpha in alphas:
        interpolated = alpha * sample1 + (1 - alpha) * sample2
        interpolations.append(interpolated.cpu())

    return torch.cat(interpolations, dim=0)


def save_samples(
    samples: torch.Tensor,
    output_dir: Path,
    save_format: str,
    feature_names: Optional[List[str]] = None,
    prefix: str = "samples"
) -> None:
    """Save samples to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_np = samples.numpy()

    if save_format in ['csv', 'both']:
        # Save as CSV
        if feature_names:
            df = pd.DataFrame(samples_np, columns=feature_names)
        else:
            df = pd.DataFrame(samples_np, columns=[f'feature_{i}' for i in range(samples_np.shape[1])])

        csv_path = output_dir / f'{prefix}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved samples to {csv_path}")

    if save_format in ['npy', 'both']:
        # Save as numpy array
        npy_path = output_dir / f'{prefix}.npy'
        np.save(npy_path, samples_np)
        logger.info(f"Saved samples to {npy_path}")


def visualize_samples(
    samples: torch.Tensor,
    output_dir: Path,
    feature_names: Optional[List[str]] = None,
    regime: Optional[str] = None
) -> None:
    """Generate visualizations of samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_np = samples.numpy()

    # Feature distribution plots
    n_features = samples_np.shape[1]
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_features):
        row = i // n_cols
        col = i % n_cols

        feature_name = feature_names[i] if feature_names else f'Feature {i}'

        axes[row, col].hist(samples_np[:, i], bins=30, alpha=0.7, density=True)
        axes[row, col].set_title(feature_name)
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Density')
        axes[row, col].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation heatmap
    if n_features > 1:
        plt.figure(figsize=(10, 8))

        if feature_names:
            df = pd.DataFrame(samples_np, columns=feature_names)
        else:
            df = pd.DataFrame(samples_np, columns=[f'F{i}' for i in range(n_features)])

        correlation_matrix = df.corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlations in Generated Samples')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved visualizations to {output_dir}")


def main():
    """Main sampling function."""
    args = parse_arguments()

    try:
        # Setup device
        device = setup_device(args.device)

        # Load model
        model = load_model(args.checkpoint, device)

        # Get feature names if regime is specified
        feature_names = None
        if args.regime:
            try:
                schema = get_schema(args.regime)
                feature_names = list(schema.features.keys())
            except Exception as e:
                logger.warning(f"Could not load schema for regime {args.regime}: {e}")

        # Setup output directory
        output_dir = Path(args.output_dir)

        # Generate samples based on mode
        if args.conditional and args.evidence:
            if not feature_names:
                raise ValueError("Feature names required for conditional sampling. Specify --regime.")

            evidence = parse_evidence(args.evidence, feature_names)
            logger.info(f"Generating {args.n} conditional samples with evidence: {evidence}")

            samples = conditional_sampling(
                model=model,
                evidence=evidence,
                feature_names=feature_names,
                n_samples=args.n,
                n_gibbs=args.n_gibbs,
                device=device
            )
            prefix = "conditional_samples"

        elif args.interpolate:
            logger.info(f"Generating interpolation with {args.n_steps} steps")

            samples = interpolate_samples(
                model=model,
                n_steps=args.n_steps,
                n_gibbs=args.n_gibbs,
                device=device
            )
            prefix = "interpolation"

        elif isinstance(model, ConditionalRBM) and args.sequence_length:
            logger.info(f"Generating {args.n} sequences of length {args.sequence_length}")

            with torch.no_grad():
                sequences = model.generate_sequence(
                    length=args.sequence_length,
                    batch_size=args.n,
                    n_gibbs=args.n_gibbs
                )

            # Save sequences with temporal structure
            for i, sequence in enumerate(sequences):
                seq_output_dir = output_dir / f"sequence_{i}"
                save_samples(
                    sequence,
                    seq_output_dir,
                    args.save_format,
                    feature_names,
                    f"timesteps"
                )

            # Also save flattened version
            samples = sequences.reshape(-1, sequences.size(-1))
            prefix = "sequence_samples"

        else:
            logger.info(f"Generating {args.n} samples")

            samples = generate_samples(
                model=model,
                n_samples=args.n,
                temperature=args.temperature,
                n_gibbs=args.n_gibbs,
                batch_size=args.batch_size,
                device=device
            )
            prefix = "samples"

        # Save samples
        save_samples(samples, output_dir, args.save_format, feature_names, prefix)

        # Generate visualizations
        if args.visualize:
            visualize_samples(samples, output_dir, feature_names, args.regime)

        logger.info(f"Sampling completed successfully. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        raise


if __name__ == "__main__":
    main()
