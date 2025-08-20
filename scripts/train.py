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
Comprehensive training script for NeuroBM models.

This script provides a complete training pipeline with support for:
- Multiple model types (RBM, DBM, CRBM)
- Flexible experiment configurations
- Comprehensive logging and monitoring
- Model evaluation and checkpointing
- Hyperparameter overrides

Usage:
    # Basic training
    python scripts/train.py --exp=base --hidden=256 --k=1 --epochs=50

    # Advanced training with PCD
    python scripts/train.py --exp=ptsd --hidden=512 --k=10 --pcd --lr=0.005

    # CRBM training for temporal data
    python scripts/train.py --exp=ptsd_pm --model=crbm --history=3

    # Resume from checkpoint
    python scripts/train.py --exp=base --resume=runs/base/checkpoint_50.pth

    # Hyperparameter sweep
    python scripts/train.py --exp=base --sweep --n_trials=20
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import wandb
from datetime import datetime

from neurobm.training.loop import TrainingLoop
from neurobm.training.callbacks import get_standard_callbacks
from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.models.dbm import DeepBoltzmannMachine
from neurobm.models.crbm import ConditionalRBM
from neurobm.data.loaders import get_data_loader, get_sequence_loader
from neurobm.data.synth import SyntheticDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train NeuroBM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--exp', type=str, required=True,
                       help='Experiment name (config file in experiments/)')

    # Model configuration overrides
    parser.add_argument('--model', type=str, choices=['rbm', 'dbm', 'crbm'],
                       help='Model type override')
    parser.add_argument('--hidden', type=int, help='Number of hidden units')
    parser.add_argument('--layers', type=str, help='Layer sizes for DBM (e.g., "100,50,25")')
    parser.add_argument('--history', type=int, help='History length for CRBM')
    parser.add_argument('--visible_type', type=str, choices=['bernoulli', 'gaussian'],
                       help='Visible unit type')

    # Training configuration overrides
    parser.add_argument('--k', type=int, help='CD-k steps')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--momentum', type=float, help='Momentum')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--pcd', action='store_true', help='Use PCD instead of CD')
    parser.add_argument('--temperature', type=float, help='Sampling temperature')

    # Data configuration overrides
    parser.add_argument('--n_samples', type=int, help='Number of training samples')
    parser.add_argument('--noise_level', type=float, help='Data noise level')

    # Training control
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    parser.add_argument('--no_save', action='store_true', help='Disable saving')
    parser.add_argument('--dry_run', action='store_true', help='Dry run without training')

    # Hyperparameter search
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of sweep trials')

    # Logging and monitoring
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    return parser.parse_args()


def load_config(exp_name: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = Path(f'experiments/{exp_name}.yaml')

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded experiment config: {config_path}")
    return config


def override_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config with command line arguments."""
    # Model overrides
    if args.model:
        config['model']['type'] = args.model
    if args.hidden:
        if 'architecture' in config['model']:
            config['model']['architecture']['n_hidden'] = args.hidden
        else:
            config['model']['hidden_units'] = args.hidden
    if args.layers and args.model == 'dbm':
        layer_sizes = [int(x) for x in args.layers.split(',')]
        config['model']['architecture']['layer_sizes'] = layer_sizes
    if args.history and args.model == 'crbm':
        config['model']['architecture']['n_history'] = args.history
    if args.visible_type:
        if 'architecture' in config['model']:
            config['model']['architecture']['visible_type'] = args.visible_type
        else:
            config['model']['visible_type'] = args.visible_type

    # Training overrides
    if args.k:
        config['training']['k_steps'] = args.k
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
        if 'optimizer' in config['training']:
            config['training']['optimizer']['learning_rate'] = args.lr
    if args.momentum:
        config['training']['momentum'] = args.momentum
        if 'optimizer' in config['training']:
            config['training']['optimizer']['momentum'] = args.momentum
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
        if 'optimizer' in config['training']:
            config['training']['optimizer']['weight_decay'] = args.weight_decay
    if args.pcd:
        config['training']['algorithm'] = 'pcd'
        config['training']['persistent'] = True
    if args.temperature:
        if 'parameters' in config['model']:
            config['model']['parameters']['temperature'] = args.temperature
        else:
            config['model']['temperature'] = args.temperature

    # Data overrides
    if args.n_samples:
        if 'generation' in config['data']:
            config['data']['generation']['n_samples'] = args.n_samples
        else:
            config['data']['n_samples'] = args.n_samples
    if args.noise_level:
        if 'generation' in config['data']:
            config['data']['generation']['noise_level'] = args.noise_level
        else:
            config['data']['noise_level'] = args.noise_level

    # Logging overrides
    if args.wandb:
        config['logging']['wandb']['enabled'] = True
    if args.tensorboard:
        config['logging']['tensorboard']['enabled'] = True

    return config


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    logger.info(f"Using device: {device}")
    return device


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model based on configuration."""
    model_type = config['model']['type'].lower()

    # Get model architecture config
    if 'architecture' in config['model']:
        arch_config = config['model']['architecture']
    else:
        # Legacy config format
        arch_config = {
            'n_visible': len(config['data']['features']),
            'n_hidden': config['model']['hidden_units'],
            'visible_type': config['model'].get('visible_type', 'bernoulli')
        }

    # Get model parameters
    if 'parameters' in config['model']:
        param_config = config['model']['parameters']
    else:
        # Legacy config format
        param_config = {
            'learning_rate': config['training'].get('learning_rate', 0.01),
            'momentum': config['training'].get('momentum', 0.9),
            'weight_decay': config['training'].get('weight_decay', 0.0001)
        }

    # Determine visible units count
    if arch_config.get('n_visible') == 'auto':
        arch_config['n_visible'] = len(config['data']['features'])

    # Create model
    if model_type == 'rbm':
        model = RestrictedBoltzmannMachine(
            n_visible=arch_config['n_visible'],
            n_hidden=arch_config['n_hidden'],
            visible_type=arch_config.get('visible_type', 'bernoulli'),
            device=device,
            **param_config
        )
    elif model_type == 'dbm':
        layer_sizes = arch_config.get('layer_sizes', [arch_config['n_visible'], arch_config['n_hidden']])
        model = DeepBoltzmannMachine(
            layer_sizes=layer_sizes,
            visible_type=arch_config.get('visible_type', 'bernoulli'),
            device=device,
            **param_config
        )
    elif model_type == 'crbm':
        model = ConditionalRBM(
            n_visible=arch_config['n_visible'],
            n_hidden=arch_config['n_hidden'],
            n_history=arch_config.get('n_history', 3),
            visible_type=arch_config.get('visible_type', 'bernoulli'),
            device=device,
            **param_config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Created {model_type.upper()} model: {model}")
    return model


def main():
    """Main training function."""
    args = parse_arguments()

    # Setup logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        log_level = getattr(logging, args.log_level.upper())
        logging.getLogger().setLevel(log_level)

    try:
        # Load and override configuration
        config = load_config(args.exp)
        config = override_config(config, args)

        # Setup device
        device = setup_device(args.device)

        if args.dry_run:
            logger.info("Dry run mode - configuration loaded successfully")
            logger.info(f"Config: {yaml.dump(config, default_flow_style=False)}")
            return

        # Create model
        model = create_model(config, device)

        # Create data loaders
        train_loader = get_data_loader(
            regime_name=config['data']['regime'],
            batch_size=config['training']['batch_size']
        )

        # Create training loop
        training_loop = TrainingLoop(model)

        # Resume from checkpoint if specified
        if args.resume:
            training_loop.load_checkpoint(args.resume)
            logger.info(f"Resumed training from: {args.resume}")

        # Train model
        logger.info("Starting training...")
        training_loop.train(train_loader, epochs=config['training']['epochs'])

        logger.info(f"Training completed successfully for experiment: {args.exp}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
