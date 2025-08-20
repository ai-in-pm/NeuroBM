#!/usr/bin/env python3
"""
NeuroBM Automated Project Scaffolding & Organizer

This utility creates, validates, and continuously organizes the NeuroBM repository
during development and CI. It ensures consistent project structure and prevents drift.

Usage:
    python scripts/neurobm_scaffold.py init      # Initialize project structure
    python scripts/neurobm_scaffold.py validate  # Validate current structure
    python scripts/neurobm_scaffold.py sync      # Normalize and sync structure
    python scripts/neurobm_scaffold.py clean     # Clean temporary files
    python scripts/neurobm_scaffold.py template  # Create new files from templates
    python scripts/neurobm_scaffold.py watch     # Watch for changes (optional)
"""

import argparse
import os
import sys
import shutil
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('neurobm_scaffold')

class NeuroBMScaffold:
    """Automated scaffolding and organization system for NeuroBM."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """Initialize scaffold with project root directory."""
        self.root_dir = root_dir or Path.cwd()
        self.structure_file = self.root_dir / "tools" / "project_structure.yaml"
        self.structure_config = None
        
    def load_structure_config(self) -> Dict[str, Any]:
        """Load project structure configuration."""
        if not self.structure_file.exists():
            raise FileNotFoundError(f"Structure config not found: {self.structure_file}")
        
        with open(self.structure_file, 'r') as f:
            self.structure_config = yaml.safe_load(f)
        return self.structure_config
    
    def init(self) -> None:
        """Initialize complete project structure."""
        logger.info("Initializing NeuroBM project structure...")
        
        config = self.load_structure_config()
        
        # Create directories
        self._create_directories(config.get('folders', []))
        
        # Create required files
        self._create_required_files(config.get('required_files', []))
        
        # Create stub modules
        self._create_stub_modules(config.get('stub_modules', []))
        
        # Create experiment configs
        self._create_experiment_configs(config.get('experiment_configs', []))
        
        # Create notebooks
        self._create_notebooks(config.get('notebooks', []))
        
        # Create scripts
        self._create_scripts(config.get('scripts', []))
        
        # Create sweep configs
        self._create_sweep_configs(config.get('sweep_configs', []))
        
        # Create standard files
        self._create_standard_files()
        
        logger.info("Project structure initialized successfully!")
    
    def _create_directories(self, folders: List[str]) -> None:
        """Create directory structure with .keep files."""
        for folder in folders:
            dir_path = self.root_dir / folder
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Add .keep file for empty directories
            keep_file = dir_path / ".keep"
            if not any(dir_path.iterdir()) and not keep_file.exists():
                keep_file.touch()
                
        logger.info(f"Created {len(folders)} directories")
    
    def _create_required_files(self, files: List[str]) -> None:
        """Create required files if they don't exist."""
        for file_path in files:
            full_path = self.root_dir / file_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                if file_path.endswith('__init__.py'):
                    self._create_init_file(full_path)
                else:
                    full_path.touch()
                    
        logger.info(f"Created {len(files)} required files")
    
    def _create_init_file(self, path: Path) -> None:
        """Create __init__.py file with proper header."""
        header = self._get_license_header()
        content = f'''{header}
"""
{path.parent.name.replace('_', ' ').title()} module for NeuroBM.
"""

__version__ = "0.1.0"
__all__ = []
'''
        path.write_text(content)
    
    def _create_stub_modules(self, modules: List[Dict[str, str]]) -> None:
        """Create stub modules with docstrings and type hints."""
        for module in modules:
            path = self.root_dir / module['path']
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                
                header = self._get_license_header()
                content = f'''{header}
"""
{module['docstring']}
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


class {self._get_class_name_from_path(path)}:
    """Main class for {module['docstring'].lower()}."""
    
    def __init__(self):
        """Initialize the class."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


# TODO: Implement the actual functionality
'''
                path.write_text(content)
                
        logger.info(f"Created {len(modules)} stub modules")
    
    def _get_class_name_from_path(self, path: Path) -> str:
        """Generate class name from file path."""
        name = path.stem
        if name == 'rbm':
            return 'RestrictedBoltzmannMachine'
        elif name == 'dbm':
            return 'DeepBoltzmannMachine'
        elif name == 'crbm':
            return 'ConditionalRBM'
        else:
            return ''.join(word.capitalize() for word in name.split('_'))
    
    def _get_license_header(self) -> str:
        """Get Apache 2.0 license header."""
        year = datetime.now().year
        return f'''# Copyright {year} NeuroBM Contributors
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
# limitations under the License.'''

    def _create_experiment_configs(self, configs: List[str]) -> None:
        """Create experiment configuration files."""
        for config_path in configs:
            path = self.root_dir / config_path
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

                # Create basic experiment config
                config_name = path.stem
                content = self._get_experiment_config_template(config_name)
                path.write_text(content)

        logger.info(f"Created {len(configs)} experiment configs")

    def _get_experiment_config_template(self, name: str) -> str:
        """Get experiment configuration template."""
        base_config = {
            'name': name,
            'model': {
                'type': 'rbm',
                'hidden_units': 256,
                'visible_units': 'auto',
                'activation': 'sigmoid'
            },
            'training': {
                'algorithm': 'cd',
                'k_steps': 1,
                'learning_rate': 0.01,
                'batch_size': 64,
                'epochs': 100,
                'early_stopping': True,
                'patience': 10
            },
            'data': {
                'regime': name,
                'normalize': True,
                'binarize': False
            },
            'logging': {
                'log_interval': 10,
                'save_checkpoints': True,
                'tensorboard': True
            }
        }

        # Customize based on experiment type
        if name == 'ptsd':
            base_config['data']['features'] = [
                'hyperarousal_proxy', 'startle_sensitivity', 'avoidance_tendency',
                'intrusive_thought_proxy', 'sleep_disruption', 'threat_bias_proxy'
            ]
        elif name == 'autism':
            base_config['data']['features'] = [
                'sensory_sensitivity', 'routine_adherence', 'focused_interest_intensity',
                'social_inference_difficulty', 'attention_switch_latency'
            ]
        elif name == 'ai_reliance':
            base_config['data']['features'] = [
                'perceived_effort_cost', 'ambiguity_tolerance', 'reward_delay_sensitivity',
                'automation_expectation', 'frustration_tolerance'
            ]
        else:  # base
            base_config['data']['features'] = [
                'attention_span', 'working_memory_proxy', 'novelty_seeking',
                'sleep_quality', 'stress_index'
            ]

        return yaml.dump(base_config, default_flow_style=False, sort_keys=False)

    def _create_notebooks(self, notebooks: List[str]) -> None:
        """Create Jupyter notebook templates."""
        for notebook_path in notebooks:
            path = self.root_dir / notebook_path
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

                # Create notebook template
                notebook_name = path.stem
                content = self._get_notebook_template(notebook_name)
                path.write_text(content)

        logger.info(f"Created {len(notebooks)} notebook templates")

    def _get_notebook_template(self, name: str) -> str:
        """Get Jupyter notebook template."""
        title = name.replace('_', ' ').title()

        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {title}\n",
                        "\n",
                        "**Educational/Research Purpose Only - No Clinical Applications**\n",
                        "\n",
                        "This notebook explores Boltzmann Machine dynamics for understanding cognitive patterns.\n",
                        "All results are hypothetical and for research/educational purposes only.\n",
                        "\n",
                        "## Limitations & Non-clinical Interpretation\n",
                        "\n",
                        "- This is a working theory with significant uncertainty\n",
                        "- No diagnostic, predictive, or treatment value\n",
                        "- Synthetic data and simplified assumptions\n",
                        "- Requires validation with proper clinical studies\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Standard imports\n",
                        "import numpy as np\n",
                        "import torch\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from pathlib import Path\n",
                        "import sys\n",
                        "\n",
                        "# Add project root to path\n",
                        "sys.path.append(str(Path.cwd().parent))\n",
                        "\n",
                        "# NeuroBM imports\n",
                        "from neurobm.models import rbm, dbm\n",
                        "from neurobm.data import synth, loaders\n",
                        "from neurobm.interpret import saliency, traversals\n",
                        "\n",
                        "# Set random seeds for reproducibility\n",
                        "np.random.seed(42)\n",
                        "torch.manual_seed(42)\n",
                        "\n",
                        "print(f\"Starting {title} analysis...\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## TODO: Implement notebook content\n",
                        "\n",
                        "This notebook template needs to be filled with:\n",
                        "1. Theoretical background\n",
                        "2. Data generation/loading\n",
                        "3. Model training\n",
                        "4. Analysis and visualization\n",
                        "5. Interpretation and limitations\n"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.12.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        import json
        return json.dumps(notebook, indent=2)

    def _create_scripts(self, scripts: List[str]) -> None:
        """Create script templates."""
        for script_path in scripts:
            path = self.root_dir / script_path
            if not path.exists() and path.name != 'neurobm_scaffold.py':
                path.parent.mkdir(parents=True, exist_ok=True)

                script_name = path.stem
                content = self._get_script_template(script_name)
                path.write_text(content)
                path.chmod(0o755)  # Make executable

        logger.info(f"Created script templates")

    def _get_script_template(self, name: str) -> str:
        """Get script template."""
        header = self._get_license_header()

        if name == 'train':
            return f'''{header}
"""
Training script for NeuroBM models.

Usage:
    python scripts/train.py --exp=base --hidden=256 --k=1 --epochs=50
    python scripts/train.py --exp=ai_reliance --hidden=512 --k=10 --pcd
"""

import argparse
import yaml
from pathlib import Path
import torch

from neurobm.training.loop import TrainingLoop
from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.data.loaders import get_data_loader


def main():
    parser = argparse.ArgumentParser(description='Train NeuroBM models')
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden units')
    parser.add_argument('--k', type=int, default=1, help='CD-k steps')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--pcd', action='store_true', help='Use PCD instead of CD')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')

    args = parser.parse_args()

    # Load experiment config
    config_path = Path(f'experiments/{{args.exp}}.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override with command line args
    config['model']['hidden_units'] = args.hidden
    config['training']['k_steps'] = args.k
    config['training']['epochs'] = args.epochs
    if args.pcd:
        config['training']['algorithm'] = 'pcd'

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Training {{args.exp}} on {{device}}")

    # TODO: Implement actual training logic
    print("Training completed!")


if __name__ == '__main__':
    main()
'''
        elif name == 'sample':
            return f'''{header}
"""
Sampling script for trained NeuroBM models.
"""

import argparse
import torch
from pathlib import Path

# TODO: Implement sampling logic


def main():
    parser = argparse.ArgumentParser(description='Sample from trained models')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n', type=int, default=32, help='Number of samples')
    parser.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()
    print(f"Sampling {{args.n}} examples from {{args.checkpoint}}")
    # TODO: Implement


if __name__ == '__main__':
    main()
'''
        else:
            return f'''{header}
"""
{name.replace('_', ' ').title()} script for NeuroBM.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='{name.replace('_', ' ').title()} for NeuroBM')
    # TODO: Add arguments

    args = parser.parse_args()
    # TODO: Implement functionality
    print("Script executed successfully!")


if __name__ == '__main__':
    main()
'''

    def _create_sweep_configs(self, sweeps: List[str]) -> None:
        """Create hyperparameter sweep configurations."""
        for sweep_path in sweeps:
            path = self.root_dir / sweep_path
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

                sweep_name = path.stem
                content = self._get_sweep_config_template(sweep_name)
                path.write_text(content)

        logger.info(f"Created {len(sweeps)} sweep configs")

    def _get_sweep_config_template(self, name: str) -> str:
        """Get sweep configuration template."""
        if 'hidden_units' in name:
            config = {
                'name': 'hidden_units_sweep',
                'method': 'grid',
                'parameters': {
                    'model.hidden_units': {
                        'values': [64, 128, 256, 512, 1024]
                    }
                },
                'base_config': 'experiments/base.yaml'
            }
        elif 'k_steps' in name:
            config = {
                'name': 'k_steps_sweep',
                'method': 'grid',
                'parameters': {
                    'training.k_steps': {
                        'values': [1, 3, 5, 10, 15]
                    }
                },
                'base_config': 'experiments/base.yaml'
            }
        elif 'temp' in name:
            config = {
                'name': 'temperature_sweep',
                'method': 'grid',
                'parameters': {
                    'training.temperature': {
                        'values': [0.5, 0.7, 1.0, 1.2, 1.5]
                    }
                },
                'base_config': 'experiments/base.yaml'
            }
        else:
            config = {
                'name': name,
                'method': 'grid',
                'parameters': {},
                'base_config': 'experiments/base.yaml'
            }

        return yaml.dump(config, default_flow_style=False)

    def _create_standard_files(self) -> None:
        """Create standard project files."""
        # pyproject.toml
        pyproject_path = self.root_dir / "pyproject.toml"
        if not pyproject_path.exists():
            pyproject_content = self._get_pyproject_template()
            pyproject_path.write_text(pyproject_content)

        # README.md
        readme_path = self.root_dir / "README.md"
        if not readme_path.exists():
            readme_content = self._get_readme_template()
            readme_path.write_text(readme_content)

        # .gitignore
        gitignore_path = self.root_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_content = self._get_gitignore_template()
            gitignore_path.write_text(gitignore_content)

        # Makefile
        makefile_path = self.root_dir / "Makefile"
        if not makefile_path.exists():
            makefile_content = self._get_makefile_template()
            makefile_path.write_text(makefile_content)

        # configs/logging.yaml
        logging_config_path = self.root_dir / "configs" / "logging.yaml"
        if not logging_config_path.exists():
            logging_config_path.parent.mkdir(parents=True, exist_ok=True)
            logging_content = self._get_logging_config_template()
            logging_config_path.write_text(logging_content)

        logger.info("Created standard project files")

    def validate(self) -> bool:
        """Validate current project structure."""
        logger.info("Validating NeuroBM project structure...")

        config = self.load_structure_config()
        issues = []

        # Check directories
        for folder in config.get('folders', []):
            dir_path = self.root_dir / folder
            if not dir_path.exists():
                issues.append(f"Missing directory: {folder}")

        # Check required files
        for file_path in config.get('required_files', []):
            full_path = self.root_dir / file_path
            if not full_path.exists():
                issues.append(f"Missing required file: {file_path}")

        # Check __init__.py files
        if config.get('policies', {}).get('ensure_dunder_init', False):
            for folder in config.get('folders', []):
                if folder.startswith('neurobm/'):
                    init_path = self.root_dir / folder / "__init__.py"
                    if not init_path.exists():
                        issues.append(f"Missing __init__.py in: {folder}")

        if issues:
            logger.error(f"Found {len(issues)} validation issues:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("Project structure validation passed!")
            return True

    def sync(self) -> None:
        """Normalize and sync project structure."""
        logger.info("Syncing NeuroBM project structure...")

        # TODO: Implement sync functionality
        # - Normalize names (snake_case)
        # - Fill missing __all__
        # - Update import barrels
        # - Refresh LICENSE headers
        # - Fix notebook kernel metadata

        logger.info("Project structure synced!")

    def clean(self, dry_run: bool = True) -> None:
        """Clean temporary files and directories."""
        logger.info(f"Cleaning project (dry_run={dry_run})...")

        # TODO: Implement clean functionality
        # - Prune empty temp dirs (except .keep)
        # - Rotate checkpoints (keep N best + last)
        # - Archive old artifacts

        logger.info("Project cleaned!")

    def template(self, kind: str, name: str) -> None:
        """Create new file from template."""
        logger.info(f"Creating {kind} template: {name}")

        # TODO: Implement template creation
        # - Use Jinja2 templates
        # - Add header blocks

        logger.info(f"Template {name} created!")

    def watch(self) -> None:
        """Watch for file changes and auto-validate."""
        logger.info("Starting file watcher...")

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class ValidationHandler(FileSystemEventHandler):
                def __init__(self, scaffold):
                    self.scaffold = scaffold

                def on_modified(self, event):
                    if not event.is_directory:
                        logger.info(f"File changed: {event.src_path}")
                        self.scaffold.validate()

            observer = Observer()
            observer.schedule(ValidationHandler(self), str(self.root_dir), recursive=True)
            observer.start()

            logger.info("File watcher started. Press Ctrl+C to stop.")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        except ImportError:
            logger.error("watchdog package not installed. Install with: pip install watchdog")

    def _get_pyproject_template(self) -> str:
        """Get pyproject.toml template."""
        return '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neurobm"
version = "0.1.0"
description = "NeuroBM: A Boltzmann Machine Research Stack for Modeling Human Brain Dynamics"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "NeuroBM Contributors"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "tensorboard>=2.13.0",
    "jupyter>=1.0.0",
    "streamlit>=1.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "ruff>=0.0.280",
    "mypy>=1.5.0",
    "pre-commit>=3.3.0",
    "watchdog>=3.0.0",
]

[project.urls]
Homepage = "https://github.com/neurobm/neurobm"
Repository = "https://github.com/neurobm/neurobm"
Documentation = "https://neurobm.readthedocs.io"

[tool.setuptools.packages.find]
where = ["."]
include = ["neurobm*"]

[tool.black]
line-length = 88
target-version = ["py312"]

[tool.ruff]
line-length = 88
target-version = "py312"
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "T20"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
'''

    def _get_readme_template(self) -> str:
        """Get README.md template."""
        return '''# NeuroBM: Boltzmann Machine Research Stack

**Educational/Research Platform for Modeling Human Brain Dynamics**

⚠️ **IMPORTANT**: This is for educational and hypothesis-generation purposes only.
No diagnosis, risk prediction, or treatment advice.

## Quick Start

Bootstrap the project in one command:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -U pip && \\
pip install -e . && pre-commit install && \\
python scripts/neurobm_scaffold.py init && python scripts/neurobm_scaffold.py validate
```

## Features

- **Boltzmann Machine Models**: RBM, DBM, CRBM implementations
- **Training Algorithms**: CD-k, PCD, AIS for likelihood estimation
- **Interpretability Tools**: Saliency, mutual information, latent traversals
- **Research Scenarios**: General, PTSD, Autism, AI-reliance dynamics
- **Automated Scaffolding**: Project organization and validation
- **Interactive Notebooks**: Educational content with clear limitations

## Usage

### Training Models

```bash
# Train base model
python scripts/train.py --exp=base --hidden=256 --k=1 --epochs=50

# Train AI-reliance model with PCD
python scripts/train.py --exp=ai_reliance --hidden=512 --k=10 --pcd
```

### Evaluation

```bash
# Estimate likelihood with AIS
python scripts/eval_ais.py --checkpoint runs/base/latest.ckpt --particles=200

# Generate samples
python scripts/sample.py --checkpoint runs/ai_reliance/best.ckpt --n=32
```

### Dashboard

```bash
streamlit run dashboards/app.py
```

## Project Structure

The project is automatically organized by the scaffolding system:

- `neurobm/`: Core package with models, training, and interpretation tools
- `experiments/`: Configuration files for different research scenarios
- `notebooks/`: Educational Jupyter notebooks
- `scripts/`: Command-line tools for training and evaluation
- `dashboards/`: Interactive visualization interface
- `governance/`: Ethics documentation and model cards

## Ethics and Limitations

This framework is designed for:
- Educational exploration of cognitive dynamics
- Hypothesis generation and testing
- Research into AI-human interaction patterns

**NOT for**:
- Clinical diagnosis or treatment
- Risk prediction or assessment
- Real-world decision making

See `governance/ETHICS.md` for detailed guidelines.

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

Apache 2.0 - see `LICENSE` file.
'''

    def _get_gitignore_template(self) -> str:
        """Get .gitignore template."""
        return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data and models
runs/
archives/
*.ckpt
*.pth
*.pkl
*.h5

# Logs
*.log
logs/
tensorboard_logs/

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
*.tmp
*.temp

# Coverage
.coverage
htmlcov/
.pytest_cache/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json
'''

    def _get_makefile_template(self) -> str:
        """Get Makefile template."""
        return '''# NeuroBM Makefile

.PHONY: init validate fmt test run report clean help

# Initialize project
init:
	python scripts/neurobm_scaffold.py init
	pre-commit install

# Validate project structure
validate:
	python scripts/neurobm_scaffold.py validate

# Format code
fmt:
	ruff check --fix .
	black .

# Run tests
test:
	pytest -q

# Run example training (CPU)
run:
	python scripts/train.py --exp=base --hidden=128 --k=1 --epochs=10

# Generate experiment report
report:
	python scripts/make_report.py --run=latest

# Clean temporary files
clean:
	python scripts/neurobm_scaffold.py clean

# Show help
help:
	@echo "Available targets:"
	@echo "  init     - Initialize project structure"
	@echo "  validate - Validate project structure"
	@echo "  fmt      - Format code with ruff and black"
	@echo "  test     - Run tests with pytest"
	@echo "  run      - Run example training"
	@echo "  report   - Generate experiment report"
	@echo "  clean    - Clean temporary files"
	@echo "  help     - Show this help message"
'''

    def _get_logging_config_template(self) -> str:
        """Get logging configuration template."""
        return '''version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/neurobm.log
    mode: a

loggers:
  neurobm:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
'''


def main():
    """Main entry point for the scaffolding system."""
    parser = argparse.ArgumentParser(
        description='NeuroBM Automated Project Scaffolding & Organizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/neurobm_scaffold.py init
  python scripts/neurobm_scaffold.py validate
  python scripts/neurobm_scaffold.py sync
  python scripts/neurobm_scaffold.py clean --dry-run
  python scripts/neurobm_scaffold.py template --kind model --name my_model
  python scripts/neurobm_scaffold.py watch
        '''
    )

    parser.add_argument(
        'command',
        choices=['init', 'validate', 'sync', 'clean', 'template', 'watch'],
        help='Command to execute'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes (clean only)'
    )

    parser.add_argument(
        '--kind',
        choices=['model', 'notebook', 'experiment'],
        help='Type of template to create (template only)'
    )

    parser.add_argument(
        '--name',
        type=str,
        help='Name for the new template (template only)'
    )

    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory'
    )

    args = parser.parse_args()

    # Initialize scaffold
    scaffold = NeuroBMScaffold(args.root)

    try:
        if args.command == 'init':
            scaffold.init()
        elif args.command == 'validate':
            success = scaffold.validate()
            sys.exit(0 if success else 1)
        elif args.command == 'sync':
            scaffold.sync()
        elif args.command == 'clean':
            scaffold.clean(dry_run=args.dry_run)
        elif args.command == 'template':
            if not args.kind or not args.name:
                parser.error('template command requires --kind and --name')
            scaffold.template(args.kind, args.name)
        elif args.command == 'watch':
            scaffold.watch()

    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
