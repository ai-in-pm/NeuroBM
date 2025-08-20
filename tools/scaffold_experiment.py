#!/usr/bin/env python3
"""
Experiment Scaffolding Tool for NeuroBM.

This tool helps researchers quickly set up new experiments with proper
structure, configuration, and documentation.

Features:
- Generate experiment configurations from templates
- Create experiment directories with proper structure
- Set up logging and output directories
- Generate documentation templates
- Create training and evaluation scripts

Usage:
    python tools/scaffold_experiment.py --name my_experiment --base base
    python tools/scaffold_experiment.py --name ptsd_study --base ptsd --type longitudinal
    python tools/scaffold_experiment.py --list-templates
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json
from datetime import datetime
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.config_manager import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Template configurations
EXPERIMENT_TEMPLATES = {
    'basic': {
        'description': 'Basic RBM experiment template',
        'model_type': 'rbm',
        'features': ['standard_training', 'evaluation', 'visualization']
    },
    'comparative': {
        'description': 'Comparative study template with multiple models',
        'model_type': 'multiple',
        'features': ['model_comparison', 'statistical_analysis', 'reporting']
    },
    'longitudinal': {
        'description': 'Longitudinal study template with temporal modeling',
        'model_type': 'crbm',
        'features': ['temporal_modeling', 'sequence_analysis', 'prediction']
    },
    'interpretability': {
        'description': 'Interpretability-focused experiment template',
        'model_type': 'rbm',
        'features': ['saliency_analysis', 'feature_importance', 'visualization']
    },
    'hyperparameter': {
        'description': 'Hyperparameter optimization experiment template',
        'model_type': 'rbm',
        'features': ['parameter_sweep', 'optimization', 'analysis']
    }
}


class ExperimentScaffolder:
    """Tool for scaffolding new experiments."""
    
    def __init__(self, project_root: Path):
        """Initialize scaffolder."""
        self.project_root = project_root
        self.experiments_dir = project_root / 'experiments'
        self.results_dir = project_root / 'results'
        self.templates_dir = project_root / 'templates'
    
    def create_experiment(
        self,
        name: str,
        base_regime: str = 'base',
        template: str = 'basic',
        description: Optional[str] = None,
        author: Optional[str] = None
    ) -> Path:
        """
        Create a new experiment with proper structure.
        
        Args:
            name: Experiment name
            base_regime: Base cognitive regime to use
            template: Experiment template type
            description: Experiment description
            author: Experiment author
            
        Returns:
            Path to created experiment directory
        """
        logger.info(f"Creating experiment: {name}")
        
        # Validate inputs
        if template not in EXPERIMENT_TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Available: {list(EXPERIMENT_TEMPLATES.keys())}")
        
        # Create experiment directory structure
        exp_dir = self.results_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['configs', 'logs', 'checkpoints', 'plots', 'reports', 'data']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)
        
        # Generate experiment configuration
        config_path = self._create_experiment_config(
            name, base_regime, template, description, author, exp_dir
        )
        
        # Create training script
        self._create_training_script(name, template, exp_dir)
        
        # Create evaluation script
        self._create_evaluation_script(name, template, exp_dir)
        
        # Create documentation
        self._create_experiment_documentation(name, template, description, author, exp_dir)
        
        # Create analysis notebook
        self._create_analysis_notebook(name, template, exp_dir)
        
        logger.info(f"✅ Experiment created successfully at: {exp_dir}")
        logger.info(f"📋 Configuration: {config_path}")
        
        return exp_dir
    
    def _create_experiment_config(
        self,
        name: str,
        base_regime: str,
        template: str,
        description: Optional[str],
        author: Optional[str],
        exp_dir: Path
    ) -> Path:
        """Create experiment configuration file."""
        template_info = EXPERIMENT_TEMPLATES[template]
        
        # Load base configuration
        base_config_path = self.experiments_dir / f"{base_regime}.yaml"
        if base_config_path.exists():
            base_config = ConfigManager.load(base_config_path, validate_config=False)
        else:
            base_config = ConfigManager._create_default_template(template_info['model_type'])
        
        # Customize configuration
        config = base_config.copy()
        config.update({
            'name': name,
            'description': description or f"{template.title()} experiment for {base_regime} regime",
            'template': template,
            'base_regime': base_regime,
            'author': author or 'Unknown',
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        })
        
        # Update output directory
        config['output'] = config.get('output', {})
        config['output']['save_dir'] = f"results/{name}"
        
        # Template-specific customizations
        if template == 'longitudinal':
            config['model']['type'] = 'crbm'
            config['model']['architecture'] = config['model'].get('architecture', {})
            config['model']['architecture']['n_history'] = 3
        
        elif template == 'comparative':
            config['comparison'] = {
                'models': ['rbm', 'dbm'],
                'metrics': ['reconstruction_error', 'likelihood', 'sparsity'],
                'statistical_tests': ['t_test', 'wilcoxon']
            }
        
        elif template == 'hyperparameter':
            config['hyperparameter_search'] = {
                'method': 'grid',
                'parameters': {
                    'model.architecture.n_hidden': [128, 256, 512],
                    'training.learning_rate': [0.005, 0.01, 0.02],
                    'training.k_steps': [1, 3, 5]
                }
            }
        
        # Save configuration
        config_path = exp_dir / 'configs' / f"{name}.yaml"
        ConfigManager.save(config, config_path, include_metadata=True)
        
        return config_path
    
    def _create_training_script(self, name: str, template: str, exp_dir: Path) -> None:
        """Create training script for the experiment."""
        script_content = f'''#!/usr/bin/env python3
"""
Training script for {name} experiment.

This script was auto-generated by the NeuroBM scaffolding tool.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.train import main as train_main
import argparse

def main():
    """Main training function."""
    # Set up arguments for this specific experiment
    sys.argv = [
        'train.py',
        '--exp={name}',
        '--config=results/{name}/configs/{name}.yaml'
    ]
    
    # Run training
    train_main()

if __name__ == "__main__":
    main()
'''
        
        script_path = exp_dir / f"train_{name}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
    
    def _create_evaluation_script(self, name: str, template: str, exp_dir: Path) -> None:
        """Create evaluation script for the experiment."""
        script_content = f'''#!/usr/bin/env python3
"""
Evaluation script for {name} experiment.

This script was auto-generated by the NeuroBM scaffolding tool.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from neurobm.training.eval import ModelEvaluator
from neurobm.models.rbm import RestrictedBoltzmannMachine
import torch

def main():
    """Main evaluation function."""
    # Load trained model
    model_path = Path("results/{name}/checkpoints/final_model.pth")
    
    if not model_path.exists():
        print("❌ No trained model found. Please run training first.")
        return
    
    # Load model
    model = RestrictedBoltzmannMachine.load_checkpoint(model_path)
    
    # Create evaluator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(model, device)
    
    # Run evaluation
    print("🔍 Running model evaluation...")
    
    # Add evaluation code here based on template
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    main()
'''
        
        script_path = exp_dir / f"evaluate_{name}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
    
    def _create_experiment_documentation(
        self,
        name: str,
        template: str,
        description: Optional[str],
        author: Optional[str],
        exp_dir: Path
    ) -> None:
        """Create experiment documentation."""
        template_info = EXPERIMENT_TEMPLATES[template]
        
        doc_content = f'''# {name.title()} Experiment

## Overview

**Experiment Name**: {name}  
**Template**: {template} ({template_info['description']})  
**Author**: {author or 'Unknown'}  
**Created**: {datetime.now().strftime('%Y-%m-%d')}  

## Description

{description or f"This experiment uses the {template} template to study cognitive patterns."}

## Objectives

- [ ] Define specific research questions
- [ ] Identify key hypotheses to test
- [ ] Determine success criteria
- [ ] Plan analysis approach

## Methodology

### Model Configuration
- **Model Type**: {template_info['model_type']}
- **Features**: {', '.join(template_info['features'])}

### Data
- **Regime**: Specify the cognitive regime being studied
- **Sample Size**: Define required sample size
- **Generation Method**: Describe data generation approach

### Analysis Plan
- **Primary Metrics**: List key evaluation metrics
- **Statistical Tests**: Define statistical analysis approach
- **Visualization**: Plan for result visualization

## Results

### Training Results
- Training curves and convergence analysis
- Model performance metrics
- Hyperparameter sensitivity analysis

### Evaluation Results
- Test set performance
- Interpretability analysis
- Comparison with baselines

## Conclusions

### Key Findings
- Summarize main results
- Discuss implications
- Note limitations

### Future Work
- Identify follow-up experiments
- Suggest improvements
- Plan extensions

## Files and Structure

```
{name}/
├── configs/           # Experiment configurations
├── logs/             # Training and evaluation logs
├── checkpoints/      # Model checkpoints
├── plots/           # Generated visualizations
├── reports/         # Analysis reports
├── data/            # Experiment-specific data
├── train_{name}.py  # Training script
├── evaluate_{name}.py # Evaluation script
└── README.md        # This documentation
```

## Usage

### Training
```bash
python train_{name}.py
```

### Evaluation
```bash
python evaluate_{name}.py
```

### Analysis
Open and run the analysis notebook: `analysis_{name}.ipynb`

---

**Note**: This experiment was created using the NeuroBM scaffolding tool.
All code and configurations should be reviewed and customized for your specific research needs.
'''
        
        readme_path = exp_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(doc_content)
    
    def _create_analysis_notebook(self, name: str, template: str, exp_dir: Path) -> None:
        """Create analysis Jupyter notebook."""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {name.title()} Experiment Analysis\n",
                        "\n",
                        "This notebook provides analysis tools for the experiment.\n",
                        "\n",
                        "## Setup"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import sys\n",
                        "from pathlib import Path\n",
                        "import torch\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "\n",
                        "# Add project root to path\n",
                        "sys.path.append(str(Path.cwd().parent.parent))\n",
                        "\n",
                        "from neurobm.models.rbm import RestrictedBoltzmannMachine\n",
                        "from neurobm.training.eval import ModelEvaluator\n",
                        "from neurobm.interpret.saliency import SaliencyAnalyzer\n",
                        "\n",
                        "print(f\"Analysis notebook for {name} experiment\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Load Results\n",
                        "\n",
                        "Load the trained model and experiment results."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Load trained model\n",
                        "model_path = Path('checkpoints/final_model.pth')\n",
                        "\n",
                        "if model_path.exists():\n",
                        "    model = RestrictedBoltzmannMachine.load_checkpoint(model_path)\n",
                        "    print(f\"✅ Loaded model from {model_path}\")\n",
                        "else:\n",
                        "    print(\"❌ No trained model found. Please run training first.\")"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Analysis\n",
                        "\n",
                        "Add your analysis code here."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Add analysis code here\n",
                        "pass"
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
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_path = exp_dir / f"analysis_{name}.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    def list_templates(self) -> None:
        """List available experiment templates."""
        print("📋 Available Experiment Templates:")
        print("=" * 50)
        
        for template_name, template_info in EXPERIMENT_TEMPLATES.items():
            print(f"\n🔧 {template_name.upper()}")
            print(f"   Description: {template_info['description']}")
            print(f"   Model Type: {template_info['model_type']}")
            print(f"   Features: {', '.join(template_info['features'])}")


def main():
    """Main scaffolding function."""
    parser = argparse.ArgumentParser(
        description='Scaffold new NeuroBM experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--base', type=str, default='base', help='Base cognitive regime')
    parser.add_argument('--template', type=str, default='basic', help='Experiment template')
    parser.add_argument('--description', type=str, help='Experiment description')
    parser.add_argument('--author', type=str, help='Experiment author')
    parser.add_argument('--list-templates', action='store_true', help='List available templates')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create scaffolder
    scaffolder = ExperimentScaffolder(project_root)
    
    if args.list_templates:
        scaffolder.list_templates()
        return
    
    try:
        # Create experiment
        exp_dir = scaffolder.create_experiment(
            name=args.name,
            base_regime=args.base,
            template=args.template,
            description=args.description,
            author=args.author
        )
        
        print(f"\n🎉 Experiment '{args.name}' created successfully!")
        print(f"📁 Location: {exp_dir}")
        print(f"\n📝 Next steps:")
        print(f"   1. Review and customize the configuration: {exp_dir}/configs/{args.name}.yaml")
        print(f"   2. Run training: python {exp_dir}/train_{args.name}.py")
        print(f"   3. Analyze results: Open {exp_dir}/analysis_{args.name}.ipynb")
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
