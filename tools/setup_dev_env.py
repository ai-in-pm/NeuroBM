#!/usr/bin/env python3
"""
Development Environment Setup Tool for NeuroBM.

This tool helps set up a complete development environment for NeuroBM,
including dependencies, pre-commit hooks, and development tools.

Features:
- Install Python dependencies
- Set up pre-commit hooks
- Configure development tools
- Create necessary directories
- Set up Jupyter kernel
- Configure IDE settings

Usage:
    python tools/setup_dev_env.py
    python tools/setup_dev_env.py --minimal
    python tools/setup_dev_env.py --jupyter-only
"""

import argparse
import logging
import sys
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DevEnvironmentSetup:
    """Development environment setup tool."""
    
    def __init__(self, project_root: Path):
        """Initialize setup tool."""
        self.project_root = project_root
        self.python_executable = sys.executable
    
    def setup_full_environment(self, minimal: bool = False) -> bool:
        """Set up complete development environment."""
        print("🚀 Setting up NeuroBM Development Environment")
        print("=" * 50)
        
        steps = [
            ("Creating directories", self.create_directories),
            ("Installing dependencies", self.install_dependencies),
        ]
        
        if not minimal:
            steps.extend([
                ("Setting up pre-commit hooks", self.setup_precommit),
                ("Configuring Jupyter", self.setup_jupyter),
                ("Creating IDE configurations", self.setup_ide_configs),
                ("Setting up development tools", self.setup_dev_tools),
            ])
        
        for step_name, step_func in steps:
            print(f"\n🔧 {step_name}...")
            try:
                if step_func():
                    print(f"  ✅ {step_name} completed")
                else:
                    print(f"  ❌ {step_name} failed")
                    return False
            except Exception as e:
                print(f"  💥 {step_name} error: {e}")
                return False
        
        print("\n🎉 Development environment setup completed!")
        self.print_next_steps()
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        directories = [
            'results',
            'logs',
            'checkpoints',
            'data/cache',
            'data/external',
            'plots',
            'reports',
            '.vscode',
            '.idea'
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        try:
            # Install main dependencies
            result = subprocess.run([
                self.python_executable, '-m', 'pip', 'install', '-e', '.'
            ], cwd=self.project_root, check=True, capture_output=True, text=True)
            
            # Install development dependencies
            dev_packages = [
                'jupyter',
                'jupyterlab',
                'pre-commit',
                'black',
                'flake8',
                'mypy',
                'pytest',
                'pytest-cov',
                'sphinx',
                'sphinx-rtd-theme',
                'nbsphinx'
            ]
            
            result = subprocess.run([
                self.python_executable, '-m', 'pip', 'install'
            ] + dev_packages, check=True, capture_output=True, text=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e.stderr}")
            return False
    
    def setup_precommit(self) -> bool:
        """Set up pre-commit hooks."""
        try:
            # Install pre-commit hooks
            result = subprocess.run([
                self.python_executable, '-m', 'pre_commit', 'install'
            ], cwd=self.project_root, check=True, capture_output=True, text=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Pre-commit setup failed: {e.stderr}")
            return True  # Non-critical failure
    
    def setup_jupyter(self) -> bool:
        """Set up Jupyter environment."""
        try:
            # Install Jupyter kernel
            result = subprocess.run([
                self.python_executable, '-m', 'ipykernel', 'install', '--user',
                '--name', 'neurobm', '--display-name', 'NeuroBM'
            ], check=True, capture_output=True, text=True)
            
            # Create Jupyter config
            jupyter_config = {
                "NotebookApp": {
                    "notebook_dir": str(self.project_root),
                    "open_browser": False,
                    "port": 8888
                }
            }
            
            jupyter_config_dir = Path.home() / '.jupyter'
            jupyter_config_dir.mkdir(exist_ok=True)
            
            config_file = jupyter_config_dir / 'jupyter_notebook_config.json'
            with open(config_file, 'w') as f:
                json.dump(jupyter_config, f, indent=2)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Jupyter setup failed: {e.stderr}")
            return True  # Non-critical failure
    
    def setup_ide_configs(self) -> bool:
        """Set up IDE configurations."""
        # VS Code settings
        vscode_settings = {
            "python.defaultInterpreterPath": self.python_executable,
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.formatting.provider": "black",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests"],
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                ".pytest_cache": True,
                ".mypy_cache": True,
                "*.egg-info": True
            },
            "jupyter.notebookFileRoot": "${workspaceFolder}",
            "jupyter.defaultKernel": "neurobm"
        }
        
        vscode_dir = self.project_root / '.vscode'
        vscode_dir.mkdir(exist_ok=True)
        
        settings_file = vscode_dir / 'settings.json'
        with open(settings_file, 'w') as f:
            json.dump(vscode_settings, f, indent=2)
        
        # VS Code launch configuration
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Train Model",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/scripts/train.py",
                    "args": ["--exp=base", "--epochs=10"],
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}"
                },
                {
                    "name": "Run Tests",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/tests/run_tests.py",
                    "args": ["--verbosity=2"],
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}"
                }
            ]
        }
        
        launch_file = vscode_dir / 'launch.json'
        with open(launch_file, 'w') as f:
            json.dump(launch_config, f, indent=2)
        
        return True
    
    def setup_dev_tools(self) -> bool:
        """Set up additional development tools."""
        # Create development scripts
        scripts_dir = self.project_root / 'dev_scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # Quick test script
        quick_test_script = scripts_dir / 'quick_test.py'
        with open(quick_test_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""Quick test script for development."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Run quick tests."""
    print("🧪 Running quick tests...")
    
    # Test imports
    try:
        from neurobm.models.rbm import RestrictedBoltzmannMachine
        from neurobm.data.synth import SyntheticDataGenerator
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test basic functionality
    try:
        rbm = RestrictedBoltzmannMachine(n_visible=5, n_hidden=8)
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(10, method='skewed')
        metrics = rbm.train_batch(data, k=1)
        print("✅ Basic functionality working")
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False
    
    print("🎉 All quick tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
''')
        
        # Make executable
        quick_test_script.chmod(0o755)
        
        # Development environment info script
        env_info_script = scripts_dir / 'env_info.py'
        with open(env_info_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""Display development environment information."""

import sys
import platform
import subprocess
from pathlib import Path

def main():
    """Display environment info."""
    print("🔍 NeuroBM Development Environment Info")
    print("=" * 50)
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Python Executable: {sys.executable}")
    
    # Check key packages
    packages = ['torch', 'numpy', 'matplotlib', 'jupyter']
    print("\\n📦 Key Packages:")
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  {package}: {version}")
        except ImportError:
            print(f"  {package}: Not installed")
    
    # Check CUDA availability
    try:
        import torch
        print(f"\\n🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
''')
        
        env_info_script.chmod(0o755)
        
        return True
    
    def print_next_steps(self) -> None:
        """Print next steps for development."""
        print("\n📋 Next Steps:")
        print("=" * 30)
        print("1. 🧪 Run quick tests:")
        print("   python dev_scripts/quick_test.py")
        print("\n2. 🔍 Validate project:")
        print("   python tools/validate_project.py")
        print("\n3. 🚀 Start Jupyter Lab:")
        print("   jupyter lab")
        print("\n4. 📚 Open example notebook:")
        print("   notebooks/01_theory_primer.ipynb")
        print("\n5. 🏃 Run training example:")
        print("   python scripts/train.py --exp=base --epochs=10")
        print("\n6. 🧪 Run full test suite:")
        print("   python tests/run_tests.py")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description='Set up NeuroBM development environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--minimal', action='store_true', help='Minimal setup only')
    parser.add_argument('--jupyter-only', action='store_true', help='Set up Jupyter only')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create setup tool
    setup_tool = DevEnvironmentSetup(project_root)
    
    if args.jupyter_only:
        print("🔧 Setting up Jupyter environment only...")
        success = setup_tool.setup_jupyter()
    else:
        success = setup_tool.setup_full_environment(minimal=args.minimal)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
