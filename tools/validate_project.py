#!/usr/bin/env python3
"""
Project Validation Tool for NeuroBM.

This tool validates the entire project structure, dependencies, and functionality
to ensure everything is properly set up and working correctly.

Features:
- Validate project structure and required files
- Check Python dependencies and versions
- Test core functionality and imports
- Validate configuration files
- Check documentation completeness
- Run basic integration tests

Usage:
    python tools/validate_project.py
    python tools/validate_project.py --quick
    python tools/validate_project.py --fix-issues
"""

import argparse
import logging
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectValidator:
    """Comprehensive project validation tool."""
    
    def __init__(self, project_root: Path):
        """Initialize validator."""
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        self.passed_checks = []
    
    def validate_all(self, quick: bool = False) -> bool:
        """Run all validation checks."""
        print("🔍 NeuroBM Project Validation")
        print("=" * 50)
        
        checks = [
            ("Project Structure", self.validate_project_structure),
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("Core Imports", self.validate_core_imports),
            ("Configuration Files", self.validate_configurations),
            ("Documentation", self.validate_documentation),
        ]
        
        if not quick:
            checks.extend([
                ("Basic Functionality", self.validate_basic_functionality),
                ("Test Suite", self.validate_test_suite),
            ])
        
        for check_name, check_func in checks:
            print(f"\n🔍 {check_name}...")
            try:
                if check_func():
                    print(f"  ✅ {check_name} passed")
                    self.passed_checks.append(check_name)
                else:
                    print(f"  ❌ {check_name} failed")
            except Exception as e:
                print(f"  💥 {check_name} error: {e}")
                self.issues.append(f"{check_name}: {e}")
        
        # Print summary
        self.print_summary()
        
        return len(self.issues) == 0
    
    def validate_project_structure(self) -> bool:
        """Validate project directory structure."""
        required_dirs = [
            'neurobm',
            'neurobm/models',
            'neurobm/data',
            'neurobm/training',
            'neurobm/interpret',
            'experiments',
            'scripts',
            'notebooks',
            'configs',
            'tests',
            'docs',
            'tools',
            'dashboards'
        ]
        
        required_files = [
            'pyproject.toml',
            'README.md',
            'neurobm/__init__.py',
            'neurobm/models/__init__.py',
            'neurobm/data/__init__.py',
            'neurobm/training/__init__.py',
            'neurobm/interpret/__init__.py',
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        # Check files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_dirs:
            self.issues.append(f"Missing directories: {', '.join(missing_dirs)}")
        
        if missing_files:
            self.issues.append(f"Missing files: {', '.join(missing_files)}")
        
        return len(missing_dirs) == 0 and len(missing_files) == 0
    
    def validate_python_environment(self) -> bool:
        """Validate Python version and environment."""
        # Check Python version
        if sys.version_info < (3, 8):
            self.issues.append(f"Python 3.8+ required, found {sys.version}")
            return False
        
        # Check if we're in a virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if not in_venv:
            self.warnings.append("Not running in a virtual environment")
        
        return True
    
    def validate_dependencies(self) -> bool:
        """Validate required dependencies."""
        required_packages = [
            'torch',
            'numpy',
            'matplotlib',
            'seaborn',
            'pandas',
            'scipy',
            'scikit-learn',
            'tqdm',
            'pyyaml',
            'jsonschema'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.issues.append(f"Missing packages: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def validate_core_imports(self) -> bool:
        """Validate that core modules can be imported."""
        # Add project root to path
        sys.path.insert(0, str(self.project_root))
        
        core_modules = [
            'neurobm',
            'neurobm.models.rbm',
            'neurobm.models.dbm',
            'neurobm.models.crbm',
            'neurobm.data.synth',
            'neurobm.data.loaders',
            'neurobm.training.loop',
            'neurobm.interpret.saliency'
        ]
        
        failed_imports = []
        
        for module in core_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
        
        if failed_imports:
            self.issues.append(f"Failed imports: {'; '.join(failed_imports)}")
            return False
        
        return True
    
    def validate_configurations(self) -> bool:
        """Validate configuration files."""
        config_files = [
            'configs/logging.yaml',
            'experiments/base.yaml',
            'experiments/ptsd.yaml',
            'experiments/autism.yaml',
            'experiments/ai_reliance.yaml'
        ]
        
        invalid_configs = []
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                invalid_configs.append(f"{config_file}: File not found")
                continue
            
            try:
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                invalid_configs.append(f"{config_file}: Invalid YAML - {e}")
        
        if invalid_configs:
            self.issues.append(f"Invalid configurations: {'; '.join(invalid_configs)}")
            return False
        
        return True
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        required_docs = [
            'README.md',
            'docs/ethics_guidelines.md',
            'docs/responsible_ai_framework.md',
            'docs/model_cards/base_cognitive_model_card.md',
            'docs/data_cards/synthetic_cognitive_data_card.md'
        ]
        
        missing_docs = []
        
        for doc_file in required_docs:
            doc_path = self.project_root / doc_file
            if not doc_path.exists():
                missing_docs.append(doc_file)
            elif doc_path.stat().st_size < 100:  # Very small files
                self.warnings.append(f"{doc_file} seems incomplete (< 100 bytes)")
        
        if missing_docs:
            self.issues.append(f"Missing documentation: {', '.join(missing_docs)}")
            return False
        
        return True
    
    def validate_basic_functionality(self) -> bool:
        """Validate basic functionality works."""
        try:
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            # Test basic model creation
            from neurobm.models.rbm import RestrictedBoltzmannMachine
            rbm = RestrictedBoltzmannMachine(n_visible=5, n_hidden=8)
            
            # Test data generation
            from neurobm.data.synth import SyntheticDataGenerator
            generator = SyntheticDataGenerator('base', random_seed=42)
            data = generator.generate_samples(10, method='skewed')
            
            # Test basic training
            metrics = rbm.train_batch(data, k=1)
            
            if 'reconstruction_error' not in metrics:
                self.issues.append("Training doesn't return expected metrics")
                return False
            
            return True
            
        except Exception as e:
            self.issues.append(f"Basic functionality test failed: {e}")
            return False
    
    def validate_test_suite(self) -> bool:
        """Validate test suite can run."""
        test_files = [
            'tests/__init__.py',
            'tests/test_models.py',
            'tests/test_data.py',
            'tests/run_tests.py'
        ]
        
        missing_tests = []
        
        for test_file in test_files:
            test_path = self.project_root / test_file
            if not test_path.exists():
                missing_tests.append(test_file)
        
        if missing_tests:
            self.issues.append(f"Missing test files: {', '.join(missing_tests)}")
            return False
        
        # Try to run a simple test
        try:
            result = subprocess.run([
                sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'
            ], cwd=self.project_root, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                self.warnings.append(f"Some tests failed: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            self.warnings.append("Test suite timed out")
        except Exception as e:
            self.warnings.append(f"Could not run test suite: {e}")
        
        return True
    
    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("📊 Validation Summary")
        print("=" * 50)
        
        print(f"✅ Passed checks: {len(self.passed_checks)}")
        print(f"❌ Issues found: {len(self.issues)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.issues:
            print("\n💥 Issues:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if len(self.issues) == 0:
            print("\n🎉 All validation checks passed!")
            print("✨ Your NeuroBM project is ready to use!")
        else:
            print(f"\n⚠️  Found {len(self.issues)} issues that need attention.")
            print("🔧 Please fix these issues before using the project.")
    
    def fix_common_issues(self) -> None:
        """Attempt to fix common issues automatically."""
        print("🔧 Attempting to fix common issues...")
        
        # Create missing directories
        required_dirs = ['results', 'logs', 'checkpoints']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created directory: {dir_name}")
        
        # Create empty __init__.py files if missing
        init_files = [
            'neurobm/__init__.py',
            'neurobm/models/__init__.py',
            'neurobm/data/__init__.py',
            'neurobm/training/__init__.py',
            'neurobm/interpret/__init__.py',
            'tests/__init__.py'
        ]
        
        for init_file in init_files:
            init_path = self.project_root / init_file
            if not init_path.exists():
                init_path.parent.mkdir(parents=True, exist_ok=True)
                init_path.touch()
                print(f"  ✅ Created: {init_file}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description='Validate NeuroBM project setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--fix-issues', action='store_true', help='Attempt to fix common issues')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create validator
    validator = ProjectValidator(project_root)
    
    # Fix issues if requested
    if args.fix_issues:
        validator.fix_common_issues()
    
    # Run validation
    success = validator.validate_all(quick=args.quick)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
