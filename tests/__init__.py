"""
NeuroBM Test Suite

This package contains comprehensive tests for all NeuroBM components.

Test Structure:
- test_models.py: Tests for RBM, DBM, and CRBM models
- test_data.py: Tests for data generation, loading, and transformation
- test_training.py: Tests for training loops and optimization
- test_interpret.py: Tests for interpretability tools
- test_integration.py: Integration tests for complete workflows

Usage:
    # Run all tests
    python tests/run_tests.py
    
    # Run specific test module
    python tests/run_tests.py --test test_models
    
    # Run with minimal output
    python tests/run_tests.py --quiet
    
    # Stop on first failure
    python tests/run_tests.py --failfast
"""

import os
import sys
import warnings
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure test environment
os.environ['NEUROBM_TEST_MODE'] = '1'

# Suppress warnings during testing
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Test configuration
TEST_CONFIG = {
    'random_seed': 42,
    'device': 'cpu',  # Use CPU for consistent testing
    'small_data_size': 50,
    'medium_data_size': 200,
    'large_data_size': 1000,
    'test_timeout': 30,  # seconds
    'tolerance': 1e-5,
}

# Make config available to test modules
__all__ = ['TEST_CONFIG']
