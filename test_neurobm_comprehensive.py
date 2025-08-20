#!/usr/bin/env python3
"""
Comprehensive test suite for NeuroBM.

This script tests all major components of the NeuroBM system to ensure
everything is working correctly after the complete implementation.
"""

import sys
import traceback
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all major modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Core models
        from neurobm.models.rbm import RestrictedBoltzmannMachine
        from neurobm.models.dbm import DeepBoltzmannMachine
        from neurobm.models.crbm import ConditionalRBM
        print("  ✅ Models imported successfully")
        
        # Data handling
        from neurobm.data.synth import SyntheticDataGenerator
        from neurobm.data.loaders import get_data_loader
        from neurobm.data.schema import get_schema
        print("  ✅ Data modules imported successfully")
        
        # Training
        from neurobm.training.loop import TrainingLoop
        from neurobm.training.callbacks import get_standard_callbacks
        from neurobm.training.eval import ModelEvaluator
        print("  ✅ Training modules imported successfully")
        
        # Interpretability
        from neurobm.interpret.saliency import SaliencyAnalyzer
        from neurobm.interpret.mutual_info import MutualInformationAnalyzer
        from neurobm.interpret.traversals import LatentTraverser
        from neurobm.interpret.tiles import FilterVisualizer
        print("  ✅ Interpretability modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_generation():
    """Test synthetic data generation."""
    print("\n🔬 Testing data generation...")
    
    try:
        from neurobm.data.synth import SyntheticDataGenerator
        from neurobm.data.schema import get_schema
        
        # Test base regime
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(n_samples=100, method='skewed')
        
        assert data.shape == (100, 5), f"Expected shape (100, 5), got {data.shape}"
        assert torch.all(data >= 0) and torch.all(data <= 1), "Data not in [0,1] range"
        print("  ✅ Base regime data generation successful")
        
        # Test PTSD regime
        generator_ptsd = SyntheticDataGenerator('ptsd', random_seed=42)
        data_ptsd = generator_ptsd.generate_samples(n_samples=50, method='skewed')
        
        assert data_ptsd.shape == (50, 6), f"Expected shape (50, 6), got {data_ptsd.shape}"
        print("  ✅ PTSD regime data generation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data generation failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation and basic operations."""
    print("\n🧠 Testing model creation...")
    
    try:
        from neurobm.models.rbm import RestrictedBoltzmannMachine
        from neurobm.models.dbm import DeepBoltzmannMachine
        from neurobm.models.crbm import ConditionalRBM
        
        device = torch.device('cpu')  # Use CPU for testing
        
        # Test RBM
        rbm = RestrictedBoltzmannMachine(
            n_visible=5,
            n_hidden=10,
            device=device
        )
        assert rbm.n_visible == 5
        assert rbm.n_hidden == 10
        print("  ✅ RBM creation successful")
        
        # Test DBM
        dbm = DeepBoltzmannMachine(
            layer_sizes=[5, 8, 6],
            device=device
        )
        assert len(dbm.layer_sizes) == 3
        print("  ✅ DBM creation successful")
        
        # Test CRBM
        crbm = ConditionalRBM(
            n_visible=5,
            n_hidden=8,
            n_history=3,
            device=device
        )
        assert crbm.n_visible == 5
        assert crbm.n_hidden == 8
        assert crbm.n_history == 3
        print("  ✅ CRBM creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_training():
    """Test basic training functionality."""
    print("\n🚀 Testing training...")
    
    try:
        from neurobm.models.rbm import RestrictedBoltzmannMachine
        from neurobm.data.synth import SyntheticDataGenerator
        
        device = torch.device('cpu')
        
        # Create model and data
        rbm = RestrictedBoltzmannMachine(
            n_visible=5,
            n_hidden=8,
            learning_rate=0.01,
            device=device
        )
        
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(n_samples=100, method='skewed')
        
        # Test single batch training
        batch_data = data[:32]  # Small batch
        metrics = rbm.train_batch(batch_data, k=1)
        
        assert 'reconstruction_error' in metrics
        assert isinstance(metrics['reconstruction_error'], float)
        print("  ✅ Basic training successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test model evaluation."""
    print("\n📊 Testing evaluation...")
    
    try:
        from neurobm.models.rbm import RestrictedBoltzmannMachine
        from neurobm.training.eval import ModelEvaluator
        from neurobm.data.synth import SyntheticDataGenerator
        
        device = torch.device('cpu')
        
        # Create and train a simple model
        rbm = RestrictedBoltzmannMachine(
            n_visible=5,
            n_hidden=8,
            device=device
        )
        
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(n_samples=100, method='skewed')
        
        # Quick training
        for _ in range(5):
            rbm.train_batch(data[:32], k=1)
        
        # Test evaluation
        evaluator = ModelEvaluator(rbm, device)
        metrics = evaluator.reconstruction_metrics(data[:50])
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        print("  ✅ Model evaluation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
        traceback.print_exc()
        return False


def test_interpretability():
    """Test interpretability tools."""
    print("\n🔍 Testing interpretability...")
    
    try:
        from neurobm.models.rbm import RestrictedBoltzmannMachine
        from neurobm.interpret.saliency import SaliencyAnalyzer
        from neurobm.data.synth import SyntheticDataGenerator
        
        device = torch.device('cpu')
        
        # Create model and data
        rbm = RestrictedBoltzmannMachine(
            n_visible=5,
            n_hidden=8,
            device=device
        )
        
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(n_samples=50, method='skewed')
        
        # Test saliency analysis
        analyzer = SaliencyAnalyzer(rbm, device)
        saliency = analyzer.compute_feature_importance(data[:10])

        assert saliency.shape == (5,)  # Feature importance per feature
        print("  ✅ Interpretability tools successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Interpretability failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run comprehensive tests."""
    print("🧪 NeuroBM Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_model_creation,
        test_training,
        test_evaluation,
        test_interpretability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print(f"\n📈 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! NeuroBM is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
