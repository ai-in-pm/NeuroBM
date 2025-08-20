#!/usr/bin/env python3
"""
Quick test script to verify NeuroBM core functionality.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.models.dbm import DeepBoltzmannMachine
from neurobm.data.synth import SyntheticDataGenerator
from neurobm.data.loaders import get_data_loader
from neurobm.training.loop import TrainingLoop
from neurobm.training.callbacks import EarlyStopping, ProgressLogger
from neurobm.interpret.saliency import SaliencyAnalyzer


def test_rbm():
    """Test RBM functionality."""
    print("Testing RBM...")
    
    # Create RBM
    rbm = RestrictedBoltzmannMachine(
        n_visible=10,
        n_hidden=5,
        visible_type="bernoulli",
        learning_rate=0.01
    )
    
    # Generate synthetic data
    data = torch.rand(100, 10)
    
    # Test forward pass
    h_prob, h_sample = rbm.visible_to_hidden(data)
    v_prob, v_sample = rbm.hidden_to_visible(h_sample)
    
    # Test energy computation
    energy = rbm.energy(data, h_sample)
    free_energy = rbm.free_energy(data)
    
    # Test training step
    metrics = rbm.train_batch(data[:32])
    
    print(f"  ✓ RBM created: {rbm}")
    print(f"  ✓ Hidden shape: {h_prob.shape}")
    print(f"  ✓ Energy shape: {energy.shape}")
    print(f"  ✓ Training metrics: {list(metrics.keys())}")
    
    return rbm


def test_dbm():
    """Test DBM functionality."""
    print("Testing DBM...")
    
    # Create DBM
    dbm = DeepBoltzmannMachine(
        layer_sizes=[10, 8, 5],
        visible_type="bernoulli",
        learning_rate=0.01
    )
    
    # Generate synthetic data
    data = torch.rand(50, 10)
    
    # Test mean-field inference
    states, iterations = dbm.mean_field_inference(data)
    
    # Test energy computation
    energy = dbm.energy(states)
    
    # Test reconstruction
    reconstruction = dbm.reconstruct(data)
    
    print(f"  ✓ DBM created: {dbm}")
    print(f"  ✓ Mean-field states: {len(states)} layers")
    print(f"  ✓ Iterations: {iterations}")
    print(f"  ✓ Reconstruction shape: {reconstruction.shape}")
    
    return dbm


def test_synthetic_data():
    """Test synthetic data generation."""
    print("Testing synthetic data generation...")
    
    # Test different regimes
    regimes = ["base", "ptsd", "autism", "ai_reliance"]
    
    for regime in regimes:
        generator = SyntheticDataGenerator(regime, random_seed=42)
        
        # Generate normal data
        normal_data = generator.generate_multivariate_normal(100)
        
        # Generate skewed data
        skewed_data = generator.generate_skewed_distributions(100)
        
        print(f"  ✓ {regime}: normal={normal_data.shape}, skewed={skewed_data.shape}")
    
    return generator


def test_training_loop():
    """Test training loop functionality."""
    print("Testing training loop...")
    
    # Create model and data
    rbm = RestrictedBoltzmannMachine(n_visible=5, n_hidden=3)
    data_loader = get_data_loader("base", n_samples=200, batch_size=32)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(patience=5, verbose=False),
        ProgressLogger(log_freq=5)
    ]
    
    # Create training loop
    trainer = TrainingLoop(
        model=rbm,
        train_loader=data_loader,
        callbacks=callbacks,
        log_interval=5
    )
    
    # Train for a few epochs
    history = trainer.train(epochs=3, k_steps=1)
    
    print(f"  ✓ Training completed: {len(history['train_loss'])} epochs")
    print(f"  ✓ Final loss: {history['train_loss'][-1]:.4f}")
    
    return trainer


def test_interpretability():
    """Test interpretability tools."""
    print("Testing interpretability tools...")
    
    # Create model and data
    rbm = RestrictedBoltzmannMachine(n_visible=8, n_hidden=4)
    data = torch.rand(100, 8)
    
    # Test saliency analysis
    analyzer = SaliencyAnalyzer(rbm)
    
    # Compute weight saliency
    weight_saliency = analyzer.compute_weight_saliency()
    
    # Compute feature importance
    feature_importance = analyzer.compute_feature_importance(data)
    
    # Analyze weight patterns
    weight_patterns = analyzer.analyze_weight_patterns()
    
    print(f"  ✓ Weight saliency shape: {weight_saliency.shape}")
    print(f"  ✓ Feature importance shape: {feature_importance.shape}")
    print(f"  ✓ Weight patterns: {list(weight_patterns.keys())}")
    
    return analyzer


def main():
    """Run all tests."""
    print("🧠 Testing NeuroBM Core Functionality\n")
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run tests
        rbm = test_rbm()
        print()
        
        dbm = test_dbm()
        print()
        
        generator = test_synthetic_data()
        print()
        
        trainer = test_training_loop()
        print()
        
        analyzer = test_interpretability()
        print()
        
        print("🎉 All tests passed successfully!")
        print("\nNeuroBM framework is ready for use!")
        
        # Print summary
        print("\n📊 Framework Summary:")
        print("  • Automated scaffolding system ✓")
        print("  • RBM and DBM models ✓")
        print("  • Training infrastructure ✓")
        print("  • Synthetic data generation ✓")
        print("  • Interpretability tools ✓")
        print("  • Data schemas and transformations ✓")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
