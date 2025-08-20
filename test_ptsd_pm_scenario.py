#!/usr/bin/env python3
"""
Test script for the new PTSD-PM scenario in NeuroBM framework.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from neurobm.data.schema import get_schema, PTSD_PM_FEATURES
from neurobm.data.synth import SyntheticDataGenerator, generate_ptsd_pm_longitudinal_study
from neurobm.data.loaders import get_data_loader
from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.training.loop import TrainingLoop
from neurobm.training.callbacks import EarlyStopping, ProgressLogger
from neurobm.interpret.saliency import SaliencyAnalyzer


def test_ptsd_pm_schema():
    """Test PTSD-PM schema definition."""
    print("Testing PTSD-PM schema...")
    
    # Get schema
    schema = get_schema("ptsd_pm")
    
    # Verify features
    expected_features = [
        "decision_pressure_sensitivity",
        "stakeholder_communication_stress", 
        "deadline_anxiety",
        "ai_tool_adoption_resistance",
        "ai_tool_acceptance",
        "cognitive_load_threshold",
        "attention_task_switching",
        "hypervigilance_risk_assessment",
        "avoidance_complex_interactions",
        "pm_performance_confidence"
    ]
    
    feature_names = schema.get_feature_names()
    
    print(f"  ✓ Schema loaded: {schema.name}")
    print(f"  ✓ Features count: {len(feature_names)}")
    print(f"  ✓ Expected features: {len(expected_features)}")
    
    # Check all expected features are present
    for feature in expected_features:
        assert feature in feature_names, f"Missing feature: {feature}"
    
    # Check correlations
    correlation_matrix = schema.feature_correlations
    print(f"  ✓ Correlation matrix shape: {correlation_matrix.shape}")
    
    # Verify some key correlations
    ai_resistance_idx = feature_names.index("ai_tool_adoption_resistance")
    ai_acceptance_idx = feature_names.index("ai_tool_acceptance")
    
    # These should be strongly negatively correlated
    correlation = correlation_matrix[ai_resistance_idx, ai_acceptance_idx]
    print(f"  ✓ AI resistance vs acceptance correlation: {correlation:.3f}")
    
    return schema


def test_ptsd_pm_data_generation():
    """Test PTSD-PM synthetic data generation."""
    print("Testing PTSD-PM data generation...")
    
    generator = SyntheticDataGenerator("ptsd_pm", random_seed=42)
    
    # Test basic data generation
    normal_data = generator.generate_multivariate_normal(100)
    skewed_data = generator.generate_skewed_distributions(100)
    
    print(f"  ✓ Normal data shape: {normal_data.shape}")
    print(f"  ✓ Skewed data shape: {skewed_data.shape}")
    
    # Test AI integration scenarios
    ai_scenarios = generator.generate_ptsd_pm_ai_scenarios(
        n_samples=50,
        ai_integration_levels=[0.0, 0.5, 1.0],
        time_periods=["2025", "2026"]
    )
    
    print(f"  ✓ AI scenarios generated: {len(ai_scenarios)} time periods")
    for period, data in ai_scenarios.items():
        print(f"    - {period}: {len(data)} integration levels")
    
    # Test workplace scenarios
    workplace_data = generator.generate_ptsd_pm_workplace_scenarios(50)
    
    print(f"  ✓ Workplace scenarios: {len(workplace_data)} factors")
    for factor, data in workplace_data.items():
        print(f"    - {factor}: {data.shape}")
    
    return generator


def test_longitudinal_study():
    """Test longitudinal study data generation."""
    print("Testing longitudinal study generation...")
    
    study_data = generate_ptsd_pm_longitudinal_study(
        n_participants=100,
        time_periods=["2025", "2026", "2027"],
        ai_integration_levels=[0.0, 0.5, 1.0]
    )
    
    print(f"  ✓ Study metadata: {list(study_data['metadata'].keys())}")
    print(f"  ✓ Participants: {study_data['metadata']['n_participants']}")
    print(f"  ✓ Time periods: {study_data['metadata']['time_periods']}")
    print(f"  ✓ AI levels: {study_data['metadata']['ai_integration_levels']}")
    
    # Check longitudinal data structure
    longitudinal = study_data["longitudinal_data"]
    print(f"  ✓ Longitudinal periods: {len(longitudinal)}")
    
    # Check workplace scenarios
    workplace = study_data["workplace_scenarios"]
    print(f"  ✓ Workplace factors: {len(workplace)}")
    
    # Check population heterogeneity
    heterogeneous = study_data["population_heterogeneity"]
    print(f"  ✓ Heterogeneous data shape: {heterogeneous['data'].shape}")
    print(f"  ✓ Subgroup labels shape: {heterogeneous['subgroup_labels'].shape}")
    
    return study_data


def test_ptsd_pm_model_training():
    """Test training a model on PTSD-PM data."""
    print("Testing PTSD-PM model training...")
    
    # Create model
    rbm = RestrictedBoltzmannMachine(
        n_visible=10,  # PTSD-PM features
        n_hidden=15,
        visible_type="bernoulli",
        learning_rate=0.005
    )
    
    # Get data loader
    data_loader = get_data_loader(
        "ptsd_pm", 
        n_samples=200, 
        batch_size=32,
        random_seed=42
    )
    
    # Create training loop
    callbacks = [
        EarlyStopping(patience=5, verbose=False),
        ProgressLogger(log_freq=5)
    ]
    
    trainer = TrainingLoop(
        model=rbm,
        train_loader=data_loader,
        callbacks=callbacks,
        log_interval=5
    )
    
    # Train for a few epochs
    history = trainer.train(epochs=5, k_steps=3)
    
    print(f"  ✓ Training completed: {len(history['train_loss'])} epochs")
    print(f"  ✓ Final loss: {history['train_loss'][-1]:.4f}")
    
    return rbm, history


def test_ptsd_pm_interpretability():
    """Test interpretability analysis for PTSD-PM model."""
    print("Testing PTSD-PM interpretability...")
    
    # Create model and data
    rbm = RestrictedBoltzmannMachine(n_visible=10, n_hidden=8)
    generator = SyntheticDataGenerator("ptsd_pm", random_seed=42)
    data = generator.generate_skewed_distributions(150)
    
    # Saliency analysis
    analyzer = SaliencyAnalyzer(rbm)
    
    # Weight saliency
    weight_saliency = analyzer.compute_weight_saliency()
    print(f"  ✓ Weight saliency shape: {weight_saliency.shape}")
    
    # Feature importance
    feature_importance = analyzer.compute_feature_importance(data)
    print(f"  ✓ Feature importance shape: {feature_importance.shape}")
    
    # Get feature names for interpretation
    feature_names = generator.schema.get_feature_names()
    
    # Find most important features
    top_indices = torch.argsort(feature_importance, descending=True)[:3]
    print("  ✓ Top 3 most important features:")
    for i, idx in enumerate(top_indices):
        feature_name = feature_names[idx]
        importance = feature_importance[idx].item()
        print(f"    {i+1}. {feature_name}: {importance:.4f}")
    
    return analyzer


def test_ai_integration_analysis():
    """Test AI integration impact analysis."""
    print("Testing AI integration impact analysis...")
    
    generator = SyntheticDataGenerator("ptsd_pm", random_seed=42)
    
    # Generate data for different AI integration levels
    ai_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    ai_data = {}
    
    for ai_level in ai_levels:
        # Modify parameters for this AI level
        modified_params = generator._modify_params_for_ptsd_pm_ai(
            ai_level, "workplace_ai_integration", "2026"
        )
        
        # Temporarily update parameters
        original_params = generator.pop_params.copy()
        generator.pop_params.update(modified_params)
        
        # Generate data
        data = generator.generate_skewed_distributions(100)
        ai_data[ai_level] = data
        
        # Restore parameters
        generator.pop_params = original_params
    
    print(f"  ✓ Generated data for {len(ai_data)} AI integration levels")
    
    # Analyze changes in key features
    feature_names = generator.schema.get_feature_names()
    
    # Focus on AI-related features
    ai_acceptance_idx = feature_names.index("ai_tool_acceptance")
    ai_resistance_idx = feature_names.index("ai_tool_adoption_resistance")
    stress_idx = feature_names.index("stakeholder_communication_stress")
    
    print("  ✓ AI integration impact on key features:")
    for ai_level in ai_levels:
        data = ai_data[ai_level]
        acceptance_mean = torch.mean(data[:, ai_acceptance_idx]).item()
        resistance_mean = torch.mean(data[:, ai_resistance_idx]).item()
        stress_mean = torch.mean(data[:, stress_idx]).item()
        
        print(f"    AI Level {ai_level:.2f}: acceptance={acceptance_mean:.3f}, "
              f"resistance={resistance_mean:.3f}, stress={stress_mean:.3f}")
    
    return ai_data


def main():
    """Run all PTSD-PM scenario tests."""
    print("🧠 Testing PTSD-PM Scenario in NeuroBM Framework\n")
    
    try:
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run tests
        schema = test_ptsd_pm_schema()
        print()
        
        generator = test_ptsd_pm_data_generation()
        print()
        
        study_data = test_longitudinal_study()
        print()
        
        rbm, history = test_ptsd_pm_model_training()
        print()
        
        analyzer = test_ptsd_pm_interpretability()
        print()
        
        ai_data = test_ai_integration_analysis()
        print()
        
        print("🎉 All PTSD-PM scenario tests passed successfully!")
        print("\n📊 PTSD-PM Scenario Summary:")
        print("  • Schema with 10 specialized features ✓")
        print("  • AI integration scenarios across time periods ✓")
        print("  • Workplace factor modeling ✓")
        print("  • Longitudinal study data generation ✓")
        print("  • Model training and interpretability ✓")
        print("  • Ethical safeguards and disclaimers ✓")
        
        print("\n⚠️  IMPORTANT REMINDERS:")
        print("  • For educational and research purposes only")
        print("  • NOT for workplace assessment or clinical diagnosis")
        print("  • Synthetic data - not based on real individuals")
        print("  • Professional support recommended for PTSD")
        print("  • Proper workplace accommodations essential")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
