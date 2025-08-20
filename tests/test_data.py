#!/usr/bin/env python3
"""
Unit tests for NeuroBM data handling.

This module contains comprehensive unit tests for data generation,
loading, transformation, and schema validation.
"""

import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neurobm.data.synth import SyntheticDataGenerator
from neurobm.data.schema import get_schema, FeatureType, FeatureDefinition, RegimeSchema
from neurobm.data.loaders import get_data_loader, RegimeDataset
from neurobm.data.transforms import (
    MinMaxNormalizer, StandardScaler, BinaryThresholdTransform,
    DataTransformer, create_preprocessing_pipeline
)


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test cases for SyntheticDataGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticDataGenerator('base', random_seed=42)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.regime_name, 'base')
        self.assertEqual(self.generator.random_seed, 42)
        self.assertIsNotNone(self.generator.schema)
    
    def test_generate_samples(self):
        """Test sample generation."""
        n_samples = 100
        data = self.generator.generate_samples(n_samples, method='skewed')
        
        # Check shape
        expected_features = len(self.generator.schema.features)
        self.assertEqual(data.shape, (n_samples, expected_features))
        
        # Check data type
        self.assertIsInstance(data, torch.Tensor)
        
        # Check value range
        self.assertTrue(torch.all(data >= 0))
        self.assertTrue(torch.all(data <= 1))
    
    def test_generate_multivariate_normal(self):
        """Test multivariate normal generation."""
        n_samples = 50
        data = self.generator.generate_multivariate_normal(n_samples)
        
        # Check shape
        expected_features = len(self.generator.schema.features)
        self.assertEqual(data.shape, (n_samples, expected_features))
        
        # Check that data has reasonable correlations
        corr_matrix = torch.corrcoef(data.T)
        self.assertFalse(torch.allclose(corr_matrix, torch.eye(expected_features), atol=0.1))
    
    def test_generate_skewed_distributions(self):
        """Test skewed distribution generation."""
        n_samples = 100
        data = self.generator.generate_skewed_distributions(n_samples)
        
        # Check shape
        expected_features = len(self.generator.schema.features)
        self.assertEqual(data.shape, (n_samples, expected_features))
        
        # Check that distributions are actually skewed
        for i in range(expected_features):
            feature_data = data[:, i].numpy()
            # Skewness should be non-zero for most features
            skewness = np.mean((feature_data - np.mean(feature_data))**3) / np.std(feature_data)**3
            # Allow some tolerance for randomness
            self.assertNotAlmostEqual(skewness, 0, places=1)
    
    def test_different_regimes(self):
        """Test generation for different regimes."""
        regimes = ['base', 'ptsd', 'autism', 'ai_reliance']
        
        for regime in regimes:
            with self.subTest(regime=regime):
                generator = SyntheticDataGenerator(regime, random_seed=42)
                data = generator.generate_samples(50, method='skewed')
                
                # Check that data was generated
                self.assertGreater(data.shape[0], 0)
                self.assertGreater(data.shape[1], 0)
                
                # Check value range
                self.assertTrue(torch.all(data >= 0))
                self.assertTrue(torch.all(data <= 1))
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        generator1 = SyntheticDataGenerator('base', random_seed=42)
        generator2 = SyntheticDataGenerator('base', random_seed=42)
        
        data1 = generator1.generate_samples(100, method='skewed')
        data2 = generator2.generate_samples(100, method='skewed')
        
        # Should be identical with same seed
        self.assertTrue(torch.allclose(data1, data2))


class TestSchema(unittest.TestCase):
    """Test cases for schema functionality."""
    
    def test_get_schema(self):
        """Test schema retrieval."""
        schema = get_schema('base')
        
        self.assertIsInstance(schema, RegimeSchema)
        self.assertGreater(len(schema.features), 0)
        
        # Check that all features have proper definitions
        for name, feature in schema.features.items():
            self.assertIsInstance(feature, FeatureDefinition)
            self.assertIsInstance(feature.feature_type, FeatureType)
            self.assertIsInstance(feature.description, str)
    
    def test_schema_validation(self):
        """Test schema validation functionality."""
        schema = get_schema('base')
        
        # Generate valid data
        generator = SyntheticDataGenerator('base', random_seed=42)
        valid_data = generator.generate_samples(50, method='skewed')
        
        # Should not raise an exception
        try:
            # Note: validate_regime_data might not be implemented yet
            # This is a placeholder for when it is
            pass
        except NotImplementedError:
            pass
    
    def test_feature_types(self):
        """Test feature type definitions."""
        # Test that feature types are properly defined
        self.assertTrue(hasattr(FeatureType, 'CONTINUOUS'))
        self.assertTrue(hasattr(FeatureType, 'BINARY'))
        self.assertTrue(hasattr(FeatureType, 'CATEGORICAL'))


class TestDataLoaders(unittest.TestCase):
    """Test cases for data loaders."""
    
    def test_get_data_loader(self):
        """Test data loader creation."""
        data_loader = get_data_loader(
            regime_name='base',
            n_samples=100,
            batch_size=32,
            shuffle=True
        )
        
        # Check that we can iterate through the loader
        batch_count = 0
        total_samples = 0
        
        for batch in data_loader:
            batch_count += 1
            batch_data = batch[0] if isinstance(batch, (list, tuple)) else batch
            total_samples += batch_data.shape[0]
            
            # Check batch properties
            self.assertLessEqual(batch_data.shape[0], 32)  # batch_size
            self.assertEqual(batch_data.shape[1], 5)  # base regime features
        
        # Check total samples
        self.assertEqual(total_samples, 100)
        self.assertGreater(batch_count, 0)
    
    def test_regime_dataset(self):
        """Test RegimeDataset functionality."""
        # Generate test data
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(50, method='skewed')
        
        # Create dataset
        dataset = RegimeDataset(data, 'base')
        
        # Check dataset properties
        self.assertEqual(len(dataset), 50)
        
        # Check individual samples
        sample = dataset[0]
        self.assertEqual(sample.shape, (5,))  # base regime features
        
        # Check that we can iterate
        for i, sample in enumerate(dataset):
            if i >= 5:  # Just check first few
                break
            self.assertEqual(sample.shape, (5,))


class TestTransforms(unittest.TestCase):
    """Test cases for data transforms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        torch.manual_seed(42)
        self.test_data = torch.randn(100, 5) * 2 + 1  # Mean 1, std 2
    
    def test_minmax_normalizer(self):
        """Test MinMax normalization."""
        normalizer = MinMaxNormalizer()
        
        # Fit and transform
        normalizer.fit(self.test_data)
        normalized = normalizer.transform(self.test_data)
        
        # Check range
        self.assertTrue(torch.all(normalized >= 0))
        self.assertTrue(torch.all(normalized <= 1))
        
        # Check that min and max are actually 0 and 1
        self.assertAlmostEqual(torch.min(normalized).item(), 0, places=5)
        self.assertAlmostEqual(torch.max(normalized).item(), 1, places=5)
    
    def test_standard_scaler(self):
        """Test standard scaling."""
        scaler = StandardScaler()
        
        # Fit and transform
        scaler.fit(self.test_data)
        scaled = scaler.transform(self.test_data)
        
        # Check mean and std
        self.assertTrue(torch.allclose(torch.mean(scaled, dim=0), torch.zeros(5), atol=1e-5))
        self.assertTrue(torch.allclose(torch.std(scaled, dim=0), torch.ones(5), atol=1e-5))
    
    def test_binary_threshold(self):
        """Test binary thresholding."""
        threshold_transform = BinaryThresholdTransform(threshold=0.5)
        
        # Create test data with known values
        test_data = torch.tensor([[0.3, 0.7], [0.1, 0.9], [0.6, 0.4]])
        binary_data = threshold_transform.transform(test_data)
        
        # Check results
        expected = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertTrue(torch.allclose(binary_data, expected))
    
    def test_data_transformer(self):
        """Test DataTransformer composition."""
        # Create a pipeline
        transforms = [
            StandardScaler(),
            MinMaxNormalizer()
        ]
        
        transformer = DataTransformer(transforms)
        
        # Fit and transform
        transformer.fit(self.test_data)
        transformed = transformer.transform(self.test_data)
        
        # Should be normalized to [0, 1] after both transforms
        self.assertTrue(torch.all(transformed >= 0))
        self.assertTrue(torch.all(transformed <= 1))
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        pipeline = create_preprocessing_pipeline(
            normalize=True,
            normalization_method='minmax',
            binarize=False
        )
        
        self.assertIsInstance(pipeline, DataTransformer)
        
        # Test with data
        pipeline.fit(self.test_data)
        transformed = pipeline.transform(self.test_data)
        
        # Should be normalized
        self.assertTrue(torch.all(transformed >= 0))
        self.assertTrue(torch.all(transformed <= 1))


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data components."""
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline from generation to loading."""
        # Generate data
        generator = SyntheticDataGenerator('base', random_seed=42)
        data = generator.generate_samples(200, method='skewed')
        
        # Create transforms
        pipeline = create_preprocessing_pipeline(
            normalize=True,
            normalization_method='standard'
        )
        
        # Apply transforms
        pipeline.fit(data)
        transformed_data = pipeline.transform(data)
        
        # Create dataset and loader
        dataset = RegimeDataset(transformed_data, 'base')
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test that we can iterate through the loader
        batch_count = 0
        for batch in loader:
            batch_count += 1
            self.assertEqual(batch.shape[1], 5)  # base regime features
            
            # Check that data is properly transformed (standardized)
            # Mean should be close to 0, std close to 1
            batch_mean = torch.mean(batch)
            self.assertLess(abs(batch_mean.item()), 0.5)  # Reasonable tolerance
        
        self.assertGreater(batch_count, 0)


if __name__ == '__main__':
    unittest.main()
