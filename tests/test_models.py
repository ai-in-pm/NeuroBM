#!/usr/bin/env python3
"""
Unit tests for NeuroBM models.

This module contains comprehensive unit tests for all model components
including RBM, DBM, and CRBM implementations.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neurobm.models.rbm import RestrictedBoltzmannMachine
from neurobm.models.dbm import DeepBoltzmannMachine
from neurobm.models.crbm import ConditionalRBM


class TestRestrictedBoltzmannMachine(unittest.TestCase):
    """Test cases for RestrictedBoltzmannMachine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.n_visible = 5
        self.n_hidden = 8
        self.rbm = RestrictedBoltzmannMachine(
            n_visible=self.n_visible,
            n_hidden=self.n_hidden,
            device=self.device,
            random_seed=42
        )
        
        # Create test data
        torch.manual_seed(42)
        self.test_data = torch.rand(32, self.n_visible)
    
    def test_initialization(self):
        """Test RBM initialization."""
        self.assertEqual(self.rbm.n_visible, self.n_visible)
        self.assertEqual(self.rbm.n_hidden, self.n_hidden)
        self.assertEqual(self.rbm.device, self.device)
        
        # Check weight matrix shape
        self.assertEqual(self.rbm.W.shape, (self.n_visible, self.n_hidden))
        
        # Check bias shapes
        self.assertEqual(self.rbm.v_bias.shape, (self.n_visible,))
        self.assertEqual(self.rbm.h_bias.shape, (self.n_hidden,))
    
    def test_visible_to_hidden(self):
        """Test visible to hidden transformation."""
        h_prob, h_sample = self.rbm.visible_to_hidden(self.test_data)
        
        # Check shapes
        self.assertEqual(h_prob.shape, (32, self.n_hidden))
        self.assertEqual(h_sample.shape, (32, self.n_hidden))
        
        # Check probability range
        self.assertTrue(torch.all(h_prob >= 0))
        self.assertTrue(torch.all(h_prob <= 1))
        
        # Check samples are binary
        self.assertTrue(torch.all((h_sample == 0) | (h_sample == 1)))
    
    def test_hidden_to_visible(self):
        """Test hidden to visible transformation."""
        h_sample = torch.randint(0, 2, (32, self.n_hidden), dtype=torch.float32)
        v_prob, v_sample = self.rbm.hidden_to_visible(h_sample)
        
        # Check shapes
        self.assertEqual(v_prob.shape, (32, self.n_visible))
        self.assertEqual(v_sample.shape, (32, self.n_visible))
        
        # Check probability range
        self.assertTrue(torch.all(v_prob >= 0))
        self.assertTrue(torch.all(v_prob <= 1))
    
    def test_gibbs_step(self):
        """Test Gibbs sampling step."""
        v_sample, v_prob, h_sample, h_prob = self.rbm.gibbs_step(self.test_data)
        
        # Check shapes
        self.assertEqual(v_sample.shape, self.test_data.shape)
        self.assertEqual(v_prob.shape, self.test_data.shape)
        self.assertEqual(h_sample.shape, (32, self.n_hidden))
        self.assertEqual(h_prob.shape, (32, self.n_hidden))
    
    def test_free_energy(self):
        """Test free energy computation."""
        free_energy = self.rbm.free_energy(self.test_data)
        
        # Check shape
        self.assertEqual(free_energy.shape, (32,))
        
        # Check that free energy is finite
        self.assertTrue(torch.all(torch.isfinite(free_energy)))
    
    def test_train_batch(self):
        """Test batch training."""
        initial_weights = self.rbm.W.clone()
        
        metrics = self.rbm.train_batch(self.test_data, k=1)
        
        # Check that weights changed
        self.assertFalse(torch.allclose(initial_weights, self.rbm.W))
        
        # Check metrics
        self.assertIn('reconstruction_error', metrics)
        self.assertIsInstance(metrics['reconstruction_error'], float)
        self.assertGreaterEqual(metrics['reconstruction_error'], 0)
    
    def test_reconstruct(self):
        """Test data reconstruction."""
        reconstructed = self.rbm.reconstruct(self.test_data, n_gibbs=1)
        
        # Check shape
        self.assertEqual(reconstructed.shape, self.test_data.shape)
        
        # Check value range
        self.assertTrue(torch.all(reconstructed >= 0))
        self.assertTrue(torch.all(reconstructed <= 1))


class TestDeepBoltzmannMachine(unittest.TestCase):
    """Test cases for DeepBoltzmannMachine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.layer_sizes = [5, 8, 6]
        self.dbm = DeepBoltzmannMachine(
            layer_sizes=self.layer_sizes,
            device=self.device
        )
        
        # Create test data
        torch.manual_seed(42)
        self.test_data = torch.rand(16, self.layer_sizes[0])
    
    def test_initialization(self):
        """Test DBM initialization."""
        self.assertEqual(self.dbm.layer_sizes, self.layer_sizes)
        self.assertEqual(len(self.dbm.weights), len(self.layer_sizes) - 1)
        
        # Check weight shapes
        for i, weight in enumerate(self.dbm.weights):
            expected_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            self.assertEqual(weight.shape, expected_shape)
    
    def test_mean_field_inference(self):
        """Test mean field inference."""
        states, energies = self.dbm.mean_field_inference(self.test_data, n_iterations=5)
        
        # Check number of layers
        self.assertEqual(len(states), len(self.layer_sizes))
        
        # Check shapes
        for i, state in enumerate(states):
            expected_shape = (16, self.layer_sizes[i])
            self.assertEqual(state.shape, expected_shape)
        
        # Check energy shape
        self.assertEqual(energies.shape, (5,))  # n_iterations
    
    def test_train_batch(self):
        """Test DBM batch training."""
        initial_weights = [w.clone() for w in self.dbm.weights]
        
        metrics = self.dbm.train_batch(self.test_data, n_iterations=3)
        
        # Check that weights changed
        for initial, current in zip(initial_weights, self.dbm.weights):
            self.assertFalse(torch.allclose(initial, current))
        
        # Check metrics
        self.assertIn('reconstruction_error', metrics)
        self.assertIsInstance(metrics['reconstruction_error'], float)


class TestConditionalRBM(unittest.TestCase):
    """Test cases for ConditionalRBM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.n_visible = 5
        self.n_hidden = 8
        self.n_history = 3
        self.crbm = ConditionalRBM(
            n_visible=self.n_visible,
            n_hidden=self.n_hidden,
            n_history=self.n_history,
            device=self.device
        )
        
        # Create test data
        torch.manual_seed(42)
        self.test_data = torch.rand(16, self.n_visible)
        self.test_history = torch.rand(16, self.n_visible * self.n_history)
    
    def test_initialization(self):
        """Test CRBM initialization."""
        self.assertEqual(self.crbm.n_visible, self.n_visible)
        self.assertEqual(self.crbm.n_hidden, self.n_hidden)
        self.assertEqual(self.crbm.n_history, self.n_history)
        
        # Check weight matrix shapes
        self.assertEqual(self.crbm.W.shape, (self.n_visible, self.n_hidden))
        self.assertEqual(self.crbm.A.shape, (self.n_visible * self.n_history, self.n_hidden))
    
    def test_visible_to_hidden_with_history(self):
        """Test visible to hidden with history."""
        h_prob, h_sample = self.crbm.visible_to_hidden(self.test_data, self.test_history)
        
        # Check shapes
        self.assertEqual(h_prob.shape, (16, self.n_hidden))
        self.assertEqual(h_sample.shape, (16, self.n_hidden))
        
        # Check probability range
        self.assertTrue(torch.all(h_prob >= 0))
        self.assertTrue(torch.all(h_prob <= 1))
    
    def test_generate_sequence(self):
        """Test sequence generation."""
        sequence = self.crbm.generate_sequence(length=10, batch_size=4, n_gibbs=1)
        
        # Check shape
        self.assertEqual(sequence.shape, (4, 10, self.n_visible))
        
        # Check value range
        self.assertTrue(torch.all(sequence >= 0))
        self.assertTrue(torch.all(sequence <= 1))
    
    def test_train_batch(self):
        """Test CRBM batch training."""
        initial_W = self.crbm.W.clone()
        initial_A = self.crbm.A.clone()
        
        metrics = self.crbm.train_batch(self.test_data, self.test_history, k=1)
        
        # Check that weights changed
        self.assertFalse(torch.allclose(initial_W, self.crbm.W))
        self.assertFalse(torch.allclose(initial_A, self.crbm.A))
        
        # Check metrics
        self.assertIn('reconstruction_error', metrics)
        self.assertIsInstance(metrics['reconstruction_error'], float)


class TestModelUtils(unittest.TestCase):
    """Test cases for model utilities."""
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Create and train a simple RBM
        rbm = RestrictedBoltzmannMachine(n_visible=5, n_hidden=8, random_seed=42)
        test_data = torch.rand(32, 5)
        
        # Train for a few steps
        for _ in range(3):
            rbm.train_batch(test_data, k=1)
        
        # Save model
        save_path = Path('test_model.pth')
        rbm.save_checkpoint(save_path)
        
        # Load model
        loaded_rbm = RestrictedBoltzmannMachine.load_checkpoint(save_path)
        
        # Check that weights are the same
        self.assertTrue(torch.allclose(rbm.W, loaded_rbm.W))
        self.assertTrue(torch.allclose(rbm.v_bias, loaded_rbm.v_bias))
        self.assertTrue(torch.allclose(rbm.h_bias, loaded_rbm.h_bias))
        
        # Clean up
        save_path.unlink()


if __name__ == '__main__':
    unittest.main()
