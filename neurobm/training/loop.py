# Copyright 2025 NeuroBM Contributors
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
# limitations under the License.
"""
Training loops for Boltzmann machines with callbacks and logging

This module provides comprehensive training infrastructure for Boltzmann machines:
- Flexible training loops with callback support
- Early stopping and checkpointing
- Learning rate scheduling
- Progress tracking and logging
- Support for both RBM and DBM training
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import logging
import time
from pathlib import Path
from tqdm import tqdm

from ..models.rbm import RestrictedBoltzmannMachine
from ..models.dbm import DeepBoltzmannMachine
from ..models.utils import TemperatureScheduler, clip_gradients
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoardLogger

logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Comprehensive training loop for Boltzmann machines.

    Supports both RBM and DBM training with callbacks, early stopping,
    checkpointing, and comprehensive logging.
    """

    def __init__(
        self,
        model: Union[RestrictedBoltzmannMachine, DeepBoltzmannMachine],
        train_loader: data.DataLoader,
        val_loader: Optional[data.DataLoader] = None,
        callbacks: Optional[List[Any]] = None,
        device: Optional[torch.device] = None,
        log_interval: int = 10,
        gradient_clip_norm: Optional[float] = None,
        mixed_precision: bool = False,
    ):
        """
        Initialize training loop.

        Args:
            model: Boltzmann machine model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            callbacks: List of training callbacks
            device: Device to use for training
            log_interval: Interval for logging metrics
            gradient_clip_norm: Maximum gradient norm (if None, no clipping)
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.device = device or torch.device('cpu')
        self.log_interval = log_interval
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Mixed precision setup
        if self.mixed_precision:
            try:
                from torch.cuda.amp import GradScaler, autocast
                self.scaler = GradScaler()
                self.autocast = autocast
            except ImportError:
                logger.warning("Mixed precision not available, falling back to FP32")
                self.mixed_precision = False

        logger.info(f"Training loop initialized for {type(model).__name__}")

    def train_epoch(
        self,
        epoch: int,
        k_steps: int = 1,
        persistent: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
            k_steps: Number of CD steps (for RBM)
            persistent: Whether to use PCD (for RBM)
            **kwargs: Additional arguments for model-specific training

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_reconstruction_error': 0.0,
            'train_sparsity': 0.0,
            'train_free_energy': 0.0,
            'n_batches': 0
        }

        epoch_start_time = time.time()

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}",
            disable=not logger.isEnabledFor(logging.INFO)
        )

        for batch_idx, batch in enumerate(pbar):
            # Extract data from batch
            if isinstance(batch, (list, tuple)):
                data_batch = batch[0]
            else:
                data_batch = batch

            # Move to device
            data_batch = data_batch.to(self.device)

            # Training step
            if self.mixed_precision:
                with self.autocast():
                    batch_metrics = self._train_batch(
                        data_batch, k_steps=k_steps, persistent=persistent, **kwargs
                    )
            else:
                batch_metrics = self._train_batch(
                    data_batch, k_steps=k_steps, persistent=persistent, **kwargs
                )

            # Gradient clipping
            if self.gradient_clip_norm is not None:
                if hasattr(self.model, 'parameters'):
                    clip_gradients(list(self.model.parameters()), self.gradient_clip_norm)

            # Update metrics
            for key, value in batch_metrics.items():
                if key.startswith('train_') or key in epoch_metrics:
                    epoch_metrics[key] += value

            epoch_metrics['n_batches'] += 1
            self.global_step += 1

            # Update progress bar
            if batch_idx % self.log_interval == 0:
                current_loss = epoch_metrics['train_loss'] / epoch_metrics['n_batches']
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

                # Call batch callbacks
                for callback in self.callbacks:
                    if hasattr(callback, 'on_batch_end'):
                        callback.on_batch_end(
                            batch=batch_idx,
                            logs=batch_metrics,
                            model=self.model
                        )

        # Average metrics over epoch
        for key in epoch_metrics:
            if key != 'n_batches':
                epoch_metrics[key] /= epoch_metrics['n_batches']

        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time

        return epoch_metrics

    def _train_batch(
        self,
        batch: torch.Tensor,
        k_steps: int = 1,
        persistent: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Train on a single batch."""
        if isinstance(self.model, RestrictedBoltzmannMachine):
            return self.model.train_batch(batch, k=k_steps, persistent=persistent)
        elif isinstance(self.model, DeepBoltzmannMachine):
            n_gibbs = kwargs.get('n_gibbs', 5)
            return self.model.joint_train_batch(batch, n_gibbs=n_gibbs)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_reconstruction_error': 0.0,
            'val_free_energy': 0.0,
            'n_batches': 0
        }

        with torch.no_grad():
            for batch in self.val_loader:
                # Extract data from batch
                if isinstance(batch, (list, tuple)):
                    data_batch = batch[0]
                else:
                    data_batch = batch

                data_batch = data_batch.to(self.device)

                # Compute validation metrics
                batch_metrics = self._validate_batch(data_batch)

                # Update metrics
                for key, value in batch_metrics.items():
                    if key in val_metrics:
                        val_metrics[key] += value

                val_metrics['n_batches'] += 1

        # Average metrics
        for key in val_metrics:
            if key != 'n_batches':
                val_metrics[key] /= val_metrics['n_batches']

        return val_metrics

    def _validate_batch(self, batch: torch.Tensor) -> Dict[str, float]:
        """Validate on a single batch."""
        # Compute reconstruction error
        if isinstance(self.model, RestrictedBoltzmannMachine):
            reconstruction = self.model.reconstruct(batch, n_gibbs=1)
            reconstruction_error = torch.mean((batch - reconstruction)**2).item()
            free_energy = torch.mean(self.model.free_energy(batch)).item()
        elif isinstance(self.model, DeepBoltzmannMachine):
            reconstruction = self.model.reconstruct(batch)
            reconstruction_error = torch.mean((batch - reconstruction)**2).item()
            free_energy = torch.mean(self.model._compute_free_energy(batch)).item()
        else:
            reconstruction_error = 0.0
            free_energy = 0.0

        return {
            'val_loss': reconstruction_error,
            'val_reconstruction_error': reconstruction_error,
            'val_free_energy': free_energy
        }

    def train(
        self,
        epochs: int,
        k_steps: int = 1,
        persistent: bool = False,
        save_best: bool = True,
        save_dir: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            epochs: Number of epochs to train
            k_steps: Number of CD steps (for RBM)
            persistent: Whether to use PCD (for RBM)
            save_best: Whether to save best model
            save_dir: Directory to save checkpoints
            **kwargs: Additional arguments for model-specific training

        Returns:
            history: Training history
        """
        logger.info(f"Starting training for {epochs} epochs")

        # Call training start callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs={}, model=self.model)

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch

                # Call epoch start callbacks
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_begin'):
                        callback.on_epoch_begin(epoch=epoch, logs={}, model=self.model)

                # Training epoch
                train_metrics = self.train_epoch(
                    epoch=epoch,
                    k_steps=k_steps,
                    persistent=persistent,
                    **kwargs
                )

                # Validation epoch
                val_metrics = self.validate_epoch(epoch)

                # Combine metrics
                epoch_logs = {**train_metrics, **val_metrics}

                # Update training history
                for key, value in epoch_logs.items():
                    if key not in self.training_history:
                        self.training_history[key] = []
                    self.training_history[key].append(value)

                # Log epoch results
                log_msg = f"Epoch {epoch+1}/{epochs}"
                log_msg += f" - train_loss: {train_metrics.get('train_loss', 0):.4f}"
                if val_metrics:
                    log_msg += f" - val_loss: {val_metrics.get('val_loss', 0):.4f}"
                log_msg += f" - time: {train_metrics.get('epoch_time', 0):.2f}s"
                logger.info(log_msg)

                # Call epoch end callbacks
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(
                            epoch=epoch,
                            logs=epoch_logs,
                            model=self.model
                        )

                # Check for early stopping
                should_stop = False
                for callback in self.callbacks:
                    if hasattr(callback, 'should_stop') and callback.should_stop():
                        should_stop = True
                        logger.info(f"Early stopping triggered by {type(callback).__name__}")
                        break

                if should_stop:
                    break

                # Save best model
                if save_best and val_metrics and save_dir:
                    val_loss = val_metrics.get('val_loss', float('inf'))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_path = save_dir / 'best_model.ckpt'
                        self.model.save_checkpoint(save_path)
                        logger.info(f"New best model saved: val_loss={val_loss:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        finally:
            # Call training end callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end(logs=self.training_history, model=self.model)

        logger.info("Training completed")
        return self.training_history

    def pretrain_dbm(
        self,
        epochs_per_layer: Union[int, List[int]] = 100,
        k_steps: int = 1,
        joint_epochs: int = 100,
        save_dir: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """
        Pretrain DBM with layer-wise pretraining followed by joint training.

        Args:
            epochs_per_layer: Epochs for each layer pretraining
            k_steps: CD steps for pretraining
            joint_epochs: Epochs for joint training
            save_dir: Directory to save checkpoints

        Returns:
            history: Training history
        """
        if not isinstance(self.model, DeepBoltzmannMachine):
            raise ValueError("DBM pretraining only available for DeepBoltzmannMachine")

        logger.info("Starting DBM pretraining")

        # Layer-wise pretraining
        self.model.pretrain_all_layers(
            data_loader=self.train_loader,
            epochs_per_layer=epochs_per_layer,
            k_steps=k_steps,
            verbose=True
        )

        # Save pretrained model
        if save_dir:
            pretrain_path = save_dir / 'pretrained_model.ckpt'
            self.model.save_checkpoint(pretrain_path)
            logger.info(f"Pretrained model saved to {pretrain_path}")

        # Joint fine-tuning
        logger.info("Starting joint fine-tuning")
        joint_history = self.train(
            epochs=joint_epochs,
            save_best=True,
            save_dir=save_dir
        )

        return joint_history

    def evaluate(self, test_loader: data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            metrics: Test metrics
        """
        logger.info("Evaluating model")

        self.model.eval()
        test_metrics = {
            'test_loss': 0.0,
            'test_reconstruction_error': 0.0,
            'test_free_energy': 0.0,
            'n_batches': 0
        }

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if isinstance(batch, (list, tuple)):
                    data_batch = batch[0]
                else:
                    data_batch = batch

                data_batch = data_batch.to(self.device)

                # Compute test metrics
                batch_metrics = self._validate_batch(data_batch)

                # Update metrics
                for key, value in batch_metrics.items():
                    test_key = key.replace('val_', 'test_')
                    if test_key in test_metrics:
                        test_metrics[test_key] += value

                test_metrics['n_batches'] += 1

        # Average metrics
        for key in test_metrics:
            if key != 'n_batches':
                test_metrics[key] /= test_metrics['n_batches']

        # Log results
        log_msg = "Test Results: "
        for key, value in test_metrics.items():
            if key != 'n_batches':
                log_msg += f"{key}: {value:.4f} "
        logger.info(log_msg)

        return test_metrics

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.training_history.copy()

    def reset_training_state(self) -> None:
        """Reset training state for fresh training."""
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
