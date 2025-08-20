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
Training callbacks for early stopping, checkpointing, and monitoring

This module provides a comprehensive callback system for training:
- Early stopping based on validation metrics
- Model checkpointing with best model saving
- Learning rate scheduling
- TensorBoard logging
- Custom metric monitoring
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import numpy as np
import logging
import time
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to stop training when validation loss stops improving."""

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimization, 'max' for maximization
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} not supported")

    def on_train_begin(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.mode == 'min' else -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Check for early stopping condition."""
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Early stopping metric '{self.monitor}' not found in logs")
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")

                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("Restored best weights")

    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.wait >= self.patience


class ModelCheckpoint(Callback):
    """Callback to save model checkpoints during training."""

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = 'min',
        period: int = 1,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.

        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor for best model
            save_best_only: Whether to save only the best model
            save_weights_only: Whether to save only weights (not full model)
            mode: 'min' for minimization, 'max' for maximization
            period: Interval (epochs) between checkpoints
            verbose: Whether to print messages
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        self.verbose = verbose

        self.epochs_since_last_save = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"Mode {mode} not supported")

        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Reset state at training start."""
        self.best = np.inf if self.mode == 'min' else -np.inf
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Save checkpoint if conditions are met."""
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = logs.get(self.monitor)
            if current is None and self.save_best_only:
                logger.warning(f"Checkpoint metric '{self.monitor}' not found in logs")
                return

            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    self.best = current
                    self._save_model(model, epoch, current)
            else:
                self._save_model(model, epoch, current)

    def _save_model(self, model: nn.Module, epoch: int, metric_value: Optional[float]) -> None:
        """Save the model."""
        # Format filepath with epoch and metric
        if hasattr(model, 'save_checkpoint'):
            # Use model's save method if available
            filepath = str(self.filepath).format(epoch=epoch+1, **{self.monitor: metric_value or 0})
            model.save_checkpoint(Path(filepath))
        else:
            # Fallback to torch.save
            filepath = str(self.filepath).format(epoch=epoch+1, **{self.monitor: metric_value or 0})
            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                torch.save(model, filepath)

        if self.verbose:
            logger.info(f"Model checkpoint saved to {filepath}")


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(
        self,
        schedule: Union[Callable[[int], float], torch.optim.lr_scheduler._LRScheduler],
        verbose: bool = True
    ):
        """
        Initialize learning rate scheduler.

        Args:
            schedule: Learning rate schedule function or PyTorch scheduler
            verbose: Whether to print LR changes
        """
        self.schedule = schedule
        self.verbose = verbose
        self.is_pytorch_scheduler = hasattr(schedule, 'step')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Update learning rate."""
        if self.is_pytorch_scheduler:
            old_lr = self.schedule.get_last_lr()[0]
            self.schedule.step()
            new_lr = self.schedule.get_last_lr()[0]
        else:
            # Custom schedule function
            new_lr = self.schedule(epoch)
            old_lr = getattr(model, 'learning_rate', None)

            # Update model learning rate if it has the attribute
            if hasattr(model, 'learning_rate'):
                model.learning_rate = new_lr

        if self.verbose and old_lr != new_lr:
            logger.info(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")

        # Add to logs
        logs['learning_rate'] = new_lr


class TensorBoardLogger(Callback):
    """TensorBoard logging callback."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        log_freq: int = 1,
        histogram_freq: int = 0,
        write_graph: bool = False
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
            log_freq: Frequency (epochs) for logging scalars
            histogram_freq: Frequency (epochs) for logging histograms
            write_graph: Whether to write model graph
        """
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.tensorboard_available = True
        except ImportError:
            logger.warning("TensorBoard not available, logging disabled")
            self.tensorboard_available = False

    def on_train_begin(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Initialize TensorBoard logging."""
        if not self.tensorboard_available:
            return

        if self.write_graph:
            # This would require a sample input to trace the model
            # For now, we skip graph writing for Boltzmann machines
            pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Log metrics to TensorBoard."""
        if not self.tensorboard_available:
            return

        if (epoch + 1) % self.log_freq == 0:
            # Log scalar metrics
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)

        if self.histogram_freq > 0 and (epoch + 1) % self.histogram_freq == 0:
            # Log parameter histograms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'{name}/weights', param.data, epoch)
                    self.writer.add_histogram(f'{name}/gradients', param.grad.data, epoch)

    def on_train_end(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Close TensorBoard writer."""
        if self.tensorboard_available:
            self.writer.close()


class MetricMonitor(Callback):
    """Monitor and track custom metrics during training."""

    def __init__(
        self,
        metrics: Dict[str, Callable],
        log_freq: int = 1,
        verbose: bool = True
    ):
        """
        Initialize metric monitor.

        Args:
            metrics: Dictionary of metric name -> metric function
            log_freq: Frequency (epochs) for computing metrics
            verbose: Whether to print metric values
        """
        self.metrics = metrics
        self.log_freq = log_freq
        self.verbose = verbose
        self.metric_history = {name: [] for name in metrics.keys()}

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Compute and log custom metrics."""
        if (epoch + 1) % self.log_freq == 0:
            for name, metric_fn in self.metrics.items():
                try:
                    value = metric_fn(model, logs)
                    self.metric_history[name].append(value)
                    logs[f'custom_{name}'] = value

                    if self.verbose:
                        logger.info(f"Custom metric {name}: {value:.4f}")

                except Exception as e:
                    logger.warning(f"Failed to compute metric {name}: {e}")

    def get_metric_history(self) -> Dict[str, List[float]]:
        """Get history of custom metrics."""
        return self.metric_history.copy()


# Alias for backward compatibility
CustomMetricLogger = MetricMonitor


class ProgressLogger(Callback):
    """Simple progress logging callback."""

    def __init__(self, log_freq: int = 1):
        """
        Initialize progress logger.

        Args:
            log_freq: Frequency (epochs) for logging progress
        """
        self.log_freq = log_freq
        self.start_time = None

    def on_train_begin(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Record training start time."""
        self.start_time = time.time()
        logger.info("Training started")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Log progress."""
        if (epoch + 1) % self.log_freq == 0:
            elapsed = time.time() - self.start_time
            logger.info(f"Epoch {epoch + 1} completed in {elapsed:.2f}s")

    def on_train_end(self, logs: Dict[str, Any], model: nn.Module) -> None:
        """Log training completion."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {total_time:.2f}s")


class TemperatureAnnealing(Callback):
    """Temperature annealing callback for Boltzmann machines."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        annealing_epochs: int = 100,
        schedule: str = 'exponential'
    ):
        """
        Initialize temperature annealing.

        Args:
            initial_temp: Starting temperature
            final_temp: Final temperature
            annealing_epochs: Number of epochs for annealing
            schedule: Annealing schedule ('exponential', 'linear', 'cosine')
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_epochs = annealing_epochs
        self.schedule = schedule

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], model: nn.Module) -> None:
        """Update model temperature."""
        if epoch < self.annealing_epochs:
            progress = epoch / self.annealing_epochs

            if self.schedule == 'exponential':
                temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
            elif self.schedule == 'linear':
                temp = self.initial_temp + progress * (self.final_temp - self.initial_temp)
            elif self.schedule == 'cosine':
                temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * \
                       (1 + np.cos(np.pi * progress))
            else:
                temp = self.initial_temp

            # Update model temperature if it has the attribute
            if hasattr(model, 'temperature'):
                model.temperature = temp
                logs['temperature'] = temp


# Utility function to create common callback combinations
def get_standard_callbacks(
    save_dir: Path,
    patience: int = 10,
    monitor: str = 'val_loss',
    tensorboard: bool = True
) -> List[Callback]:
    """
    Get a standard set of callbacks for training.

    Args:
        save_dir: Directory for saving checkpoints and logs
        patience: Patience for early stopping
        monitor: Metric to monitor
        tensorboard: Whether to include TensorBoard logging

    Returns:
        List of configured callbacks
    """
    callbacks = [
        EarlyStopping(monitor=monitor, patience=patience, verbose=True),
        ModelCheckpoint(
            filepath=save_dir / 'checkpoint_epoch_{epoch:03d}_{val_loss:.4f}.ckpt',
            monitor=monitor,
            save_best_only=True,
            verbose=True
        ),
        ProgressLogger(log_freq=1)
    ]

    if tensorboard:
        callbacks.append(TensorBoardLogger(log_dir=save_dir / 'tensorboard'))

    return callbacks
