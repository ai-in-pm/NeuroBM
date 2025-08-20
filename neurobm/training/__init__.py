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
Training module for NeuroBM.

This module provides comprehensive training infrastructure for Boltzmann machines:
- TrainingLoop: Main training orchestration with callbacks
- AnnealedImportanceSampling: Likelihood estimation via AIS
- Callbacks: Early stopping, checkpointing, logging, and monitoring
- ModelEvaluator: Comprehensive model evaluation and comparison
- Utility functions for training and evaluation
"""

from .loop import TrainingLoop
from .ais import AnnealedImportanceSampling
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoardLogger,
    ProgressLogger,
    CustomMetricLogger,
    get_standard_callbacks
)
from .eval import (
    ModelEvaluator,
    compute_model_complexity,
    statistical_significance_test
)

__version__ = "0.1.0"
__all__ = [
    # Core training
    "TrainingLoop",
    "AnnealedImportanceSampling",

    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "TensorBoardLogger",
    "ProgressLogger",
    "CustomMetricLogger",
    "get_standard_callbacks",

    # Evaluation
    "ModelEvaluator",
    "compute_model_complexity",
    "statistical_significance_test"
]
