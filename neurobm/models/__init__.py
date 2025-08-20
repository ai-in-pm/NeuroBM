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
Models module for NeuroBM.

This module provides implementations of various Boltzmann machine architectures:
- RestrictedBoltzmannMachine: Standard RBM with CD-k and PCD training
- DeepBoltzmannMachine: Multi-layer DBM with layer-wise pretraining
- ConditionalRBM: CRBM for temporal sequence modeling
- Utility functions for energy computation and sampling
"""

from .rbm import RestrictedBoltzmannMachine
from .dbm import DeepBoltzmannMachine
from .crbm import ConditionalRBM
from .utils import (
    sigmoid_with_temperature,
    sample_bernoulli,
    sample_gaussian,
    clip_gradients,
    TemperatureScheduler,
    GibbsSampler,
    compute_effective_sample_size,
    compute_partition_function_bounds,
    validate_energy_symmetry
)

__version__ = "0.1.0"
__all__ = [
    # Core models
    "RestrictedBoltzmannMachine",
    "DeepBoltzmannMachine",
    "ConditionalRBM",

    # Utility functions
    "sigmoid_with_temperature",
    "sample_bernoulli",
    "sample_gaussian",
    "clip_gradients",
    "TemperatureScheduler",
    "GibbsSampler",
    "compute_effective_sample_size",
    "compute_partition_function_bounds",
    "validate_energy_symmetry"
]
