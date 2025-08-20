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
Interpret module for NeuroBM.

This module provides comprehensive interpretability tools for Boltzmann machines:
- Saliency analysis and weight importance computation
- Mutual information analysis between visible and hidden units
- Latent space traversal and counterfactual analysis
- Filter visualization and receptive field analysis
- Model architecture visualization
"""

from .saliency import SaliencyAnalyzer
from .mutual_info import MutualInformationAnalyzer
from .traversals import LatentTraverser
from .tiles import FilterVisualizer, create_model_architecture_diagram

__version__ = "0.1.0"
__all__ = [
    # Core analyzers
    "SaliencyAnalyzer",
    "MutualInformationAnalyzer",
    "LatentTraverser",
    "FilterVisualizer",

    # Utility functions
    "create_model_architecture_diagram"
]
