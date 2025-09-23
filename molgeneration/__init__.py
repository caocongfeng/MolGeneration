"""
MolGeneration: Two-stage training for reasoning LLMs for molecule generation.

This package provides a comprehensive framework for training language models
to generate molecules through a two-stage approach:

1. Stage 1: Reasoning pre-training on chemical knowledge
2. Stage 2: Molecule generation fine-tuning with reinforcement learning

Key components:
- Data processing for chemical datasets
- Model architectures adapted for molecular generation
- Two-stage training pipeline
- Evaluation metrics for molecular properties
- Utilities for SMILES processing and chemical reasoning
"""

__version__ = "0.1.0"
__author__ = "MolGeneration Team"

from molgeneration.models import MolGenerationModel
from molgeneration.training import TwoStageTrainer
from molgeneration.data import MolecularDataset
from molgeneration.evaluation import MolecularEvaluator

__all__ = [
    "MolGenerationModel",
    "TwoStageTrainer", 
    "MolecularDataset",
    "MolecularEvaluator",
]