"""
Core model implementations for contrastive learning and classification.
"""

from .contrastive import ContrastiveTrainer
from .classifier import ConversionClassifier

__all__ = ['ContrastiveTrainer', 'ConversionClassifier'] 