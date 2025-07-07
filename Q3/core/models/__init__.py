"""
Core model implementations for contrastive learning and classification.
"""

from .classifier import ConversionClassifier
from .contrastive import ContrastiveTrainer

__all__ = ["ContrastiveTrainer", "ConversionClassifier"]
