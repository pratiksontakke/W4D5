"""
Embedding models and utilities for text representation.
"""

from .base import BaseEmbedding
from .pretrained import PretrainedEmbedding
from .fine_tuned import FineTunedEmbedding

__all__ = ['BaseEmbedding', 'PretrainedEmbedding', 'FineTunedEmbedding'] 