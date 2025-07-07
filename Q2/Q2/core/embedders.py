"""
Text embedding and vectorization module.
"""
import logging
from typing import Dict, List

import numpy as np
from config import EMBEDDING_MODELS
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ... rest of the file ...


def visualize_embeddings(self, embeddings: np.ndarray) -> None:
    """
    Visualize embeddings using dimensionality reduction.
    """
    # Remove unused scatter variable
    self._reduce_dimensions(embeddings)
    # ... rest of the method ...
