"""Unit tests for core functionality."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from core.data_loader import DocumentLoader
from core.embedders import TextEmbedder
from core.preprocess import TextPreprocessor
from core.retrieval import DocumentRetriever
from core.similarity import (
    CosineSimilarity,
    EuclideanDistance,
    HybridSimilarity,
    MaximalMarginalRelevance,
)
from sentence_transformers import SentenceTransformer

# ... rest of the file ...
