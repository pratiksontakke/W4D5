"""
Configuration settings for the Indian Legal Document Search System.
"""

from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifact paths
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
INDICES_DIR = ARTIFACTS_DIR / "indices"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Document categories
LEGAL_DOCUMENT_TYPES = [
    "income_tax_act",
    "gst_act",
    "court_judgments",
    "property_law"
]

# Embedding models configuration
EMBEDDING_MODELS = {
    "sentence-transformer": "all-MiniLM-L6-v2",  # Sentence-BERT for semantic search
    "spacy": "en_core_web_lg",                   # For legal entity recognition
    "tf-idf": None,                              # Built-in TF-IDF vectorizer
}

# Similarity methods configuration
SIMILARITY_METHODS = {
    "cosine": {
        "name": "Cosine Similarity",
        "weight": 1.0
    },
    "euclidean": {
        "name": "Euclidean Distance",
        "weight": 1.0
    },
    "mmr": {
        "name": "Maximal Marginal Relevance",
        "lambda_param": 0.6,  # Balance between relevance and diversity
        "threshold": 0.7      # Diversity threshold
    },
    "hybrid": {
        "name": "Hybrid Similarity",
        "cosine_weight": 0.6,
        "entity_weight": 0.4
    }
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    "top_k": 5,              # Number of top results to consider
    "relevance_threshold": 0.7,  # Minimum similarity score for relevance
    "test_queries": [
        "Income tax deduction for education",
        "GST rate for textile products",
        "Property registration process",
        "Court fee structure"
    ]
}

# Web UI configuration
UI_CONFIG = {
    "max_upload_size_mb": 10,
    "allowed_extensions": [".pdf", ".doc", ".docx", ".txt"],
    "results_per_page": 10,
    "cache_timeout": 3600    # 1 hour
}

# Processing configuration
PROCESSING_CONFIG = {
    "batch_size": 32,
    "num_workers": 4,
    "chunk_size": 1000,      # Number of characters per text chunk
    "overlap": 200           # Overlap between chunks
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, INDICES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
