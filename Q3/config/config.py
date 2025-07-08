"""
Configuration settings for the Sales Conversion Prediction project.
This module centralizes all configuration parameters used across the application.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "core", "models")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "core", "embeddings")

# Model configuration
MODEL_CONFIG = {
    "base_model": "sentence-transformers/all-mpnet-base-v2",
    "max_seq_length": 512,
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
}

# Training configuration
TRAIN_CONFIG = {
    "train_test_split": 0.2,
    "validation_split": 0.1,
    "random_seed": 42,
    "early_stopping_patience": 3,
}

# LangChain configuration
LANGCHAIN_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "temperature": 0.7,
    "max_tokens": 500,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4,
    "timeout": 60,
}

# Monitoring configuration
MONITORING_CONFIG = {
    "log_level": "INFO",
    "metrics_port": 9090,
    "enable_tracing": True,
}

if __name__ == "__main__":
    # Simple test to verify config is loaded correctly
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model configuration: {MODEL_CONFIG}")
