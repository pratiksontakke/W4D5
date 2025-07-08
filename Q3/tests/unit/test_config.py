"""
Unit tests for the configuration module.
"""

import os
from pathlib import Path


def test_config_paths(config, project_root):
    """Test that configuration paths are set up correctly."""
    from config.config import DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR

    # Test that paths are absolute
    assert os.path.isabs(DATA_DIR)
    assert os.path.isabs(MODELS_DIR)
    assert os.path.isabs(EMBEDDINGS_DIR)

    # Test that paths are under project root
    assert Path(DATA_DIR).is_relative_to(project_root)
    assert Path(MODELS_DIR).is_relative_to(project_root)
    assert Path(EMBEDDINGS_DIR).is_relative_to(project_root)


def test_model_config(config):
    """Test that model configuration has required parameters."""
    model_config = config["model"]

    required_params = [
        "base_model",
        "max_seq_length",
        "batch_size",
        "epochs",
        "learning_rate",
        "warmup_steps",
    ]

    for param in required_params:
        assert param in model_config, f"Missing required parameter: {param}"
        assert model_config[param] is not None


def test_train_config(config):
    """Test that training configuration has required parameters."""
    train_config = config["train"]

    # Test split ratios sum to less than 1
    total_split = train_config["train_test_split"] + train_config["validation_split"]
    assert total_split < 1.0, "Train/test/validation splits should sum to less than 1"

    # Test random seed is set
    assert isinstance(train_config["random_seed"], int)

    # Test early stopping
    assert train_config["early_stopping_patience"] > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
