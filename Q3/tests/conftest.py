"""
Pytest configuration file with shared fixtures.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent.absolute()


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory."""
    test_data_path = project_root / "tests" / "data"
    os.makedirs(test_data_path, exist_ok=True)
    return test_data_path


@pytest.fixture(scope="session")
def config():
    """Return the project configuration."""
    from config.config import (
        API_CONFIG,
        LANGCHAIN_CONFIG,
        MODEL_CONFIG,
        MONITORING_CONFIG,
        TRAIN_CONFIG,
    )

    return {
        "model": MODEL_CONFIG,
        "train": TRAIN_CONFIG,
        "langchain": LANGCHAIN_CONFIG,
        "api": API_CONFIG,
        "monitoring": MONITORING_CONFIG,
    }
