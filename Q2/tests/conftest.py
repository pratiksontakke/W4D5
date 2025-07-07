"""
Common test fixtures and configuration for the test suite.
"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env.test", override=True)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_documents(test_data_dir):
    """Create sample legal documents for testing."""
    docs = {
        "income_tax": "Section 80C: Deduction for education expenses...",
        "gst": "GST Rate for textiles: 5% on raw materials...",
        "court": "In the matter of property registration...",
        "property": "Process for property registration in Delhi...",
    }

    for doc_type, content in docs.items():
        doc_path = test_data_dir / f"{doc_type}.txt"
        doc_path.write_text(content)

    return test_data_dir


@pytest.fixture(scope="session")
def test_queries():
    """Return a set of test queries."""
    return [
        "Income tax deduction for education",
        "GST rate for textile products",
        "Property registration process",
        "Court fee structure",
    ]


@pytest.fixture(scope="session")
def embedding_config():
    """Return test configuration for embeddings."""
    return {
        "sentence-transformer": "all-MiniLM-L6-v2",
        "spacy": "en_core_web_lg",
        "tf-idf": None,
    }


@pytest.fixture(scope="session")
def similarity_config():
    """Return test configuration for similarity methods."""
    return {
        "cosine": {"weight": 1.0},
        "euclidean": {"weight": 1.0},
        "mmr": {"lambda_param": 0.6, "threshold": 0.7},
        "hybrid": {"cosine_weight": 0.6, "entity_weight": 0.4},
    }
