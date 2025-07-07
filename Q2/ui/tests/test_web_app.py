"""
Tests for the web application functionality.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import jwt
import pytest
from config import UI_CONFIG
from fastapi.testclient import TestClient
from ui.web_app import app, check_rate_limit, get_current_user

# Test client
client = TestClient(app)

# Test data
TEST_TOKEN = jwt.encode(
    {"sub": "test_user", "exp": time.time() + 3600}, "your-secret-key"
)
TEST_QUERY = "Income tax deduction for education"
TEST_DOC_TYPES = ["income_tax_act", "gst_act"]


@pytest.fixture
def auth_headers():
    """Fixture for authentication headers."""
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


@pytest.fixture
def mock_doc_loader():
    """Fixture for mocked document loader."""
    with patch("ui.web_app.doc_loader") as mock:
        yield mock


@pytest.fixture
def mock_similarity_engine():
    """Fixture for mocked similarity engine."""
    with patch("ui.web_app.similarity_engine") as mock:
        yield mock


@pytest.fixture
def mock_metrics_calculator():
    """Fixture for mocked metrics calculator."""
    with patch("ui.web_app.metrics_calculator") as mock:
        yield mock


def test_home_page():
    """Test home page rendering."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Indian Legal Document Search" in response.text


def test_upload_document_success(auth_headers, mock_doc_loader):
    """Test successful document upload."""
    # Create test file
    test_file = Path("test_doc.pdf")
    test_file.write_bytes(b"test content")

    try:
        with open(test_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test_doc.pdf", f, "application/pdf")},
                data={"document_type": "income_tax_act"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        assert response.json() == {"message": "Document uploaded successfully"}
        mock_doc_loader.process_document.assert_called_once()

    finally:
        test_file.unlink()


def test_upload_document_invalid_size(auth_headers):
    """Test document upload with invalid file size."""
    # Create large test file
    test_file = Path("large_test_doc.pdf")
    test_file.write_bytes(b"x" * (UI_CONFIG["max_upload_size_mb"] * 1024 * 1024 + 1))

    try:
        with open(test_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("large_test_doc.pdf", f, "application/pdf")},
                data={"document_type": "income_tax_act"},
                headers=auth_headers,
            )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    finally:
        test_file.unlink()


def test_upload_document_invalid_type(auth_headers):
    """Test document upload with invalid file type."""
    response = client.post(
        "/upload",
        files={"file": ("test.exe", b"test content", "application/octet-stream")},
        data={"document_type": "income_tax_act"},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_search_success(auth_headers, mock_similarity_engine, mock_metrics_calculator):
    """Test successful search query."""
    # Mock search results
    mock_results = {
        "cosine": [{"title": "Test Doc", "score": 0.9, "excerpt": "Test content"}],
        "euclidean": [{"title": "Test Doc", "score": 0.8, "excerpt": "Test content"}],
        "mmr": [{"title": "Test Doc", "score": 0.7, "excerpt": "Test content"}],
        "hybrid": [{"title": "Test Doc", "score": 0.95, "excerpt": "Test content"}],
    }
    mock_similarity_engine.search_all.return_value = mock_results

    # Mock metrics
    mock_metrics = {
        "cosine": {"precision": 0.9, "recall": 0.8, "diversity": 0.7},
        "euclidean": {"precision": 0.8, "recall": 0.7, "diversity": 0.8},
        "mmr": {"precision": 0.7, "recall": 0.9, "diversity": 0.9},
        "hybrid": {"precision": 0.95, "recall": 0.85, "diversity": 0.8},
    }
    mock_metrics_calculator.calculate_metrics.return_value = mock_metrics

    response = client.post(
        "/search",
        data={"query": TEST_QUERY, "document_types": TEST_DOC_TYPES},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "metrics" in data
    assert data["results"] == mock_results
    assert data["metrics"] == mock_metrics


def test_search_invalid_doc_type(auth_headers):
    """Test search with invalid document type."""
    response = client.post(
        "/search",
        data={"query": TEST_QUERY, "document_types": ["invalid_type"]},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "Invalid document type" in response.json()["detail"]


def test_feedback_success(auth_headers, mock_metrics_calculator):
    """Test successful feedback submission."""
    response = client.post(
        "/feedback",
        data={"query_id": "test_query", "relevant_docs": ["doc1", "doc2"]},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Feedback recorded successfully"}
    mock_metrics_calculator.update_feedback.assert_called_once()


def test_get_metrics_success(auth_headers, mock_metrics_calculator):
    """Test successful metrics retrieval."""
    # Mock metrics data
    mock_metrics = {
        "overall_precision": 0.85,
        "overall_recall": 0.8,
        "method_comparison": {
            "cosine": {"precision": 0.9, "recall": 0.8},
            "euclidean": {"precision": 0.8, "recall": 0.7},
            "mmr": {"precision": 0.7, "recall": 0.9},
            "hybrid": {"precision": 0.95, "recall": 0.85},
        },
    }
    mock_metrics_calculator.get_performance_metrics.return_value = mock_metrics

    response = client.get("/metrics", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == mock_metrics


def test_rate_limiting():
    """Test rate limiting functionality."""
    user_id = "test_user"

    # Clear rate limit store
    rate_limit_store = {}

    # Make requests up to limit
    for _ in range(100):
        assert check_rate_limit(user_id) is True

    # Next request should be blocked
    assert check_rate_limit(user_id) is False


def test_authentication():
    """Test authentication middleware."""
    # Test with invalid token
    response = client.get("/metrics", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401

    # Test with expired token
    expired_token = jwt.encode(
        {"sub": "test_user", "exp": time.time() - 3600}, "your-secret-key"
    )
    response = client.get(
        "/metrics", headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == 401

    # Test with valid token
    response = client.get("/metrics", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main(["-v", __file__])
