"""
Tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import json

from api.app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(content)
        return Path(f.name)

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_get_supported_types(client):
    """Test getting supported file types."""
    response = client.get("/supported-types")
    assert response.status_code == 200
    data = response.json()
    assert "supported_types" in data
    assert isinstance(data["supported_types"], list)
    assert ".pdf" in data["supported_types"]

def test_process_document_with_valid_pdf(client, sample_pdf):
    """Test processing a valid PDF document."""
    with open(sample_pdf, 'rb') as f:
        response = client.post(
            "/process",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "document_info" in data
    assert "classification" in data
    assert "chunks" in data
    assert "metadata" in data

def test_process_document_with_invalid_file(client):
    """Test processing an invalid file."""
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        f.write(b"This is not a PDF file")
        f.seek(0)
        response = client.post(
            "/process",
            files={"file": ("test.txt", f, "text/plain")}
        )
    assert response.status_code == 400

def test_process_document_without_file(client):
    """Test processing without a file."""
    response = client.post("/process")
    assert response.status_code == 422  # Unprocessable Entity

def test_process_document_with_empty_file(client):
    """Test processing an empty file."""
    with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
        response = client.post(
            "/process",
            files={"file": ("empty.pdf", f, "application/pdf")}
        )
    assert response.status_code == 400

# Clean up test files
def teardown_module(module):
    """Clean up temporary test files."""
    # Clean up any remaining temporary files
    for file in Path(tempfile.gettempdir()).glob("tmp*.pdf"):
        try:
            file.unlink()
        except:
            pass 