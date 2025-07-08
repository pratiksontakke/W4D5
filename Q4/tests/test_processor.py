"""
Tests for the document processor module.
"""
import pytest
from pathlib import Path
import tempfile
import shutil

from core.processor import DocumentProcessor
from config import settings

@pytest.fixture
def processor():
    """Create a document processor instance."""
    return DocumentProcessor()

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(content)
        return Path(f.name)

def test_processor_initialization(processor):
    """Test processor initialization."""
    assert processor is not None
    assert processor.pdf_converter is not None
    assert processor.classifier is not None

def test_get_supported_types(processor):
    """Test getting supported file types."""
    supported_types = processor.get_supported_types()
    assert isinstance(supported_types, list)
    assert '.pdf' in supported_types

def test_validate_document_with_valid_pdf(processor, sample_pdf):
    """Test document validation with valid PDF."""
    assert processor.validate_document(sample_pdf)

def test_validate_document_with_invalid_path(processor):
    """Test document validation with invalid path."""
    assert not processor.validate_document("nonexistent.pdf")

def test_validate_document_with_unsupported_type(processor):
    """Test document validation with unsupported file type."""
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        assert not processor.validate_document(f.name)

def test_process_document_with_invalid_path(processor):
    """Test processing document with invalid path."""
    with pytest.raises(FileNotFoundError):
        processor.process_document("nonexistent.pdf")

def test_process_document_with_unsupported_type(processor):
    """Test processing document with unsupported file type."""
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        with pytest.raises(ValueError):
            processor.process_document(f.name)

# Clean up test files
def teardown_module(module):
    """Clean up temporary test files."""
    # Clean up any remaining temporary files
    for file in Path(tempfile.gettempdir()).glob("tmp*.pdf"):
        try:
            file.unlink()
        except:
            pass 