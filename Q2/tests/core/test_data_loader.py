"""
Tests for the data_loader module.
"""

from pathlib import Path

import pytest

from core.data_loader import DocumentLoader


def test_document_loader_initialization(test_data_dir):
    """Test that DocumentLoader initializes correctly."""
    loader = DocumentLoader(data_dir=test_data_dir)
    assert isinstance(loader.data_dir, Path)
    assert loader.data_dir == test_data_dir


def test_load_text_document(sample_documents):
    """Test loading a text document."""
    loader = DocumentLoader(data_dir=sample_documents)
    content = loader.load_document(sample_documents / "income_tax.txt")
    assert "Section 80C" in content
    assert "education expenses" in content


def test_load_nonexistent_document(test_data_dir):
    """Test that loading a nonexistent document raises an error."""
    loader = DocumentLoader(data_dir=test_data_dir)
    with pytest.raises(FileNotFoundError):
        loader.load_document(test_data_dir / "nonexistent.txt")


def test_batch_document_loading(sample_documents):
    """Test loading multiple documents in batch."""
    loader = DocumentLoader(data_dir=sample_documents)
    documents = loader.load_documents(
        [sample_documents / "income_tax.txt", sample_documents / "gst.txt"]
    )

    assert len(documents) == 2
    assert any("Section 80C" in doc for doc in documents)
    assert any("GST Rate" in doc for doc in documents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
