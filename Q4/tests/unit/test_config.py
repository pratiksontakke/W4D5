"""
Unit tests for configuration module.
"""
import pytest
from pathlib import Path
from config import settings

def test_base_paths():
    """Test that base paths are correctly configured."""
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.MODELS_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    
    assert settings.MODELS_DIR.exists()
    assert settings.DATA_DIR.exists()

def test_api_settings():
    """Test API configuration settings."""
    assert isinstance(settings.API_HOST, str)
    assert isinstance(settings.API_PORT, int)
    assert isinstance(settings.API_WORKERS, int)
    
    assert settings.API_PORT > 0
    assert settings.API_WORKERS > 0

def test_document_processing_settings():
    """Test document processing configuration settings."""
    assert isinstance(settings.MAX_CHUNK_SIZE, int)
    assert isinstance(settings.MIN_CHUNK_SIZE, int)
    assert isinstance(settings.OVERLAP_SIZE, int)
    assert isinstance(settings.SUPPORTED_FORMATS, list)
    
    assert settings.MAX_CHUNK_SIZE > settings.MIN_CHUNK_SIZE
    assert settings.OVERLAP_SIZE < settings.MAX_CHUNK_SIZE
    assert all(isinstance(fmt, str) for fmt in settings.SUPPORTED_FORMATS)

def test_model_settings():
    """Test model configuration settings."""
    assert isinstance(settings.EMBEDDING_MODEL, str)
    assert isinstance(settings.CLASSIFIER_MODEL, str)
    assert isinstance(settings.DEVICE, str)
    
    assert settings.DEVICE in ["cpu", "cuda"]

def test_storage_settings():
    """Test storage configuration settings."""
    assert isinstance(settings.VECTOR_STORE_TYPE, str)
    assert isinstance(settings.VECTOR_STORE_PATH, Path)
    assert isinstance(settings.GRAPH_STORE_URI, str)
    
    assert settings.VECTOR_STORE_TYPE in ["faiss", "elasticsearch"]
    assert "bolt://" in settings.GRAPH_STORE_URI

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 