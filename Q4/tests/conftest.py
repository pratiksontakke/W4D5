"""
Pytest configuration and fixtures for the Intelligent Document Chunking System.
"""
import os
import pytest
from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import app
from config.logging import setup_logging

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests."""
    # Set environment to testing
    os.environ["ENVIRONMENT"] = "testing"
    
    # Initialize logging
    setup_logging(log_level="DEBUG")
    
    # Create test directories
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup (if needed)
    # test_dir.rmdir()  # Uncomment if you want to clean up test data

@pytest.fixture
def test_client() -> Generator:
    """Create a test client for FastAPI application."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_pdf_path() -> Path:
    """Provide path to a sample PDF file for testing."""
    return Path(__file__).parent / "test_data" / "sample.pdf"

@pytest.fixture
def sample_wiki_content() -> str:
    """Provide sample wiki content for testing."""
    return """
    = Sample Wiki Page =
    
    == Introduction ==
    This is a sample wiki page for testing purposes.
    
    == Content ==
    * Item 1
    * Item 2
    
    == Code Example ==
    ```python
    def hello():
        print("Hello, World!")
    ```
    """

@pytest.fixture
def sample_jira_issue() -> dict:
    """Provide sample Jira issue data for testing."""
    return {
        "key": "TEST-1",
        "fields": {
            "summary": "Test Issue",
            "description": "This is a test issue description",
            "issuetype": {"name": "Bug"},
            "priority": {"name": "High"},
        }
    } 