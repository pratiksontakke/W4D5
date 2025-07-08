"""
Configuration settings for the Intelligent Document Chunking System.
"""
import os
from pathlib import Path
from typing import List

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, TEMP_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Document Processing
SUPPORTED_PDF_VERSIONS = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "2.0"]
DOCUMENT_TYPES = ["technical_documentation", "article", "data_report", "general"]

# Chunking Settings
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
CODE_CHUNK_SIZE = int(os.getenv("CODE_CHUNK_SIZE", "500"))
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000

# Classification Settings
CLASSIFIER_MAX_FEATURES = int(os.getenv("CLASSIFIER_MAX_FEATURES", "5000"))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7"))

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# Performance Settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "300"))  # seconds

# Feature Flags
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Security Settings
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
ALLOWED_FILE_TYPES = [".pdf"]  # Add more as support is added

# Error Messages
ERROR_MESSAGES = {
    "file_not_found": "Document not found: {path}",
    "invalid_file_type": "Unsupported file type: {file_type}",
    "processing_error": "Error processing document: {error}",
    "validation_error": "Document validation failed: {reason}",
    "timeout_error": "Processing timeout after {timeout} seconds"
} 