"""
Integration test configuration for the Indian Legal Document Search System.
"""

import tempfile
from pathlib import Path

# Test environment configuration
TEST_ENV = {
    "max_concurrent_users": 10,
    "requests_per_user": 5,
    "request_timeout": 30,
    "max_retries": 3,
}

# Test data configuration
TEST_DATA = {
    "num_documents": 100,
    "doc_size_range": (100, 5000),  # characters
    "query_length_range": (3, 50),  # words
    "batch_size": 10,
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time": 2.0,  # seconds
    "throughput": 50,  # requests/second
    "success_rate": 0.99,  # 99%
    "memory_usage": 1024,  # MB
    "cpu_usage": 80,  # percent
}

# Test document templates
TEST_DOCUMENTS = {
    "income_tax": """
    Section {section_num}. {title}
    ({subsection}) In computing the total income of an assessee,
    being an individual or Hindu undivided family, there shall be deducted,
    in accordance with and subject to the provisions of this section,
    {deduction_details}
    """,
    "gst": """
    Chapter {chapter}. {title}
    {section}. Rate of tax for {category}
    (1) The rate of goods and services tax on {item_description}
    shall be {rate} percent.
    """,
    "property": """
    {act_name}
    Section {section}. {title}
    {content}
    Provided that {condition}
    """,
}

# Test queries with expected results
TEST_QUERIES = [
    {
        "query": "income tax deduction for education",
        "relevant_docs": ["doc1", "doc3", "doc5"],
        "expected_precision": 0.8,
        "expected_recall": 0.7,
    },
    {
        "query": "GST rate for textile products",
        "relevant_docs": ["doc2", "doc4", "doc6"],
        "expected_precision": 0.75,
        "expected_recall": 0.8,
    },
    {
        "query": "property registration process",
        "relevant_docs": ["doc7", "doc8", "doc9"],
        "expected_precision": 0.85,
        "expected_recall": 0.75,
    },
]

# Error test cases
ERROR_TEST_CASES = {
    "invalid_files": [
        ("large.txt", "x" * (11 * 1024 * 1024)),  # > 10MB
        ("binary.exe", bytes([0x00, 0xFF] * 100)),
        ("empty.txt", ""),
    ],
    "invalid_queries": [
        "",
        "a" * 1000,
        "SELECT * FROM documents",
        "<script>alert('xss')</script>",
    ],
}

# Concurrent test configuration
CONCURRENT_CONFIG = {
    "num_users": 10,
    "ramp_up_time": 5,  # seconds
    "test_duration": 60,  # seconds
    "think_time": 2,  # seconds between requests
    "scenarios": [
        {"name": "search_only", "weight": 0.7, "actions": ["search"]},
        {"name": "upload_and_search", "weight": 0.2, "actions": ["upload", "search"]},
        {"name": "browse_only", "weight": 0.1, "actions": ["browse"]},
    ],
}

# Create temporary test directories
TEST_DIR = Path(tempfile.mkdtemp()) / "integration_tests"
TEST_DATA_DIR = TEST_DIR / "data"
TEST_ARTIFACTS_DIR = TEST_DIR / "artifacts"

# Create test directories
for directory in [TEST_DATA_DIR, TEST_ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
