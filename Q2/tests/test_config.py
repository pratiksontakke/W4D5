"""
Test configuration settings for the Indian Legal Document Search System.
"""

import os
from pathlib import Path

# Test data paths
TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_ARTIFACTS_DIR = TEST_DIR / "test_artifacts"

# Sample test documents
SAMPLE_DOCUMENTS = {
    "income_tax": """
    Section 80C. Deduction for education expenses.
    (1) In computing the total income of an assessee, being an individual, there shall be deducted,
    the whole of the amount paid for education fees of up to two children.
    """,
    "gst": """
    Chapter 3: GST Rates for Textiles
    The rate of GST on textile products shall be as follows:
    (a) Raw cotton: 5%
    (b) Processed fabrics: 12%
    (c) Ready-made garments: 18%
    """,
    "property": """
    Property Registration Act, Section 17
    (1) The following documents shall be registered:
    (a) Instruments of gift of immovable property
    (b) Leases of immovable property exceeding one year
    """,
}

# Test queries
TEST_QUERIES = [
    "education tax deduction",
    "GST rate textile",
    "property registration requirements",
]

# Test embeddings
SAMPLE_EMBEDDINGS = {
    "doc1": [0.1, 0.2, 0.3],
    "doc2": [0.4, 0.5, 0.6],
    "doc3": [0.7, 0.8, 0.9],
}

# Test legal entities
SAMPLE_ENTITIES = {
    "income_tax": ["Section 80C", "education fees", "total income"],
    "gst": ["GST", "textile products", "cotton", "fabrics"],
    "property": ["Property Registration Act", "immovable property", "leases"],
}

# Test similarity scores
SAMPLE_SCORES = {
    "cosine": {"doc1": 0.85, "doc2": 0.65, "doc3": 0.45},
    "euclidean": {"doc1": 0.75, "doc2": 0.55, "doc3": 0.35},
    "mmr": {"doc1": 0.95, "doc2": 0.75, "doc3": 0.55},
    "hybrid": {"doc1": 0.90, "doc2": 0.70, "doc3": 0.50},
}

# Create test directories if they don't exist
for directory in [TEST_DATA_DIR, TEST_ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Save sample documents to test data directory
for doc_type, content in SAMPLE_DOCUMENTS.items():
    with open(TEST_DATA_DIR / f"{doc_type}.txt", "w", encoding="utf-8") as f:
        f.write(content)
