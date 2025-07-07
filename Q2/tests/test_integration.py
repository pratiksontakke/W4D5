"""
Integration tests for the Indian Legal Document Search System.
Tests end-to-end workflows and component interactions.
"""

import asyncio
import json
import logging
import multiprocessing
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import aiohttp
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    EVALUATION_CONFIG,
    LEGAL_DOCUMENT_TYPES,
    PROCESSED_DATA_DIR,
    PROCESSING_CONFIG,
    RAW_DATA_DIR,
    UI_CONFIG,
)
from core.data_loader import DocumentLoader
from core.embedders import TextEmbedder
from core.evaluation import Evaluator
from core.preprocess import TextPreprocessor
from core.retrieval import DocumentRetriever
from core.similarity import (
    CosineSimilarity,
    EuclideanDistance,
    HybridSimilarity,
    MaximalMarginalRelevance,
)
from fastapi.testclient import TestClient
from pipelines.build_index import IndexBuilder
from ui.web_app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestOfflinePipeline(unittest.TestCase):
    """Integration tests for the offline processing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment before running tests."""
        cls.index_builder = IndexBuilder()
        cls.test_docs = {
            "income_tax.txt": "Section 80C discusses education tax benefits.",
            "gst.txt": "GST rates for textiles vary by category.",
            "property.txt": "Property registration requires proper documentation.",
        }

        # Create test documents
        for filename, content in cls.test_docs.items():
            path = RAW_DATA_DIR / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def test_complete_pipeline(self):
        """Test the complete offline processing pipeline."""
        try:
            # Run the complete pipeline
            self.index_builder.run()

            # Verify artifacts were created
            self.assertTrue(ARTIFACTS_DIR.exists())
            self.assertTrue((ARTIFACTS_DIR / "embeddings.npz").exists())
            self.assertTrue((ARTIFACTS_DIR / "metadata.json").exists())

            # Verify processed documents
            self.assertTrue(PROCESSED_DATA_DIR.exists())
            processed_files = list(PROCESSED_DATA_DIR.glob("*.txt"))
            self.assertEqual(len(processed_files), len(self.test_docs))

        except Exception as e:
            self.fail(f"Pipeline failed: {str(e)}")


class TestOnlineSearch(unittest.TestCase):
    """Integration tests for the online search functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and client."""
        cls.client = TestClient(app)
        cls.test_query = "education tax deduction"

    def test_search_flow(self):
        """Test the complete search flow."""
        # Make search request
        response = self.client.post("/api/search", json={"query": self.test_query})

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check response structure
        self.assertIn("results", data)
        self.assertIn("metrics", data)

        # Verify results for each similarity method
        for method in ["cosine", "euclidean", "mmr", "hybrid"]:
            self.assertIn(method, data["results"])
            self.assertIsInstance(data["results"][method], list)
            self.assertGreater(len(data["results"][method]), 0)


class TestUIComponents(unittest.TestCase):
    """Integration tests for UI components."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and client."""
        cls.client = TestClient(app)

    def test_document_upload(self):
        """Test document upload functionality."""
        # Create test file
        test_file = RAW_DATA_DIR / "test_upload.txt"
        test_file.write_text("Test content for upload")

        # Upload file
        with open(test_file, "rb") as f:
            response = self.client.post(
                "/api/upload", files={"file": ("test_upload.txt", f)}
            )

        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertIn("file_id", response.json())

    def test_metrics_dashboard(self):
        """Test metrics dashboard data retrieval."""
        response = self.client.get("/api/metrics")

        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check metrics
        required_metrics = ["precision", "recall", "diversity"]
        for metric in required_metrics:
            self.assertIn(metric, data)


class TestErrorHandling(unittest.TestCase):
    """Integration tests for error handling."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and client."""
        cls.client = TestClient(app)

    def test_invalid_file_upload(self):
        """Test handling of invalid file uploads."""
        # Try uploading file that's too large
        large_content = "x" * (UI_CONFIG["max_upload_size_mb"] * 1024 * 1024 + 1)
        response = self.client.post(
            "/api/upload", files={"file": ("large.txt", large_content.encode())}
        )
        self.assertEqual(response.status_code, 413)

    def test_invalid_query(self):
        """Test handling of invalid search queries."""
        # Empty query
        response = self.client.post("/api/search", json={"query": ""})
        self.assertEqual(response.status_code, 400)

        # Very long query
        long_query = "test " * 1000
        response = self.client.post("/api/search", json={"query": long_query})
        self.assertEqual(response.status_code, 400)


class TestPerformanceMetrics(unittest.TestCase):
    """Integration tests for performance metrics."""

    def setUp(self):
        """Set up test environment."""
        self.evaluator = Evaluator()
        self.test_results = {
            "relevant": ["doc1", "doc2", "doc3"],
            "retrieved": ["doc1", "doc4", "doc2"],
        }

    def test_metric_calculation(self):
        """Test calculation of all performance metrics."""
        metrics = self.evaluator.calculate_all_metrics(
            self.test_results["relevant"], self.test_results["retrieved"]
        )

        # Verify metrics
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("diversity", metrics)

        # Check metric values
        self.assertGreaterEqual(metrics["precision"], 0)
        self.assertLessEqual(metrics["precision"], 1)
        self.assertGreaterEqual(metrics["recall"], 0)
        self.assertLessEqual(metrics["recall"], 1)


class TestConcurrentUsers(unittest.TestCase):
    """Integration tests for concurrent user handling."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.client = TestClient(app)
        cls.num_users = 10
        cls.requests_per_user = 5

    async def concurrent_search(self, query: str) -> List[Dict]:
        """Perform concurrent search requests."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://testserver/api/search", json={"query": query}
            ) as response:
                return await response.json()

    async def run_concurrent_searches(self):
        """Run multiple concurrent searches."""
        tasks = []
        for _ in range(self.num_users):
            for query in EVALUATION_CONFIG["test_queries"]:
                tasks.append(self.concurrent_search(query))
        return await asyncio.gather(*tasks)

    def test_concurrent_requests(self):
        """Test system handling of concurrent requests."""
        # Run concurrent searches
        results = asyncio.run(self.run_concurrent_searches())

        # Verify all requests succeeded
        self.assertEqual(
            len(results), self.num_users * len(EVALUATION_CONFIG["test_queries"])
        )

        # Check each result
        for result in results:
            self.assertIn("results", result)
            self.assertIn("metrics", result)


if __name__ == "__main__":
    # Set up test data
    test_data_dir = RAW_DATA_DIR / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Create sample documents
    sample_docs = {
        "income_tax.txt": "Section 80C discusses education tax benefits.",
        "gst.txt": "GST rates for textiles vary by category.",
        "property.txt": "Property registration requires proper documentation.",
    }

    for filename, content in sample_docs.items():
        (test_data_dir / filename).write_text(content)

    # Run the tests
    unittest.main(verbosity=2)
