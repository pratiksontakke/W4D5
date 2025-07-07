"""
Unit tests for the core components of the Indian Legal Document Search System.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
from tests.test_config import (
    SAMPLE_DOCUMENTS,
    SAMPLE_EMBEDDINGS,
    SAMPLE_ENTITIES,
    SAMPLE_SCORES,
    TEST_DATA_DIR,
    TEST_QUERIES,
)


class TestDocumentLoader(unittest.TestCase):
    """Test suite for document loading functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        self.loader = DocumentLoader(TEST_DATA_DIR)

    def test_load_text_file(self):
        """Test loading a text file."""
        content = self.loader.load_file(TEST_DATA_DIR / "income_tax.txt")
        self.assertIsInstance(content, str)
        self.assertIn("Section 80C", content)

    def test_load_multiple_files(self):
        """Test loading multiple files."""
        documents = self.loader.load_directory()
        self.assertEqual(len(documents), len(SAMPLE_DOCUMENTS))

    def test_invalid_file(self):
        """Test handling of invalid file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_file(TEST_DATA_DIR / "nonexistent.txt")


class TestTextPreprocessor(unittest.TestCase):
    """Test suite for text preprocessing functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        self.preprocessor = TextPreprocessor()

    def test_text_cleaning(self):
        """Test basic text cleaning."""
        text = "Section 80C (1) \n\n Tax deduction..."
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("\n", cleaned)
        self.assertTrue(cleaned.startswith("Section"))

    def test_entity_extraction(self):
        """Test legal entity extraction."""
        text = SAMPLE_DOCUMENTS["income_tax"]
        entities = self.preprocessor.extract_entities(text)
        self.assertIn("Section 80C", entities)

    def test_text_normalization(self):
        """Test text normalization."""
        text = "TAX Deduction"
        normalized = self.preprocessor.normalize_text(text)
        self.assertEqual(normalized, "tax deduction")


class TestTextEmbedder(unittest.TestCase):
    """Test suite for text embedding functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        self.embedder = TextEmbedder()

    def test_tfidf_embedding(self):
        """Test TF-IDF embedding generation."""
        docs = list(SAMPLE_DOCUMENTS.values())
        embeddings = self.embedder.get_tfidf_embeddings(docs)
        self.assertEqual(len(embeddings), len(docs))

    def test_sentence_embedding(self):
        """Test sentence transformer embedding."""
        text = SAMPLE_DOCUMENTS["income_tax"]
        embedding = self.embedder.get_sentence_embedding(text)
        self.assertIsInstance(embedding, np.ndarray)

    def test_embedding_similarity(self):
        """Test embedding similarity computation."""
        emb1 = np.array([0.1, 0.2, 0.3])
        emb2 = np.array([0.4, 0.5, 0.6])
        similarity = self.embedder.compute_similarity(emb1, emb2)
        self.assertIsInstance(similarity, float)
        self.assertTrue(0 <= similarity <= 1)


class TestSimilarityMethods(unittest.TestCase):
    """Test suite for similarity computation methods."""

    def setUp(self):
        """Set up test environment before each test."""
        self.embeddings = np.array(list(SAMPLE_EMBEDDINGS.values()))
        self.query_embedding = np.array([0.2, 0.3, 0.4])

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        cosine = CosineSimilarity()
        scores = cosine.compute(self.query_embedding, self.embeddings)
        self.assertEqual(len(scores), len(self.embeddings))
        self.assertTrue(all(0 <= s <= 1 for s in scores))

    def test_euclidean_distance(self):
        """Test euclidean distance computation."""
        euclidean = EuclideanDistance()
        scores = euclidean.compute(self.query_embedding, self.embeddings)
        self.assertEqual(len(scores), len(self.embeddings))

    def test_mmr(self):
        """Test maximal marginal relevance."""
        mmr = MaximalMarginalRelevance(lambda_param=0.6)
        scores = mmr.compute(self.query_embedding, self.embeddings)
        self.assertEqual(len(scores), len(self.embeddings))

    def test_hybrid_similarity(self):
        """Test hybrid similarity computation."""
        hybrid = HybridSimilarity(cosine_weight=0.6, entity_weight=0.4)
        entities1 = SAMPLE_ENTITIES["income_tax"]
        entities2 = SAMPLE_ENTITIES["gst"]
        score = hybrid.compute_with_entities(
            self.query_embedding, self.embeddings[0], entities1, entities2
        )
        self.assertTrue(0 <= score <= 1)


class TestDocumentRetriever(unittest.TestCase):
    """Test suite for document retrieval functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        self.retriever = DocumentRetriever()
        self.documents = SAMPLE_DOCUMENTS
        self.scores = SAMPLE_SCORES

    def test_rank_documents(self):
        """Test document ranking."""
        ranked = self.retriever.rank_documents(self.scores["cosine"])
        self.assertEqual(len(ranked), len(self.scores["cosine"]))
        self.assertEqual(ranked[0][0], "doc1")  # Highest scoring doc

    def test_get_top_k(self):
        """Test top-k retrieval."""
        top_k = self.retriever.get_top_k(self.scores["cosine"], k=2)
        self.assertEqual(len(top_k), 2)

    def test_aggregate_results(self):
        """Test results aggregation across methods."""
        aggregated = self.retriever.aggregate_results(self.scores)
        self.assertEqual(len(aggregated), len(self.scores))


class TestEvaluator(unittest.TestCase):
    """Test suite for evaluation metrics."""

    def setUp(self):
        """Set up test environment before each test."""
        self.evaluator = Evaluator()
        self.relevant_docs = ["doc1", "doc2"]
        self.retrieved_docs = ["doc1", "doc3", "doc2"]

    def test_precision(self):
        """Test precision calculation."""
        precision = self.evaluator.calculate_precision(
            self.relevant_docs, self.retrieved_docs
        )
        self.assertIsInstance(precision, float)
        self.assertEqual(precision, 2 / 3)

    def test_recall(self):
        """Test recall calculation."""
        recall = self.evaluator.calculate_recall(
            self.relevant_docs, self.retrieved_docs
        )
        self.assertEqual(recall, 1.0)

    def test_diversity_score(self):
        """Test diversity score calculation."""
        docs = list(SAMPLE_DOCUMENTS.values())
        diversity = self.evaluator.calculate_diversity(docs)
        self.assertTrue(0 <= diversity <= 1)


if __name__ == "__main__":
    # Create test data files if they don't exist
    for doc_type, content in SAMPLE_DOCUMENTS.items():
        path = TEST_DATA_DIR / f"{doc_type}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # Run the tests
    unittest.main(verbosity=2)
