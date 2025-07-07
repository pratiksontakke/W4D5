"""
Retrieval system for legal document search.

This module handles:
1. Document ranking and scoring
2. Top-k retrieval with filters
3. Result aggregation across methods
4. Search result caching
5. Query expansion
6. Result filtering and post-processing

The module is designed to efficiently retrieve and rank documents
while providing additional features like caching and query expansion
for improved search performance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from cachetools import TTLCache, cached
from config import (
    EVALUATION_CONFIG,
    LEGAL_DOCUMENT_TYPES,
    PROCESSING_CONFIG,
    SIMILARITY_METHODS,
    UI_CONFIG,
)
from core.similarity import SimilarityResult
from nltk.corpus import wordnet
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Container for document metadata and content."""

    doc_id: str
    title: str
    content: str
    doc_type: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Container for search results."""

    query: str
    results: List[DocumentInfo]
    scores: Dict[str, float]  # Method -> Score mapping
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "results": [doc.to_dict() for doc in self.results],
            "scores": self.scores,
            "metadata": self.metadata,
        }


class RetrievalSystem:
    """Main retrieval system class."""

    def __init__(
        self, cache_timeout: int = UI_CONFIG["cache_timeout"], cache_size: int = 1000
    ):
        """Initialize retrieval system.

        Args:
            cache_timeout: Time in seconds to keep results in cache
            cache_size: Maximum number of queries to cache
        """
        self.cache_timeout = cache_timeout
        self.cache_size = cache_size

        # Initialize caches
        self.result_cache = TTLCache(maxsize=cache_size, ttl=cache_timeout)

        # Initialize normalizer for score aggregation
        self.normalizer = MinMaxScaler()

    def retrieve(
        self,
        query: str,
        similarity_results: Dict[str, SimilarityResult],
        documents: List[DocumentInfo],
        top_k: int = EVALUATION_CONFIG["top_k"],
        doc_type_filter: Optional[str] = None,
        min_score: float = EVALUATION_CONFIG["relevance_threshold"],
    ) -> SearchResult:
        """Retrieve and rank documents based on similarity scores.

        Args:
            query: Original search query
            similarity_results: Results from different similarity methods
            documents: List of available documents
            top_k: Number of results to return
            doc_type_filter: Optional filter by document type
            min_score: Minimum relevance score

        Returns:
            SearchResult object with ranked documents
        """
        # Check cache first
        cache_key = self._make_cache_key(query, top_k, doc_type_filter, min_score)

        cached_result = self.result_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for query: {query}")
            return cached_result

        # Aggregate scores across methods
        aggregated_scores = self._aggregate_scores(similarity_results)

        # Get candidate documents
        candidates = self._get_candidates(
            documents, aggregated_scores, min_score, doc_type_filter
        )

        # Rank and select top-k
        ranked_results = self._rank_results(candidates, aggregated_scores, top_k)

        # Create search result
        result = SearchResult(
            query=query,
            results=ranked_results,
            scores={
                method: results.scores.mean()
                for method, results in similarity_results.items()
            },
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_candidates": len(candidates),
                "filters_applied": {
                    "doc_type": doc_type_filter,
                    "min_score": min_score,
                },
            },
        )

        # Cache result
        self.result_cache[cache_key] = result

        return result

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with related terms using WordNet.

        Args:
            query: Original search query
            max_expansions: Maximum number of expanded terms per word

        Returns:
            List of expanded query terms
        """
        expanded_terms = set()

        # Process each word in query
        for word in query.split():
            # Get synsets
            synsets = wordnet.synsets(word)

            # Add synonyms and hypernyms
            for synset in synsets[:max_expansions]:
                # Add lemma names (synonyms)
                expanded_terms.update(lemma.name() for lemma in synset.lemmas())

                # Add hypernyms (more general terms)
                for hypernym in synset.hypernyms():
                    expanded_terms.update(lemma.name() for lemma in hypernym.lemmas())

        # Remove original terms and normalize
        expanded_terms = {
            term.lower().replace("_", " ")
            for term in expanded_terms
            if term.lower() != word.lower()
        }

        return list(expanded_terms)

    def _aggregate_scores(
        self, similarity_results: Dict[str, SimilarityResult]
    ) -> np.ndarray:
        """Aggregate scores from different similarity methods.

        Args:
            similarity_results: Results from different methods

        Returns:
            Array of aggregated scores
        """
        all_scores = []
        weights = []

        # Collect scores and weights
        for method, result in similarity_results.items():
            # Normalize scores to [0, 1]
            normalized = self._normalize_scores(result.scores)

            # Get method weight from config
            weight = SIMILARITY_METHODS[method].get("weight", 1.0)

            all_scores.append(normalized)
            weights.append(weight)

        # Convert to arrays
        all_scores = np.array(all_scores)
        weights = np.array(weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Compute weighted average
        aggregated = np.average(all_scores, axis=0, weights=weights)

        return aggregated

    def _get_candidates(
        self,
        documents: List[DocumentInfo],
        scores: np.ndarray,
        min_score: float,
        doc_type_filter: Optional[str],
    ) -> List[DocumentInfo]:
        """Get candidate documents based on scores and filters.

        Args:
            documents: List of all documents
            scores: Aggregated similarity scores
            min_score: Minimum score threshold
            doc_type_filter: Optional document type filter

        Returns:
            List of candidate documents
        """
        candidates = []

        for idx, doc in enumerate(documents):
            # Check score threshold
            if scores[idx] < min_score:
                continue

            # Apply document type filter
            if doc_type_filter and doc.doc_type != doc_type_filter:
                continue

            # Add score to metadata
            doc.metadata["score"] = float(scores[idx])
            candidates.append(doc)

        return candidates

    def _rank_results(
        self, candidates: List[DocumentInfo], scores: np.ndarray, top_k: int
    ) -> List[DocumentInfo]:
        """Rank candidates and return top-k results.

        Args:
            candidates: List of candidate documents
            scores: Similarity scores
            top_k: Number of results to return

        Returns:
            List of top-k ranked documents
        """
        # Sort by score
        ranked = sorted(candidates, key=lambda x: x.metadata["score"], reverse=True)

        # Return top-k
        return ranked[:top_k]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range.

        Args:
            scores: Input scores

        Returns:
            Normalized scores
        """
        # Reshape for sklearn
        scores_2d = scores.reshape(-1, 1)

        # Fit and transform
        normalized = self.normalizer.fit_transform(scores_2d)

        return normalized.flatten()

    def _make_cache_key(
        self, query: str, top_k: int, doc_type_filter: Optional[str], min_score: float
    ) -> str:
        """Create cache key from query parameters.

        Args:
            query: Search query
            top_k: Number of results
            doc_type_filter: Document type filter
            min_score: Minimum score

        Returns:
            Cache key string
        """
        return f"{query}::{top_k}::{doc_type_filter}::{min_score}"


if __name__ == "__main__":
    # Example usage
    retrieval = RetrievalSystem()

    # Create sample documents
    docs = [
        DocumentInfo(
            doc_id=f"doc_{i}",
            title=f"Document {i}",
            content=f"Sample content {i}",
            doc_type=LEGAL_DOCUMENT_TYPES[i % len(LEGAL_DOCUMENT_TYPES)],
        )
        for i in range(10)
    ]

    # Create sample similarity results
    n_docs = len(docs)
    similarity_results = {
        "cosine": SimilarityResult(
            scores=np.random.random(n_docs),
            method="cosine",
            metadata={"method": "cosine"},
        ),
        "euclidean": SimilarityResult(
            scores=np.random.random(n_docs),
            method="euclidean",
            metadata={"method": "euclidean"},
        ),
    }

    try:
        # Test retrieval
        result = retrieval.retrieve(
            query="sample query",
            similarity_results=similarity_results,
            documents=docs,
            top_k=3,
        )

        print("\nSearch Results:")
        print(f"Query: {result.query}")
        print(f"Number of results: {len(result.results)}")
        print("\nTop documents:")
        for doc in result.results:
            print(f"- {doc.title} (Score: {doc.metadata['score']:.3f})")

        # Test query expansion
        expanded = retrieval.expand_query("tax payment deadline")
        print("\nExpanded query terms:", expanded)

        # Test caching
        cached_result = retrieval.retrieve(
            query="sample query",
            similarity_results=similarity_results,
            documents=docs,
            top_k=3,
        )
        print("\nCache hit:", cached_result.query)

    except Exception as e:
        print(f"Error: {str(e)}")
