"""
Similarity computation module for legal documents.

This module implements:
1. Cosine similarity
2. Euclidean distance
3. Maximal Marginal Relevance (MMR)
4. Hybrid similarity (Cosine + Entity)
5. Score normalization
6. Method comparison

The module is designed to handle both dense and sparse vectors,
with optimizations for large-scale similarity computations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from config import SIMILARITY_METHODS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Container for similarity computation results."""

    scores: np.ndarray
    method: str
    indices: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

    def get_top_k(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top k scores and indices.

        Args:
            k: Number of top results to return

        Returns:
            Tuple of (top k scores, top k indices)
        """
        if self.indices is None:
            self.indices = np.arange(len(self.scores))

        # Handle case where k > number of scores
        k = min(k, len(self.scores))

        # Get top k indices (ascending for distance, descending for similarity)
        if self.method == "euclidean":
            top_k_idx = np.argpartition(self.scores, k)[:k]
        else:
            top_k_idx = np.argpartition(-self.scores, k)[:k]

        return self.scores[top_k_idx], self.indices[top_k_idx]


class SimilarityComputer:
    """Main similarity computation class."""

    def __init__(self):
        """Initialize similarity computer with configured methods."""
        self.methods = SIMILARITY_METHODS
        self.normalizer = MinMaxScaler()

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        method: str = "cosine",
        entity_overlap: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
    ) -> SimilarityResult:
        """Compute similarity between query and documents.

        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors
            method: Similarity method to use
            entity_overlap: Optional entity overlap scores for hybrid method
            batch_size: Optional batch size for large-scale computation

        Returns:
            SimilarityResult object
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Select similarity function
        if method == "cosine":
            scores = self._compute_cosine(
                query_embedding, document_embeddings, batch_size
            )

        elif method == "euclidean":
            scores = self._compute_euclidean(
                query_embedding, document_embeddings, batch_size
            )

        elif method == "mmr":
            scores = self._compute_mmr(
                query_embedding,
                document_embeddings,
                lambda_param=self.methods["mmr"]["lambda_param"],
                threshold=self.methods["mmr"]["threshold"],
            )

        elif method == "hybrid":
            if entity_overlap is None:
                raise ValueError("Entity overlap scores required for hybrid method")

            scores = self._compute_hybrid(
                query_embedding, document_embeddings, entity_overlap, batch_size
            )

        else:
            raise ValueError(f"Unknown similarity method: {method}")

        # Create result object
        result = SimilarityResult(
            scores=scores,
            method=method,
            metadata={
                "num_documents": len(document_embeddings),
                "embedding_dim": document_embeddings.shape[1],
            },
        )

        return result

    def _compute_cosine(
        self, query: np.ndarray, documents: np.ndarray, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Compute cosine similarity.

        Args:
            query: Query vector (1 x d)
            documents: Document vectors (n x d)
            batch_size: Optional batch size

        Returns:
            Similarity scores array
        """
        # Handle sparse matrices
        if isinstance(documents, csr_matrix):
            documents = documents.toarray()

        if batch_size is None:
            # Compute all at once
            scores = self._cosine_similarity(query, documents)
        else:
            # Compute in batches
            scores = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_scores = self._cosine_similarity(query, batch)
                scores.append(batch_scores)
            scores = np.concatenate(scores)

        return scores.flatten()

    def _compute_euclidean(
        self, query: np.ndarray, documents: np.ndarray, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Compute Euclidean distance.

        Args:
            query: Query vector
            documents: Document vectors
            batch_size: Optional batch size

        Returns:
            Distance scores array
        """
        if batch_size is None:
            # Compute all at once
            distances = cdist(query, documents, metric="euclidean")
        else:
            # Compute in batches
            distances = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                batch_distances = cdist(query, batch, metric="euclidean")
                distances.append(batch_distances)
            distances = np.concatenate(distances)

        return distances.flatten()

    def _compute_mmr(
        self,
        query: np.ndarray,
        documents: np.ndarray,
        lambda_param: float = 0.6,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """Compute Maximal Marginal Relevance.

        Args:
            query: Query vector
            documents: Document vectors
            lambda_param: Balance between relevance and diversity
            threshold: Diversity threshold

        Returns:
            MMR scores array
        """
        n_docs = len(documents)

        # Initialize scores and selected set
        mmr_scores = np.zeros(n_docs)
        selected = set()

        # Compute initial similarity to query
        sim_query = self._cosine_similarity(query, documents).flatten()

        # Iteratively select documents
        for i in range(n_docs):
            # Compute MMR scores for remaining documents
            if i == 0:
                # First document: pure similarity to query
                mmr = sim_query
            else:
                # Later documents: balance similarity and diversity
                selected_docs = documents[list(selected)]
                sim_selected = self._cosine_similarity(documents, selected_docs).max(
                    axis=1
                )

                mmr = lambda_param * sim_query - (1 - lambda_param) * sim_selected

                # Zero out already selected
                mmr[list(selected)] = -np.inf

            # Select document with highest MMR
            next_idx = mmr.argmax()
            mmr_scores[next_idx] = mmr[next_idx]
            selected.add(next_idx)

            # Stop if max similarity drops below threshold
            if mmr[next_idx] < threshold:
                break

        return mmr_scores

    def _compute_hybrid(
        self,
        query: np.ndarray,
        documents: np.ndarray,
        entity_overlap: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Compute hybrid similarity (cosine + entity).

        Args:
            query: Query vector
            documents: Document vectors
            entity_overlap: Entity overlap scores
            batch_size: Optional batch size

        Returns:
            Hybrid similarity scores array
        """
        # Get weights from config
        cosine_weight = self.methods["hybrid"]["cosine_weight"]
        entity_weight = self.methods["hybrid"]["entity_weight"]

        # Compute cosine similarity
        cosine_scores = self._compute_cosine(query, documents, batch_size)

        # Normalize both score types to [0, 1]
        cosine_norm = self._normalize_scores(cosine_scores)
        entity_norm = self._normalize_scores(entity_overlap)

        # Combine scores
        hybrid_scores = cosine_weight * cosine_norm + entity_weight * entity_norm

        return hybrid_scores

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vectors.

        Args:
            a: First vector(s)
            b: Second vector(s)

        Returns:
            Similarity matrix
        """
        # Normalize vectors
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

        # Avoid division by zero
        a_norm[a_norm == 0] = 1
        b_norm[b_norm == 0] = 1

        a_normalized = a / a_norm
        b_normalized = b / b_norm

        # Compute similarity
        return np.dot(a_normalized, b_normalized.T)

    def _normalize_scores(
        self, scores: np.ndarray, feature_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """Normalize scores to specified range.

        Args:
            scores: Input scores
            feature_range: Target range for normalization

        Returns:
            Normalized scores
        """
        # Reshape for sklearn
        scores_2d = scores.reshape(-1, 1)

        # Fit and transform
        self.normalizer.feature_range = feature_range
        normalized = self.normalizer.fit_transform(scores_2d)

        return normalized.flatten()

    def compare_methods(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        entity_overlap: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> Dict[str, SimilarityResult]:
        """Compare all similarity methods.

        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors
            entity_overlap: Optional entity overlap scores
            top_k: Number of top results to consider

        Returns:
            Dictionary of method names to SimilarityResult objects
        """
        results = {}

        # Compute similarity for each method
        for method in self.methods:
            try:
                result = self.compute_similarity(
                    query_embedding,
                    document_embeddings,
                    method=method,
                    entity_overlap=entity_overlap,
                )

                # Add top-k metadata
                scores, indices = result.get_top_k(top_k)
                result.metadata.update(
                    {"top_k_scores": scores.tolist(), "top_k_indices": indices.tolist()}
                )

                results[method] = result

            except Exception as e:
                logger.error(f"Error computing {method} similarity: {str(e)}")

        return results


if __name__ == "__main__":
    # Example usage
    computer = SimilarityComputer()

    # Create sample embeddings
    query = np.random.randn(1, 100)  # 1 query, 100 dimensions
    documents = np.random.randn(50, 100)  # 50 documents, 100 dimensions

    # Create sample entity overlap scores
    entity_overlap = np.random.random(50)  # 50 documents

    try:
        # Compare all methods
        results = computer.compare_methods(query, documents, entity_overlap, top_k=3)

        # Print results
        for method, result in results.items():
            print(f"\n{method.upper()} Results:")
            print(f"Shape: {result.scores.shape}")
            print("Top 3 scores:", result.get_top_k(3)[0])
            print("Metadata:", result.metadata)

        # Demonstrate batch processing
        print("\nBatch processing example:")
        batch_result = computer.compute_similarity(
            query, documents, method="cosine", batch_size=10
        )
        print(f"Batch result shape: {batch_result.scores.shape}")

    except Exception as e:
        print(f"Error: {str(e)}")
