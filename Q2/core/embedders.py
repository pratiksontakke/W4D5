"""
Text embedding module for legal documents.

This module handles:
1. TF-IDF vectorization
2. Sentence-BERT embeddings
3. Embedding caching
4. Validation checks
5. Dimension reduction
6. Embedding visualization

The module is designed to be efficient with caching and
extensible for new embedding methods.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import EMBEDDING_MODELS, EMBEDDINGS_DIR, PROCESSING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings."""

    method: str
    dimension: int
    vocab_size: Optional[int] = None
    model_name: Optional[str] = None
    version: str = "1.0"

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            "method": self.method,
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "model_name": self.model_name,
            "version": self.version,
        }


class TextEmbedder:
    """Main text embedding class."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize embedders and cache.

        Args:
            cache_dir: Optional custom cache directory
        """
        self.cache_dir = cache_dir or EMBEDDINGS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=50000,  # Limit vocabulary size
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words="english",
        )

        # Initialize Sentence-BERT
        self.sbert = SentenceTransformer(EMBEDDING_MODELS["sentence-transformer"])

        # Initialize dimension reduction
        self.pca = None
        self.tsne = None

        # Cache for embeddings
        self._embedding_cache = {}

    def embed_documents(
        self,
        texts: List[str],
        method: str = "hybrid",
        use_cache: bool = True,
        reduce_dim: Optional[int] = None,
    ) -> Tuple[np.ndarray, EmbeddingMetadata]:
        """Embed multiple documents.

        Args:
            texts: List of documents to embed
            method: Embedding method ('tfidf', 'sbert', or 'hybrid')
            use_cache: Whether to use embedding cache
            reduce_dim: Optional dimension to reduce to

        Returns:
            Tuple of (embeddings array, metadata)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(texts, method)

        # Try to load from cache
        if use_cache:
            cached = self._load_from_cache(cache_key, method)
            if cached is not None:
                embeddings, metadata = cached
                logger.info(f"Loaded {method} embeddings from cache")
                return embeddings, metadata

        # Generate embeddings based on method
        if method == "tfidf":
            embeddings = self._embed_tfidf(texts)
            metadata = EmbeddingMetadata(
                method="tfidf",
                dimension=embeddings.shape[1],
                vocab_size=len(self.tfidf.vocabulary_),
            )

        elif method == "sbert":
            embeddings = self._embed_sbert(texts)
            metadata = EmbeddingMetadata(
                method="sbert",
                dimension=embeddings.shape[1],
                model_name=EMBEDDING_MODELS["sentence-transformer"],
            )

        elif method == "hybrid":
            # Combine TF-IDF and SBERT
            tfidf_embeddings = self._embed_tfidf(texts)
            sbert_embeddings = self._embed_sbert(texts)

            # Normalize and concatenate
            tfidf_norm = self._normalize_embeddings(tfidf_embeddings)
            sbert_norm = self._normalize_embeddings(sbert_embeddings)
            embeddings = np.hstack([tfidf_norm, sbert_norm])

            metadata = EmbeddingMetadata(
                method="hybrid",
                dimension=embeddings.shape[1],
                vocab_size=len(self.tfidf.vocabulary_),
                model_name=EMBEDDING_MODELS["sentence-transformer"],
            )

        else:
            raise ValueError(f"Unknown embedding method: {method}")

        # Reduce dimensions if requested
        if reduce_dim is not None:
            embeddings = self.reduce_dimensions(embeddings, reduce_dim)
            metadata.dimension = reduce_dim

        # Validate embeddings
        self._validate_embeddings(embeddings, texts)

        # Cache embeddings
        if use_cache:
            self._save_to_cache(cache_key, method, embeddings, metadata)

        return embeddings, metadata

    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings.

        Args:
            texts: List of documents

        Returns:
            TF-IDF matrix
        """
        # Check if vectorizer needs to be fit
        if not self.tfidf.vocabulary_:
            embeddings = self.tfidf.fit_transform(texts)
        else:
            embeddings = self.tfidf.transform(texts)

        return embeddings.toarray()

    def _embed_sbert(self, texts: List[str]) -> np.ndarray:
        """Generate Sentence-BERT embeddings.

        Args:
            texts: List of documents

        Returns:
            SBERT embeddings array
        """
        # Process in batches
        batch_size = PROCESSING_CONFIG["batch_size"]
        embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size), desc="Generating SBERT embeddings"
        ):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.sbert.encode(
                batch, show_progress_bar=False, convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def reduce_dimensions(
        self, embeddings: np.ndarray, target_dim: int, method: str = "pca"
    ) -> np.ndarray:
        """Reduce embedding dimensions.

        Args:
            embeddings: Input embeddings
            target_dim: Target dimension
            method: Reduction method ('pca' or 'tsne')

        Returns:
            Reduced embeddings
        """
        if method == "pca":
            if self.pca is None or self.pca.n_components_ != target_dim:
                self.pca = PCA(n_components=target_dim)
            reduced = self.pca.fit_transform(embeddings)

        elif method == "tsne":
            if self.tsne is None or self.tsne.n_components != target_dim:
                self.tsne = TSNE(n_components=target_dim)
            reduced = self.tsne.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")

        return reduced

    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Visualize embeddings in 2D.

        Args:
            embeddings: Input embeddings
            labels: Optional label for each embedding
            save_path: Optional path to save plot
        """
        # Reduce to 2D if needed
        if embeddings.shape[1] != 2:
            embeddings_2d = self.reduce_dimensions(embeddings, 2, "tsne")
        else:
            embeddings_2d = embeddings

        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

        if labels:
            for i, label in enumerate(labels):
                plt.annotate(
                    label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8
                )

        plt.title("Document Embeddings Visualization")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length.

        Args:
            embeddings: Input embeddings

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _generate_cache_key(self, texts: List[str], method: str) -> str:
        """Generate cache key for embeddings.

        Args:
            texts: Input texts
            method: Embedding method

        Returns:
            Cache key string
        """
        # Combine all texts and method
        content = "".join(texts) + method
        return hashlib.md5(content.encode()).hexdigest()

    def _save_to_cache(
        self, key: str, method: str, embeddings: np.ndarray, metadata: EmbeddingMetadata
    ) -> None:
        """Save embeddings to cache.

        Args:
            key: Cache key
            method: Embedding method
            embeddings: Embeddings array
            metadata: Embedding metadata
        """
        cache_path = self.cache_dir / f"{method}_{key}.npz"
        metadata_path = self.cache_dir / f"{method}_{key}_meta.json"

        # Save embeddings
        np.savez_compressed(cache_path, embeddings=embeddings)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Cached {method} embeddings to {cache_path}")

    def _load_from_cache(
        self, key: str, method: str
    ) -> Optional[Tuple[np.ndarray, EmbeddingMetadata]]:
        """Load embeddings from cache.

        Args:
            key: Cache key
            method: Embedding method

        Returns:
            Tuple of (embeddings, metadata) if found, None otherwise
        """
        cache_path = self.cache_dir / f"{method}_{key}.npz"
        metadata_path = self.cache_dir / f"{method}_{key}_meta.json"

        if cache_path.exists() and metadata_path.exists():
            # Load embeddings
            with np.load(cache_path) as data:
                embeddings = data["embeddings"]

            # Load metadata
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
                metadata = EmbeddingMetadata(**metadata_dict)

            return embeddings, metadata

        return None

    def _validate_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """Validate embedding quality and dimensions.

        Args:
            embeddings: Generated embeddings
            texts: Original texts

        Raises:
            ValueError: If validation fails
        """
        # Check dimensions
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) " f"!= text count ({len(texts)})"
            )

        # Check for NaN values
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values")

        # Check for zero vectors
        zero_vectors = np.all(embeddings == 0, axis=1)
        if zero_vectors.any():
            raise ValueError(f"Found {zero_vectors.sum()} zero vectors")

        # Check for reasonable magnitudes
        magnitudes = np.linalg.norm(embeddings, axis=1)
        if (magnitudes > 100).any() or (magnitudes < 0.01).any():
            logger.warning("Some embedding magnitudes are outside expected range")


if __name__ == "__main__":
    # Example usage
    embedder = TextEmbedder()

    # Sample legal texts
    texts = [
        """
        Section 80C of the Income Tax Act provides for deductions
        that can be claimed from gross total income.
        """,
        """
        The GST rate for textile products varies based on the
        type of product and its HSN classification.
        """,
        """
        Property registration process requires submission of
        documents and payment of stamp duty.
        """,
    ]

    try:
        # Generate embeddings with different methods
        for method in ["tfidf", "sbert", "hybrid"]:
            print(f"\nGenerating {method} embeddings:")
            embeddings, metadata = embedder.embed_documents(
                texts, method=method, use_cache=True
            )

            print(f"Shape: {embeddings.shape}")
            print(f"Metadata: {metadata.to_dict()}")

            # Visualize (for demonstration)
            if method == "hybrid":
                embedder.visualize_embeddings(
                    embeddings, labels=[f"Doc {i+1}" for i in range(len(texts))]
                )

        # Demonstrate dimension reduction
        print("\nReducing dimensions:")
        embeddings_reduced, metadata = embedder.embed_documents(
            texts, method="hybrid", reduce_dim=10
        )
        print(f"Reduced shape: {embeddings_reduced.shape}")

    except Exception as e:
        print(f"Error: {str(e)}")
