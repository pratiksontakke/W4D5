"""
Build index pipeline for legal document search system.

This module implements:
1. Document ingestion and validation
2. Text preprocessing and cleaning
3. Embedding generation
4. Index building and optimization
5. Artifact management
6. Pipeline monitoring and logging

The pipeline is designed to efficiently process large collections of legal documents
and prepare them for fast similarity search during inference.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from config import (
    ARTIFACTS_DIR,
    EMBEDDING_MODELS,
    INDICES_DIR,
    LEGAL_DOCUMENT_TYPES,
    PROCESSING_CONFIG,
    RAW_DATA_DIR,
)
from core.data_loader import DocumentLoader
from core.embedders import EmbeddingGenerator
from core.preprocess import TextPreprocessor
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Container for pipeline statistics."""

    start_time: str
    end_time: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    total_tokens: int
    embedding_dimensions: Dict[str, int]
    document_types: Dict[str, int]
    errors: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class IndexBuilder:
    """Main index building pipeline class."""

    def __init__(
        self,
        batch_size: int = PROCESSING_CONFIG["batch_size"],
        num_workers: int = PROCESSING_CONFIG["num_workers"],
    ):
        """Initialize index builder.

        Args:
            batch_size: Number of documents to process at once
            num_workers: Number of parallel workers
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize components
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.embedder = EmbeddingGenerator()

        # Initialize statistics
        self.stats = PipelineStats(
            start_time=datetime.now().isoformat(),
            end_time="",
            total_documents=0,
            processed_documents=0,
            failed_documents=0,
            total_tokens=0,
            embedding_dimensions={},
            document_types={},
            errors=[],
        )

    def build_index(
        self,
        input_dir: Path = RAW_DATA_DIR,
        output_dir: Path = INDICES_DIR,
        doc_types: Optional[List[str]] = None,
    ) -> PipelineStats:
        """Run full indexing pipeline.

        Args:
            input_dir: Directory containing raw documents
            output_dir: Directory to save indices and artifacts
            doc_types: Optional list of document types to process

        Returns:
            PipelineStats object with pipeline statistics
        """
        try:
            logger.info("Starting indexing pipeline...")

            # Validate and prepare directories
            self._prepare_directories(input_dir, output_dir)

            # Load and validate documents
            documents = self._ingest_documents(input_dir, doc_types)

            # Preprocess documents
            processed_docs = self._preprocess_documents(documents)

            # Generate embeddings
            embeddings = self._generate_embeddings(processed_docs)

            # Build indices
            self._build_indices(processed_docs, embeddings, output_dir)

            # Save artifacts and metadata
            self._save_artifacts(processed_docs, embeddings, output_dir)

            # Finalize statistics
            self._finalize_stats()

            logger.info("Indexing pipeline completed successfully")
            return self.stats

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.stats.errors.append(str(e))
            raise

    def _prepare_directories(self, input_dir: Path, output_dir: Path) -> None:
        """Prepare input and output directories.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
        """
        # Validate input directory
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different artifacts
        (output_dir / "embeddings").mkdir(exist_ok=True)
        (output_dir / "indices").mkdir(exist_ok=True)
        (output_dir / "metadata").mkdir(exist_ok=True)

    def _ingest_documents(
        self, input_dir: Path, doc_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Load and validate documents.

        Args:
            input_dir: Directory containing documents
            doc_types: Optional list of document types to process

        Returns:
            List of document dictionaries
        """
        logger.info("Ingesting documents...")

        # Get document paths
        doc_paths = list(input_dir.glob("**/*"))
        self.stats.total_documents = len(doc_paths)

        documents = []
        for path in tqdm(doc_paths, desc="Loading documents"):
            try:
                # Load document
                doc = self.loader.load_document(path)

                # Filter by document type
                if doc_types and doc["doc_type"] not in doc_types:
                    continue

                documents.append(doc)

                # Update statistics
                self.stats.processed_documents += 1
                self.stats.document_types[doc["doc_type"]] = (
                    self.stats.document_types.get(doc["doc_type"], 0) + 1
                )

            except Exception as e:
                logger.error(f"Failed to load document {path}: {str(e)}")
                self.stats.failed_documents += 1
                self.stats.errors.append(f"Load failed for {path}: {str(e)}")

        return documents

    def _preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """Clean and preprocess documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of preprocessed documents
        """
        logger.info("Preprocessing documents...")

        processed_docs = []
        for doc in tqdm(documents, desc="Preprocessing"):
            try:
                # Preprocess text
                processed_text = self.preprocessor.preprocess(
                    doc["content"], extract_entities=True
                )

                # Update document
                doc["processed_content"] = processed_text
                doc["entities"] = self.preprocessor.get_entities(doc["content"])

                processed_docs.append(doc)

                # Update statistics
                self.stats.total_tokens += len(processed_text.split())

            except Exception as e:
                logger.error(f"Failed to preprocess document {doc['doc_id']}: {str(e)}")
                self.stats.failed_documents += 1
                self.stats.errors.append(
                    f"Preprocessing failed for {doc['doc_id']}: {str(e)}"
                )

        return processed_docs

    def _generate_embeddings(self, documents: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate embeddings for documents.

        Args:
            documents: List of preprocessed documents

        Returns:
            Dictionary of embedding type to embedding matrix
        """
        logger.info("Generating embeddings...")

        embeddings = {}
        texts = [doc["processed_content"] for doc in documents]

        # Generate embeddings for each model
        for model_name, model_path in EMBEDDING_MODELS.items():
            try:
                if model_name == "tf-idf":
                    # Generate TF-IDF embeddings
                    embeddings["tf-idf"] = self.embedder.generate_tfidf(texts)

                elif model_name == "sentence-transformer":
                    # Generate Sentence-BERT embeddings
                    embeddings[
                        "sentence-transformer"
                    ] = self.embedder.generate_sentence_embeddings(texts)

                # Update statistics
                self.stats.embedding_dimensions[model_name] = embeddings[
                    model_name
                ].shape[1]

            except Exception as e:
                logger.error(f"Failed to generate {model_name} embeddings: {str(e)}")
                self.stats.errors.append(f"Embedding failed for {model_name}: {str(e)}")

        return embeddings

    def _build_indices(
        self, documents: List[Dict], embeddings: Dict[str, np.ndarray], output_dir: Path
    ) -> None:
        """Build search indices.

        Args:
            documents: List of preprocessed documents
            embeddings: Dictionary of embedding matrices
            output_dir: Output directory for indices
        """
        logger.info("Building indices...")

        index_dir = output_dir / "indices"

        # Save document ID mapping
        id_mapping = {i: doc["doc_id"] for i, doc in enumerate(documents)}

        with open(index_dir / "id_mapping.json", "w") as f:
            json.dump(id_mapping, f, indent=2)

        # Save type mapping
        type_mapping = {doc["doc_id"]: doc["doc_type"] for doc in documents}

        with open(index_dir / "type_mapping.json", "w") as f:
            json.dump(type_mapping, f, indent=2)

        # Save entity mapping
        entity_mapping = {doc["doc_id"]: doc["entities"] for doc in documents}

        with open(index_dir / "entity_mapping.json", "w") as f:
            json.dump(entity_mapping, f, indent=2)

    def _save_artifacts(
        self, documents: List[Dict], embeddings: Dict[str, np.ndarray], output_dir: Path
    ) -> None:
        """Save pipeline artifacts.

        Args:
            documents: List of preprocessed documents
            embeddings: Dictionary of embedding matrices
            output_dir: Output directory
        """
        logger.info("Saving artifacts...")

        # Save embeddings
        embedding_dir = output_dir / "embeddings"
        for model_name, embedding_matrix in embeddings.items():
            np.save(embedding_dir / f"{model_name}_embeddings.npy", embedding_matrix)

        # Save preprocessed documents
        with open(output_dir / "processed_documents.json", "w") as f:
            json.dump(documents, f, indent=2)

        # Save pipeline metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": {"batch_size": self.batch_size, "num_workers": self.num_workers},
            "stats": self.stats.to_dict(),
        }

        with open(output_dir / "metadata" / "pipeline_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _finalize_stats(self) -> None:
        """Finalize pipeline statistics."""
        self.stats.end_time = datetime.now().isoformat()


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize builder
        builder = IndexBuilder(batch_size=32, num_workers=4)

        # Create sample documents
        sample_dir = RAW_DATA_DIR / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create sample text files
        for i in range(5):
            doc_type = LEGAL_DOCUMENT_TYPES[i % len(LEGAL_DOCUMENT_TYPES)]
            content = f"""
            Sample {doc_type} document {i}.
            This is a test document for the indexing pipeline.
            It contains some legal terms like court, judge, and tax.
            """

            with open(sample_dir / f"doc_{i}.txt", "w") as f:
                f.write(content)

        # Run pipeline
        stats = builder.build_index(
            input_dir=sample_dir,
            doc_types=LEGAL_DOCUMENT_TYPES[:2],  # Only process first two types
        )

        # Print statistics
        print("\nPipeline Statistics:")
        print(f"Total documents: {stats.total_documents}")
        print(f"Processed documents: {stats.processed_documents}")
        print(f"Failed documents: {stats.failed_documents}")
        print(f"Total tokens: {stats.total_tokens}")
        print("\nEmbedding dimensions:")
        for model, dim in stats.embedding_dimensions.items():
            print(f"- {model}: {dim}")
        print("\nDocument types:")
        for doc_type, count in stats.document_types.items():
            print(f"- {doc_type}: {count}")

        if stats.errors:
            print("\nErrors encountered:")
            for error in stats.errors:
                print(f"- {error}")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
