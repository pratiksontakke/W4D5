"""
Document loading and parsing module for legal documents.

This module handles:
1. Loading documents from various formats (PDF, Word, Text)
2. Extracting metadata (title, date, document type, etc.)
3. Document validation
4. Batch processing
5. Version tracking

The module is designed to be extensible for new document types and metadata fields.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import docx
import PyPDF2
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

from config import LEGAL_DOCUMENT_TYPES, PROCESSED_DATA_DIR, RAW_DATA_DIR, UI_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata model for legal documents."""

    # Basic information
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    doc_type: str = Field(..., description="Type of legal document")
    file_path: Path = Field(..., description="Path to the original document")

    # Content information
    num_pages: int = Field(..., description="Number of pages")
    word_count: int = Field(0, description="Total word count")

    # Processing metadata
    format: str = Field(..., description="Original file format")
    processed_at: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., description="Content hash for versioning")

    @validator("doc_type")
    def validate_doc_type(cls, v):
        """Ensure document type is one of the allowed types."""
        if v not in LEGAL_DOCUMENT_TYPES:
            raise ValueError(f"Document type must be one of: {LEGAL_DOCUMENT_TYPES}")
        return v


class DocumentLoader:
    """Main document loading and processing class."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the document loader.

        Args:
            data_dir: Optional custom data directory path
        """
        self.data_dir = data_dir or RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.allowed_extensions = set(UI_CONFIG["allowed_extensions"])

        # Mapping of file extensions to parser functions
        self.parsers = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".doc": self._parse_docx,  # Assuming .doc files can be read by python-docx
            ".txt": self._parse_txt,
        }

    def load_document(
        self, file_path: Union[str, Path]
    ) -> Tuple[str, DocumentMetadata]:
        """Load and parse a single document with metadata.

        Args:
            file_path: Path to the document

        Returns:
            Tuple of (document_text, metadata)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        # Validate file exists and format is supported
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if file_path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(
                f"Unsupported file format. Allowed formats: {self.allowed_extensions}"
            )

        # Parse document
        parser = self.parsers[file_path.suffix.lower()]
        text = parser(file_path)

        # Extract metadata
        metadata = self._extract_metadata(text, file_path)

        # Validate document
        self._validate_document(text, metadata)

        return text, metadata

    def load_documents(
        self, file_paths: List[Union[str, Path]], batch_size: Optional[int] = None
    ) -> List[Tuple[str, DocumentMetadata]]:
        """Load multiple documents in batch.

        Args:
            file_paths: List of paths to documents
            batch_size: Optional batch size for processing

        Returns:
            List of (document_text, metadata) tuples
        """
        batch_size = batch_size or len(file_paths)
        results = []

        # Process in batches with progress bar
        for i in tqdm(range(0, len(file_paths), batch_size), desc="Loading documents"):
            batch = file_paths[i : i + batch_size]
            for file_path in batch:
                try:
                    result = self.load_document(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")

        return results

    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF documents.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        text = []
        with open(file_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            for page in pdf.pages:
                text.append(page.extract_text())

        return "\n".join(text)

    def _parse_docx(self, file_path: Path) -> str:
        """Parse Word documents.

        Args:
            file_path: Path to Word file

        Returns:
            Extracted text content
        """
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _parse_txt(self, file_path: Path) -> str:
        """Parse plain text documents.

        Args:
            file_path: Path to text file

        Returns:
            File content
        """
        return file_path.read_text(encoding="utf-8")

    def _extract_metadata(self, text: str, file_path: Path) -> DocumentMetadata:
        """Extract metadata from document content.

        Args:
            text: Document text content
            file_path: Path to original file

        Returns:
            DocumentMetadata object
        """
        # Generate unique document ID
        doc_id = hashlib.md5(text.encode()).hexdigest()

        # Infer document type from path or content
        doc_type = self._infer_document_type(file_path, text)

        # Extract title from first non-empty line
        title = next(
            (line.strip() for line in text.split("\n") if line.strip()), file_path.stem
        )

        # Calculate version hash for content tracking
        version = hashlib.sha256(text.encode()).hexdigest()[:8]

        return DocumentMetadata(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            file_path=file_path,
            num_pages=text.count("\f") + 1,  # Form feed character count + 1
            word_count=len(text.split()),
            format=file_path.suffix.lower(),
            version=version,
        )

    def _infer_document_type(self, file_path: Path, text: str) -> str:
        """Infer document type from path and content.

        Args:
            file_path: Document path
            text: Document content

        Returns:
            Inferred document type
        """
        # First try to infer from path
        for doc_type in LEGAL_DOCUMENT_TYPES:
            if doc_type in str(file_path).lower():
                return doc_type

        # Then try to infer from content
        text_lower = text.lower()
        type_scores = {
            doc_type: text_lower.count(doc_type.replace("_", " "))
            for doc_type in LEGAL_DOCUMENT_TYPES
        }

        # Return type with highest score, default to first type if no matches
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _validate_document(self, text: str, metadata: DocumentMetadata) -> None:
        """Validate document content and metadata.

        Args:
            text: Document content
            metadata: Document metadata

        Raises:
            ValueError: If validation fails
        """
        # Check for minimum content
        if len(text.strip()) < 100:  # Arbitrary minimum length
            raise ValueError("Document content too short")

        # Check for maximum content
        if len(text) > 10_000_000:  # 10MB text limit
            raise ValueError("Document content too large")

        # Validate metadata completeness
        if not all([metadata.title, metadata.doc_type, metadata.doc_id]):
            raise ValueError("Incomplete metadata")

    def save_processed_document(self, text: str, metadata: DocumentMetadata) -> Path:
        """Save processed document and metadata.

        Args:
            text: Processed document text
            metadata: Document metadata

        Returns:
            Path to saved document
        """
        # Create processed document directory
        doc_dir = self.processed_dir / metadata.doc_type / metadata.doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Save text content
        text_path = doc_dir / "content.txt"
        text_path.write_text(text, encoding="utf-8")

        # Save metadata
        metadata_path = doc_dir / "metadata.json"
        metadata_path.write_text(metadata.json(indent=2), encoding="utf-8")

        return doc_dir


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()

    # Create a sample text file for testing
    sample_text = """
    Income Tax Act, 1961
    Section 80C: Deduction in respect of life insurance premia,
    deferred annuity, contributions to provident fund,
    subscription to certain equity shares or debentures, etc.

    (1) In computing the total income of an assessee,
    being an individual or a Hindu undivided family,
    there shall be deducted [...]
    """

    test_file = RAW_DATA_DIR / "income_tax_act" / "section_80c.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(sample_text)

    try:
        # Load single document
        text, metadata = loader.load_document(test_file)
        print(f"\nLoaded document: {metadata.title}")
        print(f"Type: {metadata.doc_type}")
        print(f"Word count: {metadata.word_count}")
        print(f"Version: {metadata.version}")

        # Save processed document
        processed_path = loader.save_processed_document(text, metadata)
        print(f"\nSaved processed document to: {processed_path}")

        # Batch loading example
        results = loader.load_documents([test_file])
        print(f"\nBatch loaded {len(results)} documents")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup test file
        test_file.unlink()
        test_file.parent.rmdir()
