"""
Main document processor for the Intelligent Document Chunking System.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

from core.ingestion.converters.pdf_converter import PDFConverter
from core.classification.document_classifier import DocumentClassifier
from core.chunking.strategies import get_chunking_strategy
from config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main processor that orchestrates document processing pipeline."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.pdf_converter = PDFConverter()
        self.classifier = DocumentClassifier()
        
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing processing results and chunks
            
        Raises:
            FileNotFoundError: If document doesn't exist
            ValueError: If document type is unsupported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        try:
            # Start processing
            start_time = datetime.now()
            logger.info(f"Starting document processing: {file_path}")
            
            # Convert document
            content = self._convert_document(file_path)
            
            # Classify document
            classification = self.classifier.classify(content)
            logger.info(f"Document classified as: {classification['document_type']}")
            
            # Get appropriate chunking strategy
            strategy = get_chunking_strategy(
                classification['chunking_strategy']['method'],
                classification['chunking_strategy']
            )
            
            # Generate chunks
            chunks = strategy.chunk(content)
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'document_info': {
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'type': classification['document_type']
                },
                'classification': classification,
                'chunks': chunks,
                'metadata': {
                    'processing_time': processing_time,
                    'chunk_count': len(chunks),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Add document metadata if available
            if 'metadata' in content:
                result['document_info']['metadata'] = content['metadata']
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
            
    def _convert_document(self, file_path: Path) -> Dict[str, any]:
        """
        Convert document to internal format based on file type.
        
        Args:
            file_path: Path to document
            
        Returns:
            Dictionary containing document content
            
        Raises:
            ValueError: If file type is unsupported
        """
        file_type = file_path.suffix.lower()
        
        if file_type == '.pdf':
            return self.pdf_converter.convert(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    def validate_document(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if document can be processed.
        
        Args:
            file_path: Path to document
            
        Returns:
            True if document is valid and supported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False
            
        file_type = file_path.suffix.lower()
        
        if file_type == '.pdf':
            return self.pdf_converter.validate(file_path)
            
        return False
        
    def get_supported_types(self) -> List[str]:
        """Get list of supported document types."""
        return ['.pdf']  # Add more as support is added

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Test with sample document
    sample_path = Path("tests/test_data/sample.pdf")
    if sample_path.exists():
        try:
            result = processor.process_document(sample_path)
            print(f"Successfully processed document:")
            print(f"- Type: {result['document_info']['type']}")
            print(f"- Chunks: {result['metadata']['chunk_count']}")
            print(f"- Processing time: {result['metadata']['processing_time']:.2f}s")
        except Exception as e:
            print(f"Error processing document: {e}")
    else:
        print(f"Sample document not found at {sample_path}") 