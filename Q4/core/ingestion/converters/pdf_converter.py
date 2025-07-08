"""
PDF document converter for the Intelligent Document Chunking System.
"""
from pathlib import Path
from typing import Dict, List, Optional
import logging
import PyPDF2
from PyPDF2.errors import PdfReadError

from config import settings

logger = logging.getLogger(__name__)

class PDFConverter:
    """Converts PDF documents to a standardized format for processing."""
    
    def __init__(self):
        """Initialize the PDF converter."""
        self.supported_versions = settings.SUPPORTED_PDF_VERSIONS

    def convert(self, file_path: Path) -> Dict[str, any]:
        """
        Convert a PDF file to structured text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PdfReadError: If PDF is corrupted or unsupported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'keywords': reader.metadata.get('/Keywords', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                    'producer': reader.metadata.get('/Producer', ''),
                    'page_count': len(reader.pages)
                }
                
                # Extract text from each page
                pages = []
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        text = page.extract_text()
                        pages.append({
                            'page_number': page_num,
                            'content': text,
                            'layout': self._extract_layout(page)
                        })
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        pages.append({
                            'page_number': page_num,
                            'content': '',
                            'layout': {}
                        })
                
                return {
                    'metadata': metadata,
                    'pages': pages,
                    'source_path': str(file_path),
                    'document_type': 'pdf'
                }
                
        except PdfReadError as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing PDF {file_path}: {e}")
            raise

    def _extract_layout(self, page: PyPDF2.PageObject) -> Dict[str, any]:
        """
        Extract layout information from a PDF page.
        
        Args:
            page: PyPDF2 page object
            
        Returns:
            Dictionary containing layout information
        """
        try:
            mediabox = page.mediabox
            return {
                'width': float(mediabox.width),
                'height': float(mediabox.height),
                'rotation': page.get('/Rotate', 0),
                'has_images': bool(page.images)
            }
        except Exception as e:
            logger.warning(f"Error extracting layout information: {e}")
            return {}

    def validate(self, file_path: Path) -> bool:
        """
        Validate if the file is a valid PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'rb') as file:
                PyPDF2.PdfReader(file)
                return True
        except:
            return False

if __name__ == "__main__":
    # Example usage
    converter = PDFConverter()
    
    # Test with a sample PDF
    sample_path = Path("tests/test_data/sample.pdf")
    if sample_path.exists():
        try:
            result = converter.convert(sample_path)
            print(f"Successfully converted PDF with {len(result['pages'])} pages")
            print(f"Title: {result['metadata']['title']}")
        except Exception as e:
            print(f"Error converting PDF: {e}")
    else:
        print(f"Sample PDF not found at {sample_path}") 