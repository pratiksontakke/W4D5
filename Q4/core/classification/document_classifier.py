"""
Document classifier for the Intelligent Document Chunking System.
"""
from pathlib import Path
from typing import Dict, List, Optional
import logging
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """Classifies documents based on content and structure."""
    
    def __init__(self):
        """Initialize the document classifier."""
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(
            max_features=settings.CLASSIFIER_MAX_FEATURES,
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.document_types = settings.DOCUMENT_TYPES
        
    def classify(self, content: Dict[str, any]) -> Dict[str, any]:
        """
        Classify document content and identify its structure.
        
        Args:
            content: Dictionary containing document content and metadata
            
        Returns:
            Dictionary with classification results and structure info
        """
        try:
            # Extract features
            features = self._extract_features(content)
            
            # Classify document type
            doc_type = self._classify_type(features)
            
            # Analyze structure
            structure = self._analyze_structure(content, doc_type)
            
            # Determine chunking strategy
            chunking_strategy = self._determine_chunking_strategy(doc_type, structure)
            
            return {
                'document_type': doc_type,
                'structure': structure,
                'chunking_strategy': chunking_strategy,
                'confidence': self._calculate_confidence(features)
            }
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            raise
            
    def _extract_features(self, content: Dict[str, any]) -> Dict[str, any]:
        """
        Extract relevant features from document content.
        
        Args:
            content: Document content dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Text statistics
        text = self._get_full_text(content)
        doc = self.nlp(text[:1000000])  # Limit for very large docs
        
        features['avg_sentence_length'] = np.mean([len(sent) for sent in doc.sents])
        features['noun_phrases'] = len(list(doc.noun_chunks))
        features['named_entities'] = len(doc.ents)
        
        # Structure features
        features['has_sections'] = self._detect_sections(content)
        features['code_blocks'] = self._count_code_blocks(content)
        features['table_count'] = self._count_tables(content)
        
        # Metadata features
        if 'metadata' in content:
            features.update(self._extract_metadata_features(content['metadata']))
            
        return features
        
    def _classify_type(self, features: Dict[str, any]) -> str:
        """
        Determine document type based on extracted features.
        
        Args:
            features: Dictionary of document features
            
        Returns:
            Predicted document type
        """
        # Rule-based classification (can be replaced with ML model)
        if features.get('code_blocks', 0) > 5:
            return 'technical_documentation'
        elif features.get('table_count', 0) > 3:
            return 'data_report'
        elif features.get('has_sections', False):
            return 'article'
        else:
            return 'general'
            
    def _analyze_structure(self, content: Dict[str, any], doc_type: str) -> Dict[str, any]:
        """
        Analyze document structure based on content and type.
        
        Args:
            content: Document content
            doc_type: Classified document type
            
        Returns:
            Dictionary describing document structure
        """
        structure = {
            'hierarchical': self._is_hierarchical(content),
            'sections': self._get_section_info(content),
            'special_elements': self._detect_special_elements(content),
            'formatting': self._analyze_formatting(content)
        }
        
        return structure
        
    def _determine_chunking_strategy(self, doc_type: str, structure: Dict[str, any]) -> Dict[str, any]:
        """
        Determine optimal chunking strategy based on document type and structure.
        
        Args:
            doc_type: Document type
            structure: Document structure information
            
        Returns:
            Dictionary describing chunking strategy
        """
        strategy = {
            'method': 'semantic',  # Default
            'chunk_size': settings.DEFAULT_CHUNK_SIZE,
            'overlap': settings.DEFAULT_CHUNK_OVERLAP
        }
        
        if doc_type == 'technical_documentation':
            strategy['method'] = 'code_aware'
            strategy['chunk_size'] = settings.CODE_CHUNK_SIZE
        elif structure['hierarchical']:
            strategy['method'] = 'hierarchical'
            strategy['section_based'] = True
        
        return strategy
        
    def _calculate_confidence(self, features: Dict[str, any]) -> float:
        """Calculate confidence score for classification."""
        # Simplified confidence calculation
        return 0.85  # Placeholder - implement actual confidence metric
        
    def _get_full_text(self, content: Dict[str, any]) -> str:
        """Extract full text from content dictionary."""
        if 'pages' in content:
            return ' '.join(page['content'] for page in content['pages'])
        return content.get('content', '')
        
    def _detect_sections(self, content: Dict[str, any]) -> bool:
        """Detect if document has clear section structure."""
        text = self._get_full_text(content)
        # Simple heuristic - look for common section markers
        section_markers = ['Introduction', 'Background', 'Methods', 'Results', 'Discussion', 'Conclusion']
        return any(marker in text for marker in section_markers)
        
    def _count_code_blocks(self, content: Dict[str, any]) -> int:
        """Count number of code blocks in document."""
        text = self._get_full_text(content)
        # Simple heuristic - count code fence markers
        return text.count('```')
        
    def _count_tables(self, content: Dict[str, any]) -> int:
        """Count number of tables in document."""
        text = self._get_full_text(content)
        # Simple heuristic - count pipe characters and line breaks
        lines = text.split('\n')
        table_lines = sum(1 for line in lines if line.count('|') > 2)
        return table_lines // 4  # Assume average table is 4 lines
        
    def _extract_metadata_features(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """Extract features from document metadata."""
        features = {}
        features['has_title'] = bool(metadata.get('title'))
        features['has_author'] = bool(metadata.get('author'))
        features['has_keywords'] = bool(metadata.get('keywords'))
        return features
        
    def _is_hierarchical(self, content: Dict[str, any]) -> bool:
        """Determine if document has hierarchical structure."""
        text = self._get_full_text(content)
        # Look for header patterns (##, ###, etc.)
        header_count = sum(1 for line in text.split('\n') if line.strip().startswith('#'))
        return header_count > 3
        
    def _get_section_info(self, content: Dict[str, any]) -> Dict[str, any]:
        """Extract information about document sections."""
        text = self._get_full_text(content)
        sections = []
        current_level = 0
        
        for line in text.split('\n'):
            if line.strip().startswith('#'):
                level = len(line.split()[0])
                title = line.strip('#').strip()
                sections.append({
                    'level': level,
                    'title': title
                })
                current_level = max(current_level, level)
                
        return {
            'count': len(sections),
            'max_depth': current_level,
            'sections': sections[:10]  # Return first 10 sections
        }
        
    def _detect_special_elements(self, content: Dict[str, any]) -> Dict[str, bool]:
        """Detect presence of special document elements."""
        text = self._get_full_text(content)
        return {
            'has_code': '```' in text,
            'has_tables': '|' in text,
            'has_lists': any(line.strip().startswith(('- ', '* ', '1. ')) for line in text.split('\n')),
            'has_images': '![' in text or '.png' in text.lower() or '.jpg' in text.lower()
        }
        
    def _analyze_formatting(self, content: Dict[str, any]) -> Dict[str, int]:
        """Analyze document formatting patterns."""
        text = self._get_full_text(content)
        return {
            'bold_count': text.count('**'),
            'italic_count': text.count('*') - 2 * text.count('**'),
            'link_count': text.count(']('),
            'quote_count': sum(1 for line in text.split('\n') if line.strip().startswith('>'))
        }

if __name__ == "__main__":
    # Example usage
    classifier = DocumentClassifier()
    
    # Test with sample content
    sample_content = {
        'content': '# Introduction\n\nThis is a technical document.\n\n```python\ndef example():\n    pass\n```\n',
        'metadata': {
            'title': 'Technical Guide',
            'author': 'John Doe'
        }
    }
    
    try:
        result = classifier.classify(sample_content)
        print("Classification result:", result)
    except Exception as e:
        print(f"Error classifying document: {e}") 