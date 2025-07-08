"""
Chunking strategies for the Intelligent Document Chunking System.
"""
from typing import Dict, List, Optional
import logging
import re
from abc import ABC, abstractmethod

from config import settings

logger = logging.getLogger(__name__)

class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""
    
    def __init__(self, config: Dict[str, any] = None):
        """
        Initialize chunking strategy.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', settings.DEFAULT_CHUNK_SIZE)
        self.overlap = self.config.get('overlap', settings.DEFAULT_CHUNK_OVERLAP)
        
    @abstractmethod
    def chunk(self, content: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split document content into chunks.
        
        Args:
            content: Document content dictionary
            
        Returns:
            List of chunk dictionaries
        """
        pass
        
    def _create_chunk(self, text: str, metadata: Dict[str, any] = None) -> Dict[str, any]:
        """Create a chunk dictionary with metadata."""
        return {
            'content': text.strip(),
            'metadata': metadata or {},
            'length': len(text.strip())
        }

class SemanticChunking(ChunkingStrategy):
    """Chunks documents based on semantic boundaries."""
    
    def chunk(self, content: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split content at semantic boundaries (sentences, paragraphs).
        
        Args:
            content: Document content dictionary
            
        Returns:
            List of semantically coherent chunks
        """
        text = self._get_text(content)
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # Split paragraphs into sentences
            sentences = self._split_into_sentences(para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if adding this sentence would exceed chunk size
                if current_length + len(sentence) > self.chunk_size and current_chunk:
                    # Create chunk from accumulated sentences
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text))
                    
                    # Start new chunk, including overlap
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        
        # Add final chunk if there's content
        if current_chunk:
            chunks.append(self._create_chunk(' '.join(current_chunk)))
            
        return chunks
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules."""
        # Basic sentence splitting - can be improved with NLP
        sentence_endings = r'[.!?][\s]{1,2}(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _get_text(self, content: Dict[str, any]) -> str:
        """Extract text from content dictionary."""
        if 'pages' in content:
            return ' '.join(page['content'] for page in content['pages'])
        return content.get('content', '')

class CodeAwareChunking(ChunkingStrategy):
    """Chunks documents while preserving code block integrity."""
    
    def chunk(self, content: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split content while keeping code blocks intact.
        
        Args:
            content: Document content dictionary
            
        Returns:
            List of chunks with preserved code blocks
        """
        text = self._get_text(content)
        chunks = []
        
        # Split content into code and non-code sections
        sections = self._split_code_sections(text)
        
        current_chunk = []
        current_length = 0
        
        for section in sections:
            is_code = section.startswith('```')
            
            if is_code:
                # If current chunk is getting large, save it before code block
                if current_length > self.chunk_size / 2 and current_chunk:
                    chunks.append(self._create_chunk('\n'.join(current_chunk)))
                    current_chunk = []
                    current_length = 0
                
                # Add code block as its own chunk
                chunks.append(self._create_chunk(section, {'type': 'code'}))
            else:
                # Process non-code section
                sentences = self._split_into_sentences(section)
                
                for sentence in sentences:
                    if current_length + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(self._create_chunk('\n'.join(current_chunk)))
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk('\n'.join(current_chunk)))
            
        return chunks
        
    def _split_code_sections(self, text: str) -> List[str]:
        """Split text into code and non-code sections."""
        sections = []
        current_section = []
        in_code_block = False
        
        for line in text.split('\n'):
            if line.strip().startswith('```'):
                if in_code_block:
                    current_section.append(line)
                    sections.append('\n'.join(current_section))
                    current_section = []
                else:
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    current_section.append(line)
                in_code_block = not in_code_block
            else:
                current_section.append(line)
                
        if current_section:
            sections.append('\n'.join(current_section))
            
        return sections
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving line breaks."""
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            if line.strip():
                sentences.extend(re.split(r'(?<=[.!?])\s+', line))
            else:
                sentences.append(line)
                
        return sentences
        
    def _get_text(self, content: Dict[str, any]) -> str:
        """Extract text from content dictionary."""
        if 'pages' in content:
            return '\n'.join(page['content'] for page in content['pages'])
        return content.get('content', '')

class HierarchicalChunking(ChunkingStrategy):
    """Chunks documents based on their hierarchical structure."""
    
    def chunk(self, content: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split content based on document hierarchy.
        
        Args:
            content: Document content dictionary
            
        Returns:
            List of hierarchical chunks
        """
        text = self._get_text(content)
        chunks = []
        
        # Parse document structure
        sections = self._parse_sections(text)
        
        # Process each section
        for section in sections:
            # Add section header as its own chunk
            if section['title']:
                chunks.append(self._create_chunk(
                    section['title'],
                    {'type': 'header', 'level': section['level']}
                ))
            
            # Process section content
            content_chunks = self._chunk_section_content(
                section['content'],
                section['level']
            )
            chunks.extend(content_chunks)
            
        return chunks
        
    def _parse_sections(self, text: str) -> List[Dict[str, any]]:
        """Parse document into hierarchical sections."""
        sections = []
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            if line.strip().startswith('#'):
                # Save previous section if exists
                if current_section is not None:
                    current_section['content'] = '\n'.join(current_content)
                    sections.append(current_section)
                
                # Start new section
                level = len(line.split()[0])  # Count #'s
                title = line.strip('#').strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'content': ''
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section is not None:
            current_section['content'] = '\n'.join(current_content)
            sections.append(current_section)
        elif current_content:  # Content without any sections
            sections.append({
                'level': 0,
                'title': '',
                'content': '\n'.join(current_content)
            })
            
        return sections
        
    def _chunk_section_content(self, content: str, level: int) -> List[Dict[str, any]]:
        """Chunk section content while preserving context."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = self._split_into_sentences(content)
        
        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk with section context
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text,
                    {'type': 'content', 'section_level': level}
                ))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text,
                {'type': 'content', 'section_level': level}
            ))
            
        return chunks
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving structure."""
        # First split by line breaks to preserve formatting
        lines = text.split('\n')
        sentences = []
        
        for line in lines:
            if line.strip():
                # Split line into sentences
                line_sentences = re.split(r'(?<=[.!?])\s+', line)
                sentences.extend(line_sentences)
            else:
                sentences.append(line)
                
        return [s.strip() for s in sentences if s.strip()]
        
    def _get_text(self, content: Dict[str, any]) -> str:
        """Extract text from content dictionary."""
        if 'pages' in content:
            return '\n'.join(page['content'] for page in content['pages'])
        return content.get('content', '')

def get_chunking_strategy(strategy_type: str, config: Dict[str, any] = None) -> ChunkingStrategy:
    """
    Factory function to get appropriate chunking strategy.
    
    Args:
        strategy_type: Type of chunking strategy
        config: Optional configuration dictionary
        
    Returns:
        ChunkingStrategy instance
        
    Raises:
        ValueError: If strategy type is unknown
    """
    strategies = {
        'semantic': SemanticChunking,
        'code_aware': CodeAwareChunking,
        'hierarchical': HierarchicalChunking
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy_type}")
        
    return strategies[strategy_type](config)

if __name__ == "__main__":
    # Example usage
    sample_content = {
        'content': """# Introduction
        
        This is a sample document with multiple sections.
        It contains some code examples.
        
        ```python
        def example():
            return "Hello World"
        ```
        
        ## Section 1
        
        This is the first section with multiple sentences.
        It demonstrates hierarchical structure.
        
        ## Section 2
        
        Another section with code:
        
        ```python
        def another_example():
            pass
        ```
        """
    }
    
    # Test different strategies
    strategies = ['semantic', 'code_aware', 'hierarchical']
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name} strategy:")
        strategy = get_chunking_strategy(strategy_name)
        chunks = strategy.chunk(sample_content)
        print(f"Generated {len(chunks)} chunks") 