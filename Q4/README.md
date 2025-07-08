# Intelligent Document Chunking System

An adaptive document chunking system that automatically detects document types and applies appropriate chunking strategies to improve knowledge retrieval for enterprise content.

## Features

- **Document Classification**: Auto-detects content types and structure patterns
- **Adaptive Chunking**: Applies document-specific strategies for optimal context preservation
- **Multiple Strategies**:
  - Semantic chunking for natural language documents
  - Code-aware chunking for technical documentation
  - Hierarchical chunking for structured documents
- **REST API**: FastAPI-based service for document processing
- **CLI Interface**: Command-line tool for batch processing
- **Extensible Architecture**: Easy to add new document types and chunking strategies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intelligent-chunker.git
cd intelligent-chunker
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### CLI Interface

Process a single document:
```bash
python cli.py input.pdf -o output.json
```

Process a directory recursively:
```bash
python cli.py input_dir/ -r -o output_dir/
```

### REST API

Start the API server:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /process`: Process a document
- `GET /supported-types`: List supported document types
- `GET /health`: Health check

### Python API

```python
from core.processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a document
result = processor.process_document("document.pdf")

# Access chunks
for chunk in result['chunks']:
    print(chunk['content'])
```

## Configuration

Key settings in `config/settings.py`:
- `DEFAULT_CHUNK_SIZE`: Default size for document chunks
- `DEFAULT_CHUNK_OVERLAP`: Overlap between consecutive chunks
- `CODE_CHUNK_SIZE`: Chunk size for code blocks
- `CLASSIFIER_MAX_FEATURES`: Maximum features for document classification
- `ALLOWED_ORIGINS`: CORS settings for API

## Architecture

The system follows a modular architecture:

1. **Document Ingestion**:
   - Converts various document formats to internal representation
   - Currently supports PDF files (extensible)

2. **Document Classification**:
   - Analyzes document content and structure
   - Determines optimal chunking strategy
   - Uses ML-based and rule-based approaches

3. **Chunking Strategies**:
   - Semantic: Natural language-aware chunking
   - Code-aware: Preserves code block integrity
   - Hierarchical: Section-based chunking

4. **API Layer**:
   - REST API for document processing
   - Swagger/OpenAPI documentation
   - CORS support

## Development

### Project Structure

```
intelligent-chunker/
├── api/
│   └── app.py           # FastAPI application
├── core/
│   ├── ingestion/       # Document converters
│   ├── classification/  # Document classifier
│   ├── chunking/        # Chunking strategies
│   └── processor.py     # Main processor
├── config/
│   └── settings.py      # Configuration
├── tests/               # Test suite
├── cli.py              # CLI interface
└── requirements.txt     # Dependencies
```

### Testing

Run tests:
```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Future Enhancements

- Support for more document types (Word, Markdown, etc.)
- Advanced ML-based document classification
- Additional chunking strategies
- Performance optimizations
- Integration with vector stores
- Batch processing improvements
- Web interface for document processing
