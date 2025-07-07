# Indian Legal Document Search System

A sophisticated search system for Indian legal documents that compares 4 different similarity methods to find the most effective approach for legal document retrieval.

## Features

- Multiple similarity methods (Cosine, Euclidean, MMR, Hybrid)
- Document preprocessing and legal entity recognition
- Advanced embedding techniques (SBERT, TF-IDF)
- Interactive web UI with side-by-side comparison
- Comprehensive evaluation framework
- Support for multiple document formats (PDF, Word, Text)

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd legal-document-search

# Create and activate conda environment
conda create --prefix ./venv python=3.10
conda activate ./venv

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings
nano .env
```

### 3. Data Setup

```bash
# Place your legal documents in the appropriate directories:
data/raw/income_tax_act/
data/raw/gst_act/
data/raw/court_judgments/
data/raw/property_law/
```

### 4. Build Search Index

```bash
# Run the indexing pipeline
python pipelines/build_index.py
```

### 5. Run the Application

```bash
# Start the web server
python app.py
```

Visit `http://localhost:8000` to access the web interface.

## Project Structure

```
project_root/
├── app.py                  # Main entry point
├── config.py              # Configuration settings
├── core/                  # Core functionality
├── pipelines/             # Data processing pipelines
├── ui/                    # Web interface
├── artifacts/             # Generated artifacts
├── data/                  # Document storage
└── _docs/                 # Documentation
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/core/
pytest tests/pipelines/
pytest tests/ui/
```

### Code Quality

The project uses pre-commit hooks for code quality:
- black for code formatting
- isort for import sorting
- flake8 for style guide enforcement

### Documentation

Additional documentation can be found in the `_docs/` directory:
- Architecture Overview
- Technical Deep-Dive
- API Documentation
- Evaluation Results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

[Your License Here]
