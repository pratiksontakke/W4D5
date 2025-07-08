# Fine-Tuned Embeddings for Sales Conversion Prediction

A production-ready system that fine-tunes embeddings specifically for sales conversations to improve conversion prediction accuracy and enable better customer prioritization.

## Project Overview

This project addresses the challenge of accurately predicting customer conversion likelihood from call transcripts. It replaces subjective human judgment with a data-driven approach using domain-specific fine-tuned embeddings.

### Key Features

- Domain-specific embedding fine-tuning for sales conversations
- Contrastive learning to distinguish conversion patterns
- LangChain-based prediction workflow
- Comprehensive evaluation pipeline

## Project Structure

```
/project/
├── api/              # API endpoints and interfaces
├── config/           # Configuration management
├── core/             # Core ML and processing logic
│   ├── embeddings/   # Embedding models and processors
│   ├── langchain/    # LangChain components
│   └── models/       # Model definitions
├── data/             # Data management
│   ├── ingestion/    # Data ingestion pipelines
│   └── processing/   # Data processing utilities
├── monitoring/       # Monitoring and logging
├── pipelines/        # Training and evaluation pipelines
├── tests/
│   ├── integration/  # Integration tests
│   └── unit/        # Unit tests
└── utils/            # Utility functions
```

## Getting Started

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
pytest tests/
```

## Development

- Follow the architectural strategy in `_docs/2_STRATEGY.md`
- Run tests before committing changes
- Update documentation as needed

## License

MIT License - See LICENSE file for details
