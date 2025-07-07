# Indian Legal Document Search System

A powerful search system that compares different similarity methods for legal document retrieval.

## Overview

The Indian Legal Document Search System is designed to help legal professionals quickly find relevant documents using multiple similarity comparison methods. It supports searching through Income Tax Acts, GST provisions, court judgments, and property law documents.

## Key Features

- Multiple similarity methods (Cosine, Euclidean, MMR, Hybrid)
- Document upload and parsing (PDF, Word, Text)
- Real-time search with side-by-side comparison
- Performance metrics dashboard
- Legal entity recognition
- Concurrent user support

## Documentation Index

1. [Installation Guide](installation.md)
2. [User Guide](user_guide.md)
3. [API Documentation](api.md)
4. [Architecture](architecture.md)
5. [Maintenance Guide](maintenance.md)
6. [Contributing Guidelines](contributing.md)
7. [Deployment Guide](deployment.md)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-org/legal-search.git
cd legal-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
cp env.template .env
# Edit .env with your settings
```

4. Run the application:
```bash
python app.py
```

## Architecture Overview

![System Architecture](images/architecture.png)

The system follows a modular architecture with these main components:

- Document Processing Pipeline
- Embedding Generation
- Similarity Computation
- Search & Retrieval
- Web Interface
- Evaluation Framework

## License

MIT License - see [LICENSE](LICENSE) for details
