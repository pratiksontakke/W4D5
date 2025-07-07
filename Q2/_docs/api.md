# API Documentation

## Overview

The Indian Legal Document Search System provides a RESTful API for document search, upload, and retrieval operations. All endpoints return JSON responses and support standard HTTP methods.

## Base URL

```
https://api.legal-search.com/v1
```

## Authentication

API requests require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Rate Limiting

- 100 requests per minute for search operations
- 50 requests per minute for document uploads
- 200 requests per minute for retrieval operations

## Endpoints

### Search

#### POST /search

Search for documents using specified similarity methods.

**Request Body:**
```json
{
    "query": "string",
    "methods": ["cosine", "euclidean", "mmr", "hybrid"],
    "top_k": 5,
    "filters": {
        "document_type": ["income_tax", "gst"],
        "date_range": {
            "start": "2020-01-01",
            "end": "2023-12-31"
        }
    }
}
```

**Response:**
```json
{
    "results": {
        "cosine": [
            {
                "document_id": "string",
                "title": "string",
                "excerpt": "string",
                "score": 0.95,
                "metadata": {}
            }
        ],
        "euclidean": [...],
        "mmr": [...],
        "hybrid": [...]
    },
    "metrics": {
        "precision": 0.8,
        "recall": 0.75,
        "diversity": 0.85
    }
}
```

### Document Upload

#### POST /documents/upload

Upload legal documents for indexing.

**Request:**
- Content-Type: multipart/form-data
- Max file size: 10MB
- Supported formats: PDF, DOC, DOCX, TXT

**Response:**
```json
{
    "document_id": "string",
    "status": "processing",
    "metadata": {
        "filename": "string",
        "size": 1024,
        "type": "pdf"
    }
}
```

### Metrics

#### GET /metrics

Retrieve system performance metrics.

**Response:**
```json
{
    "search_metrics": {
        "average_precision": 0.85,
        "average_recall": 0.75,
        "average_response_time": 0.2
    },
    "system_metrics": {
        "uptime": 99.9,
        "total_documents": 10000,
        "index_size": "2.5GB"
    }
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

```json
{
    "error": {
        "code": "string",
        "message": "string",
        "details": {}
    }
}
```

Common error codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Pagination

For endpoints returning multiple items, use:
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 10, max: 100)

Example:
```
GET /documents?page=2&per_page=20
```

## SDK Examples

### Python

```python
from legal_search import Client

client = Client("your_api_key")

# Search documents
results = client.search(
    query="income tax deduction",
    methods=["cosine", "hybrid"],
    top_k=5
)

# Upload document
doc_id = client.upload_document(
    file_path="document.pdf",
    metadata={"type": "income_tax"}
)
```

### JavaScript

```javascript
const LegalSearch = require('legal-search');

const client = new LegalSearch('your_api_key');

// Search documents
const results = await client.search({
    query: 'income tax deduction',
    methods: ['cosine', 'hybrid'],
    topK: 5
});

// Upload document
const docId = await client.uploadDocument({
    file: documentFile,
    metadata: { type: 'income_tax' }
});
```

## Webhooks

Subscribe to events using webhooks:

```json
POST /webhooks
{
    "url": "https://your-server.com/webhook",
    "events": ["document.indexed", "search.completed"],
    "secret": "your_webhook_secret"
}
```

## Best Practices

1. Use connection pooling
2. Implement retry logic with exponential backoff
3. Cache frequently accessed results
4. Handle rate limits gracefully
5. Validate input before sending requests

## Support

For API support:
- Email: api-support@legal-search.com
- Documentation: https://docs.legal-search.com
- Status: https://status.legal-search.com
