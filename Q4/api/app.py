"""
FastAPI application for the Intelligent Document Chunking System.
"""
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.processor import DocumentProcessor
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Document Chunking API",
    description="API for processing and chunking enterprise documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
processor = DocumentProcessor()

class ProcessingResponse(BaseModel):
    """Response model for document processing."""
    document_info: Dict
    classification: Dict
    chunks: List[Dict]
    metadata: Dict

@app.post("/process", response_model=ProcessingResponse)
async def process_document(file: UploadFile = File(...)) -> Dict:
    """
    Process and chunk a document.
    
    Args:
        file: Uploaded document file
        
    Returns:
        Dictionary containing processing results
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Copy uploaded file to temporary location
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)
            
        try:
            # Validate document
            if not processor.validate_document(temp_path):
                raise ValueError("Invalid or unsupported document")
                
            # Process document
            result = processor.process_document(temp_path)
            
            return result
            
        finally:
            # Clean up temporary file
            temp_path.unlink()
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/supported-types")
async def get_supported_types() -> Dict[str, List[str]]:
    """Get list of supported document types."""
    return {
        "supported_types": processor.get_supported_types()
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 