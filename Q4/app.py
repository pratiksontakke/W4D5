"""
Main application entry point for the Intelligent Document Chunking System.
"""
from fastapi import FastAPI
from config import settings
import logging.config
import uvicorn

# Configure logging
logging.config.dictConfig(settings.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Intelligent Document Chunking System",
    description="API for intelligent document processing and retrieval",
    version="0.1.0",
)

@app.get("/")
async def root():
    """Root endpoint returning API status."""
    return {"status": "online", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting Intelligent Document Chunking System")
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True,
    ) 