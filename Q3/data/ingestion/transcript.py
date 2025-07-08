"""
Transcript Processing Module

This module handles the processing of sales call transcripts, including:
- Async processing of transcript files
- Text extraction and cleaning
- Format validation
- Conversion to standardized format for embedding generation
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import aiofiles
import pandas as pd
from pydantic import BaseModel, Field, validator

from config.config import DATA_DIR

# Define transcript data models
class Speaker(BaseModel):
    """Speaker information in the transcript."""
    id: str
    role: str = Field(..., regex='^(sales|customer|other)$')
    name: Optional[str] = None

class TranscriptSegment(BaseModel):
    """Individual segment of the transcript."""
    speaker: Speaker
    text: str
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)

    @validator('text')
    def text_not_empty(cls, v):
        """Validate that text is not empty and has meaningful content."""
        if not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v.strip()) < 3:  # Arbitrary minimum length for meaningful content
            raise ValueError('Text too short to be meaningful')
        return v.strip()

class Transcript(BaseModel):
    """Complete transcript with metadata."""
    id: str
    date: datetime
    duration: int  # in seconds
    segments: List[TranscriptSegment]
    metadata: Dict = Field(default_factory=dict)
    conversion_outcome: Optional[bool] = None

@dataclass
class TranscriptProcessor:
    """Processes sales call transcripts for embedding generation."""
    
    def __init__(self, input_dir: Optional[Path] = None):
        """Initialize the transcript processor.
        
        Args:
            input_dir: Directory containing transcript files. Defaults to DATA_DIR/raw
        """
        self.input_dir = input_dir or Path(DATA_DIR) / 'raw'
        self.input_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_file(self, file_path: Union[str, Path]) -> Transcript:
        """Process a single transcript file asynchronously.
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            Processed Transcript object
        """
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
            
        # Parse the file content (assuming JSON format)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
            
        # Convert to Transcript model (validates data)
        transcript = Transcript(**data)
        
        return transcript
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize transcript text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s.,?!]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Basic text normalization
        text = text.lower().strip()
        
        return text
    
    def extract_features(self, transcript: Transcript) -> Dict:
        """Extract relevant features from transcript for embedding generation.
        
        Args:
            transcript: Processed transcript
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'id': transcript.id,
            'date': transcript.date,
            'duration': transcript.duration,
            'num_segments': len(transcript.segments),
            'avg_confidence': sum(s.confidence for s in transcript.segments) / len(transcript.segments),
            'sales_turns': sum(1 for s in transcript.segments if s.speaker.role == 'sales'),
            'customer_turns': sum(1 for s in transcript.segments if s.speaker.role == 'customer'),
            'conversion': transcript.conversion_outcome
        }
        
        return features
    
    async def process_directory(self) -> pd.DataFrame:
        """Process all transcript files in the input directory.
        
        Returns:
            DataFrame containing processed transcripts and features
        """
        transcript_files = list(self.input_dir.glob('*.json'))
        
        # Process files concurrently
        tasks = [self.process_file(f) for f in transcript_files]
        transcripts = await asyncio.gather(*tasks)
        
        # Extract features from all transcripts
        features = [self.extract_features(t) for t in transcripts]
        
        return pd.DataFrame(features)

if __name__ == "__main__":
    # Example usage
    async def main():
        processor = TranscriptProcessor()
        
        # Create a sample transcript for testing
        sample_data = {
            "id": "call_123",
            "date": datetime.now(),
            "duration": 300,
            "segments": [
                {
                    "speaker": {"id": "S1", "role": "sales", "name": "John"},
                    "text": "Hello, how can I help you today?",
                    "timestamp": datetime.now(),
                    "confidence": 0.95
                },
                {
                    "speaker": {"id": "C1", "role": "customer"},
                    "text": "I'm interested in your product.",
                    "timestamp": datetime.now(),
                    "confidence": 0.92
                }
            ],
            "conversion_outcome": True
        }
        
        # Save sample transcript
        sample_file = processor.input_dir / "sample.json"
        with open(sample_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            sample_data["date"] = sample_data["date"].isoformat()
            for segment in sample_data["segments"]:
                segment["timestamp"] = segment["timestamp"].isoformat()
            json.dump(sample_data, f, indent=2)
        
        # Process the sample file
        try:
            df = await processor.process_directory()
            print("\nProcessed Transcripts:")
            print(df)
        except Exception as e:
            print(f"Error processing transcripts: {e}")
        finally:
            # Cleanup sample file
            sample_file.unlink()

    # Run the example
    asyncio.run(main()) 