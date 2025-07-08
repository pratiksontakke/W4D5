"""
Unit tests for the transcript processing module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from data.ingestion.transcript import (
    Speaker,
    TranscriptSegment,
    Transcript,
    TranscriptProcessor
)

@pytest.fixture
def sample_transcript_data():
    """Create sample transcript data for testing."""
    return {
        "id": "test_123",
        "date": datetime.now().isoformat(),
        "duration": 300,
        "segments": [
            {
                "speaker": {"id": "S1", "role": "sales", "name": "John"},
                "text": "Hello, how can I help you today?",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.95
            },
            {
                "speaker": {"id": "C1", "role": "customer"},
                "text": "I'm interested in your product.",
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.92
            }
        ],
        "conversion_outcome": True
    }

@pytest.fixture
def temp_transcript_file(tmp_path, sample_transcript_data):
    """Create a temporary transcript file for testing."""
    file_path = tmp_path / "test_transcript.json"
    with open(file_path, 'w') as f:
        json.dump(sample_transcript_data, f)
    return file_path

def test_speaker_validation():
    """Test speaker role validation."""
    # Valid roles
    Speaker(id="S1", role="sales")
    Speaker(id="C1", role="customer")
    Speaker(id="O1", role="other")
    
    # Invalid role
    with pytest.raises(ValueError):
        Speaker(id="X1", role="invalid")

def test_transcript_segment_validation():
    """Test transcript segment validation."""
    # Valid segment
    TranscriptSegment(
        speaker={"id": "S1", "role": "sales"},
        text="Valid text",
        timestamp=datetime.now(),
        confidence=0.9
    )
    
    # Empty text
    with pytest.raises(ValueError):
        TranscriptSegment(
            speaker={"id": "S1", "role": "sales"},
            text="",
            timestamp=datetime.now(),
            confidence=0.9
        )
    
    # Invalid confidence
    with pytest.raises(ValueError):
        TranscriptSegment(
            speaker={"id": "S1", "role": "sales"},
            text="Valid text",
            timestamp=datetime.now(),
            confidence=1.5
        )

@pytest.mark.asyncio
async def test_process_file(temp_transcript_file):
    """Test processing a single transcript file."""
    processor = TranscriptProcessor(input_dir=temp_transcript_file.parent)
    transcript = await processor.process_file(temp_transcript_file)
    
    assert isinstance(transcript, Transcript)
    assert transcript.id == "test_123"
    assert transcript.duration == 300
    assert len(transcript.segments) == 2
    assert transcript.conversion_outcome is True

def test_clean_text():
    """Test text cleaning functionality."""
    processor = TranscriptProcessor()
    
    # Test various text cleaning scenarios
    assert processor.clean_text("Hello,   World!") == "hello, world!"
    assert processor.clean_text("Special@#$Characters") == "special characters"
    assert processor.clean_text("  Extra  Spaces  ") == "extra spaces"

def test_extract_features(sample_transcript_data):
    """Test feature extraction from transcript."""
    processor = TranscriptProcessor()
    transcript = Transcript(**sample_transcript_data)
    features = processor.extract_features(transcript)
    
    assert features["id"] == "test_123"
    assert features["duration"] == 300
    assert features["num_segments"] == 2
    assert features["sales_turns"] == 1
    assert features["customer_turns"] == 1
    assert features["conversion"] is True

@pytest.mark.asyncio
async def test_process_directory(tmp_path, sample_transcript_data):
    """Test processing multiple transcript files in a directory."""
    # Create multiple test files
    for i in range(3):
        data = sample_transcript_data.copy()
        data["id"] = f"test_{i}"
        with open(tmp_path / f"transcript_{i}.json", 'w') as f:
            json.dump(data, f)
    
    processor = TranscriptProcessor(input_dir=tmp_path)
    df = await processor.process_directory()
    
    assert len(df) == 3
    assert all(df["duration"] == 300)
    assert all(df["conversion"] == True)

if __name__ == "__main__":
    pytest.main([__file__]) 