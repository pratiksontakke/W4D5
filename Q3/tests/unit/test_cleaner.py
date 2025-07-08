"""
Unit tests for the text cleaning module.
"""

import pytest
from data.processing.cleaner import TextCleaner

@pytest.fixture
def cleaner():
    """Create a TextCleaner instance with default settings."""
    return TextCleaner()

@pytest.fixture
def custom_cleaner():
    """Create a TextCleaner instance with custom settings."""
    return TextCleaner(
        remove_stopwords=True,
        remove_numbers=True,
        remove_punctuation=True,
        lowercase=True,
        custom_stopwords={'um', 'uh', 'er'}
    )

def test_normalize_whitespace(cleaner):
    """Test whitespace normalization."""
    text = "  Multiple    spaces   and\tTabs\n\nand newlines  "
    expected = "Multiple spaces and Tabs and newlines"
    assert cleaner.normalize_whitespace(text) == expected

def test_remove_special_characters(cleaner):
    """Test special character removal."""
    # Test with keeping punctuation
    text = "Hello! This is a test... With @#$ special chars."
    expected = "Hello! This is a test... With special chars."
    assert cleaner.remove_special_characters(text, keep_punctuation=True) == expected
    
    # Test without keeping punctuation
    expected_no_punct = "Hello This is a test With special chars"
    assert cleaner.remove_special_characters(text, keep_punctuation=False) == expected_no_punct

def test_normalize_entities(cleaner):
    """Test entity normalization."""
    text = (
        "Contact us at support@company.com or call +1-234-567-8900. "
        "Price: $1,299.99. Visit https://www.company.com"
    )
    expected = (
        "Contact us at [EMAIL] or call [PHONE]. "
        "Price: [MONEY]. Visit [URL]"
    )
    assert cleaner.normalize_entities(text) == expected

def test_clean_text_with_defaults(cleaner):
    """Test text cleaning with default settings."""
    text = "Hello! This is a TEST... Email: test@test.com"
    expected = "hello! this is a test... email: [EMAIL]"
    assert cleaner.clean_text(text) == expected

def test_clean_text_with_custom_settings(custom_cleaner):
    """Test text cleaning with custom settings."""
    text = "Hello! I have 123 dollars and um... some questions."
    # Should remove stopwords, numbers, punctuation, and custom stopwords
    expected = "hello dollars questions"
    assert custom_cleaner.clean_text(text) == expected

def test_clean_batch(cleaner):
    """Test batch text cleaning."""
    texts = [
        "First text with $100",
        "Second text with test@email.com"
    ]
    expected = [
        "first text with [MONEY]",
        "second text with [EMAIL]"
    ]
    assert cleaner.clean_batch(texts) == expected

def test_validate_text(cleaner):
    """Test text validation."""
    # Valid text
    assert cleaner.validate_text("This is a valid piece of text with sufficient length")
    
    # Invalid cases
    assert not cleaner.validate_text("")  # Empty string
    assert not cleaner.validate_text("Too short")  # Too short
    assert not cleaner.validate_text("@#$%^&*")  # Only special characters
    assert not cleaner.validate_text(None)  # None input

def test_error_handling(cleaner):
    """Test error handling."""
    with pytest.raises(ValueError):
        cleaner.clean_text("")
    
    with pytest.raises(ValueError):
        cleaner.clean_text(None)

if __name__ == "__main__":
    pytest.main([__file__]) 