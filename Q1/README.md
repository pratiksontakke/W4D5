# Smart Article Categorizer

A sophisticated system that automatically classifies articles into different categories using multiple embedding approaches and machine learning techniques.

## Overview

This project implements an article classification system using four different embedding approaches:
- Word2Vec/GloVe
- Sentence-BERT (all-MiniLM-L6-v2)
- OpenAI (text-embedding-ada-002)

The system currently classifies articles into the following categories:
- Tech
- Finance
- Sports
- Politics

Note: Healthcare and Entertainment categories will be added in future updates.

## Performance Metrics

### OpenAI Embeddings
- Tech: Precision: 0.960, Recall: 0.960, F1-score: 0.960
- Finance: Precision: 0.958, Recall: 0.920, F1-score: 0.939
- Sports: Precision: 1.000, Recall: 1.000, F1-score: 1.000
- Politics: Precision: 0.923, Recall: 0.960, F1-score: 0.941

### Sentence-BERT Embeddings
- Tech: Precision: 0.889, Recall: 0.970, F1-score: 0.928
- Finance: Precision: 0.879, Recall: 0.753, F1-score: 0.811
- Sports: Precision: 0.906, Recall: 0.951, F1-score: 0.928
- Politics: Precision: 0.936, Recall: 0.825, F1-score: 0.877

### Word2Vec Embeddings
- Tech: Precision: 0.877, Recall: 0.941, F1-score: 0.908
- Finance: Precision: 0.828, Recall: 0.690, F1-score: 0.752
- Sports: Precision: 0.897, Recall: 0.915, F1-score: 0.906
- Politics: Precision: 0.873, Recall: 0.840, F1-score: 0.856

## Project Structure

```
.
├── openAI/
│   ├── main.py
│   └── openai_classifier.py
├── sentenceBERT/
│   ├── main.py
│   └── sentence_bert_classifier.py
├── word2Vec/
│   ├── main.py
│   └── word2vec_classifier.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Each embedding approach has its own module that can be run independently:

```bash
# Run OpenAI embeddings classifier
python openAI/main.py

# Run Sentence-BERT classifier
python sentenceBERT/main.py

# Run Word2Vec classifier
python word2Vec/main.py
```

## Sample Predictions

Here's how each model performs on sample texts:

### Text: "Apple announces new iPhone with revolutionary AI capabilities"
- OpenAI: Tech (39.7% confidence)
- Sentence-BERT: Tech (90.5% confidence)
- Word2Vec: Tech (99.9% confidence)

### Text: "Manchester United wins dramatic match against Liverpool"
- OpenAI: Sports (38.3% confidence)
- Sentence-BERT: Sports (53.3% confidence)
- Word2Vec: Sports (99.8% confidence)

### Text: "Stock market reaches all-time high as tech sector surges"
- OpenAI: Finance (33.0% confidence)
- Sentence-BERT: Finance (85.2% confidence)
- Word2Vec: Finance (78.4% confidence)

## Model Comparison

- **Word2Vec**: Shows high confidence in predictions but slightly lower F1-scores. Best for clear-cut categories.
- **Sentence-BERT**: Provides balanced performance with good F1-scores and reasonable confidence levels.
- **OpenAI**: Achieves the highest F1-scores but shows more conservative confidence levels.

## Dataset

The project uses the AG News dataset with the following category mappings:
- World news → Politics
- Business → Finance
- Sci/Tech → Tech
- Sports → Sports

## Future Improvements

1. Add support for Healthcare and Entertainment categories
2. Implement ensemble methods to combine predictions
3. Add a web UI for real-time classification
4. Add visualization of embedding clusters
5. Expand the training dataset for better coverage

## License

MIT License

## Contributing

Feel free to open issues and pull requests for any improvements.
