# Technical Deep-Dive: Core Components

## Text Embeddings

### Sentence-BERT (SBERT)
**Core Principle:** Transform text into dense vector representations that capture semantic meaning.

**Specific Logic:**
1. Input text passes through a BERT-like transformer model
2. Special pooling layer (usually mean pooling) combines token embeddings
3. Results in fixed-size vector (typically 384-768 dimensions)
4. Vectors close in space = semantically similar texts

**Common Tooling:**
- sentence-transformers library
- Hugging Face transformers
- Model: 'all-MiniLM-L6-v2' (good balance of speed/accuracy)

**Project Motto:** _"Master the art of converting legal text into meaningful vectors that capture semantic relationships."_

### TF-IDF Vectorization
**Core Principle:** Represent documents as sparse vectors based on term frequency and inverse document frequency.

**Specific Logic:**
1. Calculate term frequency (TF) in each document
2. Calculate inverse document frequency (IDF) across corpus
3. Multiply TF × IDF for final weights
4. Results in sparse matrix where each row = document vector

**Common Tooling:**
- scikit-learn TfidfVectorizer
- scipy sparse matrices
- NLTK for tokenization

**Project Motto:** _"Understand how to weight legal terms based on their significance and uniqueness in the corpus."_

## Similarity Methods

### Cosine Similarity
**Core Principle:** Measure similarity between vectors based on the angle between them, not magnitude.

**Specific Logic:**
1. Calculate dot product of vectors
2. Divide by product of vector magnitudes
3. Results in similarity score between -1 and 1
4. 1 = identical direction, 0 = orthogonal, -1 = opposite

**Common Tooling:**
- numpy dot product
- sklearn.metrics.pairwise
- scipy.spatial.distance.cosine

**Project Motto:** _"Learn to compare legal documents based on their directional similarity in vector space."_

### Euclidean Distance
**Core Principle:** Calculate the straight-line distance between points in vector space.

**Specific Logic:**
1. Square the difference between each vector dimension
2. Sum all squared differences
3. Take the square root
4. Convert to similarity score (usually via negative exponential)

**Common Tooling:**
- numpy.linalg.norm
- scipy.spatial.distance
- sklearn.metrics.pairwise_distances

**Project Motto:** _"Master geometric distance calculations for comparing legal document embeddings."_

### Maximal Marginal Relevance (MMR)
**Core Principle:** Balance between relevance to query and diversity in results.

**Specific Logic:**
1. Select most relevant document to query
2. For each subsequent selection:
   - Score = λ(relevance) - (1-λ)(max similarity to selected docs)
   - Choose document with highest score
3. Continue until k documents selected

**Common Tooling:**
- Custom implementation usually required
- numpy for efficient matrix operations
- scipy for distance calculations

**Project Motto:** _"Learn to balance result relevance with diversity for comprehensive legal search."_

### Hybrid Similarity
**Core Principle:** Combine semantic similarity with domain-specific features (legal entities).

**Specific Logic:**
1. Calculate cosine similarity from embeddings
2. Extract and match legal entities
3. Compute entity overlap score
4. Weighted sum: 0.6×Cosine + 0.4×Entity_Score

**Common Tooling:**
- spacy for entity recognition
- Custom entity matching logic
- numpy for score combination

**Project Motto:** _"Master the art of combining multiple similarity signals for legal-domain-aware search."_

## Evaluation Metrics

### Precision & Recall
**Core Principle:** Measure accuracy and completeness of search results.

**Specific Logic:**
1. Precision = relevant_retrieved / total_retrieved
2. Recall = relevant_retrieved / total_relevant
3. Calculate for top-k results (k=5)
4. Average across test queries

**Common Tooling:**
- sklearn.metrics
- numpy for calculations
- pandas for result aggregation

**Project Motto:** _"Learn to rigorously evaluate search quality using standard IR metrics."_

### Diversity Score
**Core Principle:** Measure variety and coverage in search results.

**Specific Logic:**
1. Calculate pairwise similarities in result set
2. Average similarity = inverse diversity
3. Penalize redundant results
4. Reward coverage of different aspects

**Common Tooling:**
- scipy for distance calculations
- numpy for matrix operations
- Custom scoring functions

**Project Motto:** _"Master the measurement of result diversity for comprehensive legal search."_
