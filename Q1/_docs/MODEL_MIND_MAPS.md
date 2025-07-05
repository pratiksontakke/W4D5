# Model-Specific Mind Maps

This document applies the "Deconstruction" thinking process to each of the embedding models required by the assignment. Notice how only the "Transformation" component changes between them.

---

### 1. OpenAI (`text-embedding-ada-002`)
- **Transformation Method:** API Call.
- **Specific Logic:** Pass raw text to the OpenAI API endpoint. It directly returns a single, state-of-the-art embedding vector.
- **Tooling:** `openai` library (or LangChain wrapper).
- **Motto:** "Learn to integrate a powerful, managed, state-of-the-art API for embedding."

---

### 2. Sentence-BERT (`all-MiniLM-L6-v2`)
- **Transformation Method:** Local, Fine-Tuned Model.
- **Specific Logic:** Pass raw text to the model's `.encode()` method. It directly returns a single vector optimized for sentence-level meaning.
- **Tooling:** `sentence-transformers` library.
- **Motto:** "Learn to use a high-performance, self-hosted model specifically designed for sentence similarity and classification."

---

### 3. BERT (`bert-base-uncased`)
- **Transformation Method:** Local, General-Purpose Model.
- **Specific Logic:** Tokenize text, pass through the model, and extract the hidden state of the special `[CLS]` token as the document representation.
- **Tooling:** `transformers` library.
- **Motto:** "Learn the standard technique for adapting a general-purpose transformer for sentence-level tasks."

---

### 4. Word2Vec/GloVe
- **Transformation Method:** Averaging Non-Contextual Word Vectors.
- **Specific Logic:** Tokenize text, look up the vector for each word, and compute the average of all vectors.
- **Tooling:** `gensim` and `nltk`.
- **Motto:** "Learn the foundational baseline approach to understand its strengths (speed) and weaknesses (no context), providing a crucial point of comparison." 