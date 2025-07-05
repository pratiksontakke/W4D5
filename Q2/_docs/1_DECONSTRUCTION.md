# Deconstruction Report: Indian Legal Document Search System

---

## 1. Core Task

**Build a system that retrieves and compares Indian legal documents using four different similarity methods to determine the most effective approach for legal document search.**

---

## 2. Problem Type

**Information Retrieval (IR) / Document Similarity Search**

---

## 3. Key Components (Mind Map)

- **Input Data**
  - Indian legal documents (Income Tax Act, GST Act, court judgments, property law)
  - User queries (text input)
  - Uploaded documents (PDF/Word)

- **Preprocessing**
  - Document parsing (PDF/Word to text)
  - Text cleaning and normalization
  - Legal entity recognition (for hybrid similarity)

- **Embedding/Representation**
  - Text vectorization (e.g., TF-IDF, word embeddings, sentence embeddings)

- **Similarity Methods**
  - Cosine Similarity
  - Euclidean Distance
  - Maximal Marginal Relevance (MMR)
  - Hybrid Similarity (Cosine + Legal Entity Match)

- **Retrieval & Ranking**
  - Compute similarity scores
  - Rank documents for each method

- **Evaluation Framework**
  - Precision (top 5 results)
  - Recall
  - Diversity Score (for MMR)
  - Side-by-side comparison

- **Web UI**
  - Document upload interface
  - Query input box
  - 4-column results display
  - Performance metrics dashboard

- **Reporting**
  - Performance analysis
  - Recommendations

---

## 4. The "Motto" (Real-World Value)

**Why is this valuable?**

- **Legal Research:** Lawyers and paralegals can quickly find the most relevant statutes, case laws, or precedents for a given legal question.
- **Compliance & Auditing:** Companies can efficiently retrieve relevant legal provisions to ensure compliance with tax, GST, or property regulations.
- **Judicial Assistance:** Judges and court staff can rapidly compare similar cases or statutes to support fair and consistent rulings.

**Project Motto:**  
_"Empowering legal professionals with smarter, faster, and more reliable document search."_