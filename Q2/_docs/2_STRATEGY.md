# Architectural Strategy: Indian Legal Document Search System

---

## 1. Proposed Architecture

```
project_root/
│
├── app.py                  # Main entry point for the web UI (inference phase)
├── config.py               # Configuration settings (paths, model params, etc.)
├── requirements.txt        # Python dependencies
├── core/
│   ├── __init__.py
│   ├── data_loader.py      # Load and parse legal documents
│   ├── preprocess.py       # Text cleaning, normalization, legal entity recognition
│   ├── embedders.py        # Text vectorization (TF-IDF, embeddings)
│   ├── similarity.py       # Cosine, Euclidean, MMR, Hybrid similarity methods
│   ├── retrieval.py        # Retrieval and ranking logic
│   └── evaluation.py       # Precision, recall, diversity, metrics
│
├── pipelines/
│   ├── __init__.py
│   ├── build_index.py      # Offline pipeline: preprocess, embed, index, save artifacts
│   └── evaluate.py         # Offline pipeline: run evaluation, generate reports
│
├── ui/
│   ├── __init__.py
│   ├── web_app.py          # Web UI logic (Flask/FastAPI/Streamlit)
│   ├── components.py       # UI components: upload, query, results, dashboard
│   └── static/             # CSS, JS, images
│
├── artifacts/              # Saved models, embeddings, indices
│
├── data/                   # Raw and processed legal documents
│
└── _docs/                  # Documentation (deconstruction, strategy, analysis)
```

---

## 2. Component Mapping

- **Input Data**
  - Handled by `core/data_loader.py` (loading/parsing) and `data/` directory (storage).
- **Preprocessing**
  - Implemented in `core/preprocess.py` (text cleaning, normalization, legal entity recognition).
- **Embedding/Representation**
  - `core/embedders.py` (TF-IDF, word/sentence embeddings).
- **Similarity Methods**
  - `core/similarity.py` (Cosine, Euclidean, MMR, Hybrid similarity functions).
- **Retrieval & Ranking**
  - `core/retrieval.py` (computing similarity scores, ranking, result aggregation).
- **Evaluation Framework**
  - `core/evaluation.py` (precision, recall, diversity, side-by-side comparison logic).
- **Web UI**
  - `ui/web_app.py` (main UI logic), `ui/components.py` (upload, query, results, dashboard), `ui/static/` (assets).
- **Reporting**
  - `pipelines/evaluate.py` (performance analysis, recommendations), `_docs/` (final reports).

---

## 3. Data & Logic Flow

### Offline Phase (Training/Indexing)
- **Entry Point:** `pipelines/build_index.py`
- **Sequence:**
  1. **Load Data:** Use `core/data_loader.py` to ingest and parse all legal documents from `data/`.
  2. **Preprocess:** Clean and normalize text, extract legal entities via `core/preprocess.py`.
  3. **Embed:** Generate document embeddings (TF-IDF, word/sentence embeddings) using `core/embedders.py`.
  4. **Index:** Build and save indices/artifacts (e.g., embedding matrices, entity lists) to `artifacts/`.
  5. **Evaluate:** Optionally, run `pipelines/evaluate.py` to compute metrics (precision, recall, diversity) and save reports to `_docs/`.

### Online Phase (Inference/Search)
- **Entry Point:** `app.py` (serves UI via `ui/web_app.py`)
- **Sequence:**
  1. **Load Artifacts:** On startup, load precomputed embeddings, indices, and entity lists from `artifacts/` using `core/embedders.py` and `core/data_loader.py`.
  2. **User Input:** Accept user query and/or document upload via the web UI (`ui/components.py`).
  3. **Preprocess Query:** Clean and normalize query text, extract entities (`core/preprocess.py`).
  4. **Embed Query:** Convert query to embedding(s) (`core/embedders.py`).
  5. **Similarity Search:** For each method (Cosine, Euclidean, MMR, Hybrid), compute similarity scores between query and documents (`core/similarity.py`).
  6. **Retrieve & Rank:** Aggregate and rank top results for each method (`core/retrieval.py`).
  7. **Display Results:** Show side-by-side results and performance metrics in the UI (`ui/web_app.py`, `ui/components.py`).

---

This architecture ensures modularity, clarity, and extensibility for both experimentation and production deployment.
