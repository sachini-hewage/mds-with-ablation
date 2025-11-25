# Multi-Source Document Summariser with Ablation Studies

This repository contains a **multi-document summarisation (MDS) pipeline** designed to process news datasets (like Multi-News) and generate high-quality summaries using state-of-the-art NLP techniques. It supports **ablation experiments** to evaluate the impact of various preprocessing steps on summarisation performance.

---

## Features

- **Multi-document summarisation** for datasets with multiple sources per instance.
- **Ablation study support** to test the impact of:
  - Baseline preprocessing
  - Coreference resolution
  - Coreference + Named Entity Recognition (NER)
- **End-to-end pipeline**:
  1. Base preprocessing (cleaning, paragraph splitting)
  2. Ablation-specific processing (coref, NER)
  3. Postprocessing (filtering advertisements, generating golden summaries)
  4. Paragraph and sentence embeddings
  5. Paragraph pairing across documents
  6. HDBSCAN sentence clustering
  7. HDBSCAN paragraph clustering
  8. Summarisation using an LLM (via Ollama)
  9. Optional redundancy reduction via MMR
  10. Automatic evaluation of summaries
  11. Persistent JSON storage of results per instance

- **Embedding support** using [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` by default).
- **Flexible summarisation** via local LLM (configurable in `pipeline.py`).
- **Evaluation** using standard metrics on Multi-News dataset instances.

---

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/multi-doc-summariser.git
cd multi-doc-summariser

2. Install dependancies:
pip install -r requirements.txt

Requirements include sentence-transformers, hdbscan, scikit-learn, jsonlines, and your LLM integration (Ollama).
Ensure dataset files exist in data/multinews_100_instances.json. Preprocessed outputs will be saved in data/processed/ and evaluation results in data/results/.


