import os
import json
from pathlib import Path
from src.utils.preprocessing_utils import preprocess_first_instance
from src.utils.embedding_utils import Embedder
from src.clusterer.pairing_clusterer import PairingClusterer
from src.clusterer.hdbscan_sentence_clusterer import HDBSCANSentenceClusterer
from src.clusterer.hdbscan_paragraph_clusterer import HDBSCANParagraphClusterer
from src.summariser.summariser import Summariser
from src.utils.metadata_utils import Document, Paragraph, Sentence
from src.evaluator.metrics import SummaryEvaluator


class Pipeline:
    """
    Multi-document summarisation (MDS) pipeline orchestrator.

    Steps per ablation mode:
      1. Preprocess raw documents (clean ads, split paragraphs, optionally do coref/NER)
      2. Embed paragraphs and sentences
      3. Pair paragraphs across documents
      4. (Optional) HDBSCAN sentence clustering
      5. (Optional) HDBSCAN paragraph clustering
      6. Summarisation via Ollama LLM
      7. Evaluation: coverage, overlap, named entity retention
    """

    def __init__(self, config: dict):
        self.config = config

        #  Embedding setup 
        embed_cfg = config.get("embedding", {})
        embed_model = embed_cfg.get("model", "sentence-transformers/all-mpnet-base-v2")
        self.embedder = Embedder(model_name=embed_model)

        #  Summarisation setup 
        self.llm_model = config.get("ollama_model", "qwen3:14b")
        self.summarizer = Summariser(model=self.llm_model)

        #  Evaluation setup 
        self.evaluator = SummaryEvaluator(
            base_dir="data/processed",
            results_dir="data/results",
            model_name=embed_model
        )

    def run(self):
        """
        Execute the pipeline across all configured ablation modes.
        Performs preprocessing, embeddings, pairing/clustering, summarisation, and evaluation.
        """
        ablation_modes = self.config.get("ablation_modes", ["baseline","coref", "coref+ner"])
        print("=== Starting MDS Pipeline ===")

        for mode in ablation_modes:
            # Preprocessing
            print(f"\n[Step 1] Preprocessing with mode={mode}")
            preprocess_first_instance(mode=mode)

            # Prepare paths for cached embeddings and cluster outputs
            mode_dir = Path(f"data/processed/{mode}")
            cache_path = mode_dir / f"embeddings_cache.npz"

            pairing_file = mode_dir / "pairs.json"
            sentence_clusters_file = mode_dir / "hdbscan_sentence_clusters.json"
            paragraph_clusters_file = mode_dir / "hdbscan_paragraph_clusters.json"

            # Summaries output directory
            summaries_dir = mode_dir / "summaries"
            summaries_dir.mkdir(exist_ok=True, parents=True)

            # Load preprocessed documents
            docs_file = mode_dir / f"multi_doc_{mode}.jsonl"
            print(f"[Step 2] Loading preprocessed docs from {docs_file}")
            docs = self._load_docs(docs_file)

            # Compute embeddings
            print(f"[Step 3] Embedding {len(docs)} documents for mode={mode}")
            self.embedder.embed_documents(docs, str(cache_path))

            # Paragraph Pairing
            print(f"[Step 4] Running paragraph pairing for mode={mode}")
            clusterer = PairingClusterer(docs)
            pairs = clusterer.pair()
            with open(pairing_file, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved paragraph pairs to {pairing_file}")

            # HDBSCAN Sentence Clustering
            print(f"[Step 5] Running HDBSCAN sentence clustering for mode={mode}")
            hdb_sent_clusterer = HDBSCANSentenceClusterer(docs)
            sentence_clusters = hdb_sent_clusterer.cluster()
            with open(sentence_clusters_file, "w", encoding="utf-8") as f:
                json.dump(sentence_clusters, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved sentence clusters to {sentence_clusters_file}")

            # HDBSCAN Paragraph Clustering
            print(f"[Step 6] Running HDBSCAN paragraph clustering for mode={mode}")
            hdb_para_clusterer = HDBSCANParagraphClusterer(docs)
            paragraph_clusters = hdb_para_clusterer.cluster()
            with open(paragraph_clusters_file, "w", encoding="utf-8") as f:
                json.dump(paragraph_clusters, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved paragraph clusters to {paragraph_clusters_file}")

            # Summarisation
            print(f"[Step 7] Generating summaries for mode={mode}")
            self._generate_summaries(
                mode, summaries_dir, pairing_file, sentence_clusters_file, paragraph_clusters_file
            )

        # Evaluation
        print("\n[Step 8] Evaluating all summaries")
        self.evaluator.evaluate_all()  # computes coverage, overlap, NE retention; saves JSON + CSV

        print("\n=== Pipeline finished successfully ===")


    # Utilities
    def _load_docs(self, filepath: Path):
        """
        Load JSONL documents into Document objects with Paragraphs and Sentences.
        Necessary for embedding computation and clustering.
        """
        docs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)["document"]
                paragraphs = []
                for p in data["paragraphs"]:
                    sentences = [Sentence(**s) for s in p["sentences"]]
                    para = Paragraph(
                        sentences=sentences,
                        doc_id=p["doc_id"],
                        para_id=p["para_id"],
                        text=p["text"],
                        resolved_text=p.get("resolved_text"),
                    )
                    paragraphs.append(para)
                doc = Document(
                    doc_id=data["doc_id"],
                    paragraphs=paragraphs,
                    raw_text=data["raw_text"],
                )
                docs.append(doc)
        return docs

    def _generate_summaries(
        self, mode, summaries_dir, pairing_file, sentence_clusters_file, paragraph_clusters_file
    ):
        """
        Runs the summariser for each configured summarisation method (pairing, sentence clusters, paragraph clusters).
        Saves the generated summaries to files.
        """
        summary_tasks = [
            ("pairing", pairing_file, summaries_dir / "summary_pairing.txt"),
        #     ("sentence_clustering", sentence_clusters_file, summaries_dir / "summary_sentence_clusters.txt"),
        #     ("paragraph_clustering", paragraph_clusters_file, summaries_dir / "summary_paragraph_clusters.txt"),
        ]

        for method, input_file, output_file in summary_tasks:
            print(f"   → Summarising {method} results for mode={mode}")
            try:
                if not Path(input_file).exists():
                    print(f"Skipping {method} — input file missing.")
                    continue

                with open(input_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                summary = self.summarizer.summarize(data, method)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(summary)

                print(f"      ✓ Saved {method} summary to {output_file}")

            except Exception as e:
                print(f"Skipped {method} summarisation due to error: {e}")
