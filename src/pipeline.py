# pipeline.py
import json
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Imports for modular preprocessing
from src.utils.preprocessing_utils import (
    preprocess_instance_base,
    apply_ablation_processing,
    postprocess_instance_outputs,
)
from src.utils.embedding_utils import Embedder
from src.clusterer.pairing_clusterer import PairingClusterer
from src.clusterer.hdbscan_sentence_clusterer import HDBSCANSentenceClusterer
from src.clusterer.hdbscan_paragraph_clusterer import HDBSCANParagraphClusterer
from src.summariser.summariser import Summariser
from src.utils.metadata_utils import Document, Paragraph, Sentence
from src.evaluator.metrics import SummaryEvaluator


embedder = SentenceTransformer("all-MiniLM-L6-v2")

# For final results
ALL_RESULTS_PATH = Path("all_results.json")

class Pipeline:
    """
    Multi-document summarisation (MDS) pipeline orchestrator.

    Steps per ablation mode:
      0. Base preprocessing (clean ads, split paragraphs)- These are ablation-level agnostic.
      1. Apply ablation-specific processing (coref, coref+ NER)
      2. Postprocessing
      (filtering out extremely different sentences (those are advertisements with a high probability)
      + golden summary generation from concatenating original sources and replacing quotes with reported speech.)
      3. Embed paragraphs and sentences
      4. Pair paragraphs across documents
      5. HDBSCAN sentence clustering
      6. HDBSCAN paragraph clustering
      7. Summarisation via Ollama LLM
      8. Evaluation
      9. Writing results to a persistent .json file for iterative summary generation for multiple instances
    """

    def __init__(self, config: dict):
        self.config = config

        # Embedding setup
        embed_cfg = config.get("embedding", {})
        embed_model = embed_cfg.get("model", "sentence-transformers/all-mpnet-base-v2")
        self.embedder = Embedder(model_name=embed_model)

        # Summarisation setup
        self.llm_model = config.get("ollama_model", "qwen3:14b")
        self.summarizer = Summariser(model=self.llm_model)

        # Evaluation setup
        self.evaluator = SummaryEvaluator(
            base_dir="data/processed",
            results_dir="data/results",
            model_name=embed_model,
        )

    def run(self, instance_id: int):
        """
        Run the pipeline for a single instance in the multi_news dataset.
        instance_id: index of the instance in the dataset.
        """
        ablation_modes = self.config.get("ablation_modes", ["baseline", "coref", "coref+ner"])
        print(f"=== Starting MDS Pipeline for instance {instance_id} ===")

        # Load only the required instance
        dataset = load_dataset("multi_news", split=f"train[{instance_id}:{instance_id + 1}]")
        instance = dataset[0]

        # Step 0: Base preprocessing (common for all)
        print("\n[Step 0] Base preprocessing (ablation-independent)")
        raw_texts, base_out_file = preprocess_instance_base(
            instance,
            out_dir=Path("data/processed"),
            mode="baseline"
        )

        # Step 1: Postprocessing for similarity filtering & golden summary
        print("\n[Step 1] Generating golden summary and performing similarity filtering")
        postprocess_instance_outputs(
            raw_texts,
            results_dir=Path("data/results"),
            embedder=embedder
        )

        # Step 2: Run ablation-specific processing for each mode
        for mode in ablation_modes:
            print(f"\n[Step 2] Running ablation-specific preprocessing for mode={mode}")
            ablation_out_file = apply_ablation_processing(
                mode=mode,
                out_dir=Path("data/processed"),
                base_out_file=base_out_file
            )

            # Prepare paths
            mode_dir = Path(f"data/processed/{mode}")
            cache_path = mode_dir / "embeddings_cache.npz"
            pairing_file = mode_dir / "pairs.json"
            sentence_clusters_file = mode_dir / "hdbscan_sentence_clusters.json"
            paragraph_clusters_file = mode_dir / "hdbscan_paragraph_clusters.json"
            summaries_dir = mode_dir / "summaries"
            summaries_dir.mkdir(exist_ok=True, parents=True)

            # Load documents
            docs_file = mode_dir / f"multi_doc_{mode}.jsonl"
            print(f"\n[Step 3] Loading preprocessed docs from {docs_file}")
            try:
                docs = self._load_docs(docs_file)
            except FileNotFoundError:
                print(f" Docs file not found: {docs_file}. Skipping mode {mode}.")
                continue
            except Exception as e:
                print(f" Error loading docs for mode={mode}: {e}. Skipping.")
                continue

            # Step 4: Compute embeddings
            print(f"\n[Step 4] Embedding {len(docs)} documents for mode={mode}")
            try:
                self.embedder.embed_documents(docs, str(cache_path))
            except Exception as e:
                print(f" Embedding failed for mode={mode}: {e}. Skipping this mode.")
                continue

            # Step 5: Paragraph Pairing
            print(f"\n[Step 5] Running paragraph pairing for mode={mode}")
            original_lookup = self._load_original_lookup()
            clusterer = PairingClusterer(docs, original_lookup=original_lookup)
            try:
                pairs = clusterer.pair()
            except Exception as e:
                print(f" Pairing failed for mode={mode}: {e}. Skipping this mode.")
                continue

            with open(pairing_file, "w", encoding="utf-8") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            print(f" Saved paragraph pairs to {pairing_file}")

            # Step 6: HDBSCAN Sentence Clustering
            print(f"\n[Step 6] Running HDBSCAN sentence clustering for mode={mode}")
            try:
                hdb_sent_clusterer = HDBSCANSentenceClusterer(docs, original_lookup=original_lookup)
                sentence_clusters = hdb_sent_clusterer.cluster()
                with open(sentence_clusters_file, "w", encoding="utf-8") as f:
                    json.dump(sentence_clusters, f, indent=2, ensure_ascii=False)
                print(f" Saved sentence clusters to {sentence_clusters_file}")
            except Exception as e:
                print(f" Sentence clustering failed for mode={mode}: {e}. Skipping this step.")

            # Step 7: HDBSCAN Paragraph Clustering
            print(f"\n[Step 7] Running HDBSCAN paragraph clustering for mode={mode}")
            try:
                hdb_para_clusterer = HDBSCANParagraphClusterer(docs, original_lookup=original_lookup)
                paragraph_clusters = hdb_para_clusterer.cluster()
                with open(paragraph_clusters_file, "w", encoding="utf-8") as f:
                    json.dump(paragraph_clusters, f, indent=2, ensure_ascii=False)
                print(f" Saved paragraph clusters to {paragraph_clusters_file}")
            except Exception as e:
                print(f" Paragraph clustering failed for mode={mode}: {e}. Skipping this step.")

            # Step 8: Summarisation
            print(f"\n[Step 8] Generating summaries for mode={mode}")
            self._generate_summaries(
                mode, summaries_dir, pairing_file, sentence_clusters_file, paragraph_clusters_file
            )

        # Step 9: Evaluation
        print("\n[Step 9] Evaluating all summaries")
        try:
            self.evaluator.evaluate_all()
        except Exception as e:
            print(f" Evaluation failed: {e}")

        # Step 10: Write results to persistent output file
        print("\n[Step 10] Writing evaluation results")

        # Load per-run results from evaluator (currently at data/results/evaluation_results.json)
        per_run_results_path = Path("data/results/evaluation_results.json")

        if per_run_results_path.exists():
            try:
                with open(per_run_results_path, "r", encoding="utf-8") as f:
                    run_results = json.load(f)
            except Exception as e:
                print(f"Failed to read per-run results: {e}")
                run_results = {}
        else:
            print("Per-run results not found. Skipping append to all_results.json")
            run_results = {}


        self._append_results_to_json(run_results, ALL_RESULTS_PATH, instance_id)

        print("\n=== Pipeline finished successfully ===")











    # Utilities for pipeline

    def _load_docs(self, filepath: Path):
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist")
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
                    raw_text=data.get("raw_text", ""),
                )
                docs.append(doc)
        return docs

    def _load_original_lookup(self):
        baseline_path = Path("data/processed/baseline/multi_doc_baseline.jsonl")
        if not baseline_path.exists():
            print(" Baseline file missing — skipping original text restoration.")
            return {}

        lookup = {}
        with open(baseline_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc = data.get("document", {})
                for para in doc.get("paragraphs", []):
                    tag = f"{para['doc_id']}_{para['para_id']}"
                    lookup[tag] = para.get("text", "")
        print(f"Loaded {len(lookup)} original baseline paragraphs for text restoration.")
        return lookup

    def _generate_summaries(
        self, mode, summaries_dir, pairing_file, sentence_clusters_file, paragraph_clusters_file
    ):
        summary_tasks = [
            ("pairing", pairing_file, summaries_dir / "summary_pairing.txt"),
            ("sentence_clustering", sentence_clusters_file, summaries_dir / "summary_sentence_clusters.txt"),
            ("paragraph_clustering", paragraph_clusters_file, summaries_dir / "summary_paragraph_clusters.txt"),
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
                print(f"Saved {method} summary to {output_file}")
            except Exception as e:
                print(f"Skipped {method} summarisation due to error: {e}")




    def _append_results_to_json(self, new_results: dict, results_path: Path, instance_id: int):
        """
        Appends a single instance's evaluation results to the master JSON file.
        Each instance result is stored as an object with an identifier.
        """
        # Structure to store instance + results together
        record = {
            "instance_id": instance_id,
            "results": new_results
        }

        # Load existing results if any
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            except json.JSONDecodeError:
                data = []
        else:
            data = []

        # Append new results
        data.append(record)

        # Save back
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Appended results for instance {instance_id} to  {results_path}")

