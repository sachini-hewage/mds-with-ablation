import os
import re
import json
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import spacy
from bert_score import score as bert_score


class SummaryEvaluator:
    """
    Evaluates generated summaries with multiple metrics:

    Metrics:
      1. Coverage: proportion of source sentences represented in summary
      2. Overlap: redundancy among summary sentences
      3. Named Entity Retention: percentage of original PERSON entities retained

    Assumes file structure:
      - Summaries: data/processed/<ablation_type>/summaries/*.txt
      - Source sentences: data/results/combined_source_sentences.txt
    """

    def __init__(self, base_dir="data/processed", results_dir="data/results", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize evaluator.

        Args:
            base_dir (str): Base directory for summary files per ablation.
            results_dir (str): Directory to save evaluation results.
            model_name (str): SentenceTransformer model for embeddings.
        """
        self.base_dir = Path(base_dir)
        self.embedder = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")

        self.results = {}  # store results by ablation+method
        self.rows = []     # store results for CSV writing

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # File containing all combined source sentences
        self.source_file = self.results_dir / "combined_source_sentences.txt"


    # Load source sentences
    def collect_source_sentences(self):
        """Read all source sentences from combined_source_sentences.txt."""
        if not self.source_file.exists():
            raise FileNotFoundError(f"Source sentences file not found at {self.source_file}")

        all_sentences = []
        all_text = ""
        with open(self.source_file, "r", encoding="utf-8") as f:
            for line in f:
                sent_text = line.strip()
                if sent_text:
                    all_sentences.append(sent_text)
                    all_text += sent_text + " "

        print(f"Loaded {len(all_sentences)} source sentences from {self.source_file}")
        return all_sentences, all_text


    # Text cleaning & preprocessing as summaries contain annotation and NER tags
    def clean_summary_text(self, text: str) -> str:
        """
        Clean summary text by removing metadata, NER tags, and normalizing whitespace.

        Args:
            text (str): Raw summary text

        Returns:
            str: Cleaned summary text
        """
        text = re.sub(r'\[.*?\]', '', text)          # remove array-style metadata
        text = re.sub(r'\(metadata:.*?\)', '', text) # remove metadata in parentheses
        text = re.sub(r'<[^>]+>', '', text)          # remove NER tags like <ORG>
        text = re.sub(r'\*\*.*?\*\*', '', text)      # remove bold markers
        text = re.sub(r'\s+', ' ', text).strip()    # normalize whitespace
        return text

    def sentence_split(self, text: str):
        """Split text into sentences using spaCy pipeline."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def get_named_entities(self, text: str):
        """Extract unique named entities from text."""
        doc = self.nlp(text)
        return set([ent.text.strip().lower() for ent in doc.ents if ent.text.strip()])


    # Evaluation Metrics

    def coverage_metric(self, source_sents, summary_sents, threshold=0.4, return_details=False):
        """
        Coverage metric: fraction of source sentences represented in summary.

        A source sentence is considered 'covered' if at least one summary sentence
        has cosine similarity >= threshold.

        Args:
            source_sents (list[str]): List of source sentences.
            summary_sents (list[str]): List of summary sentences.
            threshold (float): Similarity threshold for coverage.
            return_details (bool): If True, return detailed per-sentence similarities.

        Returns:
            float or (float, dict): Coverage score, optionally with details.
        """
        if not source_sents or not summary_sents:
            return (0.0, {}) if return_details else 0.0

        # Encode sentences into embeddings
        src_embeds = self.embedder.encode(source_sents, convert_to_tensor=True)
        sum_embeds = self.embedder.encode(summary_sents, convert_to_tensor=True)

        # Cosine similarity matrix: shape (num_source, num_summary)
        sim_matrix = util.cos_sim(src_embeds, sum_embeds)

        # For each source sentence, get max similarity to any summary sentence
        max_sims = sim_matrix.max(dim=1).values

        # Count as covered if similarity >= threshold
        covered_mask = max_sims >= threshold
        coverage = covered_mask.float().mean().item()

        if not return_details:
            return coverage

        # Optional detailed breakdown per sentence
        details = {
            "threshold": threshold,
            "similarities": [
                {
                    "source_sentence": source_sents[i],
                    "max_similarity": float(max_sims[i]),
                    "is_covered": bool(covered_mask[i]),
                }
                for i in range(len(source_sents))
            ],
        }
        return coverage, details

    def overlap_metric(self, summary_sents, threshold=0.5):
        """Compute proportion of redundant sentences in summary based on cosine similarity."""
        if len(summary_sents) < 2:
            return 0.0

        embeds = self.embedder.encode(summary_sents, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeds, embeds)

        redundant_pairs = 0
        total_pairs = 0
        n = len(summary_sents)

        # Compare each pair of sentences (i < j)
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if sim_matrix[i][j] > threshold:
                    redundant_pairs += 1

        return redundant_pairs / total_pairs if total_pairs > 0 else 0.0



    # def coverage_metric(self, source_sents, summary_sents, return_details=True):
    #     """
    #     Coverage metric (semantic version using BERTScore Recall):
    #     Measures how much of the source content is represented in the summary,
    #     using deep contextual similarity instead of simple cosine thresholds.
    #
    #     A high recall score means the summary captures most of the semantic
    #     information from the source sentences even if rephrased.
    #
    #     Args:
    #         source_sents (list[str]): List of source sentences.
    #         summary_sents (list[str]): List of summary sentences.
    #         return_details (bool): If True, return detailed per-sentence recall contributions.
    #
    #     Returns:
    #         float or (float, dict):
    #             - If return_details=False: returns overall BERTScore Recall (float).
    #             - If return_details=True: returns (recall_score, details_dict),
    #               where details include per-sentence recall estimates.
    #     """
    #     # Handle empty input safely
    #     if not source_sents or not summary_sents:
    #         return (0.0, {}) if return_details else 0.0
    #
    #     candidate_text = " ".join(summary_sents)
    #     reference_text = " ".join(source_sents)
    #
    #     # Use a stronger model
    #     model_type = "microsoft/deberta-xlarge-mnli"  # stronger than default roberta-large
    #
    #     # Compute BERTScore Recall
    #     P, R, F1 = bert_score(
    #         cands=[candidate_text],
    #         refs=[reference_text],
    #         model_type=model_type,
    #         lang="en",
    #         rescale_with_baseline=True
    #     )
    #
    #     recall_score = float(R.mean().item())
    #
    #     if not return_details:
    #         return recall_score
    #
    #     # Sentence-level breakdown
    #     details = {
    #         "metric": "BERTScore Recall",
    #         "model": model_type,
    #         "overall_recall": recall_score,
    #         "sentence_breakdown": []
    #     }
    #
    #     for src in source_sents:
    #         _, R_sent, _ = bert_score(
    #             cands=[candidate_text],
    #             refs=[src],
    #             model_type=model_type,
    #             lang="en",
    #             rescale_with_baseline=True
    #         )
    #         r_val = float(R_sent.mean().item())
    #         details["sentence_breakdown"].append({
    #             "source_sentence": src,
    #             "approx_recall": r_val,
    #             "is_covered": r_val >= 0.4  # adjustable heuristic
    #         })
    #
    #     return recall_score, details

    def named_entity_retention_metric(self, source_text, summary_text):
        """
        Compute the percentage of unique PERSON entities from source retained in summary.

        Handles partial matches, deduplication, and nested names.

        Args:
            source_text (str): Full source text
            summary_text (str): Generated summary text

        Returns:
            float: Fraction of PERSON entities retained
        """
        src_doc = self.nlp(source_text)
        sum_doc = self.nlp(summary_text)

        # Extract PERSON entities
        src_persons = {ent.text.strip() for ent in src_doc.ents if ent.label_ == "PERSON"}
        sum_persons = {ent.text.strip() for ent in sum_doc.ents if ent.label_ == "PERSON"}

        # Normalize entities for comparison
        def normalize(name: str) -> str:
            return re.sub(r"[^\w\s]", "", name.lower()).strip()

        def tokens(name: str) -> set:
            return set(normalize(name).split())

        # Deduplicate nested/overlapping names
        def deduplicate_persons(persons: set) -> set:
            normed = sorted([normalize(p) for p in persons], key=len, reverse=True)
            unique = []
            for name in normed:
                if not any(name in u or u in name for u in unique):
                    unique.append(name)
            return set(unique)

        src_persons = deduplicate_persons(src_persons)
        sum_persons = deduplicate_persons(sum_persons)

        print("Source PERSON entities:", src_persons)
        print("Summary PERSON entities:", sum_persons)

        # If no PERSON entities in source, consider fully retained
        if not src_persons:
            return 1.0

        src_tokens = {s: tokens(s) for s in src_persons}
        sum_tokens = {s: tokens(s) for s in sum_persons}

        # Count retained entities (partial-match tolerant)
        retained = 0
        matched_summary = set()
        for src, src_tok in src_tokens.items():
            for sum_name, sum_tok in sum_tokens.items():
                if src_name := (src_tok & sum_tok):
                    if sum_name not in matched_summary:
                        retained += 1
                        matched_summary.add(sum_name)
                        break

        return retained / len(src_persons)

    # Evaluation Loop

    def evaluate_summary_file(self, ablation_name, summary_path, source_sents, source_text):
        """Evaluate one summary file and store metrics."""
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = self.clean_summary_text(f.read())

        summary_sents = self.sentence_split(summary_text)

        coverage = self.coverage_metric(source_sents, summary_sents)
        overlap = self.overlap_metric(summary_sents)
        importance = self.named_entity_retention_metric(source_text, summary_text)

        result = {
            "coverage": coverage,
            "overlap": round(overlap, 4),
            "importance_retention": round(importance, 4),
        }

        # Store results
        self.results[f"{ablation_name}_{summary_path.stem}"] = result
        self.rows.append({
            "ablation_type": ablation_name,
            "method": summary_path.stem,
            **result,
        })

    def evaluate_all(self):
        """Evaluate all summary files for all ablation types."""
        source_sents, source_text = self.collect_source_sentences()

        # Loop over ablation directories
        for ablation_dir in self.base_dir.iterdir():
            if not ablation_dir.is_dir() or ablation_dir.name == "results":
                continue

            summaries_dir = ablation_dir / "summaries"
            if not summaries_dir.exists():
                continue

            ablation_name = ablation_dir.name
            for summary_file in summaries_dir.glob("*.txt"):
                self.evaluate_summary_file(ablation_name, summary_file, source_sents, source_text)

        # Save JSON and CSV results
        self._write_results()


    # Save Results
    def _write_results(self):
        """Write evaluation results to JSON and CSV."""
        json_path = self.results_dir / "evaluation_results.json"
        csv_path = self.results_dir / "evaluation_results.csv"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(self.results, jf, indent=4)

        # Save CSV
        with open(csv_path, "w", newline='', encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=["ablation_type", "method", "coverage", "overlap", "importance_retention"])
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"Evaluation complete.")
        print(f"JSON results saved to: {json_path}")
        print(f"CSV results saved to: {csv_path}")


