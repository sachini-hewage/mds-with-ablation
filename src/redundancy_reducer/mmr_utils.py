# src/redundancy_reducer/mmr_utils.py
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pathlib import Path


class MMRReducer:
    """
    Reduces redundancy in summaries using Maximal Marginal Relevance (MMR).

    MMR balances:
      - Relevance: how well a sentence matches the document's overall meaning.
      - Diversity: how different a sentence is from already selected ones.

    This variant drops sentences whose MMR score falls below a given threshold,
    rather than keeping a fixed number of sentences because I do not want to limit
    by sentence count.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", lambda_param=0.8, mmr_threshold=0.1):
        """
        Args:
            model_name (str): Name of the SentenceTransformer model to use for embeddings.
            lambda_param (float): Trade-off parameter between relevance and diversity.
                                 - Close to 1 → prioritize relevance.
                                 - Close to 0 → prioritize diversity.
            mmr_threshold (float): Minimum MMR score required for a sentence to be retained.
                                   Sentences with MMR below this value are dropped.
        """
        self.model = SentenceTransformer(model_name)
        self.lambda_param = lambda_param
        self.mmr_threshold = mmr_threshold

    def _mmr(self, doc_embedding, sentence_embeddings, sentences):
        """
        Core MMR selection loop:
        Iteratively selects sentences that maximize the MMR score:
            MMR = λ * sim(sentence, doc) - (1 - λ) * max(sim(sentence, selected))

        Sentences with MMR < threshold are discarded.

        Args:
            doc_embedding (np.ndarray): Embedding representing the document (mean of sentence embeddings).
            sentence_embeddings (np.ndarray): Embeddings of each sentence.
            sentences (list[str]): List of candidate sentences.

        Returns:
            list[str]: Filtered list of sentences that exceed the MMR threshold.
        """
        selected = []
        selected_indices = []
        mmr_scores_dict = {}

        # Compute similarity of each sentence to the document
        doc_sim = util.cos_sim(sentence_embeddings, doc_embedding)

        # Select the most relevant sentence first
        first_idx = np.argmax(doc_sim)
        selected.append(sentences[first_idx])
        selected_indices.append(first_idx)
        mmr_scores_dict[first_idx] = float(doc_sim[first_idx])

        # Iteratively compute MMR for remaining sentences
        while len(selected_indices) < len(sentences):
            remaining = [i for i in range(len(sentences)) if i not in selected_indices]
            if not remaining:
                break

            mmr_scores = []
            for i in remaining:
                # Relevance: similarity to the document
                relevance = doc_sim[i]

                # Redundancy: similarity to already selected sentences
                redundancy = max(
                    util.cos_sim(sentence_embeddings[i], sentence_embeddings[j]).item()
                    for j in selected_indices
                )

                # Compute MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy
                mmr_scores.append((i, mmr_score))

            # Select the next highest scoring sentence
            best_idx, best_score = sorted(mmr_scores, key=lambda x: x[1], reverse=True)[0]
            mmr_scores_dict[best_idx] = float(best_score)

            # Add only if score exceeds the threshold
            if best_score >= self.mmr_threshold:
                selected_indices.append(best_idx)
                selected.append(sentences[best_idx])
            else:
                # Stop early if remaining sentences fall below threshold
                break

        # Sort back by original order for readability
        ordered_selected = [s for _, s in sorted(zip(selected_indices, selected))]
        return ordered_selected

    def reduce_summary(self, summary_text: str) -> str:
        """
        Apply MMR redundancy reduction to a single summary text.

        Args:
            summary_text (str): The full summary (possibly with redundancy).

        Returns:
            str: A rewritten summary containing the most relevant and least redundant sentences.
        """
        # Split text into sentences (simple split by '.')
        sentences = [s.strip() for s in summary_text.split('.') if s.strip()]

        # Skip if summary is already very short
        if len(sentences) <= 2:
            return summary_text

        # Encode each sentence and compute a document-level mean embedding
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        doc_embedding = np.mean(embeddings, axis=0)

        # Select sentences via MMR filtering
        selected_sentences = self._mmr(doc_embedding, embeddings, sentences)

        # Return concatenated text preserving ranking order
        return '. '.join(selected_sentences) + '.'

    def reduce_in_directory(self, summaries_dir: Path):
        """
        Apply MMR redundancy reduction to all summaries (*.txt) in a directory.

        Args:
            summaries_dir (Path): Path to a directory containing generated summary text files.
        """
        for summary_file in summaries_dir.glob("*.txt"):
            with open(summary_file, "r", encoding="utf-8") as f:
                text = f.read().strip()

            reduced_text = self.reduce_summary(text)

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(reduced_text)

            print(f"Reduced redundancy in {summary_file.name}")
