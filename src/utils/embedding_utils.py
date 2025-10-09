import os
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Embedder class for generating and caching embeddings for documents, paragraphs, and sentences
    using a SentenceTransformer model.

    Features:
      - Encode text into vector embeddings
      - Cache embeddings to disk (npz files) for reuse
      - Load cached embeddings back into document objects
      - Handles both paragraph-level and sentence-level embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a SentenceTransformer model.

        Args:
            model_name (str): Pretrained sentence-transformer model name
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encode a list of texts into embeddings.

        Args:
            texts (list[str]): List of text strings to encode

        Returns:
            np.ndarray: Array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=False)

    def save_embeddings(self, docs, cache_path: str):
        """
        Save paragraph and sentence embeddings into a single .npz file with string keys.

        Args:
            docs (list): List of document objects containing paragraphs and sentences
            cache_path (str): Path to save the cached embeddings
        """
        embeddings = {}

        for doc in docs:
            for para in doc.paragraphs:
                if para.embedding is not None:
                    key = f"{para.doc_id}_p{para.para_id}"
                    embeddings[key] = para.embedding
                for sent in para.sentences:
                    if sent.embedding is not None:
                        embeddings[sent.meta_tag()] = sent.embedding

        np.savez(cache_path, **embeddings)
        print(f"[Embedder] Saved embeddings to {cache_path}")

    def load_embeddings(self, docs, cache_path: str):
        """
        Load cached embeddings from a .npz file and assign them back to documents.

        Args:
            docs (list): List of document objects
            cache_path (str): Path to the cached embeddings file
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"No embeddings cache found at {cache_path}")

        data = np.load(cache_path, allow_pickle=True)

        for doc in docs:
            for para in doc.paragraphs:
                key = f"{para.doc_id}_p{para.para_id}"
                if key in data:
                    para.embedding = data[key]
                for sent in para.sentences:
                    if sent.meta_tag() in data:
                        sent.embedding = data[sent.meta_tag()]

        print(f"[Embedder] Loaded embeddings from {cache_path}")

    def embed_documents(self, docs, cache_path: str):
        """
        Compute embeddings for documents if cache not found, otherwise load from cache.
        Embeds both paragraphs and sentences.

        Args:
            docs (list): List of document objects
            cache_path (str): Path to cache or load embeddings
        """
        try:
            # Try loading precomputed embeddings from cache
            self.load_embeddings(docs, cache_path)
        except FileNotFoundError:
            print(f"[Embedder] No cache found at {cache_path}. Computing embeddings...")

            # Compute paragraph embeddings
            for doc in docs:
                para_texts = [p.text for p in doc.paragraphs]
                para_embeddings = self.encode(para_texts)
                for para, emb in zip(doc.paragraphs, para_embeddings):
                    para.embedding = emb

                # Compute sentence embeddings for each paragraph
                for para in doc.paragraphs:
                    sent_texts = [s.text for s in para.sentences]
                    sent_embeddings = self.encode(sent_texts)
                    for sent, emb in zip(para.sentences, sent_embeddings):
                        sent.embedding = emb

            # Save computed embeddings to cache
            self.save_embeddings(docs, cache_path)
