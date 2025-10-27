import numpy as np
import hdbscan


class HDBSCANSentenceClusterer:
    """
    Cluster sentence embeddings using HDBSCAN with explicit cosine distance,
    while keeping outliers as cluster -1 and preserving original text.
    """

    def __init__(self, docs, original_lookup=None, min_cluster_size=2, min_samples=1):
        self.docs = docs
        self.original_lookup = original_lookup or {}
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(self):
        # Flatten sentences and collect embeddings
        sentences = []
        embeddings = []
        for doc in self.docs:
            for para in doc.paragraphs:
                for sent in para.sentences:
                    if sent.embedding is not None:
                        sentences.append(sent)
                        embeddings.append(sent.embedding)

        if len(embeddings) == 0:
            return []

        embeddings = np.array(embeddings, dtype=np.float64)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        # Compute cosine distance matrix
        distance_matrix = 1 - np.dot(embeddings, embeddings.T)
        distance_matrix = distance_matrix.astype(np.float64)

        # HDBSCAN clustering with precomputed distance
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Organize clusters
        clusters = {}
        for idx, label in enumerate(labels):
            sent = sentences[idx]
            tag = f"{sent.doc_id}_{sent.para_id}_{sent.sent_id}"
            text = self._get_original_text(tag, sent.text)

            # Initialize cluster
            if label not in clusters:
                clusters[label] = {"sentences": []}

            clusters[label]["sentences"].append({
                "metadata_tag": tag,
                "text": text
            })

        # Convert clusters dict to list
        result = []
        for lbl, cluster in clusters.items():
            result.append({
                "cluster_id": int(lbl),  # HDBSCAN uses -1 for outliers
                "sentences": cluster["sentences"]
            })

        return result

    def _cosine_sim(self, a, b):
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _get_original_text(self, tag, fallback_text):
        """Retrieve original sentence text from lookup if available."""
        return self.original_lookup.get(tag, fallback_text)
