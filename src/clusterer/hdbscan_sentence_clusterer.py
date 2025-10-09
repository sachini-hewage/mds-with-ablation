import numpy as np
import hdbscan

class HDBSCANSentenceClusterer:
    """
    Cluster sentence embeddings using HDBSCAN with explicit cosine distance.

    This class performs clustering on precomputed sentence embeddings and assigns
    outlier sentences  to the closest cluster center based on cosine similarity.

    Steps:
        1. Flatten all sentences from all documents into a single list of sentences
           and corresponding embeddings.
        2. Normalize embeddings to unit vectors.
        3. Compute the pairwise cosine distance matrix.
        4. Perform HDBSCAN clustering using the precomputed distance matrix.
        5. Organize sentences into clusters, preserving metadata (doc_id, para_id, sent_id, text).
        6. Compute the centroid (mean embedding) for each cluster.
        7. Assign outliers to the closest cluster center using cosine similarity.
        8. Return a list of clusters with cluster IDs and associated sentences.
    """

    def __init__(self, docs, min_cluster_size=2, min_samples=1):
        """
        Initialize the sentence clusterer.

        Args:
            docs (list): List of documents, where each document has a `paragraphs` attribute,
                         and each paragraph has a `sentences` attribute.
                         Each sentence must have `embedding`, `doc_id`, `para_id`, `sent_id`, and `text`.
            min_cluster_size (int): Minimum size of a cluster. Passed to HDBSCAN.
            min_samples (int): Minimum samples parameter for HDBSCAN.
        """
        self.docs = docs
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(self):
        """
        Perform clustering on sentence embeddings and assign outliers to the closest cluster.

        Returns:
            list: List of clusters, where each cluster is a dict:
                {
                    "cluster_id": int,
                    "sentences": list of dicts with "metadata_tag" and "text"
                }
        """
        # Flatten sentences and collect embeddings
        sentences = []
        embeddings = []
        for doc in self.docs:
            for para in doc.paragraphs:
                for sent in para.sentences:
                    if sent.embedding is not None:
                        sentences.append(sent)
                        embeddings.append(sent.embedding)

        # Return empty list if no embeddings are available
        if len(embeddings) == 0:
            return []

        embeddings = np.array(embeddings, dtype=np.float64)

        # Normalize embeddings
        # Normalization ensures cosine similarity equals dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        # Compute cosine distance matrix
        distance_matrix = 1 - np.dot(embeddings, embeddings.T)
        distance_matrix = distance_matrix.astype(np.float64)  # ensure float64

        # HDBSCAN clustering with precomputed distance
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Organize clusters with metadata
        clusters = {}
        for idx, label in enumerate(labels):
            sent = sentences[idx]
            if label == -1:
                # Outliers will be assigned later
                continue
            if label not in clusters:
                clusters[label] = {"center": None, "sentences": []}
            clusters[label]["sentences"].append({
                "metadata_tag": f"{sent.doc_id}_{sent.para_id}_{sent.sent_id}",
                "text": sent.text
            })

        # Compute cluster centers (mean embedding of cluster)
        for label, cluster in clusters.items():
            cluster_embs = np.array(
                [sentences[idx].embedding for idx, lbl in enumerate(labels) if lbl == label],
                dtype=np.float64
            )
            cluster["center"] = np.mean(cluster_embs, axis=0)

        # Assign outliers to closest cluster
        for idx, label in enumerate(labels):
            if label != -1:
                continue  # Skip already clustered sentences
            sent = sentences[idx]
            best_label = None
            best_sim = -1
            for lbl, cluster in clusters.items():
                sim = self._cosine_sim(sent.embedding, cluster["center"])
                if sim > best_sim:
                    best_sim = sim
                    best_label = lbl
            if best_label is not None:
                clusters[best_label]["sentences"].append({
                    "metadata_tag": f"{sent.doc_id}_{sent.para_id}_{sent.sent_id}",
                    "text": sent.text
                })

        # Convert cluster dictionary to list format
        result = []
        for lbl, cluster in clusters.items():
            result.append({
                "cluster_id": int(lbl),
                "sentences": cluster["sentences"]
            })

        return result

    def _cosine_sim(self, a, b):
        """
        Compute cosine similarity between two vectors.

        Args:
            a (np.ndarray): Vector a.
            b (np.ndarray): Vector b.

        Returns:
            float: Cosine similarity between a and b.
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
