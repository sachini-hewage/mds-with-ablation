import numpy as np
import hdbscan


class HDBSCANParagraphClusterer:
    """
    Cluster paragraph embeddings using HDBSCAN with a cosine distance metric.

    This class performs clustering on precomputed paragraph embeddings and assigns
    outlier paragraphs (those not assigned to any cluster by HDBSCAN) to the closest
    cluster center based on cosine similarity.

    Steps:
        1. Flatten all paragraphs from all documents into a single list of paragraphs
           and corresponding embeddings.
        2. Normalize embeddings to unit vectors.
        3. Compute the pairwise cosine distance matrix.
        4. Perform HDBSCAN clustering using the precomputed distance matrix.
        5. Organize paragraphs into clusters, preserving metadata (doc_id, para_id, text).
        6. Compute the centroid (mean embedding) for each cluster.
        7. Assign outliers to the closest cluster center using cosine similarity.
        8. Return a list of clusters with cluster IDs and associated paragraphs.
    """

    def __init__(self, docs, min_cluster_size=2, min_samples=1):
        """
        Initialize the clusterer.

        Args:
            docs (list): List of documents, where each document has a `paragraphs` attribute.
                         Each paragraph must have `embedding`, `doc_id`, `para_id`, and `text`.
            min_cluster_size (int): Minimum size of a cluster. Passed to HDBSCAN.
            min_samples (int): Minimum samples parameter for HDBSCAN.
        """
        self.docs = docs
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster(self):
        """
        Perform clustering on paragraph embeddings and assign outliers to the closest cluster.

        Returns:
            list: List of clusters, where each cluster is a dict:
                {
                    "cluster_id": int,
                    "paragraphs": list of dicts with "metadata_tag" and "text"
                }
        """
        # Flatten paragraphs and collect embeddings
        paragraphs = []
        embeddings = []
        for doc in self.docs:
            for para in doc.paragraphs:
                if para.embedding is not None:
                    paragraphs.append(para)
                    embeddings.append(para.embedding)

        # Return empty if no embeddings available
        if len(embeddings) == 0:
            return []

        embeddings = np.array(embeddings, dtype=np.float64)

        #Step 2: Normalize embeddings
        # Note to self: Normalization ensures that cosine similarity is equivalent to the dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        #Compute cosine distance matrix
        distance_matrix = 1 - np.dot(embeddings, embeddings.T)
        distance_matrix = distance_matrix.astype(np.float64)

        #HDBSCAN clustering with precomputed distance
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        labels = clusterer.fit_predict(distance_matrix)

        #Organize clusters and preserve metadata
        clusters = {}
        for idx, label in enumerate(labels):
            para = paragraphs[idx]
            if label == -1:
                # Outliers will be handled later
                continue
            if label not in clusters:
                clusters[label] = {"center": None, "paragraphs": []}
            clusters[label]["paragraphs"].append({
                "metadata_tag": f"{para.doc_id}_{para.para_id}",
                "text": para.text
            })

        # Compute cluster centers (mean embedding of cluster)
        for label, cluster in clusters.items():
            cluster_embs = np.array(
                [paragraphs[idx].embedding for idx, lbl in enumerate(labels) if lbl == label],
                dtype=np.float64
            )
            cluster["center"] = np.mean(cluster_embs, axis=0)

        # Assign outliers to the closest cluster
        for idx, label in enumerate(labels):
            if label != -1:
                continue  # Skip already clustered paragraphs
            para = paragraphs[idx]
            best_label = None
            best_sim = -1  # Cosine similarity ranges from -1 to 1
            for lbl, cluster in clusters.items():
                sim = self._cosine_sim(para.embedding, cluster["center"])
                if sim > best_sim:
                    best_sim = sim
                    best_label = lbl
            if best_label is not None:
                clusters[best_label]["paragraphs"].append({
                    "metadata_tag": f"{para.doc_id}_{para.para_id}",
                    "text": para.text
                })

        #Convert cluster dictionary to list format
        result = []
        for lbl, cluster in clusters.items():
            result.append({
                "cluster_id": int(lbl),
                "paragraphs": cluster["paragraphs"]
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
