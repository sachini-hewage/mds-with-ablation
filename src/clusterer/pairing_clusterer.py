import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

class PairingClusterer:
    """
    Cluster paragraphs across multiple documents using global best-match assignment.

    This class pairs paragraphs from a base document (the longest one) with the most
    similar paragraphs from other documents using a maximum bipartite matching.
    Remaining paragraphs are attached as auxiliary to the closest matched base paragraph.

    Steps:
        1. Choose the longest document as the base document.
        2. Compute a cosine similarity matrix between base and non-base paragraphs.
        3. Assign globally most similar non-base paragraphs as counterparts.
           Each non-base paragraph is used only once.
        4. Base paragraphs without counterparts are added as auxiliary to the closest
           base paragraph that has a counterpart.
        5. All leftover non-base paragraphs are added as auxiliary to the closest
           base paragraph with a counterpart.
        6. Return pairs with metadata for base, counterpart, and auxiliary paragraphs.
    """

    def __init__(self, docs):
        """
        Initialize the PairingClusterer.

        Args:
            docs (list): List of documents, each with a `paragraphs` attribute.
                         Each paragraph must have `embedding`, `doc_id`, `para_id`, and `text`.
        """
        self.docs = docs

    def pair(self):
        """
        Perform global best-match pairing of paragraphs.

        Returns:
            list: List of dicts containing:
                {
                    "base": dict with base paragraph info,
                    "counterpart": dict with matched paragraph info,
                    "aux": list of dicts for auxiliary paragraphs
                }
        """
        # Select base document (I choose the longest one)
        base_doc = max(self.docs, key=lambda d: len(d.paragraphs))
        other_docs = [d for d in self.docs if d != base_doc]

        base_paras = base_doc.paragraphs
        non_base_paras = [p for doc in other_docs for p in doc.paragraphs]

        # Compute similarity matrix
        sim_matrix = np.zeros((len(base_paras), len(non_base_paras)))
        for i, b in enumerate(base_paras):
            for j, nb in enumerate(non_base_paras):
                sim_matrix[i, j] = self._cosine_sim(b.embedding, nb.embedding)

        # Finds the best one-to-one matches between base and non-base paragraphs using
        # bipartite matching
        graph = csr_matrix(sim_matrix)
        match = maximum_bipartite_matching(graph, perm_type='column')

        pairs = []
        matched_bases = set()
        matched_non_bases = set()

        # Assign counterparts according to global best-match
        for j, i in enumerate(match):
            if i != -1:
                base_para = base_paras[i]
                counterpart = non_base_paras[j]
                matched_bases.add(i)
                matched_non_bases.add(j)

                pairs.append({
                    "base": {
                        "doc_id": base_para.doc_id,
                        "para_id": base_para.para_id,
                        "metadata_tag": f"{base_para.doc_id}_{base_para.para_id}",
                        "text": base_para.text
                    },
                    "counterpart": {
                        "metadata_tag": f"{counterpart.doc_id}_{counterpart.para_id}",
                        "text": counterpart.text
                    },
                    "aux": []  # initialize empty auxiliary list
                })

        # Collect leftover paragraphs
        leftovers = []

        # Unmatched non-base paragraphs
        for j, para in enumerate(non_base_paras):
            if j not in matched_non_bases:
                leftovers.append(para)

        # Unmatched base paragraphs
        for i, para in enumerate(base_paras):
            if i not in matched_bases:
                leftovers.append(para)

        # Attach each leftover to the closest base with a counterpart
        base_with_counterpart_indices = [pairs.index(p) for p in pairs if p["counterpart"] is not None]

        for para in leftovers:
            best_idx = None
            best_score = -1
            for idx in base_with_counterpart_indices:
                base_para = base_paras[idx]
                sim = self._cosine_sim(base_para.embedding, para.embedding)
                if sim > best_score:
                    best_score = sim
                    best_idx = idx

            # Attach leftover paragraph as auxiliary
            pairs[best_idx]["aux"].append({
                "metadata_tag": f"{para.doc_id}_{para.para_id}",
                "text": para.text
            })

        return pairs

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
