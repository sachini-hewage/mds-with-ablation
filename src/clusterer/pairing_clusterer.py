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

    def __init__(self, docs, original_lookup=None):
        """
        Initialize the PairingClusterer.

        Args:
            docs (list): List of documents, each with a `paragraphs` attribute.
                         Each paragraph must have `embedding`, `doc_id`, `para_id`, and `text`.
            original_lookup (dict, optional): Mapping from "docid_paraid" → original paragraph text
                                              (from baseline data). If None, uses current text.
        """
        self.docs = docs
        self.original_lookup = original_lookup or {}

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
        # --- Step 1: Select base document (longest one) ---
        base_doc = max(self.docs, key=lambda d: len(d.paragraphs))
        other_docs = [d for d in self.docs if d != base_doc]

        base_paras = base_doc.paragraphs
        non_base_paras = [p for doc in other_docs for p in doc.paragraphs]

        # --- Step 2: Compute cosine similarity matrix ---
        sim_matrix = np.zeros((len(base_paras), len(non_base_paras)))
        for i, b in enumerate(base_paras):
            for j, nb in enumerate(non_base_paras):
                sim_matrix[i, j] = self._cosine_sim(b.embedding, nb.embedding)

        # --- Step 3: Perform bipartite matching for global best matches ---
        graph = csr_matrix(sim_matrix)
        match = maximum_bipartite_matching(graph, perm_type='column')

        pairs = []
        matched_bases = set()
        matched_non_bases = set()

        # --- Step 4: Assign matched pairs ---
        for j, i in enumerate(match):
            if i != -1:
                base_para = base_paras[i]
                counterpart = non_base_paras[j]
                matched_bases.add(i)
                matched_non_bases.add(j)

                base_tag = f"{base_para.doc_id}_{base_para.para_id}"
                counterpart_tag = f"{counterpart.doc_id}_{counterpart.para_id}"

                pairs.append({
                    "base": {
                        "doc_id": base_para.doc_id,
                        "para_id": base_para.para_id,
                        "metadata_tag": base_tag,
                        "text": self._get_original_text(base_tag, base_para.text)
                    },
                    "counterpart": {
                        "metadata_tag": counterpart_tag,
                        "text": self._get_original_text(counterpart_tag, counterpart.text)
                    },
                    "aux": []
                })

        # --- Step 5: Collect unmatched paragraphs ---
        leftovers = [p for j, p in enumerate(non_base_paras) if j not in matched_non_bases]
        leftovers += [p for i, p in enumerate(base_paras) if i not in matched_bases]

        # --- Step 6: Attach leftovers to closest matched base paragraph ---
        base_with_counterpart_indices = list(range(len(pairs)))

        for para in leftovers:
            best_idx = None
            best_score = -1
            for idx in base_with_counterpart_indices:
                base_para = base_paras[idx]
                sim = self._cosine_sim(base_para.embedding, para.embedding)
                if sim > best_score:
                    best_score = sim
                    best_idx = idx

            tag = f"{para.doc_id}_{para.para_id}"
            pairs[best_idx]["aux"].append({
                "metadata_tag": tag,
                "text": self._get_original_text(tag, para.text)
            })

        return pairs

    def _get_original_text(self, tag, fallback_text):
        """
        Replace paragraph text with baseline/original version if available.
        """
        return self.original_lookup.get(tag, fallback_text)

    def _cosine_sim(self, a, b):
        """
        Compute cosine similarity between two vectors safely.
        """
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
