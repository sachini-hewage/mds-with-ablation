import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


class PairingClusterer:
    """
    Cluster paragraphs across multiple documents using global best-match assignment.

    This class pairs paragraphs from a base document (the longest one) with the most
    similar paragraphs from other documents using maximum bipartite matching.
    Remaining paragraphs are attached as auxiliary to the closest matched base paragraph
    or grouped into a pseudo cluster if no strong match exists.

    Steps:
        1. Select the longest document as the base.
        2. Compute a cosine similarity matrix between base and non-base paragraphs.
        3. Assign globally most similar (but above a similarity threshold) non-base paragraphs as counterparts.
           Each non-base paragraph is used only once.
        4. Attach unmatched paragraphs above a similarity threshold as auxiliary.
        5. Remaining low-similarity paragraphs form a pseudo cluster.
        6. Return structured pairs with metadata for base, counterpart, and auxiliary paragraphs.
    """

    def __init__(self, docs, original_lookup=None):
        """
        Initialize the PairingClusterer.

        Args:
            docs (list): List of documents, each with a 'paragraphs' attribute.
                         Each paragraph must have 'embedding', 'doc_id', 'para_id', and 'text'.
            original_lookup (dict, optional): Mapping from "docid_paraid" → original paragraph text
                                              (from baseline data). If None, uses current text.
        """
        self.docs = docs
        self.original_lookup = original_lookup or {}

    def pair(self):
        """
        Perform safe global thresholded paragraph pairing across multiple documents.

        This version handles all edge cases safely:
            - No base-counterpart matches
            - All leftover paragraphs attached as auxiliary if possible
            - Low-similarity leftovers grouped into a pseudo cluster

        Returns:
            list: List of pairing dicts containing:
                {
                    "base": dict or None,
                    "counterpart": dict or None,
                    "aux": list of dicts for auxiliary paragraphs,
                    "pseudo": optional set to True for pseudo clusters
                }
        """

        # Select base document
        base_doc = max(self.docs, key=lambda d: len(d.paragraphs))
        other_docs = [d for d in self.docs if d != base_doc]

        base_paras = base_doc.paragraphs
        non_base_paras = [p for doc in other_docs for p in doc.paragraphs]

    
        if not base_paras and not non_base_paras:
            return []

        # Compute cosine similarity matrix
        sim_matrix = np.zeros((len(base_paras), len(non_base_paras)))
        for i, b in enumerate(base_paras):
            for j, nb in enumerate(non_base_paras):
                sim_matrix[i, j] = self._cosine_sim(b.embedding, nb.embedding)

        # Apply threshold and perform bipartite matching
        threshold = 0.75
        masked_sim = sim_matrix.copy()
        masked_sim[masked_sim < threshold] = 0.0

        if masked_sim.size == 0:
            match = np.array([], dtype=int)
        else:
            graph = csr_matrix(masked_sim)
            match = maximum_bipartite_matching(graph, perm_type='column')


        pairs = []
        matched_bases = set()
        matched_non_bases = set()
        pair_to_base_index = []

        # Assign matched pairs safely
        for j in range(len(non_base_paras)):
            if j >= len(match):
                continue  # skip safely if match array shorter than non_base_paras

            i = match[j]
            if i == -1 or i >= len(base_paras):
                continue  # skip invalid indices

            base_para = base_paras[i]
            counterpart = non_base_paras[j]
            matched_bases.add(i)
            matched_non_bases.add(j)
            pair_to_base_index.append(i)

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


        # Collect leftover paragraphs
        leftover_nonbase = [p for j, p in enumerate(non_base_paras) if j not in matched_non_bases]
        leftover_base = [p for i, p in enumerate(base_paras) if i not in matched_bases]
        leftovers = leftover_nonbase + leftover_base
        print(f"[DEBUG] Leftover paragraphs count: {len(leftovers)}")

        aux_threshold = 0.60
        low_similarity_leftovers = []

        if pairs:
            # Attach leftovers above aux_threshold to the closest matched base
            for para in leftovers:
                best_idx = None
                best_score = -1.0
                for idx, base_idx in enumerate(pair_to_base_index):
                    base_para = base_paras[base_idx]
                    sim = self._cosine_sim(base_para.embedding, para.embedding)
                    if sim > best_score:
                        best_score = sim
                        best_idx = idx

                tag = f"{para.doc_id}_{para.para_id}"
                if best_idx is not None and best_score >= aux_threshold:
                    pairs[best_idx]["aux"].append({
                        "metadata_tag": tag,
                        "text": self._get_original_text(tag, para.text)
                    })
                else:
                    low_similarity_leftovers.append({
                        "metadata_tag": tag,
                        "text": self._get_original_text(tag, para.text)
                    })
        else:
            # No matched pairs → all leftovers become pseudo cluster
            for para in leftovers:
                tag = f"{para.doc_id}_{para.para_id}"
                low_similarity_leftovers.append({
                    "metadata_tag": tag,
                    "text": self._get_original_text(tag, para.text)
                })

        # Attach all low-similarity leftovers as a pseudo cluster
        if low_similarity_leftovers:
            pairs.append({
                "base": None,
                "counterpart": None,
                "aux": low_similarity_leftovers,
                "pseudo": True
            })

        return pairs

    def _get_original_text(self, tag, fallback_text):
        """
        Replace paragraph text with baseline/original version if available.

        Args:
            tag (str): Paragraph metadata tag.
            fallback_text (str): Text to use if no original is found.

        Returns:
            str: Original or fallback paragraph text.
        """
        return self.original_lookup.get(tag, fallback_text)

    def _cosine_sim(self, a, b):
        """
        Compute cosine similarity between two vectors safely.

        Args:
            a (np.ndarray): Vector a.
            b (np.ndarray): Vector b.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
