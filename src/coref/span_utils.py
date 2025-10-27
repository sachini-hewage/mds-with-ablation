from typing import List, Tuple
from spacy.tokens import Doc, Span


def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
  """
  This function returns the indices of spans in a cluster that contain nouns or
  proper nouns

  doc: parsed NLP representation from Spacy
  cluster: list of spans in a cluster

  Returns the indices of spans in a cluster that contain nouns or proper nouns.
  """
  # Get the indices from the doc for each of the given spans in cluster to a
  # list
  spans = [doc[span[0]:span[1]+1] for span in cluster]

  # Get the part of speech type for each token in the span
  spans_pos = [[token.pos_ for token in span] for span in spans]

  # If any of the span_pos in any span is either a noun or a pronoun
  # then add the index of that span_pos
  span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
      if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]

  # Return the indices of spans in a cluster that contains a noun or proper noun
  return span_noun_indices




def get_cluster_head_concise_latest(doc: Doc, cluster: List[List[int]],
                                      noun_indices: List[int]) -> Tuple[Span, List[int]]:
    """
    Selects the best head span for a coreference cluster, trimming descriptors and modifiers.

    Strategies :  Extract the first noun span containing a proper noun (full_span)
                  If the root of the full_span is a proper noun, return only the PROPN tokens
                  If the root of the full_span is a proper noun, but it is part of a coordination
                  return the whole full_span without trimming
                  If the root of the full_span is not a proper noun, return the full_span

        doc (Doc): spaCy Doc object
        cluster (List[List[int]]): List of [start, end] indices for mentions (inclusive)
        noun_indices (List[int]): Indices into cluster that contain noun spans

    Returns Tuple[Span, List[int]]: Cleaned head span and its [start, end] indices (inclusive)

    """
    if not noun_indices:
        raise ValueError("noun_indices is empty; can't select a head span.")

    # Step 1: Pick the first noun span that contains at least one proper noun (PROPN)
    # Example: For mentions ["the man", "John Smith", "he"], this would pick "John Smith"
    head_idx = None
    for idx in noun_indices:
        start, end = cluster[idx]
        span = doc[start:end + 1]
        if any(tok.pos_ == "PROPN" for tok in span):
            head_idx = idx
            break

    # If no proper noun spans found, fall back to the first noun span
    if head_idx is None:
        head_idx = noun_indices[0]

    head_start, head_end = cluster[head_idx]
    full_span = doc[head_start:head_end + 1]
    #print("\n")
    #print("full_span:", full_span)
    span_root = full_span.root
    #print("span_root:", span_root)

    # Case 1: If the root of the span is a proper noun, return only the PROPN tokens
    # Example: "beautiful Mary" -> return "Mary"
    if span_root.pos_ == "PROPN":
    # If root is part of a coordination (conjunct), do not trim
    # Example: "Mary and her prosecutors" => "Mary and her prosecutors"
      if any(child.dep_ == "cc" or child.dep_ == "conj" for child in span_root.children):
          #print("Root is PROPN but part of coordination, keeping full span:", full_span)
          return full_span, [head_start, head_end]

      # Otherwise, it is safe to trim and pick just the proper noun
      # Example : Campbell, who was at his home on the Hawaiian island of Kauai => Campbell
      start = span_root.i
      end = span_root.i

      # Expand left
      while start > 0 and doc[start - 1].pos_ == "PROPN":
          start -= 1

      # Expand right
      while end + 1 < len(doc) and doc[end + 1].pos_ == "PROPN":
          end += 1

      #print("Root is a proper noun (refined):", doc[start:end + 1])
      return doc[start:end + 1], [start, end]



    # Case 2: If there's an appositive proper noun, extract it
    # Example: "Donald Tuck, the estranged husband" -> return "Donald Tuck"
    # for tok in full_span:
    #     if tok.dep_ == "appos" and tok.pos_ == "PROPN":
    #         appos_start = tok.i
    #         appos_end = tok.i

    #         # Expand left: include first names or titles (e.g., "Donald")
    #         while appos_start > 0 and doc[appos_start - 1].pos_ == "PROPN":
    #             appos_start -= 1

    #         # Expand right: include last names or middle names (e.g., "Tuck")
    #         while appos_end + 1 < len(doc) and doc[appos_end + 1].pos_ == "PROPN":
    #             appos_end += 1
    #         print("Appositive extraction: ", doc[appos_start:appos_end + 1] )
    #         return doc[appos_start:appos_end + 1], [appos_start, appos_end]


    # Case 2: If there are multiple appositive proper nouns, extract it
    # proper_noun_spans = []  # collect (start, end) for each proper noun group

    # for tok in full_span:
    #     if tok.dep_ == "appos" and tok.pos_ == "PROPN":
    #         appos_start = tok.i
    #         appos_end = tok.i

    #         # Expand left: include first names or titles (e.g., "Donald")
    #         while appos_start > 0 and doc[appos_start - 1].pos_ == "PROPN":
    #             appos_start -= 1

    #         # Expand right: include last names or middle names (e.g., "Tuck")
    #         while appos_end + 1 < len(doc) and doc[appos_end + 1].pos_ == "PROPN":
    #             appos_end += 1

    #         # Save this group (start, end)
    #         proper_noun_spans.append((appos_start, appos_end))

    # # After collecting all proper noun spans
    # if proper_noun_spans:
    #     # Build each proper noun phrase separately
    #     parts = []
    #     for start, end in proper_noun_spans:
    #         parts.append(doc[start:end + 1].text)

    #     # Join with commas
    #     final_text = ", ".join(parts)
    #     print("Comma-separated Appositives: ", final_text)

    #     # NOTE: Returning text now instead of Span
    #     return final_text,[0,0]


    # Case 3: Fallback  - when the root is not a proper noun
    # Example: "Monday's storm" -> return "Monday's storm"
    #print("Fallback: ", full_span)
    return full_span, [head_start, head_end]




def is_containing_other_spans(span: List[int], all_spans: List[List[int]])-> bool:
  """
  This function checks if a given span contains another span

  span : start and end indices of a span
  all_spans: list of start and end indices of all spans in a document

  Returns True if the given span contains any other span in all_spans
  """

  # Check is there are any spans in all_spans that start after or at the
  # beginning of a given span and ends before or at the given span which is
  # not identical to the given span itself
  return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])





def get_fast_cluster_spans(doc: Doc, clusters: List[List[int]]) -> List[List[List[int]]]:
    """
    This function converts the character spans returned by fastcoref into token
    spans
    doc: Doc object
    clusters: coreference spans from fastcoref
    Returns list of equivalent token spans from doc
    """
    # Container to hold token spans
    fast_clusters = []

    # For each cluster in document create a container to hold token spans
    for cluster in clusters:
        new_group = []

        # For each span in the cluster convert to the equivalent token span
        # from the doc and append to new_group container for each character span
        for indices in cluster:
            #print(type(tuple), tuple)
            (start, end) = indices
            #print("start, end", start, end)
            span = doc.char_span(start, end)
            #print('span', span.start, span.end)
            new_group.append([span.start, span.end-1])

        # Append each resulting token span to the bigger container
        fast_clusters.append(new_group)

    return fast_clusters


