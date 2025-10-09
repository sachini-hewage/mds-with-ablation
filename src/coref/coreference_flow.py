from typing import List, Any

import spacy
from fastcoref import LingMessCoref
from spacy.tokens import Doc, Span

from .span_utils import get_span_noun_indices, get_fast_cluster_spans, get_cluster_head_concise_latest, \
    is_containing_other_spans



# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Instantiate FastCoreference model
#model = LingMessCoref(device= "mps")
model = LingMessCoref(device= "cpu")



def coreference(text: str)-> str:
    """
    text: text to be processed
    This function carries out coreference resolution for a given text
    """
    # Convert text to SpaCy tokens
    doc = nlp(text)

    # Create clusters for the text
    clusters = get_fastcoref_clusters(doc, text)
    # print(clusters)

    # for cluster in clusters:
    #   for start,end in cluster:
    #     print(doc[start:end+1], end=',')
    #   print("\n")

    # Coreference resolution
    count = 0
    coref_text,count_new = improved_replace_corefs(doc, clusters, count)
    #print("count", count_new)

    # print("Doc:", doc)
    # print("Clusters;", clusters)

    return coref_text





def improved_replace_corefs(document: Doc, clusters: List[List[List[int]]],count) -> tuple[str, Any]:
    """
    Resolves coreferences in a document by replacing only pronouns with a concise head noun phrase.


    document (Doc): spaCy Doc object.
    clusters (List[List[List[int]]]): Each cluster is a list of [start, end] index pairs (inclusive),
                                      representing coreference spans in the document.

    Returns a string with pronoun coreferences replaced by their resolved mentions.
    """
    # Start with the original tokens with whitespace preserved
    resolved = [tok.text_with_ws for tok in document]

    # Flatten all coreference spans for overlap checking
    all_spans = [span for cluster in clusters for span in cluster]

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            # Get the best head span and its [start, end] indices
            mention_span, mention = get_cluster_head_concise_latest(document, cluster, noun_indices)
            #print("document:", document)
            #print("mention: ", mention)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    coref_start, coref_end = coref
                    coref_span = document[coref_start:coref_end + 1]

                    # Only replace pronoun spans
                    if all(tok.pos_ == "PRON" for tok in coref_span):
                        #print("Pronoun coreference:", coref_span.text, "->", mention_span.text)
                        count+=1
                        core_logic_part_consice(document, coref, resolved, mention_span)

    return "".join(resolved), count




def get_fastcoref_clusters(doc, text)->List[List[List[int]]]:
  """
  This function returns a list of token spans of coreference clusters from
  fastcoref for a given text

  doc: Doc object
  text: text to be processed

  """
  # Prediction for a given text from FastCoReference model - 1D list
  preds = model.predict(texts=[text])

  # Check if preds is empty or if the first element is not a Prediction object
  if not preds or not hasattr(preds[0], 'get_clusters'):
    # Handle the case where no clusters were found
    #print("Warning: No coreference clusters found for the text.")
    return []

  # Get the cluster as a character span (not a token span) from the prediction
  fast_clusters = preds[0].get_clusters(as_strings=False)

  # get the relevant token span
  fast_cluster_spans = get_fast_cluster_spans(doc, fast_clusters)

  return fast_cluster_spans




def core_logic_part_consice(document: Doc, coref: List[int], resolved: List[str], mention_span: Span) -> List[str]:
    """
    Replaces coreference pronouns with the mention_span text.
    Handles "I am" / "I'm" -> "John is".
    Also handles "I have" / "I've" -> "John has".


    document: Doc object (which can be converted into a token list)
    coref: list of indices of the coreference span in the token list
    resolved: list of resolved tokens (originally contains tokens as-is)
    mention_span: Span object of the mention span (indxing in the document)

    Returns the updated resolved list with the coreference replaced.
    """
    final_token = document[coref[1]]
    mention_text = mention_span.text

    # Get the full text of the coref span
    coref_span_text = document[coref[0]:coref[1] + 1].text.lower()

    # If any token from the mention appears in the coref span, skip replacement
    # for tok in mention_span:
    #     if tok.text.lower() in coref_span_text:
    #         return resolved

    # Handle possessives
    if final_token.tag_ in ["PRP$", "POS"]:
        #print("Possessive pronoun :" ,resolved[coref[0]], mention_text + "'s" )
        resolved[coref[0]] = mention_text + "'s" + final_token.whitespace_

    else:
        #print("Non possessive pronoun :" ,resolved[coref[0]], mention_text )
        resolved[coref[0]] = mention_text + final_token.whitespace_

        # Handle "I am" or "I'm" -> "John is"
        if final_token.text.lower() == "i":
            next_index = coref[1] + 1
            if next_index < len(document):
                next_token = document[next_index]
                if next_token.text.lower() in ["am", "'m"]:
                    resolved[next_index] = " is" + next_token.whitespace_
                elif next_token.text.lower() in ["have", "'ve"]:
                    resolved[next_index] = " has" + next_token.whitespace_
                elif next_token.text.lower() in ["do"]:
                    resolved[next_index] = " does" + next_token.whitespace_

    # Blank out the rest of the coref span (except the replaced token)
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""

    return resolved


