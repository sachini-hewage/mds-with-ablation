from datasets import load_dataset
import spacy
from pathlib import Path
import json
import re
from src.utils.metadata_utils import Document, Paragraph, Sentence
from src.ner.ner_tagger import NERTagger
from src.coref.corefernce_resolver import CoreferenceResolver
from sentence_transformers import util
from src.summariser.summariser import Summariser






# Advertisement removal patterns
AD_PATTERNS = [
    r"(?i)Advertisement",
    r"(?i)Sign up for.*",
    r"(?i)Click .*",
    r"(?i)Read more.*",
    r"(?i)Follow us on.*",
    r"(?i)©.*",
    r"(?i)Read:.*"
]

def clean_advertisements(text: str) -> str:
    """
    Remove advertisement-like lines from a document based on predefined regex patterns.

    Args:
        text (str): Raw document text.

    Returns:
        str: Text with advertisement lines removed.
    """
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not any(re.search(p, line.strip()) for p in AD_PATTERNS)]
    return "\n".join(cleaned_lines).strip()

def split_paragraphs(text: str):
    """
    Split document text into paragraphs based on line breaks.

    Args:
        text (str): Document text.

    Returns:
        List[str]: Non-empty paragraphs.
    """
    return [p.strip() for p in text.split("\n") if p.strip()]

def preprocess_first_instance(
    mode: str = "baseline",
    out_dir: Path = Path("data/processed"),
    results_dir: Path = Path("data/results"),
    embedder=None,  # Embedder required for similarity-based filtering
    dissimilar_thresh: float = 0.3,  # Max similarity below this threshold = very dissimilar
):
    """
    Preprocess the first MultiNews instance with optional coreference and NER,
    while filtering out trivial sentences using embedding similarity.

    Steps:
        1. Remove advertisements from raw text.
        2. Optionally resolve coreferences.
        3. Optionally tag named entities.
        4. Split text into paragraphs and sentences.
        5. Identify meaningful sentences via embedding similarity:
           - Sentences whose maximum similarity to any other sentence in the paragraph
             is below `dissimilar_thresh` are considered very dissimilar (trivial).
        6. Save cleaned Document objects in JSONL.
        7. Save meaningful sentences to a combined text file.
        8. Debug-print very dissimilar sentences.

    Args:
        mode (str): Preprocessing mode, e.g., "baseline", "coref", "coref+ner".
        out_dir (Path): Directory to store processed JSONL documents.
        results_dir (Path): Directory to store combined meaningful sentences.
        embedder: SentenceTransformer embedder for semantic similarity.
        dissimilar_thresh (float): Threshold to detect very dissimilar sentences.
    """
    # Step 0: Prepare directories
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("multi_news", split="train[0:1]")
    instance = dataset[0]
    raw_texts = instance["document"].split("|||||")

    # Initialize NLP, NER, coref modules
    nlp = spacy.load("en_core_web_sm")
    ner_tagger = NERTagger() if mode == "coref+ner" else None
    resolver = CoreferenceResolver() if mode in ["coref", "coref+ner"] else None

    out_file = mode_dir / f"multi_doc_{mode}.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    combined_sentences_file = results_dir / "combined_source_sentences.txt"

    all_sentences = []

    # Remove old JSONL if exists
    if out_file.exists():
        out_file.unlink()

    with out_file.open("w", encoding="utf-8") as f:
        for doc_idx, raw_text in enumerate(raw_texts):
            doc_id = f"doc{doc_idx}"

            # Step 1: Clean advertisements
            cleaned_text = clean_advertisements(raw_text)
            if not cleaned_text.strip():
                print(f"Skipped {doc_id} (empty after ad removal)")
                continue

            # Step 2: Coreference resolution if needed
            resolved_doc_text = resolver.resolve(cleaned_text) if resolver else cleaned_text

            # Step 3: Process paragraphs and sentences
            paragraphs = []
            for p_idx, p_text in enumerate(split_paragraphs(resolved_doc_text)):
                spacy_p = nlp(p_text)
                sentences = []

                for s_idx, sent_span in enumerate(spacy_p.sents):
                    sent_text = sent_span.text.strip()
                    if not sent_text:
                        continue

                    sent_entities = []

                    # Step 3a: Optional NER tagging
                    if mode == "coref+ner":
                        sent_entities, sent_text_tagged = ner_tagger.tag_sentence(sent_text, embed_tags=True)
                    else:
                        sent_text_tagged = sent_text

                    # Create Sentence object
                    sent = Sentence(
                        text=sent_text_tagged,
                        doc_id=doc_id,
                        para_id=p_idx,
                        sent_id=s_idx,
                        resolved_text=sent_text_tagged if mode in ["coref", "coref+ner"] else None,
                        entities=sent_entities,
                    )
                    sentences.append(sent)

                if sentences:
                    # Step 3b: Create Paragraph object
                    para = Paragraph(
                        sentences=sentences,
                        doc_id=doc_id,
                        para_id=p_idx,
                        text=" ".join([s.text for s in sentences]),
                        resolved_text=" ".join([s.text for s in sentences]) if mode in ["coref", "coref+ner"] else None,
                    )
                    paragraphs.append(para)

            if not paragraphs:
                continue

            # Step 4: Save Document object to JSONL
            doc_obj = Document(doc_id=doc_id, paragraphs=paragraphs, raw_text=cleaned_text)
            f.write(json.dumps({"mode": mode, "document": doc_obj.__dict__}, default=lambda o: o.__dict__, ensure_ascii=False) + "\n")

    #         # Step 5: Identify meaningful and very dissimilar sentences
    #         for para_text in split_paragraphs(cleaned_text):
    #             spacy_p = nlp(para_text)
    #             para_sents = [s.text.strip() for s in spacy_p.sents if s.text.strip()]
    #
    #             meaningful_sents = []
    #             very_dissimilar_sents = []
    #
    #             if len(para_sents) > 1 and embedder:
    #                 # Encode sentences and compute pairwise cosine similarity
    #                 embeds = embedder.encode(para_sents, convert_to_tensor=True)
    #                 sim_matrix = util.cos_sim(embeds, embeds)
    #                 sim_matrix.fill_diagonal_(0)  # remove self-similarity
    #                 max_sim_to_others = sim_matrix.max(dim=1).values
    #
    #                 for sent_text, max_sim in zip(para_sents, max_sim_to_others):
    #                     if max_sim < dissimilar_thresh:
    #                         very_dissimilar_sents.append(sent_text)
    #                     else:
    #                         meaningful_sents.append(sent_text)
    #             else:
    #                 meaningful_sents.extend(para_sents)
    #
    #             # Step 5a: Debug print very dissimilar sentences
    #             if very_dissimilar_sents:
    #                 print(f"[DEBUG] Very dissimilar sentences skipped: {very_dissimilar_sents}")
    #
    #             all_sentences.extend(meaningful_sents)
    #
    #         print(f"[{mode}] Processed {doc_id}: {len(paragraphs)} paragraphs after ad removal.")
    #
    # # Step 6: Save meaningful sentences
    # with combined_sentences_file.open("w", encoding="utf-8") as cf:
    #     cf.write("\n".join(all_sentences))
    #
    # print(f"Saved {out_file} with {len(raw_texts)} documents for mode={mode}")
    # print(f"Collected {len(all_sentences)} meaningful sentences into {combined_sentences_file}")
    #
    # # Step 6: Generate individual summaries for all documents and save as golden summary


    summariser = Summariser(model="qwen3:8b")
    print(f"[Preprocessing] Generating reported speech documents as golden summary...")

    # Summarize each document
    doc_summaries = []
    for i, doc_text in enumerate(raw_texts):
        print(f"[Preprocessing] Generating reported speech document {i + 1}/{len(raw_texts)}")
        summary = summariser.summarize(doc_text, method="individual")
        doc_summaries.append(summary)

    # Split summaries into sentences and filter very dissimilar ones
    all_summary_sentences = []

    for doc_summary in doc_summaries:
        doc = nlp(doc_summary)
        summary_sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if len(summary_sents) > 1 and embedder:
            # Encode sentences and compute pairwise cosine similarity
            embeds = embedder.encode(summary_sents, convert_to_tensor=True)
            sim_matrix = util.cos_sim(embeds, embeds)
            sim_matrix.fill_diagonal_(0)  # remove self-similarity
            max_sim_to_others = sim_matrix.max(dim=1).values

            for sent_text, max_sim in zip(summary_sents, max_sim_to_others):
                if max_sim >= dissimilar_thresh:  # only keep aligned sentences
                    all_summary_sentences.append(sent_text)
                else:
                    print(f"[DEBUG] Very dissimilar sentence skipped: {sent_text}")
        else:
            all_summary_sentences.extend(summary_sents)

    # Save each sentence on a new line
    with combined_sentences_file.open("w", encoding="utf-8") as cf:
        cf.write("\n".join(all_summary_sentences))

    print(
        f"[Preprocessing] Saved golden summary with {len(all_summary_sentences)} sentences to {combined_sentences_file}")
