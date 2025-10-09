from datasets import load_dataset
import spacy
from pathlib import Path
import json
import shutil
import re
from src.utils.metadata_utils import Document, Paragraph, Sentence
from src.ner.ner_tagger import NERTagger
from src.coref.corefernce_resolver import CoreferenceResolver


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
    Remove any advertisement-like lines from text based on predefined patterns.

    Args:
        text (str): Raw document text

    Returns:
        str: Cleaned text with ads removed
    """
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not any(re.search(p, line.strip()) for p in AD_PATTERNS)]
    return "\n".join(cleaned_lines).strip()


def split_paragraphs(text: str):
    """
    Split document into paragraphs using line breaks.

    Args:
        text (str): Document text

    Returns:
        List[str]: List of non-empty paragraphs
    """
    return [p.strip() for p in text.split("\n") if p.strip()]


def preprocess_first_instance(
    mode: str = "baseline",
    out_dir: Path = Path("data/processed"),
    results_dir: Path = Path("data/results"),
):
    """
    Preprocess the first MultiNews instance with a given ablation mode.

    Steps:
        1. Remove advertisements from raw text
        2. Optionally perform coreference resolution
        3. Optionally perform NER tagging
        4. Split text into paragraphs and sentences
        5. Store metadata in Document → Paragraph → Sentence dataclasses
        6. Save processed documents as JSONL
        7. Save cleaned original sentences for coverage evaluation

    Args:
        mode (str): Ablation mode. Options: "baseline", "coref", "coref+ner"
        out_dir (Path): Directory to save processed JSONL documents
        results_dir (Path): Directory to save combined original sentences
    """
    # Create ablation mode-specific folder
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    # Load MultiNews first given instance
    dataset = load_dataset("multi_news", split="train[1:2]")
    instance = dataset[0]
    raw_texts = instance["document"].split("|||||")  # documents are separated by '|||||'

    # Initialize spaCy 
    nlp = spacy.load("en_core_web_sm")

    # Initialize NER tagger and coreference resolver if required 
    ner_tagger = NERTagger() if mode == "coref+ner" else None
    resolver = CoreferenceResolver() if mode in ["coref", "coref+ner"] else None

    # Prepare output paths 
    out_file = mode_dir / f"multi_doc_{mode}.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    combined_sentences_file = results_dir / "combined_source_sentences.txt"

    all_sentences = []

    # Clear old JSONL if exists 
    if out_file.exists():
        out_file.unlink()

    # Process each document 
    with out_file.open("w", encoding="utf-8") as f:
        for doc_idx, raw_text in enumerate(raw_texts):
            doc_id = f"doc{doc_idx}"

            # Remove advertisements 
            cleaned_text = clean_advertisements(raw_text)
            if not cleaned_text.strip():
                print(f"Skipped {doc_id} (empty after ad removal)")
                continue

            # Coreference resolution if required 
            resolved_doc_text = resolver.resolve(cleaned_text) if resolver else cleaned_text

            paragraphs = []
            for p_idx, p_text in enumerate(split_paragraphs(resolved_doc_text)):
                spacy_p = nlp(p_text)
                sentences = []

                for s_idx, sent_span in enumerate(spacy_p.sents):
                    sent_text = sent_span.text.strip()
                    if not sent_text:
                        continue
                    sent_entities = []

                    # NER tagging only for coref+ner mode 
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
                    # Create Paragraph object 
                    para = Paragraph(
                        sentences=sentences,
                        doc_id=doc_id,
                        para_id=p_idx,
                        text=" ".join([s.text for s in sentences]),
                        resolved_text=" ".join([s.text for s in sentences])
                        if mode in ["coref", "coref+ner"]
                        else None,
                    )
                    paragraphs.append(para)

            if not paragraphs:
                continue

            # Create Document object 
            doc_obj = Document(doc_id=doc_id, paragraphs=paragraphs, raw_text=cleaned_text)

            # Write cleaned doc to JSONL 
            f.write(
                json.dumps({"mode": mode, "document": doc_obj.__dict__}, default=lambda o: o.__dict__, ensure_ascii=False)
                + "\n"
            )

            # Collect cleaned original sentences (pre-coref/NER) 
            for para in split_paragraphs(cleaned_text):
                for sent_span in nlp(para).sents:
                    sent_text = sent_span.text.strip()
                    if sent_text:
                        all_sentences.append(sent_text)

            print(f"[{mode}] Processed {doc_id}: {len(paragraphs)} paragraphs after ad removal.")

    # Save all cleaned original sentences to combined file 
    with combined_sentences_file.open("w", encoding="utf-8") as cf:
        cf.write("\n".join(all_sentences))

    print(f"Saved {out_file} with {len(raw_texts)} documents for mode={mode}")
    print(f"Collected {len(all_sentences)} CLEANED sentences into {combined_sentences_file}")


# if __name__ == "__main__":
#     # Clear and recreate processed/results folders before any run
#     out_dir = Path("data/processed")
#     results_dir = Path("data/results")
#
#     for d in [out_dir, results_dir]:
#         if d.exists():
#             shutil.rmtree(d)
#         d.mkdir(parents=True, exist_ok=True)
#
#     # Run all 3 ablation modes
#     for m in ["baseline", "coref", "coref+ner"]:
#         preprocess_first_instance(mode=m, out_dir=out_dir, results_dir=results_dir)
