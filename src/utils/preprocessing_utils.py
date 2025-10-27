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
    """Remove advertisement-like lines from text."""
    lines = text.split("\n")
    cleaned_lines = [line for line in lines if not any(re.search(p, line.strip()) for p in AD_PATTERNS)]
    return "\n".join(cleaned_lines).strip()


def split_paragraphs(text: str):
    """Split text into paragraphs."""
    return [p.strip() for p in text.split("\n") if p.strip()]



# BASE PREPROCESSING (ablation-independent)

def preprocess_instance_base(
    instance,
    out_dir: Path,
    mode: str = "baseline"
):
    """
    Performs ablation-independent preprocessing:
    - Clean advertisements
    - Split paragraphs/sentences
    - Build Document/Paragraph/Sentence objects
    - Save baseline JSONL
    """
    nlp = spacy.load("en_core_web_sm")
    raw_texts = instance["document"].split("|||||")

    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    out_file = mode_dir / f"multi_doc_{mode}.jsonl"

    if out_file.exists():
        out_file.unlink()

    with out_file.open("w", encoding="utf-8") as f:
        for doc_idx, raw_text in enumerate(raw_texts):
            doc_id = f"doc{doc_idx}"
            cleaned_text = clean_advertisements(raw_text)
            if not cleaned_text.strip():
                print(f"Skipped {doc_id} (empty after ad removal)")
                continue

            paragraphs = []
            for p_idx, p_text in enumerate(split_paragraphs(cleaned_text)):
                spacy_p = nlp(p_text)
                sentences = [
                    Sentence(
                        text=sent_span.text.strip(),
                        doc_id=doc_id,
                        para_id=p_idx,
                        sent_id=s_idx,
                        resolved_text=None,
                        entities=[]
                    )
                    for s_idx, sent_span in enumerate(spacy_p.sents)
                    if sent_span.text.strip()
                ]
                if sentences:
                    para = Paragraph(
                        sentences=sentences,
                        doc_id=doc_id,
                        para_id=p_idx,
                        text=" ".join(s.text for s in sentences),
                        resolved_text=None
                    )
                    paragraphs.append(para)

            if not paragraphs:
                continue

            doc_obj = Document(doc_id=doc_id, paragraphs=paragraphs, raw_text=cleaned_text)
            f.write(json.dumps({"mode": mode, "document": doc_obj.__dict__},
                               default=lambda o: o.__dict__, ensure_ascii=False) + "\n")

    print(f"[Base Preprocessing] Saved {out_file} with {len(raw_texts)} documents.")
    return raw_texts, out_file



# ABLATION-SPECIFIC PREPROCESSING (coref/ coref+ner)

def apply_ablation_processing(
    mode: str,
    out_dir: Path,
    base_out_file: Path
):
    """
    Applies ablation-dependent processing:
    - Coreference resolution (for 'coref' or 'coref+ner')
    - NER tagging (for 'coref+ner')
    - Updates JSONL with resolved text/entities
    """
    if mode == "baseline":
        print("[Ablation] Skipping ablation-dependent processing for baseline mode.")
        return base_out_file

    nlp = spacy.load("en_core_web_sm")
    ner_tagger = NERTagger() if mode == "coref+ner" else None
    resolver = CoreferenceResolver() if mode in ["coref", "coref+ner"] else None

    out_file = base_out_file
    mode_dir = out_dir / mode
    if not mode_dir.exists():
        mode_dir.mkdir(parents=True, exist_ok=True)

    updated_docs = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_data = data["document"]
            raw_text = doc_data["raw_text"]

            # Step 1: Apply coreference resolution
            if resolver:
                raw_text = resolver.resolve(raw_text)

            # Step 2: Re-split into paragraphs/sentences
            paragraphs = []
            for p_idx, p_text in enumerate(split_paragraphs(raw_text)):
                spacy_p = nlp(p_text)
                sentences = []
                for s_idx, sent_span in enumerate(spacy_p.sents):
                    sent_text = sent_span.text.strip()
                    if not sent_text:
                        continue

                    sent_entities = []
                    if ner_tagger:
                        sent_entities, sent_text = ner_tagger.tag_sentence(sent_text, embed_tags=True)

                    sentences.append(Sentence(
                        text=sent_text,
                        doc_id=doc_data["doc_id"],
                        para_id=p_idx,
                        sent_id=s_idx,
                        resolved_text=sent_text,
                        entities=sent_entities
                    ))

                if sentences:
                    para = Paragraph(
                        sentences=sentences,
                        doc_id=doc_data["doc_id"],
                        para_id=p_idx,
                        text=" ".join([s.text for s in sentences]),
                        resolved_text=" ".join([s.resolved_text for s in sentences])
                    )
                    paragraphs.append(para)

            updated_docs.append(Document(doc_id=doc_data["doc_id"], paragraphs=paragraphs, raw_text=raw_text))

    # Save updated JSONL
    out_file = mode_dir / f"multi_doc_{mode}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for doc in updated_docs:
            f.write(json.dumps({"mode": mode, "document": doc.__dict__},
                               default=lambda o: o.__dict__, ensure_ascii=False) + "\n")

    print(f"[Ablation] Completed mode={mode}. Updated file saved to {out_file}")
    return out_file



# POSTPROCESSING (ablation-independent)

def postprocess_instance_outputs(
    raw_texts,
    results_dir: Path,
    embedder=None,
    dissimilar_thresh: float = 0.3
):
    """
    Performs postprocessing on the original documents (not ablation-specific):
    - Similarity filtering
    - Sentence collection
    - Golden summary generation
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    combined_sentences_file = results_dir / "combined_source_sentences.txt"

    summariser = Summariser(model="qwen3:8b")
    nlp = spacy.load("en_core_web_sm")

    # Step 1: Generate document-level summaries
    doc_summaries = []
    print(f"[Postprocessing] Generating golden summaries...")
    for i, doc_text in enumerate(raw_texts):
        print(f"[Postprocessing] Summarizing doc {i+1}/{len(raw_texts)}")
        summary = summariser.summarize(doc_text, method="individual")
        doc_summaries.append(summary)

    # Step 2: Filter dissimilar summary sentences
    all_summary_sentences = []
    for doc_summary in doc_summaries:
        doc = nlp(doc_summary)
        summary_sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if len(summary_sents) > 1 and embedder:
            embeds = embedder.encode(summary_sents, convert_to_tensor=True)
            sim_matrix = util.cos_sim(embeds, embeds)
            sim_matrix.fill_diagonal_(0)
            max_sim_to_others = sim_matrix.max(dim=1).values

            for sent_text, max_sim in zip(summary_sents, max_sim_to_others):
                if max_sim >= dissimilar_thresh:
                    all_summary_sentences.append(sent_text)
                else:
                    print(f"[DEBUG] Skipped dissimilar: {sent_text}")
        else:
            all_summary_sentences.extend(summary_sents)

    with combined_sentences_file.open("w", encoding="utf-8") as cf:
        cf.write("\n".join(all_summary_sentences))

    print(f"[Postprocessing] Saved {len(all_summary_sentences)} summary sentences → {combined_sentences_file}")
    return combined_sentences_file
