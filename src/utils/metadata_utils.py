from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


# Sentence-level data structure
@dataclass
class Sentence:
    """
    Represents a single sentence within a paragraph.

    Attributes:
        text (str): Original sentence text.
        doc_id (str): ID of the parent document.
        para_id (int): ID of the parent paragraph.
        sent_id (int): Sentence ID within the paragraph.
        resolved_text (Optional[str]): Text after coreference resolution or normalization.
        entities (List[Dict]): Named entity information (type, tag, etc.).
        embedding (Optional[np.ndarray]): Vector embedding of the sentence.
    """
    text: str
    doc_id: str
    para_id: int
    sent_id: int
    resolved_text: Optional[str] = None
    entities: List[Dict] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def meta_tag(self) -> str:
        """
        Returns a unique identifier string for this sentence,
        combining document ID, paragraph ID, and sentence ID.
        Example: "doc1_p0_s2"
        """
        return f"{self.doc_id}_p{self.para_id}_s{self.sent_id}"



# Paragraph-level data structure
@dataclass
class Paragraph:
    """
    Represents a paragraph within a document.

    Attributes:
        sentences (List[Sentence]): List of Sentence objects in the paragraph.
        doc_id (str): ID of the parent document.
        para_id (int): Paragraph ID within the document.
        text (str): Original paragraph text.
        resolved_text (Optional[str]): Paragraph text after coreference resolution.
        embedding (Optional[np.ndarray]): Vector embedding of the paragraph.
        aux_attached (List[Dict]): Any auxiliary paragraphs or sentences attached (e.g., during pairing or clustering).
    """
    sentences: List[Sentence]
    doc_id: str
    para_id: int
    text: str
    resolved_text: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    aux_attached: List[Dict] = field(default_factory=list)



# Document-level data structure
@dataclass
class Document:
    """
    Represents a complete document.

    Attributes:
        doc_id (str): Unique identifier for the document.
        paragraphs (List[Paragraph]): List of Paragraph objects in the document.
        raw_text (str): Original unprocessed text of the document.
    """
    doc_id: str
    paragraphs: List[Paragraph]
    raw_text: str
