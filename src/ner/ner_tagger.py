import spacy

nlp = spacy.load("en_core_web_sm")

class NERTagger:
    """
    Detects named entities and assigns unique canonical tags.
    Can also embed tags directly in the text.
    """
    def __init__(self):
        self.entity_canon_map = {}   # {"Alice Smith": "PERSON_1"}
        self.entity_counter = {}     # counts per entity type

    def tag_sentence(self, text: str, embed_tags: bool = True):
        """
        Detect named entities in a sentence.
        Args:
            text (str): Original sentence text.
            embed_tags (bool): If True, replaces entity mentions with text<tag>.
        Returns:
            tagged_entities (list of dicts): metadata for each entity
            new_text (str): text with embedded tags (if embed_tags=True)
        """
        spacy_doc = nlp(text)
        tagged_entities = []
        new_text = text

        # We replace entities from end to start to avoid messing up indices
        # That is after the tagging, the string indices are different.
        entities_sorted = sorted(spacy_doc.ents, key=lambda e: e.start_char, reverse=True)

        for ent in entities_sorted:
            ent_text = ent.text
            ent_type = ent.label_

            # Assign a unique canonical tag
            if ent_text not in self.entity_canon_map:
                count = self.entity_counter.get(ent_type, 0) + 1
                self.entity_counter[ent_type] = count
                tag = f"{ent_type}_{count}"
                self.entity_canon_map[ent_text] = tag
            else:
                tag = self.entity_canon_map[ent_text]

            # Store metadata
            tagged_entities.append({
                "type": ent_type,
                "text": ent_text,
                "canon": ent_text,
                "tag": tag
            })

            # Embed tag in text if requested
            if embed_tags:
                # Replace the exact substring at the character level
                start, end = ent.start_char, ent.end_char
                new_text = new_text[:start] + f"{ent_text}<{tag}>" + new_text[end:]

        return tagged_entities, new_text
