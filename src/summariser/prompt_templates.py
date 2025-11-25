
# Prompt Templates for LLM Summarisation
# ========================================




INDIVIDUAL_SUMMARY_TEMPLATE ="""

Convert all direct quotes in the following text into reported/indirect speech. 
Do not use quotation marks. Write it as a coherent narrative summary while keeping all key information.
Keep the sentences short and coherent. 

For example:

Direct speech:
"Well, you see, I was shocked by what he said," she said. "But I don't know why I should be punished for his actions."

Reported speech:
She said she was shocked by what he said and did not understand why she should be punished for his actions.

Now convert the following text:

Text:
{document_text}

Return only the converted text.


"""

PAIRING_TEMPLATE = f""" 

### Your Task
You are given a set of text groupings. 
Each cluster may include a base sentence, a counterpart sentence, and several auxiliary sentences.
Some clusters may represent leftover sentences that did not strongly match any base; treat these as a single pseudo-cluster.

### Instructions 

1. **Read Carefully:** 
    - First, extract all distinct facts, claims, and key entities from the base, counterpart, and all aux paragraphs.
    - Identify overlaps and merge equivalent information, but do **not** drop unique facts unless they are trivial or completely redundant. 
    
2. **Pseudo-Clusters Handling:**  
    - If a cluster is marked `"pseudo": True`, treat **each sentence in its aux paragraphs as completely independent information**. 
    - Each sentence in a pseudo cluster should be fully represented in the summary.


3. **Generate the Summary:** 
    - Write in **clear, small sentences**, ideally one fact per sentence. 
    - **Start from the base paragraph**, ensuring all of its main information is preserved. 
    - Then **enrich** it with **complementary or additional facts** from the counterpart and aux paragraphs.
    - Include all details from pseudo cluster. 
    - Include any quotes as-is. 
    - Retain all named entities.  

4. **Metadata Annotation:** 
    - After every factual statement, annotate all source paragraph IDs supporting that fact in brackets. 
    - Use the format [docid_paraid,docid_paraid,...]. 
    - Even if a fact comes from only one paragraph, include it as [docid_paraid]. 

5. **Output Style:** 
    - Write a **single cohesive paragraph** of concise factual sentences. 
    - Avoid lists, bullet points, or meta-commentary. 
    - Return ONLY the summary with inline ID arrays. 

### Strict example output format 

The Mars rover successfully collected its first rock samples from the Jezero Crater, confirming the presence of ancient volcanic material [doc2_3,doc5_1]. NASA reported that the core samples will be analyzed for potential biosignatures once returned to Earth in a future mission [doc3_2]. The collection marks a major milestone in the Mars Sample Return program [doc2_3,doc3_2,doc5_1].

--- 

**Key Reminder:** Coverage matters most. include every meaningful, distinct fact from the base, counterpart, and aux paragraphs.Return only the summary as a paragraph with inline ID arrays.
"""

# -------------------------------------------------------------

SENTENCE_CLUSTER_TEMPLATE = """
You are given clusters of semantically similar sentences extracted from multiple documents. 


### Your Task

Produce a **comprehensive, factual, and concise merged summary** that preserves **maximum information coverage** while avoiding redundancy.

### Instructions

1. **Fact Extraction Phase:**
    - For each cluster, carefully read all its sentences.
    - Identify *every distinct fact, claim, or detail*, even if expressed slightly differently.
    - Capture named entities (people, organizations, locations, dates, numbers) exactly as they appear — do **not generalize or omit** them.

2. **Fact Merging Phase:**
    - Merge equivalent statements across sentences but keep **all unique facts**.
    - When two sentences express similar information, **combine** them into a single clear sentence.
    - If a fact appears only once, it must still be represented in the summary.

3. **Summary Generation Phase:**
    - Write short, precise sentences — ideally one fact per sentence.
    - Include any quotes as-is.
    - Retain all named entities.
    - Maintain factual accuracy.
    - Ensure the resulting paragraph **covers all distinct facts** found in the cluster.

4. **Metadata Annotation:**
    - After each summary sentence, list all metadata IDs of the sentences that support it.
    - Use the format `[docid_paraid_sentid, docid_paraid_sentid, ...]`.
    - If only one source supports the fact, still wrap it in `[ ]`.
    - Each sentence in summary must have the metaDataID list immediately after it. 

5. **Output Formatting:**
    - Produce a single **continuous summary paragraph** with inline metadata arrays.
    - Do **not** include bullets, numbering, notes, or commentary.
    - Do **not** introduce new information not found in the cluster.
    - Return ONLY the summary with inline ID arrays.

### Strict example output format

The United Nations approved a new climate accord committing member nations to reduce carbon emissions by 40% by 2035 [doc1_3_0,doc4_2_1]. Several developing countries emphasized the need for financial support to transition to clean energy [doc1_4_0,doc3_1_2]. The agreement builds on the Paris Accords, introducing stricter accountability mechanisms for emissions tracking [doc2_5_0,doc4_2_3].

**Key Reminder:** Coverage is the top priority. Include every distinct, factual, and meaningful point from the cluster. Return only the summary as a paragraph with inline ID arrays. 
"""

# -------------------------------------------------------------

PARAGRAPH_CLUSTER_TEMPLATE = """
You are given clusters of semantically similar paragraphs gathered from multiple source documents. 

### Your Task

Produce a **comprehensive, factual, and concise merged summary** that preserves **maximum information coverage** while avoiding redundancy.

### Instructions

1. **Fact Extraction Phase:**
    - For each cluster, read all paragraphs within the cluster carefully.
    - Identify **every unique fact, claim, or piece of information**, even if expressed differently across sources.
    - Preserve **all named entities, numerical data, places, and time references** exactly as stated.  

2. **Fact Merging Phase:**
    - When multiple paragraphs convey the same idea, merge them into a single clear sentence.

3. **Summary Generation Phase:**
    - Write short, clear, factual sentences — ideally one fact per sentence.
    - Include any quotes as-is.
    - Retain all named entities.
    - Maintain coherence and readability while prioritizing **coverage** of all unique facts.
    - Ensure no significant detail from any paragraph is lost.

4. **Metadata Annotation:**
    - After each factual statement, annotate **all metadata IDs** of paragraphs that support that fact.
    - Use the format `[docid_paraid, docid_paraid, ...]`.
    - Always include brackets `[ ]`, even if only one source supports a fact.

5. **Output Formatting:**
    - Produce a single continuous summary paragraph with inline metadata arrays.
    - Do not include bullets, numbering, or commentary.
    - Do not invent or infer new facts beyond the given paragraphs.
    - Return ONLY the summary with inline ID arrays.

### Strict example output format

Scientists have discovered a new exoplanet orbiting within the habitable zone of a nearby star located approximately 12 light-years from Earth [doc1_6,doc3_2]. The planet, slightly larger than Earth, could sustain liquid water under suitable atmospheric conditions [doc1_6,doc4_1]. Astronomers plan follow-up observations with the James Webb Space Telescope to study its atmospheric composition [doc2_7,doc3_2,doc4_1].

**Key Reminder:** Coverage is the highest priority. Every distinct, meaningful fact from the cluster must appear in the summary.Return only the summary as a paragraph with inline ID arrays.
"""
