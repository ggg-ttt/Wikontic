![Wikontic logo](/media/wikontic.png)

# Wikontic

**Build ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs**

---

## 🚀 Overview

Knowledge Graphs (KGs) provide structured, verifiable representations of knowledge, enabling fact grounding and empowering large language models (LLMs) with up-to-date, real-world information. However, creating high-quality KGs from open-domain text is challenging due to issues like redundancy, inconsistency, and lack of alignment with formal ontologies.

**Wikontic** is a multi-stage pipeline for constructing ontology-aligned KGs from unstructured text using LLMs and Wikidata. It extracts candidate triples from raw text, then refines them through ontology-based typing, schema validation, and entity deduplication—resulting in compact, semantically coherent graphs.

---

## 📁 Repository Structure

- `preprocessing/constraint-preprocessing.ipynb`  
  Jupyter notebook for collecting constraint rules from Wikidata.

- `utils/`  
  Utilities for LLM-based triple extraction and alignment with Wikidata ontology rules.

- `utils/ontology_mappings/`  
  JSON files containing ontology mappings from Wikidata.

- `utils/structured_dynamic_index_utils_with_db.py`  
  - `Aligner` class: ontology alignment  
  - `StructuredInferenceWithDB` class: triple extraction

- `utils/openai_utils.py`  
  `LLMTripletExtractor` class for LLM-based triple extraction.

- `pages/` and `Wikontic.py`  
  Code for the web service for knowledge graph extraction and visualization.

- `Dockerfile`  
  For building a containerized web service.

---

## 🏁 Getting Started

1. **Set up the ontology and KG databases:**
   ```
   ./setup_db.sh
   ```

2. **Launch the web service:**
   ```
   streamlit run Wikontic.py
   ```

---

Enjoy building knowledge graphs with Wikontic!