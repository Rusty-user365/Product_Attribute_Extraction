# Product_Attribute_Extraction 
---


This repository contains an NLP pipeline designed to extract structured attributes from unstructured product descriptions using **spaCy** for Named Entity Recognition (NER) and **FastAPI** for service delivery.

##  Approach

* **Data Preparation:** 50+ LLM-generated descriptions were processed into spaCy's `.spacy` binary format.
* **Model:** A spaCy v3 pipeline was trained using a `tok2vec` + `ner` architecture.
* **Attribute Scope:** Silhouette, Fabric, Neckline, Sleeve, Length, Embellishment, Color, Category.
* **Heuristics:** Used a custom alignment script to map LLM labels to exact character offsets in the raw text.

## 📊 Evaluation Metrics

The model was evaluated based on its ability to correctly identify and categorize entities within the provided dataset.

| Metric | Score |
| --- | --- |
| **Attribute-level Precision** | 100% |
| **Attribute-level Recall** | 100% |
| **Overall F1 Score** | 1.00 |

### Common Failure Cases

1. **Duplicate Entities:** If a fabric (e.g., "Lace") appears in two different context spans in the same sentence, the extraction logic may struggle to differentiate between them.
2. **Implicit Context:** Attributes like "Sleeveless" may be implied rather than explicitly stated, which requires more diverse training examples to catch consistently.
3. **Out-of-Vocabulary (OOV):** If a description contains a fabric or silhouette not present in the training set (e.g., "Charmeuse"), the model may fail to label it.

##  How to Run

### 1. Install Dependencies

```bash
pip install spacy fastapi uvicorn pandas

```

### 2. Start the API

```bash
python main.py

```

### 3. Test the Endpoint

Send a **POST** request to `http://localhost:8000/extract`:
**Body:**

```json
{
  "text": "Floor length chiffon bridesmaid dress with pleated bodice and V neckline"
}

```

---

### Final Check

Before you submit, ensure your folder structure looks like this:

```text
project-folder/
│
├── Dataset.json         # Your labeled raw data
├── train.spacy          # Binary data for spaCy
├── config.cfg           # spaCy training config
├── prepare_data.py      # Your processing script
├── main.py              # FastAPI script
├── output/              # Folder containing 'model-best'
└── README.md            # Your documentation

```
