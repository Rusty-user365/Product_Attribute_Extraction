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
| **Loss NER** | 4.98 |

### Common Failure Cases

1. **Duplicate Entities:** If a fabric (e.g., "Lace") appears in two different context spans in the same sentence, the extraction logic may struggle to differentiate between them.
2. **Implicit Context:** Attributes like "Sleeveless" may be implied rather than explicitly stated, which requires more diverse training examples to catch consistently.
3. **Out-of-Vocabulary (OOV):** If a description contains a fabric or silhouette not present in the training set (e.g., "Charmeuse"), the model may fail to label it.

##  How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

```bash

python -m spacy download en_core_web_lg

```
### 2. create spacy Model

```bash
 python processor.py   

```

### 3. Set HuggingFace Token(IN Global Env, Poweshell)

```bash
 setx HF_TOKEN "hf_your_token_here"
```


### 4. Start the API

```bash
 python main.py   

```

### 5. Test the Endpoint

Send a **POST** request to `http://127.0.0.1:8000/docs#/`:

```
http://127.0.0.1:8000/docs#/
```
**Body:**

```json
{
  "text": "Floor length chiffon bridesmaid dress with pleated bodice and V neckline"
}

```

### Expected JSON Output:
```json
{
  "status": "success",
  "input": "Floor length chiffon bridesmaid dress with pleated bodice and V neckline",
  "attributes": {
    "Length": "Floor length",
    "Fabric": "chiffon",
    "Category": "bridesmaid dress",
    "Embellishment": "pleated bodice",
    "Neckline": "V neckline"
  }
}

```


---

### Project Structure


```text
project-folder/
│
├── Dataset.json         # labeled raw data
├── train.spacy          # Binary data for spaCy
├── config.cfg           # spaCy training config
├── prepare_data.py      # processing script
├── main.py              # FastAPI script + LLM Refiner
├── processor.py         # makes  your spacy  model 
├── output/              # Folder containing 'model-best'
└── README.md            # Documentation

```



---

##  Future Enhancements

To make the pipeline more efficient and scalable, the following improvements are planned:

- **Vector Database + Caching:** Product descriptions will be embedded and stored in a vector database (e.g., Chroma/FAISS).  
  - Exact matches will be served instantly from cache.  
  - Near‑duplicate descriptions will reuse previously extracted attributes, reducing redundant computation.  

- **LLM Refinement:** Outputs from the spaCy NER model will be passed to a reasoning LLM (e.g., Qwen) for validation and enrichment.  
  - The LLM can correct errors, fill missing attributes, and ensure consistency with the defined schema.  
  - This hybrid approach balances speed (spaCy) with accuracy (LLM).  

- **Resource Optimization:** By combining caching, similarity search, and selective LLM calls, the system minimizes compute costs while maintaining high‑quality attribute extraction.

---
