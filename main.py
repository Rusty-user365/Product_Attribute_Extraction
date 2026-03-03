import spacy
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

# 1. Setup Models and DB
nlp = spacy.load("./output/model-best")
# Using Chroma's default embedding function (Sentence Transformers)
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="product_attributes")

# 2. Simple Exact-Match Cache
cache = {}

app = FastAPI()

class ProductRequest(BaseModel):
    text: str

@app.post("/extract")
async def hybrid_extract(request: ProductRequest):
    desc = request.text.strip()

    # --- STEP 1: EXACT MATCH CACHE ---
    if desc in cache:
        return {"source": "cache", "attributes": cache[desc]}

    # --- STEP 2: VECTOR SIMILARITY SEARCH ---
    # Query for the closest previous result
    results = collection.query(query_texts=[desc], n_results=1)
    
    # distance < 0.1 usually means very high similarity
    if results['distances'] and results['distances'][0][0] < 0.1:
        neighbor_attr = results['metadatas'][0][0]
        # In a real app, you'd pass neighbor_attr to LLM for 'refinement'
        # For now, we return it as a 'fuzzy cache' hit
        return {"source": "vector_db", "attributes": neighbor_attr}

    # --- STEP 3: HYBRID PIPELINE (spaCy + LLM Logic) ---
    doc = nlp(desc)
    spacy_attributes = {ent.label_: ent.text for ent in doc.ents}

    # LOGIC: If spaCy found fewer than 3 attributes, trigger "Refinement"
    # This is where you would call your LLM (OpenAI/Anthropic/Ollama)
    if len(spacy_attributes) < 3:
        final_attributes = await call_llm_refiner(desc, spacy_attributes)
        source = "llm_refined"
    else:
        final_attributes = spacy_attributes
        source = "spacy_ner"

    # --- STEP 4: STORAGE FOR FUTURE REUSE ---
    cache[desc] = final_attributes
    collection.add(
        documents=[desc],
        metadatas=[final_attributes],
        ids=[str(hash(desc))]
    )

    return {"source": source, "attributes": final_attributes}

async def call_llm_refiner(text, partial_attrs):
    """
    Placeholder for your LLM call.
    You would send the text and the 'partial_attrs' found by spaCy.
    """
    # Example: response = openai.ChatCompletion.create(...)
    return partial_attrs # Return partial for now