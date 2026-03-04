import spacy
import json
import re
import os
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient


# Qwen 2.5 7B 
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 

# Load spaCy model
try:
    nlp = spacy.load("./output/model-best")
except:
    print("Warning: Model not found. Please train the model first.")
    nlp = spacy.blank("en")

app = FastAPI(title="Product Attribute Extractor API (HF Edition)")
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

class DescriptionRequest(BaseModel):
    text: str

# --- LLM Refinement via Hugging Face API ---
def refine(text: str, partial_attrs: dict) -> dict:
    prompt = f"""
    You are an assistant that fills and normalizes fashion product attributes.
    Product description: "{text}"
    Partial attributes: {partial_attrs}

    Rules:
    - Only fill attributes that are None or empty.
    - Normalize values into canonical fashion labels.
    - If description implies defaults (e.g., corset -> sleeveless), fill them.
    - If no info, set to "None".
    - Return ONLY valid JSON.

    Required JSON Structure:
    {{
      "Silhouette": "None",
      "Fabric": "None",
      "Neckline": "None",
      "Sleeve": "None",
      "Length": "None",
      "Embellishment": "None",
      "Color": "None",
      "Category": "None"
    }}
    """
    
    try:
        # Use the serverless Inference API
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            # This forces the model to output a JSON object if the model supports it
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content
        
        # Safe JSON extraction in case there is markdown backticks
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw_output)
        
    except Exception as e:
        print(f"HF API Error: {e}")
        return {"error": "LLM refinement failed", "partial": partial_attrs}

# --- API Endpoint ---
@app.post("/extract")
async def extract(request: DescriptionRequest):
    # Stage 1: spaCy prediction (The "Cheap" Pass)
    doc = nlp(request.text)
    results = {}
    for ent in doc.ents:
        label = ent.label_.capitalize()
        results[label] = results.get(label, ent.text)

    # Stage 2: Check for missing values
    required_fields = ["Silhouette","Fabric","Neckline","Sleeve","Length","Embellishment","Color","Category"]
    
    # Ensure all required fields exist in results
    for field in required_fields:
        if field not in results:
            results[field] = "None"

    missing = [f for f in required_fields if results[f] in ["None", "", None]]

    # Stage 3: Call HF LLM if spaCy left gaps
    if missing:
        refined = refine(request.text, results)
        return {
            "status": "success",
            "source": "spaCy + HuggingFace Refiner",
            "attributes": refined
        }
    else:
        return {
            "status": "success",
            "source": "spaCy (Complete)",
            "attributes": results
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)