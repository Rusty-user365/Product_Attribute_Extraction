from fastapi import FastAPI
from pydantic import BaseModel
import spacy

# Load the best model from the training output
# If you haven't finished training yet, this will error out.
try:
    nlp = spacy.load("./output/model-best")
except:
    print("Warning: Model not found. Please train the model first.")
    nlp = spacy.blank("en")

app = FastAPI(title="Product Attribute Extractor API")

class DescriptionRequest(BaseModel):
    text: str

@app.post("/extract")
async def extract(request: DescriptionRequest):
    doc = nlp(request.text)
    
    # Organize found entities into a dictionary
    results = {}
    for ent in doc.ents:
        label = ent.label_.capitalize()
        if label in results:
            results[label] = f"{results[label]}, {ent.text}"
        else:
            results[label] = ent.text
            
    return {
        "status": "success",
        "input": request.text,
        "attributes": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)