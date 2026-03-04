import spacy
from spacy.tokens import DocBin
import json
import os
import subprocess

def create_spacy_data(input_file, output_file):
    nlp = spacy.load("en_core_web_lg")
    doc_bin = DocBin()
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            item = json.loads(line)
            text = item['text']
            labels = item['labels']
            
            doc = nlp.make_doc(text)
            ents = []
            
            for label_name, value in labels.items():
                sub_values = [v.strip() for v in str(value).split(';')]
                for val in sub_values:
                    if val.lower() in ["none", "unknown", "n/a"]:
                        continue
                    start = text.lower().find(val.lower())
                    if start != -1:
                        end = start + len(val)
                        span = doc.char_span(start, end, label=label_name.upper(), alignment_mode="contract")
                        if span:
                            ents.append(span)
            
            doc.ents = spacy.util.filter_spans(ents)
            doc_bin.add(doc)
            count += 1

    doc_bin.to_disk(output_file)
    print(f"Successfully processed {count} products into {output_file}")

def run_spacy_training(train_file, output_dir="output"):
    # Create output folder if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Generate config file
    if not os.path.exists("config.cfg"):
        subprocess.run([
            "python", "-m", "spacy", "init", "config", "config.cfg",
            "--lang", "en", "--pipeline", "ner", "--optimize", "accuracy"
        ], check=True)

    # Step 2: Train the model
    subprocess.run([
        "python", "-m", "spacy", "train", "config.cfg",
        "--output", output_dir,
        "--paths.train", train_file,
        "--paths.dev", train_file
    ], check=True)

if __name__ == "__main__":
    train_file = "train.spacy"
    create_spacy_data("Dataset_Assignment.json", train_file)
    run_spacy_training(train_file)
