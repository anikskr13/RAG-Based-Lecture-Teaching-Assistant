"""
Step 4: Create vector embeddings for all transcript chunks using Ollama (bge-m3).
Reads merged JSON transcripts, generates embeddings, and saves
everything as a pandas DataFrame in embeddings.joblib.
"""

import os
import json
import requests
import pandas as pd
import joblib

# ─── Configuration ───────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"
TRANSCRIPT_DIR = "newTranscripts"  # folder with merged chunks (output of merge_chunks)
OUTPUT_FILE = "embeddings.joblib"


def create_embeddings(text_list, model=EMBEDDING_MODEL):
    """Send a list of texts to Ollama and return their vector embeddings."""
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "input": text_list,  # use "input" for batch, not "prompt"
    })
    response.raise_for_status()
    return response.json()["embeddings"]  # "embeddings" for batch, "embedding" for single


def build_embedding_store(transcript_dir=TRANSCRIPT_DIR, output_file=OUTPUT_FILE):
    """Read all transcript JSONs, create embeddings, and save to a joblib file."""
    files = os.listdir(transcript_dir)
    all_chunks = []
    chunk_id = 0

    for file in files:
        if not file.endswith(".json"):
            continue

        filepath = os.path.join(transcript_dir, file)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [c["text"] for c in data["chunk"]]
        print(f"📝 Creating embeddings for {file} ({len(texts)} chunks)...")
        embeddings = create_embeddings(texts)

        for i, chunk in enumerate(data["chunk"]):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]
            all_chunks.append(chunk)
            chunk_id += 1

        print(f"✅ Embeddings created for {file}")

    df = pd.DataFrame.from_records(all_chunks)
    joblib.dump(df, output_file)
    print(f"\n💾 {output_file} created — {len(df)} total chunks embedded")


if __name__ == "__main__":
    build_embedding_store()