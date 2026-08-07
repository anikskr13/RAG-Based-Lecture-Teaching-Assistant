"""
Step 5: Process incoming queries against the RAG embedding store.
Loads embeddings, finds the most relevant transcript chunks via cosine similarity,
builds a polished prompt, and gets an answer from a local LLM.
"""

import requests
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity


# ─── Configuration ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
TOP_RESULTS = 5

# Choose your LLM backend by changing DEFAULT_BACKEND
# Options: "ollama", "lmstudio_4b", "lmstudio_2b"
#
# Performance notes (tested):
#   - lmstudio_2b (gemma-4-e2b) → excellent results, runs on 4GB VRAM ⭐ recommended
#   - lmstudio_4b (gemma-4-e4b) → excellent results, needs 6GB VRAM
#   - ollama (qwen2.5-coder:3b)  → worse output quality, runs on 4GB VRAM
DEFAULT_BACKEND = "lmstudio_2b"


# ─── Embedding ───────────────────────────────────────────────────────────────

def create_embeddings(text_list, model=EMBEDDING_MODEL):
    """Send a list of texts to Ollama and return their vector embeddings."""
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": model,
        "input": text_list,
    })
    r.raise_for_status()
    return r.json()["embeddings"]


# ─── LLM Backends ───────────────────────────────────────────────────────────

def inference_ollama(prompt, model="qwen2.5-coder:3b"):
    """Get a response from Ollama. Works on 4GB VRAM but output quality is weaker."""
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
    })
    r.raise_for_status()
    return r.json()["response"]


def inference_lmstudio(prompt, model="google/gemma-4-e4b", temperature=0):
    """Get a response from LM Studio (gemma-4-e4b). Excellent output, needs 6GB VRAM."""
    r = requests.post("http://127.0.0.1:1234/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def inference_lmstudio_2b(prompt, model="google/gemma-4-e2b", temperature=0):
    """Get a response from LM Studio (gemma-4-e2b). Excellent output, runs on 4GB VRAM. ⭐"""
    r = requests.post("http://127.0.0.1:1234/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# Map backend names to functions
LLM_BACKENDS = {
    "ollama": inference_ollama,
    "lmstudio_4b": inference_lmstudio,
    "lmstudio_2b": inference_lmstudio_2b,
}


# ─── Helper Functions ────────────────────────────────────────────────────────

def seconds_to_timestamp(seconds):
    """Convert seconds (float) to a human-readable H:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))


def find_relevant_chunks(df, query, top_n=TOP_RESULTS):
    """Embed the query and return the top_n most similar chunks from the DataFrame."""
    query_embedding = create_embeddings([query])[0]
    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [query_embedding],
    ).flatten()

    top_indices = similarities.argsort()[::-1][:top_n]
    return df.loc[top_indices].sort_values(["number", "start"]).reset_index(drop=True)


def build_context(results_df):
    """Format the retrieved chunks into a clean, readable context string."""
    context = ""
    for i, (_, row) in enumerate(results_df.iterrows(), start=1):
        context += f"""
====================

Chunk {i}

Video: {row['number']}
Title: {row['title']}
Time: {seconds_to_timestamp(row['start'])} - {seconds_to_timestamp(row['end'])}

Subtitle: {row['text']}

"""
    return context


def build_prompt(context, query):
    """Build the final prompt to send to the LLM."""
    return f"""
You are a teaching assistant.

Retrieved subtitle chunks:

{context}

Question:
{query}

Answer ONLY from the retrieved subtitle chunks.
If multiple chunks are relevant, include all of them.
Do not invent information.

Format:

Video:
Title:
Time:
Subtitle:

If nothing matches, reply exactly:
I couldn't find any matching subtitle in the provided videos.
"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load the embedding store
    print("🔄 Loading embeddings...")
    df = joblib.load("embeddings.joblib")
    print(f"✅ Loaded {len(df)} chunks\n")

    # Get the user's question
    query = input("❓ Ask a question: ")

    # Find the most relevant chunks
    print(f"\n🔍 Searching top {TOP_RESULTS} relevant chunks...")
    results = find_relevant_chunks(df, query)

    # Display retrieved chunks
    for _, row in results.iterrows():
        print(f"  📌 [{row['title']}] {seconds_to_timestamp(row['start'])} — {row['text'][:80]}...")

    # Build prompt and get LLM response
    context = build_context(results)
    prompt = build_prompt(context, query)

    print(f"\n🤖 Asking LLM ({DEFAULT_BACKEND})...")
    inference_fn = LLM_BACKENDS[DEFAULT_BACKEND]
    response = inference_fn(prompt)

    print("\n" + "=" * 50)
    print(response)
    print("=" * 50)

    # Save response
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response)
    print("\n💾 Response saved to response.txt")


if __name__ == "__main__":
    main()