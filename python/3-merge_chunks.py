"""
Step 3: Merge small transcript chunks into larger ones for better embedding context.
Groups every N consecutive chunks together for improved semantic meaning.
"""

import os
import json
import math

# ─── Configuration ───────────────────────────────────────────────────────────
CHUNKS_TO_MERGE = 5               # number of consecutive chunks to merge into one
INPUT_DIR = "../Transcripts"          # raw Whisper transcripts
OUTPUT_DIR = "../newTranscripts"      # merged chunks (used by mp3_to_json for embeddings)


def merge_chunks(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, n=CHUNKS_TO_MERGE):
    """Merge every n small chunks into a single larger chunk per transcript file."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        old_chunks = data["chunk"]
        old_count = len(old_chunks)
        num_groups = math.ceil(old_count / n)

        new_chunks = []
        for i in range(num_groups):
            start_idx = i * n
            end_idx = min((i + 1) * n, old_count)
            group = old_chunks[start_idx:end_idx]

            new_chunks.append({
                "number": group[0]["number"],
                "title": group[0]["title"],
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(c["text"] for c in group),
            })

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"chunk": new_chunks, "text": data["text"]}, f, indent=4)

        print(f"✅ {filename}: {old_count} chunks → {len(new_chunks)} merged chunks")

    print("💾 All transcripts merged and saved")


if __name__ == "__main__":
    merge_chunks()transcripts merged and saved")


if __name__ == "__main__":
    merge_chunks()