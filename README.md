# 🎓 RAG-Based Teaching Assistant

A **Retrieval-Augmented Generation (RAG)** pipeline that converts lecture videos into an intelligent Q&A assistant. Ask questions about your lectures and get answers with exact video timestamps.

## 🔄 Pipeline Overview

```
Videos (MP4) → Audio (MP3) → Transcripts (JSON) → Merged Chunks → Embeddings → Ask Questions → LLM Response
```

| Step | Script | What it does |
|------|--------|-------------|
| 1️⃣ | `process_video_to_mp3(1st).py` | Extracts audio from all videos in `Videos/` using ffmpeg |
| 2️⃣ | `speech_to_textSingleMp3(2nd).py` | Transcribes a single MP3 using Whisper (for testing) |
| 3️⃣ | `speech_to_text_ALL_mp3(3rd).py` | Batch transcribes all MP3s → saves JSON in `Transcripts/` |
| 4️⃣ | `merge_chunks(5th).py` | Merges every 5 small chunks into bigger ones for better context |
| 5️⃣ | `mp3_to_json(4th).py` | Creates vector embeddings (`bge-m3` via Ollama) → `embeddings.joblib` |
| 6️⃣ | `process_incoming_Query(6th).py` | Takes your question, finds relevant chunks, gets LLM answer |

---

## ⚙️ System Prerequisites

- **Python**: `3.12` (or `3.10+`)
- **FFmpeg**: Required for audio extraction. Must be added to system PATH.
  - **Windows**: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt update && sudo apt install ffmpeg`
- **Ollama**: Required for generating embeddings (`bge-m3`)
- **LM Studio** (or Ollama): Required for LLM inference backend

---

## 💻 Environment Setup

It is strongly recommended to run this project inside a dedicated Conda or virtual environment (e.g. `ml`).

### Option A: Using Conda (Recommended)

```bash
# 1. Create a Conda environment named 'ml' with Python 3.12
conda create -n ml python=3.12 -y

# 2. Activate the environment
conda activate ml

# 3. Install PyTorch with CUDA (if using GPU for Whisper) or CPU fallback
# GPU (CUDA 12.1 example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining project dependencies
pip install -r whisper/requirements.txt
pip install pandas scikit-learn joblib requests py-cpuinfo
```

### Option B: Using Python Virtual Environment (`venv`)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r whisper/requirements.txt
pip install pandas scikit-learn joblib requests py-cpuinfo
```

---

## 🤖 LLM Model Comparison (Tested)

| Model | Backend | VRAM Required | Output Quality |
|-------|---------|--------------|----------------|
| `google/gemma-4-e2b` | LM Studio | 4GB | ⭐ Excellent (recommended) |
| `google/gemma-4-e4b` | LM Studio | 6GB | ⭐ Excellent |
| `qwen2.5-coder:3b` | Ollama | 4GB | Weaker output quality |

> **Recommendation:** Use `gemma-4-e2b` via LM Studio — best quality at lowest VRAM cost.

---

## 🚀 Step-by-Step Execution Guide

```bash
# 1. Start Ollama service and pull embedding model
ollama pull bge-m3

# 2. Start LM Studio server and load 'google/gemma-4-e2b' (listening on localhost:1234)

# 3. Place your MP4 lecture videos in the Videos/ directory

# 4. Run the full pipeline in sequence:
python process_video_to_mp3(1st).py        # Step 1: Video -> Audio
python speech_to_text_ALL_mp3(3rd).py       # Step 2: Audio -> Transcripts (JSON)
python merge_chunks(5th).py                 # Step 3: Merge short chunks
python mp3_to_json(4th).py                  # Step 4: Generate vector embeddings

# 5. Query your lecture assistant:
python process_incoming_Query(6th).py
```

---

## 📁 Project Directory Layout

```
.
├── Videos/              # Input MP4 lecture videos
├── Audios/              # Generated MP3 audio files
├── Transcripts/         # Raw Whisper JSON outputs
├── newTranscripts/      # Merged context chunk JSONs
├── whisper/             # Whisper source submodule
├── embeddings.joblib    # Serialized DataFrame containing vector store
└── response.txt         # Output text file for the latest LLM query response
```

---

## 🛠️ Configuration & Customization

Each Python script contains a `# ─── Configuration ───` section at the top for easy modification:

- **LLM Backend**: Change `DEFAULT_BACKEND` in `process_incoming_Query(6th).py` (`"lmstudio_2b"`, `"lmstudio_4b"`, `"ollama"`).
- **Chunk Merging**: Adjust `CHUNKS_TO_MERGE = 5` in `merge_chunks(5th).py`.
- **Whisper Model**: Change `MODEL_SIZE = "small"` in `speech_to_text_ALL_mp3(3rd).py` (`"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"`).
- **Directory Paths**: Configurable via `INPUT_DIR` / `OUTPUT_DIR` / `TRANSCRIPT_DIR` constants at the top of each script.
