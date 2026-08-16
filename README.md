# 🎓 RAG-Based Teaching Assistant

<div align="center">

**An intelligent Q&A system that converts lecture videos into knowledge using Retrieval-Augmented Generation (RAG)**

Convert your video lectures into an AI-powered assistant that answers questions with exact timestamps and relevant context.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Status](https://img.shields.io/badge/Status-Working-brightgreen)]()

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [How It Works](#how-it-works)

---

## 🎯 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. **Extracts** audio from lecture videos (MP4 → MP3)
2. **Transcribes** audio to text using OpenAI Whisper
3. **Chunks** transcripts into meaningful segments
4. **Embeds** chunks into vectors using Ollama (bge-m3 model)
5. **Searches** for relevant chunks using cosine similarity
6. **Generates** intelligent answers using an LLM (LM Studio or Ollama)

Perfect for studying video lectures, creating searchable knowledge bases, or building educational chatbots.

---

## ✨ Features

✅ **Fully Automated Pipeline** - Convert videos to Q&A in 5 simple steps  
✅ **GPU Accelerated** - Whisper & embeddings run on CUDA (with CPU fallback)  
✅ **Context-Aware** - Merges small chunks for better semantic understanding  
✅ **Timestamp Tracking** - Answers include exact video timestamps  
✅ **Configurable** - Easy to customize models, chunk sizes, and backends  
✅ **Production Ready** - Error handling, progress logging, and validation  

---

## 🔄 Pipeline Overview

```
Videos (MP4) → Audio (MP3) → Transcripts (JSON) → Merged Chunks → Embeddings → Ask Questions → LLM Response
```

| Step | Script | What it does |
|------|--------|-------------|
| 1️⃣ | `1-video_to_audio.py` | Extracts audio from all videos in `Videos/` using ffmpeg |
| 2️⃣ | `2-speech_to_text_ALL_mp3.py` | Batch transcribes all MP3s → saves JSON in `Transcripts/` |
| 3️⃣ | `3-merge_chunks.py` | Merges every 5 small chunks into bigger ones for better context |
| 4️⃣ | `4-generate_embeddings.py` | Creates vector embeddings (`bge-m3` via Ollama) → `embeddings.joblib` |
| 5️⃣ | `5-process_query.py` | Takes your question, finds relevant chunks, gets LLM answer |

---

## 🔧 Prerequisites

### System Requirements

| Component | Requirement | Installation |
|-----------|------------|--------------|
| **Python** | 3.10+ | [python.org](https://www.python.org/) |
| **FFmpeg** | Audio extraction | `winget install ffmpeg` (Windows) / `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Linux) |
| **Ollama** | Embeddings (`bge-m3`) | [ollama.ai](https://ollama.ai/) |
| **LM Studio** (Optional) | LLM inference | [lmstudio.ai](https://lmstudio.ai) |

### Hardware Recommendations

| GPU VRAM | Setup | LLM Model | Notes |
|----------|-------|-----------|-------|
| **4GB** | Laptop/Budget | `gemma-4-e2b` (LM Studio) | ⭐ Best value |
| **6GB** | Mid-range | `gemma-4-e4b` (LM Studio) | Better quality |
| **8GB+** | High-end | Any model | Full flexibility |

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup (2 minutes)

```bash
# Clone or download this project
cd RAG-Teaching-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Start Services (1 minute)

```bash
# Start Ollama (in a new terminal)
ollama serve

# In another terminal, pull embedding model
ollama pull bge-m3

# Start LM Studio and load gemma-4-e2b (listening on localhost:1234)
# See LM Studio documentation for setup
```

### 3️⃣ Run Pipeline (5-30 minutes depending on video length)

```bash
cd python

# Step 1: Convert videos to MP3
python 1-video_to_audio.py

# Step 2: Transcribe to text
python 2-speech_to_text_ALL_mp3.py

# Step 3: Merge chunks for context
python 3-merge_chunks.py

# Step 4: Generate embeddings
python 4-generate_embeddings.py

# Step 5: Ask questions!
python 5-process_query.py
```

### 4️⃣ Try It Out!

```
❓ Ask a question: What is machine learning?
🔍 Searching top 5 relevant chunks...
  📌 [Lecture 2] 0:05:30 — Introduction to supervised learning...
  📌 [Lecture 3] 0:12:15 — Classification algorithms explained...

🤖 Asking LLM (lmstudio_2b)...
==================================================
Machine learning is a subset of AI that enables systems
to learn and improve from experience without being 
explicitly programmed. The system learns patterns from data...
==================================================

💾 Response saved to ../response.txt
```

---

## 💻 Installation Guide

### Option A: Conda (Recommended for Whisper with GPU)

```bash
# Create environment with Python 3.12
conda create -n rag python=3.12 -y
conda activate rag

# Install PyTorch with GPU support (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Verify installation
python -c "import whisper, torch; print(f'✅ GPU Available: {torch.cuda.is_available()}')"
```

### Option B: Virtual Environment (venv)

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import whisper; print('✅ Whisper installed')"
```

---

## 📁 Project Structure

```
RAG-Teaching-Assistant/
│
├── python/                           # Core scripts (run in order)
│   ├── 1-video_to_audio.py          # MP4 → MP3
│   ├── 2-speech_to_text_ALL_mp3.py  # MP3 → Transcripts
│   ├── 3-merge_chunks.py             # Chunk optimization
│   ├── 4-generate_embeddings.py      # Vector creation
│   └── 5-process_query.py            # Q&A interface
│
├── notebooks/                        # Jupyter notebooks (educational)
│   ├── process_video_to_mp3(1st).ipynb
│   ├── speech_to_text_ALL_mp3(3rd).ipynb
│   ├── merge_chunks.ipynb
│   ├── read_chunks(4th).ipynb
│   └── Full_RAG_Workflow(4th).ipynb
│
├── Videos/                           # Input: Your MP4 lecture videos
├── Audios/                           # Output: Extracted MP3 files
├── Transcripts/                      # Output: Raw Whisper JSON
├── newTranscripts/                   # Output: Merged chunk JSON
│
├── embeddings.joblib                 # Output: Vector database
├── response.txt                      # Output: Latest LLM response
│
├── requirements.txt                  # Python dependencies
├── WORKFLOW.md                       # Detailed workflow guide
└── README.md                         # This file
```

---

## ⚙️ Configuration

Each Python script has a **Configuration** section. Customize these variables:

### `1-video_to_audio.py`
```python
INPUT_DIR = "../Videos"      # Where MP4 files are
OUTPUT_DIR = "../Audios"     # Where MP3 files go
```

### `2-speech_to_text_ALL_mp3.py`
```python
AUDIO_DIR = "../Audios"
OUTPUT_DIR = "../Transcripts"
MODEL_SIZE = "small"         # tiny, base, small, medium, large
```

### `3-merge_chunks.py`
```python
CHUNKS_TO_MERGE = 5          # Merge every N chunks
INPUT_DIR = "../Transcripts"
OUTPUT_DIR = "../newTranscripts"
```

### `4-generate_embeddings.py`
```python
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"
TRANSCRIPT_DIR = "../newTranscripts"
OUTPUT_FILE = "../embeddings.joblib"
```

### `5-process_query.py`
```python
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
TOP_RESULTS = 5              # Retrieve top 5 chunks
DEFAULT_BACKEND = "lmstudio_2b"  # or "lmstudio_4b", "ollama"
EMBEDDING_FILE = "../embeddings.joblib"
```

---

## 📖 Usage Guide

### Running Individual Steps

```bash
cd python

# Step 1: Convert videos (only runs for new videos)
python 1-video_to_audio.py

# Step 2: Full transcription
python 2-speech_to_text_ALL_mp3.py

# Step 3: Merge chunks
python 3-merge_chunks.py

# Step 4: Generate embeddings (takes longest, requires Ollama)
python 4-generate_embeddings.py

# Step 5: Interactive Q&A
python 5-process_query.py
```

### Running the Complete Pipeline

```bash
# Run all steps in sequence
cd python && \
python 1-video_to_audio.py && \
python 2-speech_to_text_ALL_mp3.py && \
python 3-merge_chunks.py && \
python 4-generate_embeddings.py && \
python 5-process_query.py
```

---

## 🐛 Troubleshooting

### Common Issues

#### ❌ `ModuleNotFoundError: No module named 'whisper'`
```bash
# Solution: Install from requirements
pip install -r requirements.txt
```

#### ❌ `ConnectionError: Cannot connect to Ollama at localhost:11434`
```bash
# Solution: Start Ollama
ollama serve

# Verify connection
curl http://localhost:11434/api/status
```

#### ❌ `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
```bash
# Windows:
winget install ffmpeg

# macOS:
brew install ffmpeg

# Linux:
sudo apt install ffmpeg
```

#### ❌ `CUDA out of memory` during Step 2
```bash
# Use smaller Whisper model in 2-speech_to_text_ALL_mp3.py
MODEL_SIZE = "tiny"  # Instead of "small"
```

#### ❌ LLM response is slow
```bash
# Ensure LM Studio is running with gemma-4-e2b model loaded
# Check 5-process_query.py:
DEFAULT_BACKEND = "lmstudio_2b"  # ⭐ Recommended
```

---

## 🧠 How It Works

### Speech-to-Text (Whisper)
OpenAI's Whisper model transcribes audio into text with timestamps and segments.

### Chunk Merging
Small chunks are combined (default: 5 per group) to create larger context windows for better embeddings.

### Vector Embeddings (bge-m3)
Each merged chunk is converted to a dense vector (768 dimensions) using `bge-m3` via Ollama for fast similarity search.

### Query Processing
When you ask a question, it's also embedded, then **cosine similarity** finds the top-5 most relevant chunks.

### LLM Generation
The top chunks are formatted into a prompt and sent to an LLM (LM Studio or Ollama) for final answer generation.

### Why RAG?
✅ **Grounded Answers** - LLM only uses provided chunks (no hallucinations)  
✅ **Fast** - Similarity search is much faster than re-processing all video  
✅ **Explainable** - Can show which video segments informed the answer  
✅ **Scalable** - Add more videos without retraining

---

## ❓ FAQ

**Q: Can I use this for languages other than English?**  
A: Yes! Whisper supports 99+ languages. Change `language="en"` to any ISO-639-1 code.

**Q: What if I don't have a GPU?**  
A: The pipeline runs on CPU (slower). Use smaller models: `MODEL_SIZE = "tiny"`.

**Q: Can I use this with other LLMs?**  
A: Yes! Add new backend functions in `5-process_query.py`.

**Q: How do I update my knowledge base with new videos?**  
A: Add new MP4s to `Videos/` and re-run all 5 steps.

---

## 📬 Support

- Check the **Troubleshooting** section
- Review **WORKFLOW.md** for detailed guidance
- Check script docstrings for specific functions

---

<div align="center">

**Happy learning! 🎓✨**

*Built with ❤️ for educational excellence*

</div>
