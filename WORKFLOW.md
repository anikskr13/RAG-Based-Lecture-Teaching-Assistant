# RAG Pipeline Workflow

Complete step-by-step guide for the RAG (Retrieval-Augmented Generation) pipeline.

---

## 🎬 Pipeline Overview

```
Videos (MP4)
    ↓
[1] Video → Audio (MP3)
    ↓
Audios/ (MP3 files)
    ↓
[2] Speech to Text (Whisper)
    ↓
Transcripts/ (Raw JSON transcripts)
    ↓
[3] Merge Chunks (Combine small chunks into larger ones for better context)
    ↓
newTranscripts/ (Merged JSON transcripts)
    ↓
[4] Generate Embeddings (Ollama)
    ↓
embeddings.joblib (Vector embeddings database)
    ↓
[5] Process User Query (Find relevant chunks + LLM response)
    ↓
response.txt (Final answer)
```

---

## ⚙️ Step-by-Step Execution

### **Step 1: Convert Videos to Audio** 🎥 → 🔊
**File:** `python/1-video_to_audio.py`

Extracts audio from MP4 videos using ffmpeg.

```bash
python python/1-video_to_audio.py
```

**Input:** `Videos/` (MP4 files)  
**Output:** `Audios/` (MP3 files)

---

### **Step 2: Transcribe Audio to Text** 🔊 → 📝
**File:** `python/2-speech_to_text_ALL_mp3.py`

Converts all MP3 audio files to text using OpenAI Whisper.

```bash
python python/2-speech_to_text_ALL_mp3.py
```

**Input:** `Audios/` (MP3 files)  
**Output:** `Transcripts/` (Raw JSON transcripts with chunks)

---

### **Step 3: Merge Chunks** 📝 → 📦
**File:** `python/3-merge_chunks.py`

Merges small transcript chunks into larger ones (5 chunks per group).  
**Why?** Larger chunks have better context for embeddings.

```bash
python python/3-merge_chunks.py
```

**Input:** `Transcripts/` (Raw chunks)  
**Output:** `newTranscripts/` (Merged chunks)

---

### **Step 4: Generate Embeddings** 🧠
**File:** `python/4-generate_embeddings.py`

Creates vector embeddings for all merged chunks using Ollama (bge-m3 model).  
**Requires:** Ollama running on `http://localhost:11434`

```bash
python python/4-generate_embeddings.py
```

**Input:** `newTranscripts/` (Merged chunks)  
**Output:** `embeddings.joblib` (Pandas DataFrame with embeddings)

---

### **Step 5: Process User Query** 🤖 ❓
**File:** `python/5-process_query.py`

Answers user questions using:
1. Query embedding (via Ollama)
2. Cosine similarity to find top 5 relevant chunks
3. LLM response (LM Studio or Ollama)

```bash
python python/5-process_query.py
```

**Input:** `embeddings.joblib`, user query  
**Output:** `response.txt` (LLM answer)

---

## 📋 Folder Structure

```
Lec-3/
├── python/                      # Python scripts (run in order)
│   ├── 1-video_to_audio.py
│   ├── 2-speech_to_text_ALL_mp3.py
│   ├── 3-merge_chunks.py
│   ├── 4-generate_embeddings.py
│   ├── 5-process_query.py
│   └── _deprecated-single_transcription.py
│
├── notebooks/                   # Jupyter notebooks (same steps as .py files)
│   ├── process_video_to_mp3(1st).ipynb
│   ├── speech_to_text_ALL_mp3(3rd).ipynb
│   ├── merge_chunks.ipynb
│   ├── read_chunks(4th).ipynb
│   └── Full_RAG_Workflow(4th).ipynb
│
├── Audios/                      # MP3 files (output of step 1)
├── Transcripts/                 # Raw transcripts (output of step 2)
├── newTranscripts/              # Merged transcripts (output of step 3)
├── Videos/                      # Input MP4 videos
│
├── embeddings.joblib            # Vector embeddings (output of step 4)
├── response.txt                 # Final response (output of step 5)
├── README.md
└── WORKFLOW.md                  # This file
```

---

## 🚀 Quick Start

Run the complete pipeline:

```bash
cd python

# Step 1: Convert videos to audio
python 1-video_to_audio.py

# Step 2: Transcribe all audio files
python 2-speech_to_text_ALL_mp3.py

# Step 3: Merge chunks for better context
python 3-merge_chunks.py

# Step 4: Generate embeddings (make sure Ollama is running!)
python 4-generate_embeddings.py

# Step 5: Ask a question
python 5-process_query.py
```

---

## 🔧 Requirements

- Python 3.8+
- Libraries: `whisper`, `ollama`, `pandas`, `joblib`, `torch`, `scikit-learn`
- **Ollama** running for embeddings and LLM inference

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| Whisper model not found | Run `whisper --update` or download model manually |
| Ollama connection error | Ensure Ollama is running: `ollama serve` |
| GPU out of memory | Use smaller model size in config (e.g., `tiny` instead of `small`) |
| Embeddings.joblib not found | Run step 4 first before running step 5 |

---

## 📌 Key Concepts

- **RAG**: Retrieval-Augmented Generation - combines retrieval + generation for better answers
- **Chunks**: Segments of transcript text
- **Embeddings**: Vector representations of text for semantic search
- **Cosine Similarity**: Measure of how similar two text embeddings are

