"""
Step 2: Batch transcribe ALL MP3 files in Audios/ using Whisper.
Saves JSON transcripts (with chunks + full text) to Transcripts/.
"""

import os
import json
import whisper
import torch


# ─── Configuration ───────────────────────────────────────────────────────────
AUDIO_DIR = "../Audios"              # folder containing MP3 audio files
OUTPUT_DIR = "../Transcripts"         # folder where raw Whisper JSON transcripts are saved
MODEL_SIZE = "small"               # Whisper model size (tiny, base, small, medium, large, turbo)


def get_device():
    """Detect and return the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            print(f"✅ CPU detected: {info['brand_raw']}")
        except ImportError:
            print("✅ Using CPU")
        return "cpu"


def transcribe_all(audio_dir=AUDIO_DIR, output_dir=OUTPUT_DIR, model_size=MODEL_SIZE):
    """Transcribe every MP3 in audio_dir and save structured JSON to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    # Whisper model sizes: tiny, base, small, medium, large, turbo
    # turbo is fast but doesn't support translation tasks
    print(f"🔄 Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size).to(device)
    print("✅ Model loaded successfully")

    audios = sorted(os.listdir(audio_dir))

    for audio in audios:
        if not audio.endswith(".mp3"):
            continue

        audio_number = audio.split("-")[0]
        audio_name = audio.split("-")[1][:-4]  # strip .mp3

        print(f"🎙️  Transcribing {audio}...")
        result = model.transcribe(
            audio=os.path.join(audio_dir, audio),
            fp16=(device == "cuda"),
            language="en",
            word_timestamps=False,
        )

        # Extract clean chunks from Whisper segments
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": audio_number,
                "title": audio_name,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            })

        # Save transcript with chunks + full text
        transcript = {"chunk": chunks, "text": result["text"]}
        output_path = os.path.join(output_dir, f"{audio_number}-{audio_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=4)

    print("💾 All files transcribed and saved")


if __name__ == "__main__":
    transcribe_all()


if __name__ == "__main__":
    transcribe_all()
