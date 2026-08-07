"""
Step 2: Transcribe a single MP3 file using Whisper (for quick testing).
For batch transcription, use speech_to_text_ALL_mp3(3rd).py instead.
"""

import whisper
import torch


# ─── Configuration ───────────────────────────────────────────────────────────
TEST_AUDIO_PATH = "Audios/1 - (Audio).mp3"
MODEL_SIZE = "small"
LANGUAGE = "en"


def get_device():
    """Detect and return the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            print(f"✅ Using CPU: {info['brand_raw']}")
        except ImportError:
            print("✅ Using CPU")
        return "cpu"


def transcribe_single(audio_path=TEST_AUDIO_PATH, model_size=MODEL_SIZE, language=LANGUAGE):
    """Transcribe a single audio file and return the result dict."""
    device = get_device()

    # Whisper model sizes: tiny, base, small, medium, large, turbo
    # turbo is fast but doesn't support translation tasks
    model = whisper.load_model(model_size).to(device)

    result = model.transcribe(
        audio=audio_path,
        fp16=(device == "cuda"),
        language=language,
        word_timestamps=False,
    )
    return result


if __name__ == "__main__":
    result = transcribe_single("Audios/1 - (Audio).mp3")
    print(result["text"])