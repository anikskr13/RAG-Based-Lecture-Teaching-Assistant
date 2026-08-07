"""
Step 1: Convert all lecture videos (MP4) in Videos/ to MP3 audio files in Audios/.
Uses ffmpeg under the hood.
"""

import os
import subprocess


# ─── Configuration ───────────────────────────────────────────────────────────
INPUT_DIR = "Videos"               # folder containing raw MP4 video files
OUTPUT_DIR = "Audios"              # folder where extracted MP3 audio files will be saved


def convert_videos_to_mp3(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """Extract audio from every video file in input_dir and save as MP3."""
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)
    for file in files:
        tutorial_number = file.split(".")[0]
        tutorial_name = file.split(".")[1]

        output_path = os.path.join(output_dir, f"{tutorial_number}-{tutorial_name}.mp3")
        if os.path.exists(output_path):
            print(f"⏭️  Skipping {file} (already converted)")
            continue

        print(f"🔄 Converting {tutorial_number} - {tutorial_name} to MP3...")
        subprocess.run(
            ["ffmpeg", "-i", os.path.join(input_dir, file), output_path],
            check=True,
        )

    print("✅ All videos converted from MP4 to MP3")


if __name__ == "__main__":
    convert_videos_to_mp3()