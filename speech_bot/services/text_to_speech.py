import subprocess
import json
import os
import re
import winsound
import wave
import time
from config import PIPER_MODEL_PATH, TTS_OUTPUT_PATH, INPUT_JSON_PATH

def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

def get_audio_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def text_to_speech(text):
    text = remove_non_ascii(text)

    os.makedirs(os.path.dirname(TTS_OUTPUT_PATH), exist_ok=True)

    # Use relative paths
    piper_exe = "piper.exe"
    model_path = os.path.relpath(PIPER_MODEL_PATH).replace("\\", "/")
    output_path = os.path.relpath(TTS_OUTPUT_PATH).replace("\\", "/")

    command = [
        piper_exe,
        "-m", model_path,
        "-f", output_path
    ]

    print("[📤 Running Piper]:", ' '.join(command))

    try:
        # Generate the audio file
        result = subprocess.run(
            command,
            input=text,
            capture_output=True,
            text=True,
            check=True
        )
        print("[✅ Piper STDOUT]:", result.stdout)
        print("[✅ Piper STDERR]:", result.stderr)

        # Get audio duration
        duration = get_audio_duration(output_path)
        print(f"[⏱️ Audio Duration]: {duration:.2f} seconds")

        # Play the audio and wait for it to finish
        winsound.PlaySound(output_path, winsound.SND_FILENAME)
        time.sleep(duration)  # Wait for the audio to finish playing
        
        return duration

    except subprocess.CalledProcessError as e:
        print("[❌ Piper Failed]")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("Error Output:", e.stderr)
        raise
