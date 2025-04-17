import whisper
import time
import os

# Load Whisper model (tiny for CPU efficiency)
model = whisper.load_model("tiny")

def transcribe_file(audio_path):
    print(f"[*] Transcribing: {audio_path}")
    start_time = time.time()
    
    result = model.transcribe(audio_path)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Save transcription to file
    with open("transcriptions.txt", "a", encoding="utf-8") as f:
        f.write(f"File: {os.path.basename(audio_path)}\n")
        f.write(f"Time Taken: {elapsed:.2f} seconds\n")
        f.write("Transcription:\n")
        f.write(" " + result["text"] + "\n")
        f.write("=" * 60 + "\n\n")
    
    print(f"[+] Transcription: {result['text']}")
    return result["text"]
