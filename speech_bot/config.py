import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL_PATH = os.path.join(BASE_DIR, "en_US-kathleen-low.onnx")
TTS_OUTPUT_PATH = os.path.join(BASE_DIR, "static", "output.wav")
INPUT_JSON_PATH = os.path.join(BASE_DIR, "input.json")

# Ollama Configuration
OLLAMA_HOST = "localhost"  # Change this to your desired host
OLLAMA_PORT = "11435"     # Change this to your desired port
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_MODEL = "llama2:7b"  # Change this to your desired model

print(f"Final OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
