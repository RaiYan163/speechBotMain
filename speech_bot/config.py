import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL_PATH = os.path.join(BASE_DIR, "en_US-kathleen-low.onnx")
TTS_OUTPUT_PATH = os.path.join(BASE_DIR, "static", "output.wav")
INPUT_JSON_PATH = os.path.join(BASE_DIR, "input.json")

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")  # Default to localhost
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")      # Default port
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
