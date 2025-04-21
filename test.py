# app.py
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import soundfile as sf
import io
import requests
import tempfile
import os
import base64
import soundfile as sf
import whisper
import wave
import time
from pydub import AudioSegment
import json
import logging
from datetime import datetime
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create transcript file
TRANSCRIPT_FILE = "transcript.json"

def initialize_transcript():
    """Initialize a fresh transcript file"""
    initial_data = {
        "sessions": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "application_version": "1.0"
    }
    with open(TRANSCRIPT_FILE, 'w') as f:
        json.dump(initial_data, f, indent=2)
    logger.info("Created fresh transcript.json file")

def load_transcript():
    """Load existing transcript or create new one"""
    if os.path.exists(TRANSCRIPT_FILE):
        try:
            with open(TRANSCRIPT_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Corrupted transcript file, creating new one")
            initialize_transcript()
            return load_transcript()
    else:
        initialize_transcript()
        return load_transcript()

def save_transcript(transcript_data):
    """Save transcript data to JSON file"""
    with open(TRANSCRIPT_FILE, 'w') as f:
        json.dump(transcript_data, f, indent=2)

def log_to_transcript(step_name, start_time, end_time, additional_info=None, conversation_data=None):
    """Log timing information to transcript file"""
    duration = end_time - start_time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        "timestamp": timestamp,
        "step": step_name,
        "duration_seconds": f"{duration:.3f}",
        "start_time": start_time,
        "end_time": end_time
    }
    
    if additional_info:
        log_entry["additional_info"] = additional_info
    
    # Load existing transcript
    transcript_data = load_transcript()
    
    # If this is a new session (first step)
    if step_name == "audio_reception":
        transcript_data["sessions"].append({
            "session_id": len(transcript_data["sessions"]) + 1,
            "start_time": timestamp,
            "steps": [],
            "conversation": []  # Add conversation log
        })
    
    # Add step to current session
    current_session = transcript_data["sessions"][-1]
    current_session["steps"].append(log_entry)
    
    # If this is speech recognition step, add user input to conversation
    if step_name == "speech_recognition" and additional_info and "text" in additional_info:
        current_session["conversation"].append({
            "role": "user",
            "text": additional_info["text"],
            "timestamp": timestamp
        })
    
    # If this is LLM processing step, add LLM response to conversation
    if step_name == "llm_processing" and additional_info and "response" in additional_info:
        current_session["conversation"].append({
            "role": "assistant",
            "text": additional_info["response"],
            "timestamp": timestamp
        })
    
    # If this is the last step, add end time to session
    if step_name == "total_processing":
        current_session["end_time"] = timestamp
        current_session["total_duration"] = f"{duration:.3f}"
    
    # Save updated transcript
    save_transcript(transcript_data)
    
    return duration

app = Flask(__name__)

# Initialize transcript file when application starts
initialize_transcript()

# Initialize models
# Load Whisper model (tiny is faster but less accurate, you can use "base", "small", "medium", or "large" for better accuracy)
logger.info("Loading Whisper model...")
try:
    whisper_model = whisper.load_model("tiny")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

# Initialize VAD model
logger.info("Loading VAD model...")
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=False)
    get_speech_timestamps = utils[0]
    logger.info("VAD model loaded successfully")
except Exception as e:
    logger.error(f"Error loading VAD model: {e}")
    vad_model = None
    get_speech_timestamps = None

def check_voice_activity(audio_path, threshold_seconds=0.5):
    """
    Check if there's significant voice activity in the audio file.
    Returns True if voice detected for more than threshold_seconds.
    """
    if vad_model is None:
        logger.warning("VAD model not loaded, skipping voice check")
        return True
        
    try:
        # Load audio
        wav, sample_rate = sf.read(audio_path)
        # Convert to mono if stereo
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        # Convert to float32
        wav = wav.astype(np.float32)
        
        # Get speech timestamps with more sensitive settings
        speech_timestamps = get_speech_timestamps(
            wav, 
            vad_model,
            threshold=0.3,  # Lower threshold to detect more speech (was 0.4)
            sampling_rate=sample_rate,
            min_speech_duration_ms=100,  # Detect even shorter speech segments (was 250)
            min_silence_duration_ms=300,  # Shorter silences between words (was 500)
            window_size_samples=512  # Smaller window size for more precise detection
        )
        
        # Calculate total speech duration
        total_speech_time = sum(
            (ts['end'] - ts['start']) for ts in speech_timestamps
        ) / sample_rate
        
        logger.info(f"Detected speech duration: {total_speech_time:.2f} seconds")
        logger.info(f"Number of speech segments detected: {len(speech_timestamps)}")
        
        # Lower the threshold for minimum speech duration
        return total_speech_time >= 0.2  # Lower threshold (was 0.5)
        
    except Exception as e:
        logger.error(f"Error in VAD processing: {e}")
        return True  # Default to true on error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    total_start_time = time.time()
    audio_file = request.files['audio']
    
    # Audio file reception
    reception_start = time.time()
    temp_raw_path = tempfile.mktemp(suffix='.webm')
    audio_file.save(temp_raw_path)
    reception_duration = log_to_transcript("audio_reception", reception_start, time.time())
    
    # Audio conversion
    conversion_start = time.time()
    try:
        logger.info(f"Converting audio from {temp_raw_path}")
        audio = AudioSegment.from_file(temp_raw_path)
        temp_wav_path = tempfile.mktemp(suffix='.wav')
        audio.export(temp_wav_path, format="wav")
        logger.info(f"Converted to WAV at {temp_wav_path}")
        
        os.unlink(temp_raw_path)
        conversion_duration = log_to_transcript("audio_conversion", conversion_start, time.time())
        
        # Voice activity detection
        vad_start = time.time()
        if not check_voice_activity(temp_wav_path):
            os.unlink(temp_wav_path)
            vad_duration = log_to_transcript("voice_activity_detection", vad_start, time.time())
            return jsonify({'status': 'no_speech'}), 200
        vad_duration = log_to_transcript("voice_activity_detection", vad_start, time.time())
            
    except Exception as e:
        if os.path.exists(temp_raw_path):
            os.unlink(temp_raw_path)
        logger.error(f"Error converting audio: {e}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    
    # Speech recognition
    recognition_start = time.time()
    try:
        if whisper_model is None:
            return jsonify({'error': 'Whisper model not loaded'}), 500
            
        logger.info("Transcribing with Whisper...")
        result = whisper_model.transcribe(temp_wav_path)
        text_input = result["text"].strip()
        logger.info(f"Transcribed text: {text_input}")
        
        if not text_input:
            os.unlink(temp_wav_path)
            recognition_duration = log_to_transcript("speech_recognition", recognition_start, time.time())
            return jsonify({'error': 'Could not transcribe speech - no text detected'}), 400
        recognition_duration = log_to_transcript("speech_recognition", recognition_start, time.time(), 
                                               additional_info={"text": text_input})
    except Exception as e:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        logger.error(f"Error in transcription: {e}")
        return jsonify({'error': f'Error in transcription: {str(e)}'}), 500
    
    # LLM processing
    llm_start = time.time()
    try:
        logger.info(f"Sending to Ollama: {text_input}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": f"IMPORTANT: Be extremely brief. Respond with only 1-2 very short sentences. No greetings or explanations. Question: {text_input}",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code != 200:
            os.unlink(temp_wav_path)
            llm_duration = log_to_transcript("llm_processing", llm_start, time.time())
            return jsonify({'error': f'Failed to get response from LLM (Status: {response.status_code})'}), 500
        
        llm_response = response.json()["response"]
        logger.info(f"LLM response: {llm_response[:100]}...")
        llm_duration = log_to_transcript("llm_processing", llm_start, time.time(),
                                       additional_info={"response": llm_response})
    except requests.exceptions.RequestException as e:
        os.unlink(temp_wav_path)
        logger.error(f"Error connecting to Ollama: {e}")
        return jsonify({
            'error': 'Failed to connect to Ollama LLM. Make sure Ollama is running and the model is available.',
            'details': str(e)
        }), 500
    except Exception as e:
        os.unlink(temp_wav_path)
        logger.error(f"Error in LLM processing: {e}")
        return jsonify({'error': f'Error in LLM processing: {str(e)}'}), 500
    
    # Text-to-speech generation
    tts_start = time.time()
    try:
        temp_speech_path = tempfile.mktemp(suffix='.wav')
        temp_text_path = tempfile.mktemp(suffix='.txt')
        
        with open(temp_text_path, 'w') as f:
            f.write(llm_response)
        
        voice_model_path = "en_US-lessac-medium.onnx"
        
        if not os.path.exists(voice_model_path):
            logger.error(f"Voice model not found at {voice_model_path}")
            return jsonify({'error': f'Voice model not found at {voice_model_path}. Please download it first.'}), 500
        
        cmd = [
            "piper",
            "--model", voice_model_path,
            "--output_file", temp_speech_path
        ]
        
        logger.info(f"Running TTS command: {' '.join(cmd)}")
        
        with open(temp_text_path, 'r') as text_file:
            process = subprocess.Popen(
                cmd, 
                stdin=text_file,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Piper failed: {stderr.decode()}")
        
        os.unlink(temp_text_path)
        
        # Convert audio to base64
        with open(temp_speech_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        os.unlink(temp_wav_path)
        os.unlink(temp_speech_path)
        
        tts_duration = log_to_transcript("text_to_speech", tts_start, time.time())
        
        # Log total processing time
        total_duration = log_to_transcript("total_processing", total_start_time, time.time(), {
            "reception_duration": reception_duration,
            "conversion_duration": conversion_duration,
            "vad_duration": vad_duration,
            "recognition_duration": recognition_duration,
            "llm_duration": llm_duration,
            "tts_duration": tts_duration
        })
        
        return jsonify({
            'input_text': text_input,
            'llm_response': llm_response,
            'audio': audio_data,
            'processing_time': f"{total_duration:.2f}"
        })
        
    except Exception as e:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        if os.path.exists(temp_speech_path):
            os.unlink(temp_speech_path)
        if os.path.exists(temp_text_path):
            os.unlink(temp_text_path)
        
        logger.error(f"Error in TTS processing: {e}")
        return jsonify({
            'input_text': text_input,
            'llm_response': llm_response,
            'error': f'Error generating speech: {str(e)}',
            'audio': None
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')