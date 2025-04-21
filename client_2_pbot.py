# app.py
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import soundfile as sf
import io
import requests
import tempfile
import os
import whisper
import time
from pydub import AudioSegment
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
        
        # Get speech timestamps with more aggressive settings
        speech_timestamps = get_speech_timestamps(
            wav, 
            vad_model,
            threshold=0.4,  # Lower threshold to detect more speech
            sampling_rate=sample_rate,
            min_speech_duration_ms=250,  # Detect shorter speech segments
            min_silence_duration_ms=500  # Shorter silences between words
        )
        
        # Calculate total speech duration
        total_speech_time = sum(
            (ts['end'] - ts['start']) for ts in speech_timestamps
        ) / sample_rate
        
        logger.info(f"Detected speech duration: {total_speech_time:.2f} seconds")
        return total_speech_time >= threshold_seconds
        
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
    
    start_time = time.time()
    audio_file = request.files['audio']
    
    # First save the blob as is
    temp_raw_path = tempfile.mktemp(suffix='.webm')
    audio_file.save(temp_raw_path)
    
    # Convert to WAV using pydub
    try:
        logger.info(f"Converting audio from {temp_raw_path}")
        audio = AudioSegment.from_file(temp_raw_path)
        temp_wav_path = tempfile.mktemp(suffix='.wav')
        audio.export(temp_wav_path, format="wav")
        logger.info(f"Converted to WAV at {temp_wav_path}")
        
        # Clean up the raw file
        os.unlink(temp_raw_path)
        
        # Check for voice activity
        if not check_voice_activity(temp_wav_path):
            os.unlink(temp_wav_path)
            return jsonify({'status': 'no_speech'}), 200
            
    except Exception as e:
        if os.path.exists(temp_raw_path):
            os.unlink(temp_raw_path)
        logger.error(f"Error converting audio: {e}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    
    # Process with Whisper
    try:
        if whisper_model is None:
            return jsonify({'error': 'Whisper model not loaded'}), 500
            
        logger.info("Transcribing with Whisper...")
        result = whisper_model.transcribe(temp_wav_path)
        text_input = result["text"].strip()
        logger.info(f"Transcribed text: {text_input}")
        
        if not text_input:
            os.unlink(temp_wav_path)
            return jsonify({'error': 'Could not transcribe speech - no text detected'}), 400
    except Exception as e:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        logger.error(f"Error in transcription: {e}")
        return jsonify({'error': f'Error in transcription: {str(e)}'}), 500
    
    # Send text to Ollama LLM with instruction for short responses
    try:
        logger.info(f"Sending to Ollama: {text_input}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": f"IMPORTANT: Be extremely brief. Respond with only 1-2 very short sentences. No greetings or explanations. Question: {text_input}",
                "stream": False
            },
            timeout=30  # Add timeout to prevent hanging
        )
        
        if response.status_code != 200:
            os.unlink(temp_wav_path)
            return jsonify({'error': f'Failed to get response from LLM (Status: {response.status_code})'}), 500
        
        llm_response = response.json()["response"]
        # print(f"LLM RESPONSE: {llm_response}")  # Print to console
        # logger.info(f"LLM response: {llm_response}")

        
        try:
            api_response = requests.post(
                "http://localhost:5001/api/receive",  # Replace with actual API URL
                json={
                    "response": llm_response,
                    "input_text": text_input
                },
                timeout=3
            )
            logger.info(f"API response: {api_response.status_code}")
        except Exception as e:
            logger.error(f"API error: {e}")
            
        # Clean up files
        os.unlink(temp_wav_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        return jsonify({
            'input_text': text_input,
            'llm_response': llm_response,
            'processing_time': f"{processing_time:.2f}"
        })
        
    except requests.exceptions.RequestException as e:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        logger.error(f"Error connecting to Ollama: {e}")
        return jsonify({
            'error': 'Failed to connect to Ollama LLM. Make sure Ollama is running and the model is available.',
            'details': str(e)
        }), 500
    except Exception as e:
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        logger.error(f"Error in LLM processing: {e}")
        return jsonify({'error': f'Error in LLM processing: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')