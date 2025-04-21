import tkinter as tk
from tkinter import messagebox, scrolledtext
import json
import os
from datetime import datetime
import requests
import threading
import sounddevice as sd
import numpy as np
import wave
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
import whisper
import time
import queue

class SpeechLLMSystem:
    def __init__(self, server_url="http://localhost:11434/api/generate"):
        """
        Initialize the Speech-to-LLM system
        
        Args:
            server_url (str): URL for the LLM server
            model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
        """
        # Audio settings
        self.samplerate = 16000
        self.block_duration = 0.5
        self.blocksize = int(self.samplerate * self.block_duration)
        self.audio_queue = queue.Queue()
        self.recording = False
        self.waiting_for_response = False
        
        # Initialize models
        self.whisper_model = whisper.load_model("tiny")
        self.vad_model = load_silero_vad()
        
        # State variables
        self.speech_started = False
        self.buffer = []
        self.last_voice_time = 0
        self.last_transcription = ""
        
    def get_last_transcription(self):
        return self.last_transcription
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        if indata is not None and indata.size > 0:
            self.audio_queue.put(indata.copy())
            
    def start_recording(self, device=None):
        """Start recording audio"""
        if self.recording:
            return
            
        self.recording = True
        self.buffer = []
        
        try:
            self.stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.samplerate,
                callback=self.audio_callback,
                blocksize=self.blocksize,
                dtype=np.float32
            )
            self.stream.start()
            
            # Start processing in a separate thread
            self.process_thread = threading.Thread(target=self.process_audio_loop)
            self.process_thread.daemon = True
            self.process_thread.start()
            
        except Exception as e:
            raise Exception(f"Failed to start recording: {str(e)}")
            
    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
    def process_audio_loop(self):
        """Main audio processing loop"""
        while self.recording:
            if self.waiting_for_response:
                time.sleep(0.1)
                continue
                
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    continue
                    
                # Process with VAD
                audio_tensor = torch.tensor(chunk[:, 0])
                speech_timestamps = get_speech_timestamps(
                    audio_tensor,
                    self.vad_model,
                    sampling_rate=self.samplerate
                )
                
                speech_detected = len(speech_timestamps) > 0
                if speech_detected:
                    self.buffer.append(chunk)
                    self.speech_started = True
                    self.last_voice_time = time.time()
                elif self.speech_started and (time.time() - self.last_voice_time > 1.0):
                    self.process_completed_speech()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                
    def process_completed_speech(self):
        """Process completed speech segment"""
        if not self.buffer:
            return
            
        # Save audio to temp file
        full_audio = np.concatenate(self.buffer)
        with wave.open(self.temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes((full_audio * 32767).astype(np.int16).tobytes())
            
        # Transcribe
        result = self.whisper_model.transcribe(self.temp_file)
        transcribed_text = result["text"].strip()
        
        if transcribed_text and self.on_transcription_callback:
            self.on_transcription_callback(transcribed_text)
            
        # Get LLM response
        self.waiting_for_response = True
        response = self.send_to_llm(transcribed_text)
        
        if response and self.on_response_callback:
            self.on_response_callback(response)
            
        # Save conversation
        self.save_conversation(transcribed_text, response)
        
        # Reset state
        self.buffer = []
        self.speech_started = False
        self.waiting_for_response = False
        
    def send_to_llm(self, text):
        """Send text to LLM server and get response"""
        try:
            response = requests.post(
                self.server_url,
                json={"text": text}
            )
            response.raise_for_status()
            return response.json().get("reply", "No reply received")
        except Exception as e:
            return f"Error: {str(e)}"
            
    def save_conversation(self, question, answer):
        """Save conversation to JSON file"""
        try:
            with open(self.transcription_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
                
            conversations.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": answer
            })
            
            with open(self.transcription_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")


