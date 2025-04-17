import torch
from silero_vad import get_speech_timestamps, load_silero_vad
import time

class VADDetector:
    def __init__(self):
        """Initialize the VAD model"""
        print("[VADDetector] Initializing Silero VAD model")
        self.model = load_silero_vad()
        self.speech_started = False
        self.buffer = []
        self.last_voice_time = 0
        print("[VADDetector] Model loaded successfully")

    def process_audio_chunk(self, audio_chunk, sampling_rate=16000):
        """Process an audio chunk and detect speech"""
        if audio_chunk is None:
            print("[VADDetector] Received empty audio chunk")
            return False, None

        # Convert audio to tensor
        audio_tensor = torch.tensor(audio_chunk[:, 0])
        print(f"[VADDetector] Processing audio chunk of shape: {audio_tensor.shape}")
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            self.model, 
            sampling_rate=sampling_rate
        )

        # Check if speech is detected
        speech_detected = len(speech_timestamps) > 0
        if speech_detected:
            print(f"[VADDetector] Speech detected with {len(speech_timestamps)} segments")
        else:
            print("[VADDetector] No speech detected in chunk")
        
        return speech_detected, audio_chunk

    def update_buffer(self, audio_chunk, speech_detected):
        """Update the audio buffer based on speech detection"""
        if speech_detected:
            self.buffer.append(audio_chunk)
            self.speech_started = True
            self.last_voice_time = time.time()
            print(f"[VADDetector] Added chunk to buffer. Buffer size: {len(self.buffer)}")
            return False
        elif self.speech_started and (time.time() - self.last_voice_time > 1.0):
            # Speech ended, return the buffer
            buffer_copy = self.buffer.copy()
            self.buffer = []
            self.speech_started = False
            print(f"[VADDetector] Speech ended. Processing buffer of size: {len(buffer_copy)}")
            return buffer_copy
        return False 