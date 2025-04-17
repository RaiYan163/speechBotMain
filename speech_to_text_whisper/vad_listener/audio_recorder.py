import sounddevice as sd
import queue
import numpy as np
import wave
import os
from datetime import datetime

class AudioRecorder:
    def __init__(self, samplerate=16000, block_duration=0.5):
        print("[AudioRecorder] Initializing audio recorder")
        self.samplerate = samplerate
        self.block_duration = block_duration
        self.blocksize = int(samplerate * block_duration)
        self.audio_queue = queue.Queue()
        self.recording = False
        self.temp_file = "temp_chunk.wav"  # Single temporary file for chunks
        self.chunk_dir = "audio_chunks"
        os.makedirs(self.chunk_dir, exist_ok=True)
        self.all_audio_frames = []  # Store all audio frames
        self.recording_count = 0  # Counter for recordings

        # Print device info
        print("\n[AudioRecorder] Using default input device:")
        device_info = sd.query_devices(kind='input')
        print(f"- Name: {device_info['name']}")
        print(f"- Channels: {device_info['max_input_channels']}")
        print(f"- Sample Rate: {device_info['default_samplerate']}")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input stream"""
        if status:
            print(f"[AudioRecorder] Audio callback status: {status}")
        if indata is not None and indata.size > 0:
            self.audio_queue.put(indata.copy())

    def start_recording(self, device=None):
        """Start recording audio"""
        print(f"[AudioRecorder] Starting recording with device: {device if device is not None else 'default'}")
        self.recording = True
        self.all_audio_frames = []  # Clear previous frames
        try:
            self.stream = sd.InputStream(
                device=device,  # None will use the default device
                channels=1,
                samplerate=self.samplerate,
                callback=self.audio_callback,
                blocksize=self.blocksize,
                dtype=np.float32
            )
            self.stream.start()
            print("[AudioRecorder] Audio stream started successfully")
        except Exception as e:
            print(f"[AudioRecorder] Error starting audio stream: {e}")
            raise

    def stop_recording(self):
        """Stop recording audio"""
        print("[AudioRecorder] Stopping recording")
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            print("[AudioRecorder] Audio stream stopped and closed")

    def get_next_chunk(self):
        """Get the next audio chunk from the queue"""
        try:
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is not None and chunk.size > 0:
                    self.all_audio_frames.append(chunk)  # Store the chunk
                    return chunk
            return None
        except queue.Empty:
            return None

    def save_temp_chunk(self, audio_frames):
        """Save audio frames to a temporary WAV file"""
        full_audio = np.concatenate(audio_frames)
        
        with wave.open(self.temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes((full_audio * 32767).astype(np.int16).tobytes())
        
        return self.temp_file

    def save_audio_chunk(self, audio_frames):
        """Save audio frames to a WAV file"""
        self.recording_count += 1
        filename = os.path.join(self.chunk_dir, f"recording_{self.recording_count}.wav")
        
        print(f"[AudioRecorder] Saving complete recording")
        full_audio = np.concatenate(audio_frames)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes((full_audio * 32767).astype(np.int16).tobytes())
        
        print(f"[AudioRecorder] Saved recording to: {filename}")
        return filename 