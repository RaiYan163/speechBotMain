import subprocess
import wave
import os
import time
import winsound
from typing import Optional, Tuple
from pathlib import Path

class PiperTTS:
    def __init__(self, model_path: str, output_path: str):
        """
        Initialize Piper TTS with model and output paths
        
        Args:
            model_path: Path to the Piper model file (.onnx)
            output_path: Path where audio files will be saved
        """
        self.model_path = os.path.abspath(model_path)
        self.output_path = os.path.abspath(output_path)
        self.piper_exe = "piper.exe"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Validate model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def _remove_non_ascii(self, text: str) -> str:
        """Remove non-ASCII characters from text"""
        return ''.join(char for char in text if ord(char) < 128)

    def _get_audio_duration(self, file_path: str) -> float:
        """Get duration of WAV file in seconds"""
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)

    def generate_speech(self, text: str, play_audio: bool = True, wait_until_done: bool = True) -> Tuple[bool, float, Optional[str]]:
        """
        Generate speech from text and optionally play it
        
        Args:
            text: Text to convert to speech
            play_audio: Whether to play the audio after generating
            wait_until_done: Whether to wait for audio playback to complete
            
        Returns:
            Tuple containing:
            - Success status (bool)
            - Audio duration in seconds (float)
            - Error message if any (str or None)
        """
        try:
            # Clean text
            text = self._remove_non_ascii(text)
            
            # Prepare command
            command = [
                self.piper_exe,
                "-m", self.model_path.replace("\\", "/"),
                "-f", self.output_path.replace("\\", "/")
            ]
            
            print(f"[📤 Running Piper]: {' '.join(command)}")
            
            # Generate audio
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
            duration = self._get_audio_duration(self.output_path)
            print(f"[⏱️ Audio Duration]: {duration:.2f} seconds")
            
            # Play audio if requested
            if play_audio:
                winsound.PlaySound(self.output_path, winsound.SND_FILENAME)
                if wait_until_done:
                    time.sleep(duration)
            
            return True, duration, None
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Piper failed: {str(e)}\nCommand: {e.cmd}\nReturn code: {e.returncode}\nError Output: {e.stderr}"
            print("[❌ Error]:", error_msg)
            return False, 0.0, error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print("[❌ Error]:", error_msg)
            return False, 0.0, error_msg

    def cleanup(self) -> bool:
        """Delete generated audio file"""
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            return True
        except Exception as e:
            print(f"[❌ Cleanup Error]: {str(e)}")
            return False

    @property
    def last_output_path(self) -> str:
        """Get path to last generated audio file"""
        return self.output_path

