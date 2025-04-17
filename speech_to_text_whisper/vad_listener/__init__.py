import threading
import time
from .audio_recorder import AudioRecorder
from .vad_detector import VADDetector
from transcriber import transcribe_file

class VADListener:
    def __init__(self, update_callback=None):
        """Initialize the VAD listener with audio recorder and VAD detector"""
        print("[VADListener] Initializing VAD listener")
        self.audio_recorder = AudioRecorder()
        self.vad_detector = VADDetector()
        self.update_callback = update_callback
        self.recording = False
        self.thread = None
        self.waiting_for_response = False  # Flag to track if we're waiting for LLM response
        self.current_device = None  # Store the current device
        print("[VADListener] Initialization complete")

    def start(self, device=None):
        """Start the VAD listener"""
        if self.recording:
            print("[VADListener] Already recording, ignoring start command")
            return

        print("[VADListener] Starting VAD listener")
        self.recording = True
        self.current_device = device  # Store the device
        self.audio_recorder.start_recording(device)
        
        def process_loop():
            print("[VADListener] Starting processing loop")
            empty_chunk_count = 0
            while self.recording:
                try:
                    # Check if we're waiting for LLM response
                    if self.waiting_for_response:
                        time.sleep(0.1)  # Sleep for a bit to avoid busy waiting
                        continue

                    # Get next audio chunk
                    chunk = self.audio_recorder.get_next_chunk()
                    
                    if chunk is None:
                        empty_chunk_count += 1
                        if empty_chunk_count % 100 == 0:  # Print every 100 empty chunks
                            print(f"[VADListener] Received {empty_chunk_count} empty chunks")
                        time.sleep(0.01)  # Small delay to prevent CPU overuse
                        continue
                    
                    empty_chunk_count = 0  # Reset counter when we get a valid chunk
                    
                    # Process with VAD
                    speech_detected, audio_chunk = self.vad_detector.process_audio_chunk(
                        chunk, 
                        self.audio_recorder.samplerate
                    )
                    
                    # Update buffer and check for completed speech
                    completed_buffer = self.vad_detector.update_buffer(audio_chunk, speech_detected)
                    
                    if completed_buffer:
                        print("[VADListener] Processing completed speech segment")
                        # Save to temporary file
                        filename = self.audio_recorder.save_temp_chunk(completed_buffer)
                        
                        # Transcribe the audio file
                        try:
                            transcribed_text = transcribe_file(filename)
                            if transcribed_text and self.update_callback:
                                print("[VADListener] Calling update callback with transcribed text")
                                self.waiting_for_response = True  # Set flag to wait for response
                                # Stop recording while waiting for response
                                self.audio_recorder.stop_recording()
                                self.update_callback(transcribed_text)
                        except Exception as e:
                            print(f"[VADListener] Error transcribing audio: {e}")

                except Exception as e:
                    print(f"[VADListener] Error in processing loop: {e}")
                    break

        # Start processing in a separate thread
        self.thread = threading.Thread(target=process_loop, daemon=True)
        self.thread.start()
        print("[VADListener] Processing thread started")

    def resume_listening(self):
        """Resume speech detection after receiving LLM response"""
        print("[VADListener] Resuming speech detection")
        self.waiting_for_response = False
        # Restart recording with the same device
        if not self.audio_recorder.recording:
            self.audio_recorder.start_recording(self.current_device)

    def stop(self):
        """Stop the VAD listener"""
        print("[VADListener] Stopping VAD listener")
        self.recording = False
        self.audio_recorder.stop_recording()
        if self.thread:
            self.thread.join()
            print("[VADListener] Processing thread joined") 