import tkinter as tk
from tkinter import messagebox, scrolledtext
from vad_listener import VADListener
from middleware import LLMConnector
import os
import json
from datetime import datetime

class SpeechApp:
    def __init__(self, root):
        """Initialize the SpeechApp"""
        self.root = root
        self.root.title("Speech to Text with LLM")
        
        # Initialize fresh transcription.json
        self.initialize_transcription_file()
        
        # Initialize VAD listener
        self.vad_listener = VADListener(self.process_transcription)
        
        # Initialize server connector
        self.server_connector = LLMConnector(
            server_url="http://127.0.0.1:5000/talk"  # Default local server URL
        )
        
        # Create GUI elements
        self.create_widgets()
        
        # Set initial state
        self.is_recording = False

    def initialize_transcription_file(self):
        """Create a fresh transcription.json file"""
        try:
            # Create an empty list in a new transcription.json file
            with open("transcription.json", "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("[GUI] Created fresh transcription.json file")
        except Exception as e:
            print(f"[GUI] Error creating transcription.json: {e}")
            messagebox.showerror("Error", f"Failed to create transcription.json: {str(e)}")
            raise

    def create_widgets(self):
        # Create main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control buttons frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Center buttons container
        button_container = tk.Frame(control_frame)
        button_container.pack(expand=True)
        
        # Start Recording button (now "Talk")
        self.start_button = tk.Button(
            button_container,
            text="Talk",
            command=self.start_recording,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop Recording button (now just "Stop")
        self.stop_button = tk.Button(
            button_container,
            text="Stop",
            command=self.stop_recording,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(
            main_frame,
            text="Ready to start",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=5)
        
        # Create conversation display area
        self.create_conversation_display(main_frame)

    def create_conversation_display(self, parent):
        # Create frame for conversation display
        display_frame = tk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create scrollbar
        scrollbar = tk.Scrollbar(display_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create text widget for conversation
        self.conversation_text = scrolledtext.ScrolledText(
            display_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Arial", 11)
        )
        self.conversation_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=self.conversation_text.yview)
        
        # Make text widget read-only
        self.conversation_text.config(state=tk.DISABLED)

    def load_previous_conversations(self):
        """Load and display previous conversations from transcription.json"""
        try:
            if os.path.exists("transcription.json"):
                with open("transcription.json", 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                    
                for conv in conversations:
                    self.update_conversation_display(f"Time: {conv['timestamp']}\n")
                    self.update_conversation_display(f"Question: {conv['question']}\n")
                    self.update_conversation_display(f"Answer: {conv['answer']}\n")
                    self.update_conversation_display("-" * 50 + "\n")
        except Exception as e:
            print(f"Error loading previous conversations: {str(e)}")

    def start_recording(self):
        try:
            self.vad_listener.start()
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Recording... Speak now")
            self.update_conversation_display("System: Recording started\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")

    def stop_recording(self):
        try:
            self.vad_listener.stop()
            self.is_recording = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Recording stopped")
            self.update_conversation_display("System: Recording stopped\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")

    def process_transcription(self, transcribed_text):
        """Process transcribed text and handle LLM response"""
        if not transcribed_text:
            print("[GUI] No transcribed text received")
            return
            
        try:
            print("[GUI] Processing transcribed text")
            # Disable recording button while waiting for response
            self.start_button.config(state=tk.DISABLED)
            self.status_label.config(text="Waiting for LLM response...")
            
            # Display transcribed text (question) in bold
            self.update_conversation_display(f"[{datetime.now().strftime('%H:%M:%S')}] ")
            self.update_conversation_display("Transcribed: ", bold=True)
            self.update_conversation_display(f"{transcribed_text}\n")
            
            # Send transcribed text to server and get response (answer)
            print("[GUI] Sending text to LLM server")
            response = self.server_connector.send_transcription(transcribed_text)
            
            if response:
                print("[GUI] Received response from LLM server")
                # Display server's response (answer) in bold
                self.update_conversation_display(f"[{datetime.now().strftime('%H:%M:%S')}] ")
                self.update_conversation_display("LLM Response: ", bold=True)
                self.update_conversation_display(f"{response}\n")
                self.update_conversation_display("-" * 50 + "\n")
            else:
                print("[GUI] No response received from LLM server")
                self.update_conversation_display("System: No response received from server\n")
                
            # Resume listening for new speech
            print("[GUI] Resuming speech detection")
            self.vad_listener.resume_listening()
            
            # Re-enable recording button
            self.start_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ready to start")
            
        except Exception as e:
            print(f"[GUI] Error processing transcription: {e}")
            self.update_conversation_display(f"System: Error processing transcription: {str(e)}\n")
            # Resume listening even if there's an error
            self.vad_listener.resume_listening()
            # Re-enable recording button
            self.start_button.config(state=tk.NORMAL)
            self.status_label.config(text="Ready to start")

    def update_conversation_display(self, text, bold=False):
        """Update the conversation display with optional bold text"""
        self.conversation_text.config(state=tk.NORMAL)
        if bold:
            self.conversation_text.tag_configure("bold", font=("Arial", 11, "bold"))
            self.conversation_text.insert(tk.END, text, "bold")
        else:
            self.conversation_text.insert(tk.END, text)
        self.conversation_text.see(tk.END)
        self.conversation_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechApp(root)
    root.mainloop()


