import requests
import json
from typing import Dict, Any
import os
from datetime import datetime

class LLMConnector:
    def __init__(self, server_url: str):
        """
        Initialize the server connector.
        
        Args:
            server_url (str): The URL of the server endpoint (e.g., "http://127.0.0.1:5000/talk")
        """
        self.server_url = server_url
        self.transcription_file = "transcription.json"
        
        # Initialize transcription file if it doesn't exist
        if not os.path.exists(self.transcription_file):
            with open(self.transcription_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def save_conversation(self, question: str, answer: str) -> None:
        """
        Save the question (transcribed text) and answer (server response) to transcription.json
        
        Args:
            question (str): The text transcribed from audio (question)
            answer (str): The response received from the server (answer)
        """
        try:
            # Read existing conversations
            with open(self.transcription_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            # Add new conversation
            conversations.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,  # The transcribed text from audio
                "answer": answer      # The response from the server
            })
            
            # Save updated conversations
            with open(self.transcription_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")

    def send_transcription(self, transcribed_text: str) -> str:
        """
        Send transcribed text to the server and get response.
        
        Args:
            transcribed_text (str): The text that was transcribed from audio
            
        Returns:
            str: The response from the server
        """
        try:
            # Make the API request with the transcribed text
            response = requests.post(
                self.server_url,
                json={"text": transcribed_text}
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            try:
                # Parse the JSON response
                data = response.json()
                server_response = data.get("reply", "No reply received")
                
                # Save the transcribed text and server response
                self.save_conversation(transcribed_text, server_response)
                
                return server_response
            except json.JSONDecodeError:
                print("Error: Could not decode JSON. Raw response:")
                print(response.text)
                return "Error: Invalid response format"
                
        except requests.exceptions.RequestException as e:
            print("Request failed:", str(e))
            return f"Error: {str(e)}"
        except Exception as e:
            print("Unexpected error:", str(e))
            return "Error: An unexpected error occurred" 