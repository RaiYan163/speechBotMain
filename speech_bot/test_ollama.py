import requests
import sys
import json
from config import OLLAMA_BASE_URL

def test_ollama_connection():
    print(f"\nTesting connection to Ollama server at: {OLLAMA_BASE_URL}")
    
    try:
        # Test basic connection
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            print("✅ Successfully connected to Ollama server!")
            print("Available models:", response.json())
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            print("Response:", response.text)
            return False

        # Test model generation
        print("\nTesting model generation...")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": "llama2:7b",
                "prompt": "Tell me something cool about physics.",
                "stream": False  # Explicitly set streaming to false
            }
        )
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if "response" in response_data:
                    print("✅ Successfully tested model generation!")
                    print("Response:", response_data["response"])
                    return True
                else:
                    print("❌ Unexpected response format")
                    print("Full response:", response_data)
                    return False
            except json.JSONDecodeError as e:
                print("❌ Failed to parse response as JSON")
                print("Raw response:", response.text)
                return False
        else:
            print(f"❌ Model generation failed with status code: {response.status_code}")
            print("Response:", response.text)
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server. Please check if:")
        print("1. Ollama is running")
        print("2. The host and port in config.py are correct")
        print("3. Your firewall settings")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Ollama connection test...")
    success = test_ollama_connection()
    sys.exit(0 if success else 1) 