import re
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

def clean_response(text):
    text = text.strip()

    
    disallowed_prefixes = [
        "Sure, here is the answer:",
        "Sure, here's an answer:",
        "Answer:",
        "Sure,",
        "Here's the answer:",
        "here's the answer:",
        "This is the answer:",
    ]
    for prefix in disallowed_prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()

   
    match = re.match(
        r"(?i)^sure,\s*(here('|')?s|this is)?\s*(an\s)?answer:?\s*",
        text
    )
    if match:
        return text[match.end():].strip()

    return text

def call_llm(prompt):
    instruction = (
        "Strictly answer the question in 2 short sentences max using only English alphabet letters. "
        "No symbols or emojis. You generate answer, nothing else, like a part of a conversation. "
        "Give direct answer, no additional sentences. Here is the question:\n\n"
    )
    final_prompt = instruction + prompt

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": False
        })

        if response.status_code == 200:
            raw_response = response.json().get("response", "[No response text]")
            return clean_response(raw_response)
        else:
            raise Exception(f"Ollama returned status {response.status_code}: {response.text}")

    except Exception as e:
        return f"[Error calling LLM: {str(e)}]"
