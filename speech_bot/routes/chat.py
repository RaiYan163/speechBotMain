from flask import Blueprint, request, jsonify
from services.LLM_client import call_llm
from services.text_to_speech import text_to_speech
import traceback

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/talk", methods=["POST"])
def talk():
    try:
        data = request.get_json()
        prompt = data.get("text", "").strip()

        if not prompt:
            return jsonify({"error": "Empty input"}), 400

        print("[User Input]:", prompt)

        # Get LLM response
        response = call_llm(prompt)
        print("[LLM Response]:", response)

        # Convert to speech and play - this will block until audio finishes
        audio_duration = text_to_speech(response)
        print(f"[Audio Playback Complete] Duration: {audio_duration:.2f} seconds")

        # Only after audio finishes, send the response
        return jsonify({"reply": response})

    except Exception as e:
        print("[ERROR in /talk]:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
