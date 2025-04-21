from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/api/receive', methods=['POST'])
def receive_llm_response():
    data = request.json
    print("LLM RESPONSE RECEIVED:")
    print(f"Input: {data.get('input_text')}")
    print(f"Response: {data.get('response')}")
    return {"status": "success"}, 200

if __name__ == '__main__':
    # Run on a different port than your main app
    app.run(host='0.0.0.0', port=5001, debug=True)