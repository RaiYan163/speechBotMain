import requests

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    try:
        res = requests.post("http://127.0.0.1:5000/talk", json={"text": user_input})
        res.raise_for_status()  # Catch HTTP errors (500, 404, etc.)

        try:
            data = res.json()
            print("Gemma:", data.get("reply"))
        except requests.exceptions.JSONDecodeError:
            print("Error: Could not decode JSON. Raw response:")
            print(res.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", str(e))
