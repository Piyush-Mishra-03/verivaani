import requests
import json

SCALEDOWN_API_KEY = "UpYGg7q97W9CbWfctJ1XA3y6e06QwyZ18WhtjVLx"  # paste your key here
SCALEDOWN_URL = "https://api.scaledown.xyz/compress/raw/"

headers = {
    "x-api-key": SCALEDOWN_API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "context": "You are a helpful assistant.",
    "prompt": "What language is this text written in? Reply with ONE word only. Text: भारत की GDP 20% बढ़ी है",
    "model": "gpt-4o",
    "scaledown": {
        "rate": "auto"
    }
}

print("Sending request to ScaleDown...")
response = requests.post(SCALEDOWN_URL, headers=headers, json=payload, timeout=30)
result = response.json()

print("\n=== FULL RESPONSE ===")
print(json.dumps(result, indent=2))

print("\n=== ALL KEYS IN RESPONSE ===")
for key, value in result.items():
    print(f"  {key}: {str(value)[:100]}")