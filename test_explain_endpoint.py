import httpx
import json

url = "http://127.0.0.1:8000/api/investigate/stream"
headers = {
    "X-API-Key": "your_fraud_api_key_here",
    "Content-Type": "application/json"
}

body = {
    "transaction_id": "TXN-123456",
    "risk_score": 85.5,
    "amount": 450.0,
    "hour": 2,
    "merchant_category": "ATM",
    "shap_features": [
        {"feature": "V14", "value": -2.3, "impact": 2.3},
        {"feature": "V4", "value": 1.8, "impact": 1.8},
        {"feature": "V17", "value": -1.5, "impact": 1.5}
    ],
    "features": {
        "v14": -2.3,
        "v4": 1.8,
        "v17": -1.5,
        "v10": 0.5
    }
}

print("Sending request to explain endpoint...")
try:
    with httpx.stream("POST", url, headers=headers, json=body, timeout=30.0) as response:
        print("Status code:", response.status_code)
        if response.status_code != 200:
            print("Error response:", response.read().decode())
        else:
            for line in response.iter_lines():
                if line:
                    print("Line:", line)
except Exception as e:
    import traceback
    traceback.print_exc()
