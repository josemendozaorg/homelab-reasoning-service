import requests
import sys
import json

URL = "http://192.168.0.160:8090"

def test_reasoning():
    print(f"Testing reasoning service at {URL}...")
    
    # Check health first
    try:
        resp = requests.get(f"{URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"Health check failed: {resp.status_code}")
            sys.exit(1)
        print("Health check passed.")
    except Exception as e:
        print(f"Health check error: {e}")
        sys.exit(1)
        
    # Reason request
    query = "What is 2+2? Explain your reasoning."
    payload = {
        "query": query,
        "max_iterations": 3
    }
    
    print(f"Sending query: {query}")
    try:
        resp = requests.post(f"{URL}/v1/reason", json=payload, timeout=180) # High timeout for LLM
        if resp.status_code == 200:
            data = resp.json()
            print("Reasoning Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Reasoning failed: {resp.status_code} {resp.text}")
            sys.exit(1)
    except Exception as e:
         print(f"Reasoning error: {e}")
         sys.exit(1)

if __name__ == "__main__":
    test_reasoning()
