import requests
import json
import time
import sys

# URL from test_inference.py
# URL from test_inference.py
URL = "http://localhost:8000"

def test_strawberry():
    print(f"Testing reasoning service at {URL}...")
    
    query = "how many r's are in Strawberry"
    payload = {
        "query": query,
        "max_iterations": 5
    }
    
    print(f"Sending query: '{query}'")
    start_time = time.time()
    
    try:
        resp = requests.post(f"{URL}/v1/reason", json=payload, timeout=180)
        end_time = time.time()
        duration = end_time - start_time
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\nResponse received in {duration:.2f} seconds:")
            print(json.dumps(data, indent=2))
        else:
            print(f"\nRequest failed in {duration:.2f} seconds with status code {resp.status_code}")
            print(resp.text)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_strawberry()
