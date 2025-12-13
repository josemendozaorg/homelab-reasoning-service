import requests
import sys
import json
import time

URL = "https://dko8wckoc0c0o8k8gkwgo8sg.josemendoza.dev"

def test_fast_inference():
    print(f"Testing fast inference at {URL}...")
    
    # Wait loop
    for i in range(10):
        try:
            resp = requests.get(f"{URL}/health", timeout=2)
            if resp.status_code == 200:
                print("Health check passed.")
                break
        except:
            pass
        print(f"Waiting for service... {i+1}/10")
        time.sleep(5)
    
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{URL}/health", timeout=5)
        print(f"Health Status: {resp.status_code}")
    except Exception as e:
        print(f"Health Check Error: {e}")

    print("Testing /v1/test-inference endpoint...")
    try:
        start = time.time()
        resp = requests.post(f"{URL}/v1/test-inference", timeout=10)
        total_time = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            print("Response:")
            print(json.dumps(data, indent=2))
            print(f"Total Client Latency: {total_time:.2f}ms")
            
            if data['duration_ms'] < 2000:
                print("SUCCESS: Inference was fast!")
            else:
                 print(f"WARNING: Inference took {data['duration_ms']}ms")
        else:
            print(f"Failed: {resp.status_code} {resp.text}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    # Check if reason endpoint is at least reachable (422 is reachable)
    print("Testing /v1/reason endpoint (connectivity check)...")
    try:
        resp = requests.post(f"{URL}/v1/reason", json={}, timeout=5)
        print(f"Reason Endpoint Status: {resp.status_code} (Expect 422 for empty body)")
    except Exception as e:
        print(f"Reason Endpoint Error: {e}")
        
    sys.exit(1 if resp.status_code != 200 else 0)

if __name__ == "__main__":
    test_fast_inference()
