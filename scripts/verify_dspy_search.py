import requests
import json
import time
import sys

# URL from test_inference.py
URL = "http://localhost:8000"

def verify_search():
    print(f"Testing reasoning service at {URL}...")
    
    # Query about a recent event (or something changing) to trigger search
    query = "What is the current stock price of NVIDIA?"
    payload = {
        "query": query,
        "max_iterations": 3 # Should search quickly
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
            
            # Check for search in trace
            trace = data.get("reasoning_trace", [])
            has_search = False
            for step in trace:
                if "Search Requested" in step or "Search Results" in step:
                    has_search = True
                    break
            
            if has_search:
                print("SUCCESS: Search was triggered.")
                print("Trace snippets:")
                for step in trace:
                    print(step[:200] + "..." if len(step) > 200 else step)
            else:
                print("FAILURE: Search was NOT triggered.")
                print(json.dumps(data, indent=2))
                sys.exit(1)
                
        else:
            print(f"\nRequest failed in {duration:.2f} seconds with status code {resp.status_code}")
            print(resp.text)
            sys.exit(1)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_search()
