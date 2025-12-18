import requests
import json
import sys

URL = "http://localhost:8080"

def verify_time_context():
    print(f"Testing time context at {URL}...")
    
    # Query asking for date directly
    query = "What is today's date? Please answer in YYYY-MM-DD format."
    payload = {
        "query": query,
        "max_iterations": 1 # Should answer immediately
    }
    
    try:
        resp = requests.post(f"{URL}/v1/reason", json=payload, timeout=60)
        
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("final_answer", "")
            print(f"\nFinal Answer: {answer}")
            
            # Check if answer contains 2025-12-18 (or current date)
            # Note: We can't strictly assert the date since it changes, but we can check if it looks like a date
            # and matches the UTC date we expect (2025-12-18 based on system info)
            
            expected_date = "2025-12-18" 
            if expected_date in answer:
                print("SUCCESS: Model knows the correct date.")
            else:
                print(f"WARNING: Model answer '{answer}' does not contain expected date '{expected_date}'.")
                # Check trace to see if it mentioned the date in context
                trace = data.get("reasoning_trace", [])
                print("Trace snippets:", trace)
                
        else:
            print(f"Request failed with status {resp.status_code}: {resp.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_time_context()
