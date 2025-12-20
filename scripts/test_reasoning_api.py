
import requests
import json
import sys

def test_stream(query="What date is today?"):
    url = "http://localhost:8080/v1/reason/stream"
    payload = {
        "query": query,
        "max_iterations": 5,
        "history": []
    }
    
    print(f"Testing Query: {query}")
    print("-" * 50)
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                            
                            # Check for token event
                            if 'token' in data:
                                token = data['token']
                                node = data.get('node', 'unknown')
                                # Print token with node indicator
                                # prevent newline for tokens to see stream
                                if node == 'mcts_final':
                                    print(f"\n[FINAL ANSWER TOKEN]: {token}", end='', flush=True)
                                else:
                                    # print(f"[{node}] {token}", end='', flush=True)
                                    pass 
                                    
                            # Check for other events
                            elif 'message' in data:
                                print(f"\n[MESSAGE]: {data['message']}")
                                
                        except json.JSONDecodeError:
                            print(f"\n[RAW DATA]: {data_str}")
                    elif decoded_line.startswith('event: '):
                        event_type = decoded_line[7:]
                        # print(f"\n[EVENT]: {event_type}")
                        if event_type == 'done':
                            print("\n[DONE]")
                            
    except Exception as e:
        print(f"\nExample failed: {e}")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What date is today?"
    test_stream(query)
