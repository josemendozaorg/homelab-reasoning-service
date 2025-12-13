
import requests
import sys

# Configuration
COOLIFY_URL = "http://192.168.0.160:8000"
API_TOKEN = "3|lgnzzflq0JOXkm04qrJRiW2aoiiNwTli3NcskP3K0da2412c"
APP_UUID = "dko8wckoc0c0o8k8gkwgo8sg"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def get_logs():
    print(f"Fetching logs for {APP_UUID}...")
    
    # Get runtime logs
    log_url = f"{COOLIFY_URL}/api/v1/applications/{APP_UUID}/logs?lines=100"
    print(f"Fetching runtime logs from {log_url}")
    resp = requests.get(log_url, headers=headers)
    if resp.status_code == 200:
        print("--- RUNTIME LOGS ---")
        print(resp.text)
        print("--- END RUNTIME LOGS ---")
    else:
        print(f"Failed to get runtime logs: {resp.status_code}")
        print(f"Response: {resp.text}") # Print error message if any



if __name__ == "__main__":
    get_logs()
