import requests
import sys

COOLIFY_URL = "http://192.168.0.160:8000"
API_TOKEN = "3|lgnzzflq0JOXkm04qrJRiW2aoiiNwTli3NcskP3K0da2412c"
APP_UUID = "dko8wckoc0c0o8k8gkwgo8sg"  # From previous output

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def get_logs():
    print(f"Fetching logs for {APP_UUID}...")
    # Based on API docs/patterns, logs usually at /api/v1/applications/{uuid}/logs
    # But previous script suggests this path.
    # Note: If this fails, I might need to check how Coolify API exposes logs.
    # Actually deploy script had: f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/logs"
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{APP_UUID}/logs", headers=headers)
    if resp.status_code == 200:
        print(resp.text)
    else:
        print(f"Failed: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    get_logs()
