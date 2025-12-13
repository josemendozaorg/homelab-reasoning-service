import requests
import sys
import time

COOLIFY_URL = "http://192.168.0.160:8000"
API_TOKEN = "3|lgnzzflq0JOXkm04qrJRiW2aoiiNwTli3NcskP3K0da2412c"
APP_UUID = "dko8wckoc0c0o8k8gkwgo8sg" 

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def expose_port():
    print(f"Exposing port 8090:8080 for {APP_UUID}...")
    payload = {
        "ports_mappings": "8090:8080"
    }
    resp = requests.patch(f"{COOLIFY_URL}/api/v1/applications/{APP_UUID}", json=payload, headers=headers)
    
    if resp.status_code == 200:
        print("Port mapping updated.")
    else:
        print(f"Failed to update port mapping: {resp.status_code} {resp.text}")
        sys.exit(1)

    # Restart to apply
    print("Restarting application...")
    resp = requests.post(f"{COOLIFY_URL}/api/v1/applications/{APP_UUID}/restart", headers=headers)
    if resp.status_code == 200:
        print("Restart triggered.")
        restart_uuid = resp.json().get('uuid')
        print(f"Deployment UUID: {restart_uuid}")
    else:
        print(f"Restart failed: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    expose_port()
