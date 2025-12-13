import requests
import sys
import json

COOLIFY_URL = "http://192.168.0.160:8000"
API_TOKEN = "3|lgnzzflq0JOXkm04qrJRiW2aoiiNwTli3NcskP3K0da2412c"
APP_UUID = "dko8wckoc0c0o8k8gkwgo8sg" 

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def check_deployments():
    print(f"Checking deployments for {APP_UUID}...")
    
    # Try to get deployments list
    # Usually /api/v1/applications/{uuid}/deployments or similar? Not standard in v1 docs clearly.
    # Let's try listing general deployments or checking the status properly.
    # Actually, coolify v1 API is a bit sparse. 
    # Let's try getting the application details again to see if 'status' or 'deployment_uuid' is active.
    
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{APP_UUID}", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print("App Details:")
        
        # Decode custom_labels if present
        import base64
        if data.get("custom_labels"):
            try:
                decoded = base64.b64decode(data["custom_labels"]).decode('utf-8')
                data["custom_labels_decoded"] = decoded
            except Exception:
                pass
                
        print(json.dumps(data, indent=2))
        
    # Check if we can get deployment logs for the specific UUID from previous step if we had it.
    # But let's assume we want to see if a build failed.
    pass

if __name__ == "__main__":
    check_deployments()
