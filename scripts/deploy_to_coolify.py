
import requests
import time
import sys

# Configuration
COOLIFY_URL = "http://192.168.0.160:8000"
API_TOKEN = "3|lgnzzflq0JOXkm04qrJRiW2aoiiNwTli3NcskP3K0da2412c"
PROJECT_UUID = "occs0c800c04wckokg8880ss"
ENV_NAME = "production"
SOURCE_ID = 2
REPO_NAME = "josemendozaorg/homelab-reasoning-service"
BRANCH = "feature/langgraph-reasoning"
APP_NAME = "homelab-reasoning-service" # For display/check
DESTINATION_UUID = "b0kc0gkk0wwggwo80wkcooog" # Server UUID

ENVS = {
    "REASONING_OLLAMA_BASE_URL": "http://192.168.0.140:11434",
    "REASONING_OLLAMA_MODEL": "deepseek-r1:14b",
    "REASONING_MAX_REASONING_ITERATIONS": "5",
    "REASONING_MAX_CONTEXT_TOKENS": "16000",
    "REASONING_TEMPERATURE": "0.7",
    "PORT": "8080"
}

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def log(msg):
    print(f"[Deploy] {msg}")

def check_response(resp):
    if resp.status_code >= 400:
        log(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)
    return resp.json()

def find_existing_app():
    log("Checking for existing application...")
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications", headers=headers)
    apps = check_response(resp)
    
    for app in apps:
        # Check against repo name or custom name if possible (though name usually includes random suffix)
        # Check loose match because one is full URL
        if REPO_NAME in app.get("git_repository", ""):
            return app
    return None

def create_app():
    log(f"Creating application for {REPO_NAME}...")
    # Try /api/v1/applications/public for public repos
    payload = {
        "project_uuid": PROJECT_UUID,
        "environment_name": ENV_NAME,
        "server_uuid": DESTINATION_UUID,
        "git_repository": "https://github.com/" + REPO_NAME,
        "git_branch": BRANCH,
        "build_pack": "dockerfile",
        "ports_exposes": "8080", 
        # "ports_mappings": "8080:8080" # REMOVED: Causes bind error on host
    }
    
    endpoint = f"{COOLIFY_URL}/api/v1/applications/public"
    log(f"Trying endpoint: {endpoint}")
    resp = requests.post(endpoint, json=payload, headers=headers)
    return check_response(resp)


def get_envs(app_uuid):
    log(f"Fetching envs for {app_uuid}...")
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/envs", headers=headers)
    return check_response(resp)

def set_env(app_uuid, key, value):
    # First get existing envs to decide on POST (create) vs PATCH (update)
    existing_envs = get_envs(app_uuid)
    log(f"DEBUG: Found {len(existing_envs)} envs: {[e.get('key') for e in existing_envs]}")
    # print(existing_envs) # Uncomment for full debug
    # API v1 returns list of dicts. Structure usually: [{'key': 'FOO', 'value': 'bar', 'uuid': '...'}]
    
    target_env = next((e for e in existing_envs if e.get('key') == key), None)
    
    if target_env:
        log(f"Updating env {key}={value}...")
        env_uuid = target_env.get('uuid')
        
        # Try DELETE then CREATE (Nuclear option since PATCH is failing)
        log(f"Deleting existing env {key} ({env_uuid})...")
        if env_uuid:
             try:
                 requests.delete(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/envs/{env_uuid}", headers=headers)
             except:
                 pass
        
        # Create new
        payload = {
            "key": key,
            "value": value,
            "is_preview": False,
            "is_literal": True
        }
        resp = requests.post(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/envs", json=payload, headers=headers)
             
    else:
        log(f"Creating env {key}={value}...")
        # Create new
        payload = {
            "key": key,
            "value": value,
            "is_preview": False,
            "is_literal": True
        }
        resp = requests.post(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/envs", json=payload, headers=headers)

    check_response(resp)
    return True

def deploy(app_uuid):
    log(f"Triggering deployment for {app_uuid}...")
    
    # Primary method: Restart (deploys latest code)
    # Primary method: Deploy (ensures git pull)
    # url = f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/restart"
    # log(f"Trying {url}...")
    # resp = requests.post(url, headers=headers)
    # if resp.status_code == 200:
    #     data = resp.json()
    #     log(f"Deployment/Restart started: {data}")
    #     return data.get("uuid")
    
    # log(f"Restart failed: {resp.status_code}. Trying direct deploy...")
    
    # Fallback: Query param deploy
    url = f"{COOLIFY_URL}/api/v1/deploy"
    params = {"uuid": app_uuid}
    resp = requests.post(url, params=params, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        log(f"Deployment started: {data}")
        return data.get("uuid")
        
    log(f"All deployment methods failed. Status: {resp.status_code}")
    log(resp.text)
    sys.exit(1)


def get_logs(app_uuid):
    # Try to get build logs
    # Note: URL might need adjustment based on API. 
    # Usually /api/v1/applications/{uuid}/logs or similar.
    log(f"Fetching logs for {app_uuid}...")
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/logs", headers=headers)
    if resp.status_code == 200:
        print(resp.text)
    else:
        log(f"Failed to get logs: {resp.status_code}")

def main():
    app = find_existing_app()
    
    if app:
        log(f"Found existing app: {app['uuid']}")
        app_uuid = app['uuid']
        # get_logs(app_uuid) # Uncomment to debug
    else:
        log("App not found. Creating new...")
        app_data = create_app()
        app_uuid = app_data.get("uuid")
        log(f"App created with UUID: {app_uuid}")
    
    # Clear ports mappings (FQDN update was rejected by API)
    log("Clearing ports mappings...")
    payload = {
        "ports_mappings": None
    }
    # Clear ports mappings (FQDN update was rejected by API)
    log("Updating config (Branch & Ports)...")
    payload = {
        "ports_mappings": None,
        "git_branch": BRANCH
    }
    resp = requests.patch(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}", json=payload, headers=headers)
    log(f"Config update response: {resp.status_code}")
    if resp.status_code != 200:
        log(f"Update failed: {resp.text}")
    
    # Verify update
    check_resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}", headers=headers)
    if check_resp.status_code == 200:
        app_details = check_resp.json()
        log(f"Verified FQDN: {app_details.get('fqdn')}")
        log(f"Verified Ports: {app_details.get('ports_mappings')}")

    
    fqdn = "http://reasoning.coolify-homelab.josemendoza.dev"
    log(f"Application URL should be: {fqdn} (Please configure in UI if missing)")

    # Set Envs
    for k, v in ENVS.items():
        set_env(app_uuid, k, v)
        
    # Deploy
    deploy_uuid = deploy(app_uuid)
    
    log("Waiting for 30s before checking logs...")
    time.sleep(30)
    
    # Try to verify via health check from script logic or just print info
    log("Please check http://192.168.0.160:8080/health manually or via curl.")

if __name__ == "__main__":
    main()
