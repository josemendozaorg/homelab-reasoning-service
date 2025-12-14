
import requests
import time
import sys
import argparse
import os
import base64

# Configuration
COOLIFY_URL = "http://192.168.0.160:8000"
# NOTE: API_TOKEN must be set in environment variables (CI/CD)
API_TOKEN = os.environ.get("COOLIFY_API_TOKEN")

if not API_TOKEN:
    print("Error: COOLIFY_API_TOKEN environment variable not set.")
    sys.exit(1)

PROJECT_UUID = "occs0c800c04wckokg8880ss"
SOURCE_ID = 2
REPO_NAME = "josemendozaorg/homelab-reasoning-service"
DESTINATION_UUID = "b0kc0gkk0wwggwo80wkcooog" # Server UUID

# Base Envs
BASE_ENVS = {
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

def get_app_name(branch, env):
    # Sanitize branch name for URL/Coolify naming
    safe_branch = branch.lower().replace("/", "-").replace("_", "-")
    if env == "production":
        return "homelab-reasoning" # Production name
    elif env == "development":
        return "homelab-reasoning-dev"
    else:
        return f"homelab-reasoning-{safe_branch}"

def get_domain(branch, env):
    safe_branch = branch.lower().replace("/", "-").replace("_", "-")
    if env == "production":
        return "https://reasoning.josemendoza.dev"
    elif env == "development":
        return "https://reasoning-dev.josemendoza.dev"
    else:
        # Preview
        return f"https://reasoning-{safe_branch}.josemendoza.dev"

def find_existing_app(app_name):
    log(f"Checking for existing application '{app_name}'...")
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications", headers=headers)
    apps = check_response(resp)
    
    for app in apps:
        # Check matching name 
        if app.get("name") == app_name:
            return app
        # Fallback: check if the derived domain matches matches 
        # (This is harder reliably without query, so relying on name/repo combo if possible)
        # But Coolify adds random suffixes often.
        # Let's try to match by git_branch AND repo if name match fails
        pass 
        
    return None

# NOTE: Since Coolify might not allow searching by name efficiently, we might need to rely on creating and handling 
# the "already exists" error or iterating all. iterating all is safer.
def find_app_by_branch_and_repo(branch):
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications", headers=headers)
    apps = check_response(resp)
    for app in apps:
        # Ensure we are looking at this repo
        if REPO_NAME in app.get("git_repository", ""):
            # And specific branch
            if app.get("git_branch") == branch:
                return app
    return None

def create_app(branch, domain):
    log(f"Creating application for {REPO_NAME} (Branch: {branch})...")
    payload = {
        "project_uuid": PROJECT_UUID,
        "environment_name": "production", # Using 'production' env in Coolify logic, though logically it maps to our env
        "server_uuid": DESTINATION_UUID,
        "git_repository": "https://github.com/" + REPO_NAME,
        "git_branch": branch,
        "build_pack": "dockerfile",
        "ports_exposes": "8080", 
        "name": get_app_name(branch, "preview" if "feature" in branch else "production"), # Attempt to set name
        "description": f"Auto-deployed for branch {branch}"
    }
    
    endpoint = f"{COOLIFY_URL}/api/v1/applications/public"
    resp = requests.post(endpoint, json=payload, headers=headers)
    return check_response(resp)


def get_envs(app_uuid):
    log(f"Fetching envs for {app_uuid}...")
    resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/envs", headers=headers)
    return check_response(resp)

def set_env(app_uuid, key, value):
    existing_envs = get_envs(app_uuid)
    target_env = next((e for e in existing_envs if e.get('key') == key), None)
    
    if target_env:
        # Only update if changed
        if target_env.get('value') == value:
            return True
            
        log(f"Updating env {key}...")
        env_uuid = target_env.get('uuid')
        # Delete old
        requests.delete(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}/envs/{env_uuid}", headers=headers)
        
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
    url = f"{COOLIFY_URL}/api/v1/deploy"
    params = {"uuid": app_uuid}
    resp = requests.post(url, params=params, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        log(f"Deployment started: {data.get('uuid')}")
        return data.get("uuid")
        
    log(f"Deployment failed. Status: {resp.status_code}")
    log(resp.text)
    sys.exit(1)

def configure_traefik(app_uuid, fqdn):
    hostname = fqdn.replace("http://", "").replace("https://", "").strip("/")
    log(f"Configuring Traefik labels for host: {hostname}")
    
    labels = f"""traefik.enable=true
traefik.http.middlewares.gzip.compress=true
traefik.http.middlewares.redirect-to-https.redirectscheme.scheme=https
traefik.http.routers.http-0-{app_uuid}.entryPoints=http
traefik.http.routers.http-0-{app_uuid}.middlewares=gzip
traefik.http.routers.http-0-{app_uuid}.rule=Host(`{hostname}`) && PathPrefix(`/`)
traefik.http.routers.http-0-{app_uuid}.service=http-0-{app_uuid}
traefik.http.routers.https-0-{app_uuid}.entryPoints=https
traefik.http.routers.https-0-{app_uuid}.rule=Host(`{hostname}`) && PathPrefix(`/`)
traefik.http.routers.https-0-{app_uuid}.service=http-0-{app_uuid}
traefik.http.routers.https-0-{app_uuid}.tls=true
traefik.http.routers.https-0-{app_uuid}.tls.certresolver=letsencrypt
traefik.http.services.http-0-{app_uuid}.loadbalancer.server.port=8080
traefik.docker.network=coolify
caddy_0.encode=zstd gzip
caddy_0.handle_path.0_reverse_proxy={{upstreams 8080}}
caddy_0.handle_path=/*
caddy_0.header=-Server
caddy_0.try_files={{path}} /index.html /index.php
caddy_0={fqdn}
caddy_ingress_network=coolify"""

    encoded_labels = base64.b64encode(labels.encode('utf-8')).decode('utf-8')
    
    label_payload = {
        "custom_labels": encoded_labels
    }
    requests.patch(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}", json=label_payload, headers=headers)

def main():
    parser = argparse.ArgumentParser(description='Deploy to Coolify')
    parser.add_argument('--branch', required=True, help='Git branch to deploy')
    parser.add_argument('--env', required=True, choices=['preview', 'development', 'production'], help='Target environment')
    args = parser.parse_args()
    
    branch = args.branch
    env = args.env
    
    desired_domain = get_domain(branch, env)
    log(f"Starting deployment for Branch: {branch} -> Env: {env}")
    log(f"Target Domain: {desired_domain}")
    
    # Check if app exists for this branch
    app = find_app_by_branch_and_repo(branch)
    
    if app:
        log(f"Found existing app: {app['uuid']}")
        app_uuid = app['uuid']
    else:
        log("App not found for this branch. Creating new...")
        app_data = create_app(branch, desired_domain)
        app_uuid = app_data.get("uuid")
        log(f"App created with UUID: {app_uuid}")
    
    # Update General Config (Branch)
    log("Updating config (Branch)...")
    payload = {
        "git_branch": branch,
        "ports_mappings": None # Clean up potentially bad ports
    }
    requests.patch(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}", json=payload, headers=headers)
    
    # Update Domain (if different)
    check_resp = requests.get(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}", headers=headers)
    if check_resp.status_code == 200:
        current_data = check_resp.json()
        if current_data.get('fqdn') != desired_domain:
             log(f"Updating FQDN to {desired_domain}...")
             # Coolify API for FQDN update usually in general patch
             requests.patch(f"{COOLIFY_URL}/api/v1/applications/{app_uuid}", json={"fqdn": desired_domain}, headers=headers)

    # Configure Traefik Labels (CRITICAL for SSL/Routing)
    configure_traefik(app_uuid, desired_domain)
    
    # Set Envs
    for k, v in BASE_ENVS.items():
        set_env(app_uuid, k, v)
        
    # Deploy
    deploy_uuid = deploy(app_uuid)
    
    log(f"Deployment triggered successfully! UUID: {deploy_uuid}")
    log(f"Monitor at: {desired_domain}")

if __name__ == "__main__":
    main()
