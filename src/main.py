"""FastAPI application for the LangGraph Reasoning Service."""
import logging
from contextlib import asynccontextmanager
import hashlib

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.config import settings
from src.api import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of a file."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except FileNotFoundError:
        return "0"

# Calculate hashes at startup
STYLE_HASH = get_file_hash("src/static/style.css")
APP_HASH = get_file_hash("src/static/app.js")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting LangGraph Reasoning Service")
    logger.info(f"Ollama endpoint: {settings.ollama_base_url}")
    logger.info(f"Model: {settings.ollama_model}")
    logger.info(f"Max iterations: {settings.max_reasoning_iterations}")
    logger.info(f"Static Assets - Style Hash: {STYLE_HASH}, App Hash: {APP_HASH}")

    yield

    # Shutdown
    logger.info("Shutting down LangGraph Reasoning Service")


app = FastAPI(
    title="LangGraph Reasoning Service",
    description="""
    Inference-Time Scaling with Self-Correcting Reasoning.

    This service implements System 2 reasoning using:
    - DeepSeek-R1's native <think> tokens for chain-of-thought
    - LangGraph for stateful workflow management
    - Self-correction loop: Reason → Critique → Refine

    Designed for complex reasoning tasks that benefit from
    deliberative, iterative problem-solving.
    """,
    version=settings.app_version,
    lifespan=lifespan
)

# Include API routes
app.include_router(router)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/api/info")
async def info():
    """Return service information."""
    return {
        "service": "LangGraph Reasoning Service",
        "version": settings.app_version,
        "model": settings.ollama_model
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the UI with dynamic cache busting."""
    try:
        with open('src/static/index.html', 'r') as f:
            html_content = f.read()
        
        # Inject dynamic versions
        html_content = html_content.replace("{{STYLE_VERSION}}", STYLE_HASH)
        html_content = html_content.replace("{{APP_VERSION}}", APP_HASH)
        
        return HTMLResponse(
            content=html_content, 
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
