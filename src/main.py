"""FastAPI application for the LangGraph Reasoning Service."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import settings
from src.api import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting LangGraph Reasoning Service")
    logger.info(f"Ollama endpoint: {settings.ollama_base_url}")
    logger.info(f"Model: {settings.ollama_model}")
    logger.info(f"Max iterations: {settings.max_reasoning_iterations}")

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
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "LangGraph Reasoning Service",
        "version": "1.0.0",
        "model": settings.ollama_model,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
