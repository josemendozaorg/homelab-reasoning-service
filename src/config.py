"""Configuration management for the reasoning service."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_base_url: str = "http://192.168.0.140:11434"
    ollama_model: str = "deepseek-r1:14b"

    # Reasoning Configuration
    max_reasoning_iterations: int = 5
    max_context_tokens: int = 16000
    temperature: float = 0.7

    # API Configuration
    api_host: str = "0.0.0.0"
    # Application Version
    # Application Version
    app_version: str = "0.2.0"
    commit_hash: str = "local"

    class Config:
        env_prefix = "REASONING_"
        env_file = ".env"


settings = Settings()

