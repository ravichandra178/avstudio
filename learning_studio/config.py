from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "learning-studio"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"
    neo4j_database: str = "neo4j"
    s3_bucket: str = ""
    local_storage_path: str = ".local_storage"
    aws_region: str = "us-east-1"
    ai_provider_order: str = "gemini,ollama,github_models,groq,openai"
    question_level: str = "AP Telangana Group 1 screening exam"
    study_group: str = "Group 1"
    fallback_for_textbook: str = ""
    ollama_endpoint: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:14b"
    ollama_keep_alive: str = "10m"
    ollama_timeout_seconds: float = 300
    ollama_context_tokens: int = 8192
    ollama_max_output_tokens: int = 512
    question_generation_timeout_seconds: float = 5
    chat_fallback_timeout_seconds: float = 12
    github_token: str = ""
    github_models_endpoint: str = "https://models.inference.ai.azure.com/chat/completions"
    github_models_name: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_endpoint: str = "https://api.openai.com/v1/chat/completions"
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"
    gemini_model: str = "models/gemini-3.8-flash"
    embedding_endpoint: str = "https://api.openai.com/v1/embeddings"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    max_upload_bytes: int = 50 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def providers(self) -> list[str]:
        return [item.strip().lower() for item in self.ai_provider_order.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
