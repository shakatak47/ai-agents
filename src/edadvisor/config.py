from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    google_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-1.5-flash")

    # LangSmith
    langchain_tracing_v2: str = Field(default="false")
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="edadvisor-genai")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")

    # Vector store
    vector_store_path: str = Field(default="data/vector_store")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    session_ttl_seconds: int = Field(default=7200)

    # Database
    database_url: str = Field(default="sqlite:///./edadvisor.db")

    # RAG
    top_k_retrieval: int = Field(default=20)
    top_k_final: int = Field(default=5)
    similarity_threshold: float = Field(default=0.72)
    confidence_escalation_threshold: float = Field(default=0.65)
    active_prompt_version: str = Field(default="v2")

    # Corpus
    corpus_dir: str = Field(default="data/corpus")
    chunk_strategy: str = Field(default="hierarchical")
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)

    @property
    def vector_store_dir(self) -> Path:
        p = Path(self.vector_store_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def prompts_dir(self) -> Path:
        return Path("prompts")


settings = Settings()
