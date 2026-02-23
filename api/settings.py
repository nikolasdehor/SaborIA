from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    chroma_persist_dir: str = "./data/chroma"
    log_level: str = "INFO"

    # RAG config
    chunk_size: int = 1024
    chunk_overlap: int = 128
    retriever_k: int = 6
    retriever_k_full_menu: int = 50  # k for single-menu retrieval (needs all chunks)


settings = Settings()
