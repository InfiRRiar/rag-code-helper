from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # secrets
    hf_token: str = Field(
        validation_alias="HF_TOKEN",
        json_schema_extra={"secret": True}
    )
    
    # public data
    base_embeddings_url: str = Field(
        default="http://127.0.0.1:8080",
        json_schema_extra={"public": True}
    )
    bi_encoder_name: str = Field(
        default="jinaai/jina-code-embeddings-1.5b",
        json_schema_extra={"public": True}
    )
    instruct_llm_base_url: str = Field(
        default="http://127.0.0.1:11434",
        json_schema_extra={"public": True}
    )
    instruct_llm_name: str = Field(
        default="hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
        json_schema_extra={"public": True}
    )
    k: int = Field(
        default=3,
        json_schema_extra={"public": True}
    )
    prompts_dir: str = Field(
        default="src/srv/prompt_templates/",
        json_schema_extra={"public": True}
    )
    
    model_config = SettingsConfigDict(env_file=".env")
    
settings = Settings()