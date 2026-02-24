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
    
    model_config = SettingsConfigDict(env_file=".env")
    
settings = Settings()