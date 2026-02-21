from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import StrictStr

class Settings(BaseSettings):
    hf_token: StrictStr
    
    model_config = SettingsConfigDict(env_file=".env")
    
secrets = Settings()