from pydantic_settings import BaseSettings, SettingsConfigDict
import os

env_state = os.getenv("ENV_STATE", "dev")   # env.dev 로드

class Settings(BaseSettings):
    RRP_MODEL_PATH: str
    OPA_MODEL_PATH: str
    
    model_config = SettingConfigDict(
        env_file=(".env", f"env.{env_state}", "env.local"),
        env_file_encoding="utf-8",
        extra="ignore" # 정의되지 않은 변수는 무시. "allow"는 미리 정의된 변수 이외에도 허용
    )
    
setting = Settings()