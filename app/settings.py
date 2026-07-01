from pydantic_settings import BaseSettings, SettingsConfigDict
import os

env_state = os.getenv("ENV_STATE", "dev")   # env.dev 로드

class Settings(BaseSettings):
    RRP_MODEL_PATH: str = "app/model/rrp_model.pk1"
    # OPA_MODEL_PATH: str = "app/model/opa_model.pk1"
    RRP_TRAIN_DATA_PATH: str = "app/model/train_data/rrp_train_data.csv"
    
    model_config = SettingsConfigDict(
        env_file=(".env", f"env.{env_state}", "env.local"),
        env_file_encoding="utf-8",
        extra="ignore" # 정의되지 않은 변수는 무시. "allow"는 미리 정의된 변수 이외에도 허용
    )
    
setting = Settings()