from fastapi  import FastAPI
import joblib

import logging
import sys

from app.model_manager import ModelManager
from app.settings import setting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

app = FastAPI()
    
rrp_model_path = setting.RRP_MODEL_PATH
model_manager = ModelManager(rrp_model_path)
model_manager.load_rrp_model()

app.state.model_manager = model_manager

from app.reinspection_risk_prediction.rrp_controller import router as rrp_router
app.include_router(rrp_router)

@app.get("/")
def home():
    return {
        "name" : "inspection_schedule_ai_assist"
    }