from fastapi  import FastAPI
import joblib

from app.model_manager import ModelManager
from app.settings import setting

app = FastAPI()
    
from app.reinspection_risk_prediction.rrp_controller import router as rrp_router
app.include_router(rrp_router)


rrp_model_path = setting.RRP_MODEL_PATH
model_manager = ModelManager(rrp_model_path)
model_manager.load_rrp_model()

app.state.model_manager = model_manager

@app.get("/")
def home():
    return {
        "name" : "inspection_schedule_ai_assist",
        "rrp_model" : app.state.model_manager.rrp_model,
        "rrp_model_load" : model_manager.load_rrp_model()
    }