from fastapi  import FastAPI
import joblib

from model_manager import ModelManager
from settings import setting

app = FastAPI()
    
from app.reinspection_risk_prediction.rrp_controller import router as rrp_router
from app.optimal_period_advisor.opa_controller import router as opa_router
app.include_router(rrp_router, opa_router)


rrp_model_path = setting.RRP_MODEL_PATH
opa_model_path = setting.OPA_MODEL_PATH
model_manager = ModelManager(rrp_model_path, opa_model_path)
model_manager.load_rrp_model
# model_manager.load_opa_model

app.state.model_manager = model_manager

@app.get("/")
def home():
    return {
        "name" : "inspection_schedule_ai_assist",
    }