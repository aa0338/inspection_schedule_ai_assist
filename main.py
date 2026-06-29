from fastapi  import FastAPI

app = FastAPI()
    
from reinspection_risk_prediction.rrp_controller import router as rrp_router
app.include_router(rrp_router)

@app.get("/")
def home():
    return {
        "name" : "inspection_schedule_ai_assist",
    }