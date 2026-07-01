from typing import Annotated
from fastapi import Request, Depends
from app.reinspection_risk_prediction.rrp_service import RrpService
from app.model_manager import ModelManager

def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager

def get_rrp_service(model_manager: Annotated[ModelManager, Depends(get_model_manager)]) -> RrpService:
    return RrpService(model_manager)
