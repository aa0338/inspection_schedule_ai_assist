from typing import Annotated
from fastapi import Request, Depends
from reinspection_risk_prediction.rrp_service import RrpService
from optimal_period_advisor.opa_service import OpaService
from model_manager import ModelManager

def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager

def get_rrp_service(model_manager: Annotated[ModelManager, Depends(get_model_manager)]) -> RrpService:
    return RrpService(model_manager)

def get_opa_service(model_manager: Annotated[ModelManager, Depends(get_model_manager)]) -> OpaService:
    return OpaService(model_manager)