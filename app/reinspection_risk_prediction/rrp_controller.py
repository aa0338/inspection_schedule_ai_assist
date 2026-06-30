from typing import Annotated
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.reinspection_risk_prediction.rrp_service import RrpService
from app.dependencies import get_rrp_service, get_model_manager
from app.Schemes import BatchRequest

router = APIRouter(prefix="/rrp")


@router.post("/predict")
def get_reinspection_risk(featuresList: BatchRequest, service: Annotated[RrpService, Depends(get_rrp_service)]):
    return service.predict_reinspection_risk(featuresList)
    
@router.post("/graph")
def get_rrp_graph(service: Annotated[RrpService, Depends(get_rrp_service)]):
    return service.generate_rrp_graph()