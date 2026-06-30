from typing import Annotated
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.optimal_period_advisor.opa_service import OpaService
from app.Schemes import BatchRequest
from app.dependencies import get_opa_service

router = APIRouter(prefix="/opa")

@router.post("/predict")
def get_optimized_period(featuresList: BatchRequest, service: Annotated[OpaService, Depends(get_opa_service)]):
    return service.predict_optimal_period(featuresList)