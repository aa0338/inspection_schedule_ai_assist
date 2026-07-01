from typing import Annotated
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
import logging

from app.reinspection_risk_prediction.rrp_service import RrpService
from app.dependencies import get_rrp_service, get_model_manager
from app.Schemes import PredictBatchRequest, TrainDataBatchRequest

# uvicorn의 기본 로거 가져오기
logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/rrp")

@router.get("/")
def rrp_home():
    return {
        "name": "reinspection_risk_prediction",
        "model": "RandomForestClassifier"
    }

@router.post("/predict")
async def get_reinspection_risk(features_list: PredictBatchRequest, service: Annotated[RrpService, Depends(get_rrp_service)]):
    results =  service.predict_reinspection_risk(features_list)
    return results

@router.post("/train-data")
async def accumulate_training_data_reinspection_risk_model(partial_train_data: TrainDataBatchRequest, service: Annotated[RrpService, Depends(get_rrp_service)]):
    result = service.accumulate_data(partial_train_data)
    return result

@router.post("/debug/train-data")
async def debug_accumulate_training_data_reinspection_risk_model(partial_train_data: TrainDataBatchRequest):
    result = partial_train_data.model_dump_json()
    logger.info(result)
    return result

@router.post("/train")
async def train_reinspection_risk_model(service: Annotated[RrpService, Depends(get_rrp_service)]):
    result = service.model_learning()
    return result
