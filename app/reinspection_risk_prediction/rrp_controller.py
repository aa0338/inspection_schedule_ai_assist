from typing import Annotated
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.reinspection_risk_prediction.rrp_service import RrpService
from app.dependencies import get_rrp_service, get_model_manager
from app.Schemes import PredictBatchRequest, TrainDataBatchRequest

router = APIRouter(prefix="/rrp")

@router.get("/")
def rrp_home():
    return {
        "name": "reinspection_risk_prediction",
        "model": "RandomForestClassifier"
    }

@router.post("/predict")
def get_reinspection_risk(features_list: PredictBatchRequest, service: Annotated[RrpService, Depends(get_rrp_service)]):
    return service.predict_reinspection_risk(features_list)

@router.post("/train-data")
def accumulate_training_data_reinspection_risk_model(partial_train_data: TrainDataBatchRequest, service: Annotated[RrpService, Depends(get_rrp_service)]):
    return service.accumulate_data(partial_train_data)

@router.post("/train")
def train_reinspection_risk_model(service: Annotated[RrpService, Depends(get_rrp_service)]):
    return service.model_learning()
