from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    schedule_detail_id: int = Field(..., description="점검 상세 일정 식별자. 결과 매칭을 위한 식별자임")
    inspection_item_id: int = Field(..., description="점검 항목 식별자")
    past_inspection_count: int = Field(..., ge=0, description="과거 점검 횟수")
    past_fail_count: int = Field(..., ge=0, description="과거 불합격 횟수")
    past_reinspection_count: int = Field(..., ge=0, description="과거 재점검 횟수")
    day_after_last_inspection: int|None = Field(None, ge=0, description="최근 점검 후 경과일")
    day_after_last_reinspection: int|None = Field(None, ge=0, description="최근 재점검 후 경과일")
    month: int = Field(..., ge=1, le=12, description="월")
    quarter: int = Field(..., ge=1, le=4, description="분기")
    category_id: int = Field(..., description="점검 카테고리 식별자")
    building_id: int|None = Field(None, description="건물 식별자")
    room_id: int|None = Field(None, description="방 식별자")
    equipment_id: int|None = Field(None, description="장비 식별자")
    period_type_id: int = Field(..., description="점검주기 식별자")

class PredictBatchRequest(BaseModel):
    items: list[PredictRequest] = Field(..., description="예측을 위한 Feature 목록")

class TrainDataRequest(BaseModel):
    schedule_detail_id: int = Field(..., description="점검 상세 일정 식별자. 결과 매칭을 위한 식별자임")
    inspection_item_id: int = Field(..., description="점검 항목 식별자")
    past_inspection_count: int = Field(..., ge=0, description="과거 점검 횟수")
    past_fail_count: int = Field(..., ge=0, description="과거 불합격 횟수")
    past_reinspection_count: int = Field(..., ge=0, description="과거 재점검 횟수")
    day_after_last_inspection: int|None = Field(None, ge=0, description="최근 점검 후 경과일")
    day_after_last_reinspection: int|None = Field(None, ge=0, description="최근 재점검 후 경과일")
    month: int = Field(..., ge=1, le=12, description="월")
    quarter: int = Field(..., ge=1, le=4, description="분기")
    category_id: int = Field(..., description="점검 카테고리 식별자")
    building_id: int|None = Field(None, description="건물 식별자")
    room_id: int|None = Field(None, description="방 식별자")
    equipment_id: int|None = Field(None, description="장비 식별자")
    period_type_id: int = Field(..., description="점검주기 식별자")
    result_reinspected: int|None = Field(None, ge=0, le=1, description="재점검 발생 이력 (예측 대상)")

class TrainDataBatchRequest(BaseModel):
    items: list[TrainDataRequest]
    first: bool = False

class AccumulateDataResponse(BaseModel):
    success: bool
    recieved_count: int
    message: str

class TrainResponse(BaseModel):
    success: bool
    message: str

class PredictResponse(BaseModel):    
    schedule_detail_id: int
    reinspection_probability: float
    reinspection_risk: str