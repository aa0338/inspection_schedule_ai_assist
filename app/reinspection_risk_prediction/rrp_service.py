import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from fastapi import Request
from pathlib import Path

from app.model_manager import ModelManager
from app.settings import setting
from app.Schemes import PredictBatchRequest, TrainDataBatchRequest, AccumulateDataResponse, TrainResponse, PredictResponse

rrp_model_path = Path(setting.RRP_MODEL_PATH)
rrp_train_data_path = Path(setting.RRP_TRAIN_DATA_PATH)

class RrpService:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def accumulate_data(self, partial_train_data: TrainDataBatchRequest):
        rrp_train_data_path.parent.mkdir(parents=True, exist_ok=True)


        df = pd.DataFrame([item.model_dump() for item in partial_train_data.items])
        df.to_csv(
            rrp_train_data_path,
            mode="w" if partial_train_data.first else "a",
            header=partial_train_data.first,
            index=False,
            encoding="utf-8"
        )

        result = AccumulateDataResponse(success=True, recieved_count=len(df), message="학습 데이터 누적 완료")
        return result
        

    def model_learning(self):
        if(not rrp_train_data_path.exists()):
            result = TrainResponse(success=False, message="누적된 학습 데이터가 없습니다.")
            return result

        df = pd.read_csv(rrp_train_data_path)
        if(len(df) < 30):
            result = TrainResponse(success=False, message=f"최소 30개 이상의 학습 데이터가 필요합니다. 현재 {len(df)}개 입니다.")
            return result

        X = df.drop(["schedule_detail_id", "result_reinspected"], axis=1)
        
        y = df["result_reinspected"]
        
        # 결측치 처리
        X["day_after_last_inspection"] = X["day_after_last_inspection"].fillna(9999)
        X["day_after_last_reinspection"] = X["day_after_last_reinspection"].fillna(9999)
        X["building_id"] = X["building_id"].fillna(0)
        X["room_id"] = X["room_id"].fillna(0)
        X["equipment_id"] = X["equipment_id"].fillna(0)
        
        new_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=109,
            class_weight="balanced"
        )
        new_model.fit(X, y)
        
        # 모델 저장
        joblib.dump(new_model, rrp_model_path)

        self.model_manager.load_rrp_model()

        result = TrainResponse(success=True, message="학습 성공. 재점검 발생 위험도 예측 모델 생성 완료")
        return result
        
        # 학습 후에 모델 교체 로드되도록 할 것.
        # 학습-모델생성, 모델로드 간 충돌 안되도록 할 것.

    
    def predict_reinspection_risk(self, features_list: PredictBatchRequest):
        df = pd.DataFrame([item.model_dump() for item in features_list.items])
        
        # 결과 매칭용 상세일정 식별자 저장
        schedule_detail_ids = df["schedule_detail_id"]
        
        # 모델에 필요없는 컬럼 제거
        if ("result_reinspected" in df.columns):
            df = df.drop("result_reinspected", axis=1)
        X = df.drop("schedule_detail_id", axis=1)
        
        # 결측치 처리
        X["day_after_last_inspection"] = X["day_after_last_inspection"].fillna(9999)
        X["day_after_last_reinspection"] = X["day_after_last_reinspection"].fillna(9999)
        X["building_id"] = X["building_id"].fillna(0)
        X["room_id"] = X["room_id"].fillna(0)
        X["equipment_id"] = X["equipment_id"].fillna(0)
        
        # 예측: model 주입해야한다.
        probabilities = self.model_manager.predict_probability_rrp(X)[:, 1] # 결과 클래스 1인 확률 추출
        
        results = []
        
        for schedule_detail_id, probability in zip(schedule_detail_ids, probabilities):
            if probability >= 0.8:
                risk_level = "HIGH"
            elif probability >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            result = PredictResponse(schedule_detail_id=int(schedule_detail_id), reinspection_probability=float(probability), reinspection_risk=risk_level)
            results.append(result)
            
        return results
