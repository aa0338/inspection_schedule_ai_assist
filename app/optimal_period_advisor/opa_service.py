import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

from model_manager import ModelManager
from settings import setting

opa_model_path = setting.OPA_MODEL_PATH

class OpaService:
    def __init__(self, model_manager: ModelManager):
        self.model = model_manager.opa_model

    def model_learning(self, featuresList, results):
        dfx = pd.DataFrame([item.model_dump() for item in featuresList])
        X = dfx.drop("schedule_detail_id", axis=1)
        
        dfy = pd.DataFrame([item.model_dump() for item in results])
        y = dfy["result_period"]
        
        # 결측치 처리
        X["day_after_last_inspection"] = X["day_after_last_inspection"].fillna(9999)
        X["day_after_last_reinspection"] = X["day_after_last_reinspection"].fillna(9999)
        X["building_id"] = X["building_id"].fillna(0)
        X["room_id"] = X["room_id"].fillna(0)
        X["equipment_id"] = X["equipment_id"].fillna(0)
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        
        new_model = DecisionTreeClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=109,
            class_weight="balanced"
        )
        new_model.fit(X_train, y_train)
        
        # 모델 저장
        joblib.dump(new_model, opa_model_path)
        
        # 학습 후에 모델 교체 로드되도록 할 것.
        # 학습-모델생성, 모델로드 간 충돌 안되도록 할 것.

    def predict_optimal_period(self, featuresList):
        df = pd.DataFrame([item.model_dump() for item in featuresList])
        
        # 결과 매칭용 상세일정 식별자 저장
        schedule_detail_ids = df["schedule_detail_id"]
        
        # 모델에 필요없는 컬럼 제거
        X = df.drop("schedule_detail_id", axis=1)
        
        # 결측치 처리
        X["day_after_last_inspection"] = X["day_after_last_inspection"].fillna(9999)
        X["day_after_last_reinspection"] = X["day_after_last_reinspection"].fillna(9999)
        X["building_id"] = X["building_id"].fillna(0)
        X["room_id"] = X["room_id"].fillna(0)
        X["equipment_id"] = X["equipment_id"].fillna(0)
        
        # 예측:
        predictions = self.model.predict(X)
        
        results = []
        
        for schedule_detail_id, prediction in zip(schedule_detail_ids, predictions):

            results.append(
                {
                    "schedule_detail_id": schedule_detail_id,
                    "recommandation_period": prediction,
                }
            )
            
        return {
            "results": results
        }