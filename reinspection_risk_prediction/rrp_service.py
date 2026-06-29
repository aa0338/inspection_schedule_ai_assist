import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class RrpService:
    
    def model_learning(features, results):
        dfx = pd.DataFrame([item.model_dump() for item in features])
        X = dfx.drop("schedule_detail_id", axis=1)
        
        dfy = pd.DataFrame([item.model_dump() for item in results])
        y = dfy["result_reinspected"]
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3
        )
        model.fit(X_train, y_train)
        
        # 모델 저장
        joblib.dump(model, "models/rrp_model.pk1")

    
    def predict_reinspection_risk(features):
        df = pd.DataFrame([item.model_dump() for item in features])
        
        # 결과 매칭용 상세일정 식별자 저장
        schedule_detail_ids = df["schedule_detail_id"]
        
        # 모델에 필요없는 컬럼 제거
        X = df.drop("schedule_detail_id", axis=1)
        
        # 결측치 처리
        # ...
        
        # 모델 로드
        model = joblib.load("models/rrp_model.pk1")
        # fastapi에서 한번만 로드하도록 변경할 것.
        # 학습-모델생성, 모델로드 간 충동 안되도록 할 것.
        # 학습 후에 모델 교체 로드되도록 할 것.
        
        # 예측: model 주입해야한다.
        probabilities = model.predict_proba(X)[:, 1] # 결과 클래스 1인 확률 추출
        
        results = []
        
        for schedule_detail_id, probability in zip(schedule_detail_ids, probabilities):
            if probability >= 0.8:
                risk_level = "HIGH"
            elif probability >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            results.append(
                {
                    "schedule_detail_id": schedule_detail_id,
                    "reinspection_probability": probability,
                    "reinspection_risk": risk_level
                }
            )
            
        return {
            "results": results
        }
    
    def generate_rrp_graph():
        pass