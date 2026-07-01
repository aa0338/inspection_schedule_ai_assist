import joblib
import threading
from pathlib import Path

class ModelManager():
    def __init__(self, rrp_model_path: str):
        self.rrp_model_path = Path(rrp_model_path)
        self.rrp_model = None
        self.lock = threading.Lock()
        
    def load_rrp_model(self):
        if not self.rrp_model_path.exists():
            self.rrp_model = None
            return False

        loaded_rrp_model = joblib.load(self.rrp_model_path)
        
        with self.lock:
            self.rrp_model = loaded_rrp_model
        return True
            
    def predict_probability_rrp(self, features):
        if self.rrp_model is None:
            raise RuntimeError("Model is not loaded")
        
        with self.lock:
            return self.rrp_model.predict_proba(features)
        
