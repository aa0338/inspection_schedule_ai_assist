import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

from model_manager import ModelManager
from settings import setting

opa_model_path = setting.OPA_MODEL_PATH

class OpaService:
    pass