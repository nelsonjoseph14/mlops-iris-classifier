import joblib
import os

def load_model():
    model_path = os.path.join("models", "model.joblib")
    model = joblib.load(model_path)
    return model
