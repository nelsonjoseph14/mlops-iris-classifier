import sys
import os

# Add the parent directory (mlops-iris-classifier) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier

def test_data_shape():
    X, y = load_and_preprocess()
    assert X.shape[0] == y.shape[0]

def test_model_training():
    X, y = load_and_preprocess()
    model = RandomForestClassifier()
    model.fit(X, y)
    assert hasattr(model, "predict")
