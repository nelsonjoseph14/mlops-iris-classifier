from flask import request, jsonify
from app.model import load_model
import numpy as np

def register_routes(app):
    model = load_model()

    @app.route("/")
    def home():
        return "Iris Classifier API"

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json["features"]  # e.g., [5.1, 3.5, 1.4, 0.2]
        prediction = model.predict([np.array(data)])
        return jsonify({"prediction": int(prediction[0])})
