from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load ML models
with open("WeightPredictor.pkl", "rb") as file:
    weight_model = pickle.load(file)

with open("Experience_Category.pkl", "rb") as file:
    experience_model = pickle.load(file)

with open("HealthScorePredictor.pkl", "rb") as file:
    health_model = pickle.load(file)

# Define feature lists
weight_features = ["Age", "Height (cm)", "Weight (kg)", "Sleep Hours", "Daily Calories Intake", "TotalBurn"]
experience_features = [
    "Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM", "Resting_BPM",
    "Session_Duration (hours)", "Calories_Burned", "Fat_Percentage",
    "Water_Intake (liters)", "BMI", "Experience_Level_Code", "Gender_Code", "Workout_Code"
]
health_features = [
    "Resting Heart Rate (bpm)", "BMI", "Body Fat (%)", "Calories Burned",
    "Water Intake (liters)", "Workout Duration (mins)", "Sleep Hours"
]

# ✅ Helper function to get MongoDB connection
def get_db():
    MONGO_URL = os.getenv("MONGO_URL")
    client = MongoClient(MONGO_URL)
    return client.get_database("fitness_db")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running. Use /predict_weight, /predict_experience, or /predict_health for predictions."})

@app.route("/predict_weight", methods=["POST"])
def predict_weight():
    try:
        data = request.get_json()
        missing_features = [feature for feature in weight_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        input_features = [data[feature] for feature in weight_features]
        input_array = np.array([input_features])
        prediction = weight_model.predict(input_array)

        db = get_db()  # ✅ Get database connection inside function
        db.weight_predictions.insert_one({"input": data, "predicted_weight": float(prediction[0])})

        return jsonify({"predicted_weight": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_experience", methods=["POST"])
def predict_experience():
    try:
        data = request.get_json()
        missing_features = [feature for feature in experience_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        input_features = [data[feature] for feature in experience_features]
        input_array = np.array([input_features])
        prediction = experience_model.predict(input_array)

        db = get_db()  # ✅ Get database connection inside function
        db.experience_predictions.insert_one({"input": data, "predicted_experience_category": int(prediction[0])})

        return jsonify({"predicted_experience_category": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_health", methods=["POST"])
def predict_health():
    try:
        data = request.get_json()
        missing_features = [feature for feature in health_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        input_features = [data[feature] for feature in health_features]
        input_array = np.array([input_features])
        prediction = health_model.predict(input_array)

        db = get_db()  # ✅ Get database connection inside function
        db.health_predictions.insert_one({"input": data, "predicted_health_score": float(prediction[0])})

        return jsonify({"predicted_health_score": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
