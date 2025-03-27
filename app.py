from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load the trained models correctly
with open("WeightPredictor.pkl", "rb") as file:
    weight_model = pickle.load(file)

with open("Experience_Category.pkl", "rb") as file:
    experience_model = pickle.load(file)

with open("HealthScorePredictor.pkl", "rb") as file:
    health_model = pickle.load(file)  # ✅ FIXED: Load using pickle

# Define expected feature names for each model
weight_features = ["Age", "Weight (kg)", "Sleep Hours", "Daily Calories Intake", "TotalBurn"]
experience_features = [
    "Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM", "Resting_BPM",
    "Session_Duration (hours)", "Calories_Burned", "Fat_Percentage",
    "Water_Intake (liters)", "BMI", "Experience_Level_Code", "Gender_Code", "Workout_Code"
]
health_features = [
    "Resting Heart Rate (bpm)", "BMI", "Body Fat (%)", "Calories Burned",
    "Water Intake (liters)", "Workout Duration (mins)", "Sleep Hours"
]

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

        return jsonify({"predicted_health_score": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
