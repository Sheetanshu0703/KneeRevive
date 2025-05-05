from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta 

app = Flask(__name__)
CORS(app)

# MongoDB setup with connection 
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.server_info()  
    db = client["kneeReviveDB"]
    collection = db["kneeData"]
except Exception as e:
    print(f"⚠️ MongoDB Connection Failed: {e}")
    collection = None  

# Load ML model
model = joblib.load("C:/Users/sheet/Desktop/KneeRevive/KneeRevive/kneerevive_rf_model_new.pkl")

@app.route("/")
def home():
    return "Backend is running!"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        input_data = np.array([[data['x'], data['y'], data['z'], data['gx'], data['gy'], data['gz']]])
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/record", methods=["POST"])
def record_data():
    if not collection:
        return jsonify({"error": "MongoDB connection not established"}), 500
    data = request.json
    try:
        if "user_id" not in data:
            return jsonify({"error": "user_id is required"}), 400

        data["timestamp"] = datetime.utcnow()
        prediction_input = np.array([[data['x'], data['y'], data['z'], data['gx'], data['gy'], data['gz']]])
        prediction = model.predict(prediction_input)[0]
        data["prediction"] = str(prediction)

        collection.insert_one(data)
        return jsonify({"status": "saved", "prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/history", methods=["GET"])
def get_history():
    if not collection:
        return jsonify({"error": "MongoDB connection not established"}), 500
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    three_days_ago = datetime.utcnow() - timedelta(days=3)
    data = list(collection.find({
        "user_id": user_id,
        "timestamp": {"$gte": three_days_ago}
    }, {"_id": 0}))

    return jsonify(data)

@app.route("/assessment", methods=["GET"])
def assessment():
    if not collection:
        return jsonify({"error": "MongoDB connection not established"}), 500
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    three_days_ago = datetime.utcnow() - timedelta(days=3)
    records = list(collection.find({
        "user_id": user_id,
        "timestamp": {"$gte": three_days_ago}
    }))

    if not records:
        return jsonify({"message": "No data found"}), 404

    total = len(records)
    abnormal = sum(1 for r in records if r["prediction"] != "normal")

    avg_accel_mag = np.mean([
        (r["x"]**2 + r["y"]**2 + r["z"]**2)**0.5 for r in records
    ])

    assessment_result = {
        "total_readings": total,
        "abnormal_percentage": round((abnormal / total) * 100, 2),
        "average_acceleration_magnitude": round(avg_accel_mag, 3)
    }

    return jsonify(assessment_result)

if __name__ == "__main__":
    app.run(debug=True)
