from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
import requests
import os
import tensorflow as tf

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app)

# MongoDB setup
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["kneeReviveDB"]
    collection = db["kneeData"]
except Exception as e:
    print(f"⚠️ MongoDB Connection Failed: {e}")
    collection = None

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/Users/sheet/Desktop/kneeRevive/KneeRevive/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["normal", "abnormal"]

def predict_model(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return CLASS_NAMES[int(np.argmax(output))]

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
        input_data = np.array([[data['x'], data['y'], data['z'], data['gx'], data['gy'], data['gz'], data.get('knee_angle', 0)]])
        prediction = predict_model(input_data)
        return jsonify({"prediction_class": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/record", methods=["POST"])
def record_data():
    if collection is None:
        return jsonify({"error": "MongoDB connection not established"}), 500

    data = request.json
    try:
        if "user_id" not in data:
            return jsonify({"error": "user_id is required"}), 400

        data["timestamp"] = datetime.utcnow()

        input_data = np.array([[data['x'], data['y'], data['z'], data['gx'], data['gy'], data['gz'], data.get('knee_angle', 0)]])
        prediction = predict_model(input_data)

        data["prediction"] = prediction
        collection.insert_one(data)
        return jsonify({"status": "saved", "prediction_class": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/history", methods=["GET"])
def get_history():
    if collection is None:
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
    if collection is None:
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
    abnormal = sum(1 for r in records if str(r["prediction"]).lower() != "normal")

    avg_accel_mag = np.mean([
        (r["x"]**2 + r["y"]**2 + r["z"]**2)**0.5 for r in records
    ])

    return jsonify({
        "total_readings": total,
        "abnormal_percentage": round((abnormal / total) * 100, 2),
        "average_acceleration_magnitude": round(avg_accel_mag, 3)
    })

@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    if collection is None:
        return jsonify({"error": "MongoDB connection not established"}), 500

    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message")

    if not user_id or not user_message:
        return jsonify({"error": "user_id and message are required"}), 400

    try:
        three_days_ago = datetime.utcnow() - timedelta(days=3)
        records = list(collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": three_days_ago}
        }).sort("timestamp", -1).limit(20))

        if not records:
            return jsonify({"response": "I couldn't find any recent data for you."})

        summary_lines = []
        for r in records:
            ts = r["timestamp"].strftime('%Y-%m-%d %H:%M')
            line = f"{ts} - Prediction: {r['prediction']}, Accel: ({r['x']:.2f},{r['y']:.2f},{r['z']:.2f}), Gyro: ({r['gx']:.2f},{r['gy']:.2f},{r['gz']:.2f})"
            summary_lines.append(line)

        prompt = f"""You are a helpful and empathetic virtual physiotherapy assistant.

Here is the recent knee movement data for user '{user_id}' over the past 3 days:
{chr(10).join(summary_lines)}

The user says: "{user_message}"

Based on the above sensor data and user's message, reply with a personalized response. Give motivational, health-related or corrective feedback."""

        response = requests.post(
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt}]
                }]
            }
        )

        reply_data = response.json()
        print("Gemini API Raw Response:", reply_data)

        reply = reply_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't generate a response.")
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
