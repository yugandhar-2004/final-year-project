from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("rf_mq135_model.pkl")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        mq2 = float(request.args.get("mq2"))
        mq135 = float(request.args.get("mq135"))
        temp = float(request.args.get("temp"))
        hum = float(request.args.get("hum"))

        features = np.array([[mq2, mq135, temp, hum]])
        prediction = model.predict(features)[0]

        return jsonify({
            "next_1hr_mq135": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return "AI Air Pollution Prediction API Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
