import requests
import time
import joblib
import numpy as np

BLYNK_TOKEN = "4qIpimYTVNo4UzladIzDBI8ez343XRDz"
BLYNK_URL = "https://blynk.cloud/external/api"

model = joblib.load("rf_mq135_model.pkl")

def get_value(pin):
    r = requests.get(f"{BLYNK_URL}/get", params={
        "token": BLYNK_TOKEN,
        "pin": pin
    })
    return float(r.text)

def set_value(pin, value):
    requests.get(f"{BLYNK_URL}/update", params={
        "token": BLYNK_TOKEN,
        "pin": pin,
        "value": value
    })

while True:
    try:
        mq2 = get_value("V3")
        mq135 = get_value("V2")
        temp = get_value("V0")
        hum = get_value("V1")

        features = np.array([[mq2, mq135, temp, hum]])
        prediction = model.predict(features)[0]

        set_value("V4", round(prediction, 2))
        print("Prediction sent to Blynk:", prediction)

    except Exception as e:
        print("Error:", e)

    time.sleep(60)  # every 1 minute
