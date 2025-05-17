import requests
import pandas as pd
import time
import random
import json

# Sample prediction data
sample_inputs = [
    [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23],
    [5.0, 30.0, 5.0, 1.1, 500.0, 2.8, 36.5, -121.0],
    [3.2, 10.0, 6.0, 1.0, 1200.0, 3.5, 38.0, -120.0]
]

columns = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
monitor_log = []

previous_prediction = None
drift_threshold = 0.5  # adjust this if needed

for i, data_point in enumerate(sample_inputs):
    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": [data_point]
        }
    }

    try:
        start_time = time.time()
        response = requests.post("http://127.0.0.1:1234/invocations",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps(payload))
        end_time = time.time()
        latency = end_time - start_time

        response_json = response.json()
        if isinstance(response_json, dict) and "predictions" in response_json:
            prediction = response_json["predictions"][0]
        else:
            raise Exception(f"Unexpected prediction format: {response_json}")

        # Drift detection
        if previous_prediction is not None:
            drift = abs(prediction - previous_prediction)
            if drift > drift_threshold:
                print(f"⚠️ Drift detected! Change from {previous_prediction:.4f} to {prediction:.4f} (Δ={drift:.4f})")
        previous_prediction = prediction

        print(f"[{i}] Prediction: {prediction:.4f} | Latency: {latency:.4f}s")

        monitor_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": data_point,
            "prediction": prediction,
            "latency_seconds": latency
        })

    except Exception as e:
        print(f"❌ Error on input {i}: {e}")

    time.sleep(2)  # simulate delay

# Save log
df_log = pd.DataFrame(monitor_log)
df_log.to_csv("monitoring_log.csv", index=False)
print("📊 Monitoring log saved to 'monitoring_log.csv'")
