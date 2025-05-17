import pandas as pd
import matplotlib.pyplot as plt

# Load monitoring log
df = pd.read_csv("monitoring_log.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Drift detection
drift_threshold = 0.5
drift_points = []
previous_prediction = None

for idx, prediction in enumerate(df['prediction']):
    if previous_prediction is not None:
        drift = abs(prediction - previous_prediction)
        if drift > drift_threshold:
            drift_points.append(idx)
    previous_prediction = prediction

# Plot predictions over time
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['prediction'], marker='o', label='Predictions', color='blue')

# Highlight drift points
if drift_points:
    plt.scatter(df.iloc[drift_points]['timestamp'], df.iloc[drift_points]['prediction'],
                color='red', label='Drift Detected (Δ > 0.5)', zorder=5)

plt.title("Prediction Monitoring with Drift Detection")
plt.xlabel("Timestamp")
plt.ylabel("Predicted Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("monitoring_plot_with_drift.png")
plt.show()
