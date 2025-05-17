import pandas as pd
import matplotlib.pyplot as plt

# Load the log
df = pd.read_csv("monitoring_log.csv")

# Optional: Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot predictions over time
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['prediction'], marker='o', linestyle='-')
plt.title("Predictions Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Predicted Value")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
plt.savefig("monitoring_plot.png")
plt.show()
