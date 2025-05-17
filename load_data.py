from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Combine features and target into one DataFrame
df = X.copy()
df['target'] = y

# Save it to a CSV file (optional, nice for checking)
df.to_csv('housing_data.csv', index=False)

print("✅ Dataset loaded and saved as 'housing_data.csv'")
