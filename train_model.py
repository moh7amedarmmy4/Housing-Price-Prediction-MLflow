import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the dataset
df = pd.read_csv('housing_data.csv')
X = df.drop(columns=['target'])
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("housing_price_prediction")

with mlflow.start_run():
    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Log model parameters (none for Linear Regression but still an example)
    mlflow.log_param("model_type", "LinearRegression")

    # Evaluate
    score = model.score(X_test, y_test)  # R^2 Score

    # Log metrics
    mlflow.log_metric("r2_score", score)

    # Log the model itself
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model trained and logged to MLflow with R^2 score: {score:.4f}")
