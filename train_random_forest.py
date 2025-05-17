import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load the dataset
df = pd.read_csv('housing_data.csv')
X = df.drop(columns=['target'])
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("housing_price_prediction")

for n_estimators in [10, 50, 100, 200]:
    with mlflow.start_run():
        # Model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)

        # Evaluate
        score = model.score(X_test, y_test)

        # Log metrics
        mlflow.log_metric("r2_score", score)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ Trained RandomForest with {n_estimators} estimators, R² = {score:.4f}")
