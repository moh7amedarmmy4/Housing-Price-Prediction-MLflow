# 🏡 Housing Price Prediction using MLflow

This project demonstrates how to use MLflow to manage a complete machine learning lifecycle — from training models and tracking experiments to deployment and monitoring.

---

## 📌 Project Overview

- Predict housing prices using the California Housing Dataset
- Use MLflow to log parameters, metrics, and model artifacts
- Register and deploy the best model as a REST API
- Monitor the model for real-time predictions and drift detection

---

## 🧰 Tools Used

- Python 3.9
- scikit-learn
- MLflow
- pandas, matplotlib
- PowerShell & requests (for API testing)

---

### 📥 Dataset

The dataset (`housing_data.csv`) is included in the `/data` folder. It is based on the California Housing dataset from the 1990 U.S. Census.


## How to run

1. Activate your virtual environment
2. Train models and log to MLflow:
3. Launch MLflow UI:
Visit: `http://127.0.0.1:5000`

4. Deploy best model as API:

5. Run monitoring + drift detection:


---

## 📊 Results

- Best model: Random Forest (200 estimators)
- R² score: 0.8062
- Live predictions working via API
- Drift detected when Δ prediction > 0.5

---

## 👤 Author

**Mohamed Mohamed**  
Student ID: 2105954  
May 2025

