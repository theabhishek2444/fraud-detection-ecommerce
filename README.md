# Intelligent Fraud Detection System

Real-time fraud detection for e-commerce orders using Ensemble ML and Graph Analytics.

## Tech Stack
- ML: XGBoost, Scikit-learn, Isolation Forest, NetworkX
- Backend: FastAPI, PostgreSQL, Redis
- Frontend: React.js, Recharts

## Setup
1. Clone the repo
2. Install dependencies: `pip install fastapi uvicorn joblib scikit-learn xgboost networkx pandas`
3. Download dataset from Kaggle: https://kaggle.com/competitions/ieee-fraud-detection
4. Place train_transaction.csv and train_identity.csv in /data folder
5. Run backend: `cd backend && uvicorn main:app --reload`

## Dataset
IEEE-CIS Fraud Detection — Kaggle
Files needed: train_transaction.csv, train_identity.csv
Place both in /data folder locally (not committed to GitHub)

## Team
- Abhishek — ML Architecture & Fraud Scoring Model
- Member 2 — Backend & API
- Member 3 — Anomaly Detection & Graph Analytics
- Member 4 — Frontend & Dashboard
- Member 5 — Data & Evaluation

## Status
🚧 In Development
