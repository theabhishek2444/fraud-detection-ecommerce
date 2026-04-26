# FastAPI entry point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
xgb_model = joblib.load("../ml/fraud_model.pkl")
iso_model = joblib.load("../ml/isolation_forest.pkl")
with open("../ml/fraud_graph.pkl", "rb") as f:
    fraud_graph = pickle.load(f)
print("All models loaded! ✅")

class Order(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: float
    card2: Optional[float] = 0
    card3: Optional[float] = 0
    card4: Optional[str] = "visa"
    card5: Optional[float] = 0
    card6: Optional[str] = "debit"
    addr1: Optional[float] = 0
    addr2: Optional[float] = 0
    P_emaildomain: Optional[str] = "gmail.com"
    C1: Optional[float] = 1
    D1: Optional[float] = 1

@app.get("/")
def home():
    return {"status": "Fraud Detection API is running ✅"}

@app.get("/health")
def health():
    return {
        "xgb_model": "loaded",
        "iso_model": "loaded",
        "graph": "loaded",
        "nodes": fraud_graph.number_of_nodes(),
        "edges": fraud_graph.number_of_edges()
    }

@app.post("/score-order")
def score_order(order: Order):
    feature_cols = xgb_model.get_booster().feature_names
    order_dict = order.dict()
    feature_vector = {}

    for col in feature_cols:
        if col in order_dict:
            feature_vector[col] = order_dict[col]
        else:
            feature_vector[col] = 0

    df_order = pd.DataFrame([feature_vector])

    le = LabelEncoder()
    for col in df_order.select_dtypes(include='object').columns:
        df_order[col] = le.fit_transform(df_order[col].astype(str))

    xgb_score = xgb_model.predict_proba(df_order)[0][1]
    iso_raw = iso_model.decision_function(df_order)[0]
    iso_score = max(0, min(1, (0.5 - iso_raw)))
    combined = (0.7 * xgb_score) + (0.3 * iso_score)

    return {
        "xgb_score": round(float(xgb_score) * 100, 1),
        "iso_score": round(float(iso_score) * 100, 1),
        "combined_score": round(float(combined) * 100, 1),
        "verdict": "FRAUD" if combined > 0.5 else "LEGITIMATE",
        "risk_level": "HIGH" if combined > 0.7 else "MEDIUM" if combined > 0.4 else "LOW"
    }
