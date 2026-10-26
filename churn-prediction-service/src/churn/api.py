"""REST API serving the trained churn model.

    uvicorn churn.api:app --reload      (from src/)
"""
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.config import MODEL_PATH
from churn.data import clean

YesNo = Literal["Yes", "No"]
ServiceOpt = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: ServiceOpt
    OnlineBackup: ServiceOpt
    DeviceProtection: ServiceOpt
    TechSupport: ServiceOpt
    StreamingTV: ServiceOpt
    StreamingMovies: ServiceOpt
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 280.3}}}


class Prediction(BaseModel):
    churn_probability: float
    risk_band: Literal["low", "medium", "high"]
    contact: bool = Field(description="True if the customer should get a retention offer")


@lru_cache
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No model at {MODEL_PATH}. Run `python -m churn.train` first.")
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.get("/health")
def health():
    try:
        meta = load_model()["meta"]
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"status": "ok", "model": meta["model"], "trained_at": meta["trained_at"],
            "threshold": meta["threshold"]}


def _predict(customers: list[Customer]) -> list[Prediction]:
    bundle = load_model()
    X = clean(pd.DataFrame([c.model_dump() for c in customers]))
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    t = bundle["meta"]["threshold"]
    return [Prediction(churn_probability=round(float(p), 4),
                       risk_band="high" if p >= 0.5 else "medium" if p >= t else "low",
                       contact=bool(p >= t)) for p in proba]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer] = Body(max_length=1000)):
    return _predict(customers)
