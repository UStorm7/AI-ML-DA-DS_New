import numpy as np
import pandas as pd

from churn.data import clean
from churn.features import NUMERIC, FeatureEngineer, preprocessor

ROW = {"customerID": "x", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
       "tenure": 0, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
       "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
       "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
       "PaymentMethod": "Mailed check", "MonthlyCharges": 60.0, "TotalCharges": " ", "Churn": "No"}


def test_blank_total_charges_become_zero():
    df = clean(pd.DataFrame([ROW]))
    assert df.loc[0, "TotalCharges"] == 0
    assert df.loc[0, "Churn"] == 0


def test_engineered_features_are_finite_for_new_customer():
    X = FeatureEngineer().transform(clean(pd.DataFrame([ROW])))
    assert X.loc[0, "n_services"] == 3  # PhoneService, MultipleLines, OnlineSecurity
    assert X.loc[0, "charges_per_service"] == 20
    assert X.loc[0, "price_increase_ratio"] == 1.0
    assert np.isfinite(X[NUMERIC].to_numpy(float)).all()


def test_preprocessor_handles_unseen_category():
    train = FeatureEngineer().transform(clean(pd.DataFrame([ROW, {**ROW, "tenure": 30, "TotalCharges": "1800"}])))
    prep = preprocessor().fit(train)
    unseen = train.iloc[[0]].assign(PaymentMethod="Crypto")
    assert prep.transform(unseen).shape[1] == prep.transform(train).shape[1]
