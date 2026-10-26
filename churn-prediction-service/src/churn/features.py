"""Feature engineering + preprocessing, all inside one sklearn Pipeline so the
exact same transformations run at training time and in the API."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod", "tenure_bucket"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "n_services",
           "charges_per_service", "avg_monthly_spend", "price_increase_ratio"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain features. Stateless, so there is no leakage across CV folds."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["n_services"] = (X[SERVICES] == "Yes").sum(axis=1)
        X["charges_per_service"] = X["MonthlyCharges"] / X["n_services"].clip(lower=1)
        tenure = X["tenure"].clip(lower=1)
        X["avg_monthly_spend"] = np.where(X["tenure"] > 0, X["TotalCharges"] / tenure, X["MonthlyCharges"])
        # >1 means the customer pays more now than their historical average,
        # a proxy for recent price increases or upgrades.
        X["price_increase_ratio"] = X["MonthlyCharges"] / X["avg_monthly_spend"].replace(0, np.nan)
        X["price_increase_ratio"] = X["price_increase_ratio"].fillna(1.0)
        X["tenure_bucket"] = pd.cut(X["tenure"], [-1, 6, 12, 24, 48, 100],
                                    labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]).astype(str)
        return X


def preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
