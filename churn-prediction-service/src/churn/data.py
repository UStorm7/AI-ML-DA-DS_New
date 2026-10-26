"""Load and clean the IBM Telco Customer Churn dataset (7,043 customers)."""
from urllib.request import urlretrieve

import pandas as pd

from churn.config import DATA_PATH, DATA_URL

TARGET = "Churn"
ID_COL = "customerID"


def download(path=DATA_PATH) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)


def load(path=DATA_PATH) -> pd.DataFrame:
    download(path)
    return clean(pd.read_csv(path))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges is a string with " " for brand-new customers (tenure 0):
    # they have not been billed yet, so 0 is the correct value, not NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", "0"))
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if TARGET in df:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET, ID_COL]), df[TARGET]
