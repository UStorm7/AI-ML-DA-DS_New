from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42

# Retention-campaign economics used to pick the decision threshold. These are
# assumptions; change them and re-run `python -m churn.train`.
#   - contacting a customer costs CONTACT_COST (call + discount)
#   - an offer saves a would-be churner with probability SAVE_RATE
#   - a saved customer is worth MONTHS_SAVED x their monthly charges
CONTACT_COST = 50.0
SAVE_RATE = 0.30
MONTHS_SAVED = 12
