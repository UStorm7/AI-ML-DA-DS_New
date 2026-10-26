"""Turn churn probabilities into a contact / don't-contact decision by expected profit."""
import numpy as np

from churn.config import CONTACT_COST, MONTHS_SAVED, SAVE_RATE


def campaign_profit(y_true, proba, monthly_charges, threshold: float) -> float:
    """Realised profit of contacting everyone with proba >= threshold (on labelled data)."""
    contact = np.asarray(proba) >= threshold
    value_saved = SAVE_RATE * MONTHS_SAVED * np.asarray(monthly_charges)
    gains = np.where(np.asarray(y_true) == 1, value_saved, 0.0)
    return float(np.sum(contact * (gains - CONTACT_COST)))


def profit_curve(y_true, proba, monthly_charges, thresholds=None):
    thresholds = np.linspace(0.05, 0.95, 91) if thresholds is None else thresholds
    profits = np.array([campaign_profit(y_true, proba, monthly_charges, t) for t in thresholds])
    return thresholds, profits


def best_threshold(y_true, proba, monthly_charges) -> tuple[float, float]:
    t, p = profit_curve(y_true, proba, monthly_charges)
    i = int(np.argmax(p))
    return float(t[i]), float(p[i])

