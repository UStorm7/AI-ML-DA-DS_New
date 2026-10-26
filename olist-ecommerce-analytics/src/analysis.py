"""Analyses on top of the warehouse: cohorts, RFM segmentation, delay impact."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def cohort_matrix(cohort_long: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    """Pivot (cohort_month, month_number, active_customers) into a retention % matrix."""
    pivot = cohort_long.pivot(index="cohort_month", columns="month_number", values="active_customers")
    pivot = pivot.reindex(columns=range(0, max_months + 1))
    retention = pivot.div(pivot[0], axis=0) * 100
    retention.index = pd.to_datetime(retention.index).strftime("%Y-%m")
    return retention


def rfm_table(orders: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / Frequency / Monetary per customer, scored 1-5 and mapped to segments."""
    snapshot = snapshot or orders["purchased_at"].max() + pd.Timedelta(days=1)
    rfm = orders.groupby("customer_id").agg(
        recency_days=("purchased_at", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum"),
    )
    # Recency: lower is better -> reverse the labels. Frequency is ~97% ones on
    # this dataset, so it is bucketed by business rule instead of quintiles.
    rfm["R"] = pd.qcut(rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = np.select([rfm["frequency"] >= 3, rfm["frequency"] == 2], [5, 3], default=1)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["segment"] = rfm.apply(_segment, axis=1)
    return rfm


def _segment(row: pd.Series) -> str:
    r, f, m = row["R"], row["F"], row["M"]
    if f >= 3 and r >= 4:
        return "Champions"
    if f >= 3:
        return "Loyal - lapsing"
    if r >= 4 and m >= 4:
        return "New high-value"
    if r >= 4:
        return "New low-value"
    if m >= 4:
        return "At-risk high-value"
    if r <= 2:
        return "Hibernating"
    return "Needs attention"


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.groupby("segment").agg(
        customers=("R", "size"),
        avg_recency_days=("recency_days", "mean"),
        avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        total_spend=("monetary", "sum"),
    )
    out["pct_customers"] = out["customers"] / out["customers"].sum() * 100
    out["pct_revenue"] = out["total_spend"] / out["total_spend"].sum() * 100
    return out.sort_values("total_spend", ascending=False).round(2)


def late_delivery_effect(orders: pd.DataFrame) -> dict:
    """Quantify how late delivery relates to review scores.

    1. Welch t-test and Mann-Whitney U on review score, late vs on-time.
    2. Logistic regression for P(negative review) controlling for order value,
       freight, item count, category, and customer state, so the late effect is
       not just "expensive / remote orders get worse reviews".
    """
    df = orders.dropna(subset=["review_score", "is_late"]).copy()
    df["negative"] = (df["review_score"] <= 2).astype(int)
    df["late"] = df["is_late"].astype(int)
    late, on_time = df.loc[df.late == 1, "review_score"], df.loc[df.late == 0, "review_score"]

    t = stats.ttest_ind(late, on_time, equal_var=False)
    u = stats.mannwhitneyu(late, on_time, alternative="less")

    top_cats = df["main_category"].value_counts().index[:15]
    df["category"] = np.where(df["main_category"].isin(top_cats), df["main_category"], "other")
    df["log_value"] = np.log1p(df["payment_value"].fillna(0))
    df["log_freight"] = np.log1p(df["freight_value"].fillna(0))
    model = smf.logit(
        "negative ~ late + log_value + log_freight + n_items + C(category) + C(customer_state)",
        data=df,
    ).fit(disp=False)
    coef, (lo, hi) = model.params["late"], model.conf_int().loc["late"]

    return {
        "n_orders": int(len(df)),
        "late_share_pct": round(df.late.mean() * 100, 2),
        "mean_review_late": round(late.mean(), 3),
        "mean_review_on_time": round(on_time.mean(), 3),
        "neg_rate_late_pct": round(df.loc[df.late == 1, "negative"].mean() * 100, 2),
        "neg_rate_on_time_pct": round(df.loc[df.late == 0, "negative"].mean() * 100, 2),
        "welch_t": round(t.statistic, 2),
        "welch_p": float(t.pvalue),
        "mannwhitney_p": float(u.pvalue),
        "adjusted_odds_ratio_late": round(float(np.exp(coef)), 2),
        "adjusted_or_95ci": [round(float(np.exp(lo)), 2), round(float(np.exp(hi)), 2)],
    }


def repeat_purchase_by_first_experience(orders: pd.DataFrame) -> pd.DataFrame:
    """Does a bad first order (late / low review) reduce the chance of buying again?"""
    o = orders.sort_values("purchased_at")
    first = o.groupby("customer_id").head(1).set_index("customer_id")
    n_orders = o.groupby("customer_id")["order_id"].nunique()
    first = first.assign(repeat=(n_orders.reindex(first.index) > 1).astype(int))
    first["first_experience"] = np.select(
        [(first["review_score"] <= 2).fillna(False).to_numpy(bool),
         first["is_late"].fillna(False).to_numpy(bool)],
        ["negative review", "late, not negative"],
        default="on time & ok",
    )
    return (first.groupby("first_experience")["repeat"]
                 .agg(customers="size", repeat_rate_pct=lambda s: round(s.mean() * 100, 2)))
