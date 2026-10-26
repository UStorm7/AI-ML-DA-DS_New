"""Run every analysis, write figures to reports/figures and numbers to reports/results.json."""
import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from src import analysis, plot_style
from src.build_warehouse import build
from src.config import FIG_DIR, REPORT_DIR
from src.queries import run


def main() -> None:
    plot_style.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    con = build()
    orders = con.execute("SELECT * FROM fct_delivered").df()
    results: dict = {}

    # 1. Monthly KPIs -------------------------------------------------------
    kpis = run(con, "monthly_kpis")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(kpis["order_month"], kpis["gmv"] / 1e6, color=plot_style.BLUE, marker="o", ms=4)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("R$%.1fM"))
    ax.set_title("Monthly GMV, delivered orders")
    plot_style.save(fig, FIG_DIR / "01_monthly_gmv.png")
    results["totals"] = {
        "orders": int(kpis["orders"].sum()),
        "gmv_brl": round(float(kpis["gmv"].sum()), 0),
        "aov_brl": round(float(orders["payment_value"].mean()), 2),
        "customers": int(orders["customer_id"].nunique()),
    }

    # 2. Cohort retention ---------------------------------------------------
    retention = analysis.cohort_matrix(run(con, "cohort_retention"), max_months=12)
    retention = retention.loc["2017-01":"2018-03"]  # cohorts with >= 5 months of follow-up
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(retention.iloc[:, 1:], cmap="Blues", annot=True, fmt=".1f",
                annot_kws={"size": 7}, cbar_kws={"label": "% of cohort active"}, ax=ax)
    ax.grid(False)
    ax.set(xlabel="Months since first purchase", ylabel="First-purchase cohort")
    ax.set_title("Monthly retention by cohort (%): almost nobody comes back")
    plot_style.save(fig, FIG_DIR / "02_cohort_retention.png")
    results["retention_month1_avg_pct"] = round(float(retention[1].mean()), 2)
    repeat_share = (orders.groupby("customer_id")["order_id"].nunique() > 1).mean()
    results["repeat_customer_pct"] = round(float(repeat_share) * 100, 2)

    # 3. Delivery delay vs reviews -----------------------------------------
    buckets = run(con, "review_by_delay_bucket")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = buckets["delay_bucket"].str[3:]
    colors = [plot_style.BLUE if "early" in b else plot_style.ORANGE for b in buckets["delay_bucket"]]
    bars = ax.bar(labels, buckets["pct_negative"], color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.1f%%", color=plot_style.INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Share of 1-2 star reviews by delivery vs. promised date")
    ax.grid(axis="x", visible=False)
    plot_style.save(fig, FIG_DIR / "03_negative_reviews_by_delay.png")
    results["review_by_delay_bucket"] = buckets.to_dict(orient="records")
    results["late_delivery_effect"] = analysis.late_delivery_effect(orders)

    repeat = analysis.repeat_purchase_by_first_experience(orders)
    results["repeat_by_first_experience"] = repeat.reset_index().to_dict(orient="records")

    # 4. Late rate by state -------------------------------------------------
    states = run(con, "late_rate_by_state")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    s = states.sort_values("late_pct")
    ax.barh(s["customer_state"], s["late_pct"], color=plot_style.BLUE, height=0.65)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Late-delivery rate by customer state (states with 300+ orders)")
    ax.grid(axis="y", visible=False)
    plot_style.save(fig, FIG_DIR / "04_late_rate_by_state.png")
    results["late_rate_by_state_top5"] = states.head(5).to_dict(orient="records")

    # 5. RFM segments -------------------------------------------------------
    rfm = analysis.rfm_table(orders)
    seg = analysis.segment_summary(rfm)
    seg.to_csv(REPORT_DIR / "rfm_segments.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    seg_sorted = seg.sort_values("pct_revenue")
    y = range(len(seg_sorted))
    ax.barh([i + 0.2 for i in y], seg_sorted["pct_customers"], height=0.38,
            color=plot_style.BLUE, label="% of customers")
    ax.barh([i - 0.2 for i in y], seg_sorted["pct_revenue"], height=0.38,
            color=plot_style.ORANGE, label="% of revenue")
    ax.set_yticks(list(y), seg_sorted.index)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="lower right")
    ax.set_title("RFM segments: share of customers vs. share of revenue")
    ax.grid(axis="y", visible=False)
    plot_style.save(fig, FIG_DIR / "05_rfm_segments.png")
    results["rfm_segments"] = seg.reset_index().to_dict(orient="records")

    # 6. Category table ----------------------------------------------------
    cats = run(con, "category_performance")
    cats.to_csv(REPORT_DIR / "category_performance.csv", index=False)
    results["category_performance_top10"] = cats.head(10).to_dict(orient="records")

    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    con.close()


if __name__ == "__main__":
    main()
