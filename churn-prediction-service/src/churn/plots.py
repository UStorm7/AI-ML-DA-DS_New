import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from churn import business  # noqa: E402
from churn.config import FIG_DIR  # noqa: E402

COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#1baf7a",
          "lightgbm_tuned": "#eb6834", "final (calibrated)": "#0b0b0b"}
INK_2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.edgecolor": GRID, "axes.titlelocation": "left",
                     "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def curves(y_te, proba_final, models, X_tr, y_tr, X_te):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    probas = {n: models[n].fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
              for n in ["logistic_regression", "random_forest", "lightgbm_tuned"]}
    probas["final (calibrated)"] = proba_final
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        prec, rec, _ = precision_recall_curve(y_te, p)
        style = "--" if name == "final (calibrated)" else "-"
        a1.plot(fpr, tpr, style, color=COLORS[name], label=name, lw=1.6)
        a2.plot(rec, prec, style, color=COLORS[name], label=name, lw=1.6)
    a1.plot([0, 1], [0, 1], ":", color=INK_2, lw=1)
    a1.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    a2.axhline(y_te.mean(), ls=":", color=INK_2, lw=1)
    a2.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    a2.legend(loc="upper right", fontsize=8)
    _save(fig, "roc_pr_curves.png")


def calibration(y_te, proba_cal, proba_raw):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for p, name, c in [(proba_raw, "before calibration", "#2a78d6"), (proba_cal, "after isotonic", "#eb6834")]:
        frac, mean = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=5, color=c, label=name)
    ax.plot([0, 1], [0, 1], ":", color=INK_2, lw=1, label="perfect")
    ax.set(title="Calibration (test set)", xlabel="Predicted churn probability", ylabel="Observed churn rate")
    ax.legend()
    _save(fig, "calibration.png")


def profit(y_te, proba, monthly, threshold):
    t, p = business.profit_curve(y_te, proba, monthly)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t, p, color="#2a78d6")
    ax.axvline(threshold, ls="--", color="#eb6834", lw=1.5)
    ax.annotate(f"chosen threshold {threshold:.2f}\n(selected on train OOF)", (threshold, p.max()),
                xytext=(8, -30), textcoords="offset points", color=INK_2, fontsize=9)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(title="Retention campaign profit vs. decision threshold (test set)",
           xlabel="Contact customers with churn probability ≥ threshold", ylabel="Profit ($)")
    _save(fig, "profit_curve.png")


def shap_summary(pipeline, X_te, n=12) -> list[dict]:
    feats = pipeline.named_steps["features"].transform(X_te)
    Xt = pipeline.named_steps["prep"].transform(feats)
    names = pipeline.named_steps["prep"].get_feature_names_out()
    values = shap.TreeExplainer(pipeline.named_steps["model"]).shap_values(Xt)
    values = values[1] if isinstance(values, list) else values
    shap.summary_plot(values, pd.DataFrame(Xt, columns=names), max_display=n, show=False)
    plt.title("SHAP: what pushes churn risk up (right) or down (left)", loc="left", fontweight="bold")
    plt.gcf().savefig(FIG_DIR / "shap_summary.png")
    plt.close("all")
    imp = np.abs(values).mean(axis=0)
    order = np.argsort(imp)[::-1][:n]
    return [{"feature": names[i], "mean_abs_shap": round(float(imp[i]), 4)} for i in order]
