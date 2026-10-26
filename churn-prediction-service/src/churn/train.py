"""Train, compare, calibrate and evaluate churn models; save the best one.

    python -m churn.train            (run from the src/ directory, or pip install -e .)
"""
import json
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from churn import business, plots
from churn.config import MODEL_PATH, RANDOM_STATE, REPORT_DIR
from churn.data import load, split_xy
from churn.features import FeatureEngineer, preprocessor

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}


def make_pipeline(model, scale_numeric=True) -> Pipeline:
    return Pipeline([("features", FeatureEngineer()),
                     ("prep", preprocessor(scale_numeric)),
                     ("model", model)])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "baseline (prior)": make_pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": make_pipeline(LogisticRegression(C=0.5, max_iter=2000)),
        "random_forest": make_pipeline(RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE), False),
        "lightgbm": make_pipeline(lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=1, verbose=-1), False),
    }


def tune_lightgbm(X, y) -> Pipeline:
    search = RandomizedSearchCV(
        candidate_models()["lightgbm"],
        param_distributions={
            "model__n_estimators": [150, 250, 400, 600],
            "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "model__num_leaves": [7, 15, 31],
            "model__min_child_samples": [20, 40, 80],
            "model__reg_lambda": [0, 1, 5, 10],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
        },
        n_iter=25, scoring="average_precision", cv=CV, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    print(f"  tuned lightgbm CV PR-AUC={search.best_score_:.4f} params={search.best_params_}")
    return search.best_estimator_


def evaluate(y, proba, monthly, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "threshold": round(threshold, 3),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "pct_contacted": round(pred.mean() * 100, 2),
        "campaign_profit": round(business.campaign_profit(y, proba, monthly, threshold), 0),
    }


def main() -> None:
    df = load()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"train={len(X_tr)}  test={len(X_te)}  churn rate={y.mean():.3f}")

    # 1. Compare model families with 5-fold CV on the training set only.
    models = candidate_models()
    models["lightgbm_tuned"] = tune_lightgbm(X_tr, y_tr)
    cv_rows = []
    for name, pipe in models.items():
        s = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring=SCORING, n_jobs=-1)
        cv_rows.append({"model": name,
                        "roc_auc": s["test_roc_auc"].mean(), "roc_auc_std": s["test_roc_auc"].std(),
                        "pr_auc": s["test_pr_auc"].mean(), "pr_auc_std": s["test_pr_auc"].std(),
                        "brier": -s["test_brier"].mean()})
        print(f"  {name:<22} ROC-AUC {cv_rows[-1]['roc_auc']:.4f}  PR-AUC {cv_rows[-1]['pr_auc']:.4f}")
    cv_table = pd.DataFrame(cv_rows).sort_values("pr_auc", ascending=False).round(4)
    cv_table.to_csv(REPORT_DIR / "cv_model_comparison.csv", index=False)

    best_name = cv_table.iloc[0]["model"]
    print(f"best by CV PR-AUC: {best_name}")

    # 2. Calibrate the winner so probabilities mean what they say (needed for
    #    the expected-profit decision rule).
    final = CalibratedClassifierCV(models[best_name], method="isotonic", cv=CV)

    # 3. Choose the threshold on out-of-fold predictions from TRAIN (never on test).
    oof = cross_val_predict(final, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
    threshold, _ = business.best_threshold(y_tr, oof, X_tr["MonthlyCharges"])
    print(f"profit-optimal threshold (OOF on train) = {threshold:.2f}")

    # 4. Fit on all training data, evaluate once on the untouched test set.
    final.fit(X_tr, y_tr)
    proba_te = final.predict_proba(X_te)[:, 1]
    test = evaluate(y_te, proba_te, X_te["MonthlyCharges"], threshold)
    baselines = {
        "contact_nobody": 0.0,
        "contact_everyone": round(business.campaign_profit(y_te, np.ones(len(y_te)), X_te["MonthlyCharges"], 0.5), 0),
        "contact_month_to_month": round(business.campaign_profit(
            y_te, (X_te["Contract"] == "Month-to-month").astype(float), X_te["MonthlyCharges"], 0.5), 0),
        "default_threshold_0.5": round(business.campaign_profit(y_te, proba_te, X_te["MonthlyCharges"], 0.5), 0),
    }
    print(json.dumps({"test": test, "profit_baselines": baselines}, indent=2))

    # 5. Figures and explanations.
    plots.curves(y_te, proba_te, models, X_tr, y_tr, X_te)
    plots.calibration(y_te, proba_te, models[best_name].fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
    plots.profit(y_te, proba_te, X_te["MonthlyCharges"], threshold)
    top_features = plots.shap_summary(models["lightgbm_tuned"].fit(X_tr, y_tr), X_te)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    meta = {
        "model": best_name, "calibration": "isotonic (5-fold)", "threshold": round(threshold, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(X_tr), "features": list(X.columns),
    }
    joblib.dump({"pipeline": final, "meta": meta}, MODEL_PATH)
    results = {"meta": meta, "cv": cv_table.to_dict(orient="records"), "test": test,
               "profit_baselines": baselines, "top_shap_features": top_features}
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
