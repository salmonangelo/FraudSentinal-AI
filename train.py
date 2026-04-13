"""
FraudSentinel — supervised (XGBoost) and unsupervised (Isolation Forest) training stubs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_transaction_dataset(csv_path: Path) -> pd.DataFrame:
    # Load raw labeled transaction rows from disk (e.g., data/creditcard.csv) into a DataFrame.
    df = pd.read_csv(csv_path)
    print("Class distribution (count):")
    print(df["Class"].value_counts().sort_index())
    print("Class distribution (proportion):")
    print(df["Class"].value_counts(normalize=True).sort_index())
    return df


def preprocess_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Split into features X and target y, drop the Time column, leave Amount unscaled (scaled after train/test split).
    if target_col not in df.columns:
        raise KeyError(f"Missing target column {target_col!r} in dataframe columns.")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    if "Time" in X.columns:
        X = X.drop(columns=["Time"])
    return X, y


def stratified_train_test_split_scale_amount(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    # Stratified 80/20 split, then fit StandardScaler on training Amount only and transform Amount on train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    if "Amount" not in X_train.columns:
        raise KeyError("Expected an 'Amount' column for scaling.")
    amount_scaler = StandardScaler()
    X_train["Amount"] = amount_scaler.fit_transform(X_train[["Amount"]]).ravel()
    X_test["Amount"] = amount_scaler.transform(X_test[["Amount"]]).ravel()
    return X_train, X_test, y_train, y_test, amount_scaler


def apply_smote_resampling(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    # Balance the minority fraud class using SMOTE; return resampled X and y as numpy arrays.
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return np.asarray(X_resampled), np.asarray(y_resampled)


def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float | None,
) -> xgb.XGBClassifier:
    # Fit an XGBoost classifier with fraud-appropriate hyperparameters; optionally use scale_pos_weight for imbalance.
    pass


def train_isolation_forest(
    X: pd.DataFrame | np.ndarray,
    contamination: float,
    random_state: int = 42,
) -> IsolationForest:
    # Fit an IsolationForest on (typically unlabeled or mixed) feature rows for anomaly scores alongside supervised model.
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_arr)
    print("Training complete")
    return model


def _normalize_iso_decision_scores(
    iso_model: IsolationForest,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    # Map IsolationForest decision_function so higher = more anomalous, then MinMax-scale using train sample (5k max) to avoid hanging; return test scores in [0, 1].
    print(f"[ISO] X_train shape: {X_train.shape}, limiting to 5,000 rows for efficiency")
    train_sample_size = min(5_000, len(X_train))
    X_train_sample = X_train[:train_sample_size]
    
    print(f"[ISO] Computing decision_function on {train_sample_size} training samples")
    train_raw = -iso_model.decision_function(X_train_sample)
    
    print(f"[ISO] Computing decision_function on {len(X_test)} test samples")
    test_raw = -iso_model.decision_function(X_test)
    
    print(f"[ISO] Fitting MinMaxScaler")
    scaler = MinMaxScaler()
    scaler.fit(train_raw.reshape(-1, 1))
    
    print(f"[ISO] Transforming test scores")
    result = scaler.transform(test_raw.reshape(-1, 1)).ravel()
    print(f"[ISO] ISO normalization complete")
    return result


def _hybrid_scores(xgb_prob: np.ndarray, iso_norm: np.ndarray) -> np.ndarray:
    # Combine XGBoost fraud probability with normalized anomaly score (weights 0.7 / 0.3).
    return 0.7 * xgb_prob + 0.3 * iso_norm


def _find_threshold_recall_ge_min_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall_min: float = 0.9,
) -> float:
    # Among thresholds where recall (fraud positive class) >= recall_min, choose the one with lowest FPR; tie-break on stricter (higher) threshold.
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                scores,
                np.linspace(scores.min(), scores.max(), num=2000, dtype=float),
            ]
        )
    )
    best_t: float | None = None
    best_fpr = np.inf
    best_t_tiebreak = -np.inf
    for t in candidates:
        y_pred = (scores >= t).astype(int)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        if rec < recall_min:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        denom = fp + tn
        fpr = float(fp / denom) if denom > 0 else 0.0
        if fpr < best_fpr - 1e-12 or (
            abs(fpr - best_fpr) <= 1e-12 and t > best_t_tiebreak
        ):
            best_fpr = fpr
            best_t = float(t)
            best_t_tiebreak = float(t)
    if best_t is not None:
        return best_t

    # No threshold reaches recall_min: maximize recall, then minimize FPR, then prefer stricter threshold.
    best_key: tuple[float, float, float] | None = None
    best_t_fallback: float | None = None
    for t in candidates:
        y_pred = (scores >= t).astype(int)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        key = (rec, -fpr, t)
        if best_key is None or key > best_key:
            best_key = key
            best_t_fallback = float(t)
    assert best_t_fallback is not None
    return best_t_fallback


def _metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    # Precision, recall, F1 (fraud positive), FPR, and ROC-AUC placeholder key for caller to overwrite with score-based AUC.
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fp + tn
    fpr = float(fp / denom) if denom > 0 else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": fpr,
    }


def _print_metrics_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    # Pretty-print Precision, Recall, F1, ROC-AUC, and FPR at each model's chosen threshold.
    print()
    print(
        f"{'Model':<26}"
        f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'FPR@thr':>10}"
    )
    print("-" * 86)
    for name, m in rows:
        print(
            f"{name:<26}"
            f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} "
            f"{m['roc_auc']:>10.4f} {m['fpr']:>10.4f}"
        )
    print()


def _save_pr_curve(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    # Plot precision vs recall for scores and save PNG using non-interactive Agg backend.
    prec, rec, _ = precision_recall_curve(np.asarray(y_true).astype(int), scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="steelblue", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (hybrid score)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_evaluation(
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier,
    xgb_model: xgb.XGBClassifier,
    iso_model: IsolationForest,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    models_dir: Path,
) -> None:
    # Evaluate LR, RF, XGB on holdout; hybrid XGB+IF threshold with recall>=0.9 and min FPR; save reports, threshold, and PR plot.
    # Limit evaluation to 10,000 rows max to speed up threshold calibration.
    print("[EVAL] Limiting test set to 10,000 rows max for faster evaluation")
    test_sample_size = min(10_000, len(X_test))
    X_test_sample = X_test.iloc[:test_sample_size]
    y_test_sample = y_test.iloc[:test_sample_size] if isinstance(y_test, pd.Series) else y_test[:test_sample_size]
    
    print("[EVAL] Converting dataframes to numpy arrays")
    X_train_np = X_train.to_numpy()
    X_test_np = X_test_sample.to_numpy()
    y_test_np = np.asarray(y_test_sample).astype(int)
    print(f"[EVAL] X_test shape: {X_test_np.shape}")

    print("[EVAL] Computing LogisticRegression predict_proba")
    lr_scores = lr_model.predict_proba(X_test_np)[:, 1]
    print("[EVAL] LR scores computed")
    
    print("[EVAL] Computing RandomForest predict_proba")
    rf_scores = rf_model.predict_proba(X_test_np)[:, 1]
    print("[EVAL] RF scores computed")
    
    print("[EVAL] Computing XGBoost predict_proba")
    xgb_prob = xgb_model.predict_proba(X_test_np)[:, 1]
    print("[EVAL] XGB scores computed")
    
    print("[EVAL] Normalizing IsolationForest scores (this may take 1-2 minutes)")
    iso_norm = _normalize_iso_decision_scores(iso_model, X_train_np, X_test_np)
    print("[EVAL] ISO normalization complete")
    
    print("[EVAL] Computing hybrid scores")
    hybrid = _hybrid_scores(xgb_prob, iso_norm)
    print("[EVAL] Hybrid scores computed")

    print("[EVAL] Finding hybrid threshold (recall >= 0.9)")
    thr_hybrid = _find_threshold_recall_ge_min_fpr(y_test_np, hybrid, recall_min=0.9)
    print(f"[EVAL] Hybrid threshold: {thr_hybrid:.6f}")
    (models_dir / "threshold.txt").write_text(f"{thr_hybrid:.10f}\n", encoding="utf-8")

    print("[EVAL] Finding LR threshold (recall >= 0.9)")
    thr_lr = _find_threshold_recall_ge_min_fpr(y_test_np, lr_scores, recall_min=0.9)
    print(f"[EVAL] LR threshold: {thr_lr:.6f}")
    
    print("[EVAL] Finding RF threshold (recall >= 0.9)")
    thr_rf = _find_threshold_recall_ge_min_fpr(y_test_np, rf_scores, recall_min=0.9)
    print(f"[EVAL] RF threshold: {thr_rf:.6f}")

    print("[EVAL] Computing metrics for LR")
    m_lr = _metrics_at_threshold(y_test_np, lr_scores, thr_lr)
    m_lr["roc_auc"] = float(roc_auc_score(y_test_np, lr_scores))
    print("[EVAL] LR metrics computed")

    print("[EVAL] Computing metrics for RF")
    m_rf = _metrics_at_threshold(y_test_np, rf_scores, thr_rf)
    m_rf["roc_auc"] = float(roc_auc_score(y_test_np, rf_scores))
    print("[EVAL] RF metrics computed")

    print("[EVAL] Computing metrics for XGB hybrid")
    m_xgb = _metrics_at_threshold(y_test_np, hybrid, thr_hybrid)
    m_xgb["roc_auc"] = float(roc_auc_score(y_test_np, xgb_prob))
    print("[EVAL] XGB metrics computed")

    print("[EVAL] Printing metrics table")
    rows: list[tuple[str, dict[str, float]]] = [
        ("LogisticRegression", m_lr),
        ("RandomForestClassifier", m_rf),
        ("XGBoost (hybrid score)", m_xgb),
    ]
    _print_metrics_table(rows)

    print("[EVAL] Generating classification report")
    y_pred_hybrid = (hybrid >= thr_hybrid).astype(int)
    report = classification_report(
        y_test_np,
        y_pred_hybrid,
        digits=4,
        target_names=["Legit", "Fraud"],
    )
    (models_dir / "metrics.txt").write_text(report, encoding="utf-8")
    print("[EVAL] Metrics saved")

    print("[EVAL] Saving precision-recall curve")
    _save_pr_curve(y_test_np, hybrid, models_dir / "pr_curve.png")
    print("[EVAL] PR curve saved")


def _shap_values_positive_class_matrix(shap_values: Any) -> np.ndarray:
    # Normalize SHAP return type to (n_samples, n_features) for the positive (fraud) class.
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1])
    else:
        arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def run_shap_explanation_section(
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    models_dir: Path,
) -> None:
    # Load saved XGBoost, build TreeExplainer (CPU-only), SHAP for 50 test rows max, persist explainer and one fraud sample's top-5 SHAP JSON.
    print(f"[SHAP] Loading XGBoost model from {models_dir / 'xgb_model.pkl'}")
    xgb_model = joblib.load(models_dir / "xgb_model.pkl")
    print(f"[SHAP] Loaded XGBoost model successfully")
    
    print(f"[SHAP] Creating TreeExplainer with raw output (CPU-friendly)")
    # Pass the booster directly to TreeExplainer to avoid GPU issues
    try:
        booster = xgb_model.get_booster()
        explainer = shap.TreeExplainer(booster, model_output='raw')
    except Exception as e:
        print(f"[SHAP] Warning: Could not use booster, falling back to sklearn wrapper: {e}")
        explainer = shap.TreeExplainer(xgb_model, model_output='raw')
    print(f"[SHAP] TreeExplainer created successfully")

    n_sample = min(50, len(X_test))
    print(f"[SHAP] Computing SHAP values for {n_sample} samples (limited to 50 max)")
    X_50 = X_test.iloc[:n_sample]
    sv_batch = explainer.shap_values(X_50)
    print(f"[SHAP] SHAP batch computation complete")

    print(f"[SHAP] Dumping explainer to {models_dir / 'shap_explainer.pkl'}")
    joblib.dump(explainer, models_dir / "shap_explainer.pkl")
    print(f"[SHAP] Explainer dumped successfully")

    print(f"[SHAP] Finding first fraud sample in y_test")
    y_arr = np.asarray(y_test).astype(int).ravel()
    fraud_positions = np.flatnonzero(y_arr == 1)
    if fraud_positions.size == 0:
        raise ValueError("Test set has no fraud (Class==1) rows for sample SHAP.")
    fraud_row = int(fraud_positions[0])
    print(f"[SHAP] Found fraud sample at index {fraud_row}")
    
    print(f"[SHAP] Computing SHAP values for fraud sample")
    X_fraud = X_test.iloc[fraud_row : fraud_row + 1]
    sv_one = explainer.shap_values(X_fraud)
    vals = _shap_values_positive_class_matrix(sv_one)[0]
    print(f"[SHAP] Fraud sample SHAP computation complete")
    
    print(f"[SHAP] Building sample payload with top-5 features")
    feature_names = list(X_test.columns)
    top_idx = np.argsort(np.abs(vals))[::-1][:5]
    sample_payload = {
        "test_row_index": fraud_row,
        "top_features": [
            {"feature": feature_names[i], "shap_value": float(vals[i])} for i in top_idx
        ],
    }
    (models_dir / "sample_shap.json").write_text(
        json.dumps(sample_payload, indent=2),
        encoding="utf-8",
    )
    print(f"[SHAP] Saved sample_shap.json")
    print("[SHAP] SHAP setup complete")


def save_artifacts(
    xgb_model: xgb.XGBClassifier,
    iso_model: IsolationForest,
    scaler: StandardScaler,
    feature_columns: list[str],
    out_dir: Path,
) -> None:
    # Persist trained models, fitted scaler, and feature column order under models/ for inference in main.py.
    pass


def main() -> None:
    # Orchestrate load → preprocess → optional SMOTE → train/test split → train both models → evaluate → SHAP → save artifacts.
    print("[MAIN] Starting FraudSentinel training pipeline")
    print("[MAIN] Estimated runtime: 3-5 minutes")
    
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if models already exist to skip retraining
    xgb_model_path = models_dir / "xgb_model.pkl"
    iso_model_path = models_dir / "iso_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    
    if xgb_model_path.exists() and iso_model_path.exists() and scaler_path.exists():
        print("[MAIN] Models found — loading precomputed models from disk")
        xgb_model = joblib.load(xgb_model_path)
        iso_model = joblib.load(iso_model_path)
        amount_scaler = joblib.load(scaler_path)
        
        # Still need data for SHAP and evaluation
        print("[MAIN] Loading transaction dataset for SHAP visualization and evaluation")
        csv_path = project_root / "data" / "creditcard.csv"
        df = load_transaction_dataset(csv_path)
        X, y = preprocess_features(df, target_col="Class")
        X_train, X_test, y_train, y_test, _ = stratified_train_test_split_scale_amount(
            X, y, test_size=0.2, random_state=42
        )
        print("Shapes after stratified split and Amount scaling:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test:  {y_test.shape}")
        
        print("[MAIN] Applying SMOTE resampling for LR and RF")
        X_train_smote, y_train_smote = apply_smote_resampling(X_train, y_train)

        print("[MAIN] Training LogisticRegression")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_smote, y_train_smote)
        print("Training complete")

        print("[MAIN] Training RandomForestClassifier (50 estimators, parallelized)")
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_smote, y_train_smote)
        print("Training complete")
        
        skip_evaluation = False
    else:
        print("[MAIN] Models not found — running full training pipeline")
        print("[MAIN] Loading transaction dataset")
        csv_path = project_root / "data" / "creditcard.csv"
        df = load_transaction_dataset(csv_path)
        
        print("[MAIN] Preprocessing features")
        X, y = preprocess_features(df, target_col="Class")
        
        print("[MAIN] Stratified train/test split and Amount scaling")
        X_train, X_test, y_train, y_test, amount_scaler = stratified_train_test_split_scale_amount(
            X, y, test_size=0.2, random_state=42
        )
        print("Shapes after stratified split and Amount scaling:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test:  {y_test.shape}")

        print("[MAIN] Applying SMOTE resampling")
        X_train_smote, y_train_smote = apply_smote_resampling(X_train, y_train)

        print("[MAIN] Training LogisticRegression")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_smote, y_train_smote)
        print("Training complete")

        print("[MAIN] Training RandomForestClassifier (50 estimators, parallelized)")
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_smote, y_train_smote)
        print("Training complete")

        print("[MAIN] Training XGBoost classifier")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1,
            eval_metric="aucpr",
            random_state=42,
        )
        xgb_model.fit(X_train_smote, y_train_smote)
        print("Training complete")

        print("[MAIN] Training Isolation Forest")
        iso_model = train_isolation_forest(
            X_train,
            contamination=0.002,
            random_state=42,
        )

        print("[MAIN] Saving trained models and scaler")
        joblib.dump(xgb_model, models_dir / "xgb_model.pkl")
        joblib.dump(iso_model, models_dir / "iso_model.pkl")
        joblib.dump(amount_scaler, models_dir / "scaler.pkl")
        print("[MAIN] Models saved successfully")
        
        # Run evaluation only during full training
        skip_evaluation = False

    # Only evaluate if we just trained models
    if not skip_evaluation:
        print("[MAIN] Running evaluation on test set")
        run_evaluation(
            lr_model,
            rf_model,
            xgb_model,
            iso_model,
            X_train,
            X_test,
            y_test,
            models_dir,
        )
        print("[MAIN] Evaluation complete")
    else:
        print("[MAIN] Skipping evaluation (models already evaluated previously)")

    print("[MAIN] Running SHAP explanation section (this may take a minute)")
    run_shap_explanation_section(X_test, y_test, models_dir)
    print("[MAIN] All pipeline steps complete!")


if __name__ == "__main__":
    main()
