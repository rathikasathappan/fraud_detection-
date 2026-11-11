"""
Phase 2: Tabular Baseline Models (Logistic Regression & XGBoost)
Trains on tabular onboarding features using a strict time-based and fraud-ring
aware train/test split. Evaluates on full test set and cold-start subgroup.
Saves metrics to data/processed/baseline_metrics.json.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_split_data(
    merchants_path: str = "data/synthetic/merchants.csv",
    meta_path: str = "data/synthetic/rings_metadata.json",
    train_ratio: float = 0.75
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs time-based split combined with fraud-ring isolation so that
    no fraud ring appears in both train and test partitions.
    """
    df = pd.read_csv(merchants_path)
    df["dt"] = pd.to_datetime(df["registration_date"])
    df = df.sort_values("dt").reset_index(drop=True)
    
    cutoff_idx = int(len(df) * train_ratio)
    train_cutoff_time = df.loc[cutoff_idx, "dt"]
    
    # Load rings metadata to isolate rings
    ring_to_merchants = {}
    merchant_to_ring = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            ring_to_merchants = json.load(f)
            for ring_id, m_list in ring_to_merchants.items():
                for m in m_list:
                    merchant_to_ring[m] = ring_id

    # Identify which rings have early members (train) vs late (test)
    train_rings = set()
    test_rings = set()
    
    # Initial assignment by time
    initial_train_mask = df["dt"] < train_cutoff_time
    train_candidates = df[initial_train_mask]
    test_candidates = df[~initial_train_mask]
    
    for _, row in train_candidates.iterrows():
        ring = merchant_to_ring.get(row["merchant_id"])
        if ring and ring != "NONE":
            train_rings.add(ring)
            
    # For any ring in train_rings, all its members belong to train
    # For rings not in train, they belong to test
    is_train = []
    for _, row in df.iterrows():
        ring = merchant_to_ring.get(row["merchant_id"])
        if ring and ring in train_rings:
            is_train.append(True)
        elif ring and ring in test_rings:
            is_train.append(False)
        else:
            is_train.append(row["dt"] < train_cutoff_time)
            
    df["is_train"] = is_train
    train_df = df[df["is_train"]].copy().drop(columns=["dt", "is_train"])
    test_df = df[~df["is_train"]].copy().drop(columns=["dt", "is_train"])
    
    logger.info(f"Train set: {len(train_df)} rows (Fraud: {train_df['fraud_label'].sum()}, {train_df['fraud_label'].mean():.2%})")
    logger.info(f"Test set:  {len(test_df)} rows (Fraud: {test_df['fraud_label'].sum()}, {test_df['fraud_label'].mean():.2%})")
    
    return train_df, test_df

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculates PR-AUC, precision, recall, F1, and FPR."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    pr_auc = float(average_precision_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def train_and_evaluate_baselines(output_dir: str = "data/processed"):
    """
    Trains Logistic Regression and XGBoost baseline classifiers.
    Saves metrics and models.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    train_df, test_df = load_and_split_data()
    
    # Save train/test partitions for subsequent phases
    train_df.to_csv(os.path.join(output_dir, "train_tabular.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_tabular.csv"), index=False)
    
    feature_cols = [
        "business_category", "business_age_days",
        "phone_verified", "email_verified", "document_verified",
        "transaction_count"
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df["fraud_label"].values
    
    X_test = test_df[feature_cols]
    y_test = test_df["fraud_label"].values
    
    # Separate cold-start slice of test set (merchants with zero transactions)
    cold_start_mask = test_df["transaction_count"] == 0
    X_test_cold = X_test[cold_start_mask]
    y_test_cold = y_test[cold_start_mask]
    
    # Preprocessor for tabular features
    categorical_cols = ["business_category"]
    numerical_cols = ["business_age_days", "phone_verified", "email_verified", "document_verified", "transaction_count"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    X_test_cold_proc = preprocessor.transform(X_test_cold)
    
    # Save preprocessor
    joblib.dump(preprocessor, "models/tabular_preprocessor.pkl")
    
    results = {}
    
    # -----------------------------------------------------------------
    # Model 1: Logistic Regression (with class_weight='balanced')
    # -----------------------------------------------------------------
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train_proc, y_train)
    
    lr_pred_proba = lr.predict_proba(X_test_proc)[:, 1]
    lr_cold_pred_proba = lr.predict_proba(X_test_cold_proc)[:, 1]
    
    results["logistic_regression"] = {
        "all_merchants": compute_metrics(y_test, lr_pred_proba),
        "cold_start_group": compute_metrics(y_test_cold, lr_cold_pred_proba)
    }
    joblib.dump(lr, "models/logistic_regression_baseline.pkl")
    
    # -----------------------------------------------------------------
    # Model 2: XGBoost Classifier (with scale_pos_weight)
    # -----------------------------------------------------------------
    logger.info("Training XGBoost Baseline...")
    import xgboost as xgb
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = float(neg_count / max(1, pos_count))
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="aucpr"
    )
    xgb_model.fit(X_train_proc, y_train)
    
    xgb_pred_proba = xgb_model.predict_proba(X_test_proc)[:, 1]
    xgb_cold_pred_proba = xgb_model.predict_proba(X_test_cold_proc)[:, 1]
    
    results["xgboost_baseline"] = {
        "all_merchants": compute_metrics(y_test, xgb_pred_proba),
        "cold_start_group": compute_metrics(y_test_cold, xgb_cold_pred_proba)
    }
    joblib.dump(xgb_model, "models/xgboost_baseline.pkl")
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Baseline training complete! Results saved to data/processed/baseline_metrics.json")
    return results

if __name__ == "__main__":
    metrics = train_and_evaluate_baselines()
    print("\n" + "=" * 60)
    print("PHASE 2 BASELINE EVALUATION REPORT")
    print("=" * 60)
    for model_name, m_dict in metrics.items():
        print(f"\nModel: {model_name.upper()}")
        print("  [All Test Merchants]")
        for k, v in m_dict["all_merchants"].items():
            print(f"    - {k:20s}: {v}")
        print("  [Cold-Start Test Merchants (Zero Transactions)]")
        for k, v in m_dict["cold_start_group"].items():
            print(f"    - {k:20s}: {v}")

