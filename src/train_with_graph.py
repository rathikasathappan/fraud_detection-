"""
Phase 3: Train XGBoost with Graph Features & Compare with Baseline
Combines tabular onboarding signals with graph priors (bank/device/IP degree,
connected component size, flagged fraud neighbors, Louvain community rate).
Computes performance especially on cold-start (zero transaction) merchants.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculates PR-AUC, ROC-AUC, precision, recall, F1, and FPR."""
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

def train_and_compare_graph_model(
    graph_data_path: str = "data/processed/merchants_with_graph.csv",
    meta_path: str = "data/synthetic/rings_metadata.json",
    output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Trains XGBoost on Tabular + Graph Features with calibrated class weights.
    Evaluates cold-start merchants and compiles comparison table.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    df = pd.read_csv(graph_data_path)
    df["dt"] = pd.to_datetime(df["registration_date"])
    df = df.sort_values("dt").reset_index(drop=True)
    
    # Ring isolation + time-based split
    cutoff_idx = int(len(df) * 0.75)
    train_cutoff_time = df.loc[cutoff_idx, "dt"]
    
    merchant_to_ring = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            rings_dict = json.load(f)
            for ring_id, m_list in rings_dict.items():
                for m in m_list:
                    merchant_to_ring[m] = ring_id
                    
    # Align train_merchant_ids strictly with the saved train_tabular.csv
    train_tabular_path = os.path.join(output_dir, "train_tabular.csv")
    if os.path.exists(train_tabular_path):
        train_ref = pd.read_csv(train_tabular_path)
        train_ids_set = set(train_ref["merchant_id"])
        df["is_train"] = df["merchant_id"].isin(train_ids_set)
    else:
        df["is_train"] = df["dt"] < train_cutoff_time
    train_df = df[df["is_train"]].copy()
    test_df = df[~df["is_train"]].copy()
    
    # Define features
    tabular_features = [
        "business_category", "business_age_days",
        "phone_verified", "email_verified", "document_verified",
        "transaction_count"
    ]
    graph_features = [
        "bank_share_count", "device_share_count", "ip_share_count",
        "connected_component_size", "neighbours_flagged_fraud",
        "louvain_community_fraud_rate"
    ]
    all_features = tabular_features + graph_features
    
    categorical_cols = ["business_category"]
    numerical_cols = [c for c in all_features if c != "business_category"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )
    
    X_train_raw = train_df[all_features]
    y_train = train_df["fraud_label"].values
    
    X_test_raw = test_df[all_features]
    y_test = test_df["fraud_label"].values
    
    # Cold-start test slice
    cold_mask = test_df["transaction_count"] == 0
    X_test_cold_raw = X_test_raw[cold_mask]
    y_test_cold = y_test[cold_mask]
    
    # Fit preprocessor
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    X_test_cold = preprocessor.transform(X_test_cold_raw)
    
    # Save graph feature preprocessor
    joblib.dump(preprocessor, "models/graph_preprocessor.pkl")
    
    # Train XGBoost with graph priors
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = float(neg_count / max(1, pos_count))
    
    logger.info(f"Training XGBoost + Graph features (scale_pos_weight={scale_pos_weight:.2f})...")
    xgb_graph = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        scale_pos_weight=scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="aucpr"
    )
    xgb_graph.fit(X_train, y_train)
    
    test_preds_proba = xgb_graph.predict_proba(X_test)[:, 1]
    cold_preds_proba = xgb_graph.predict_proba(X_test_cold)[:, 1]
    
    graph_metrics = {
        "all_merchants": compute_metrics(y_test, test_preds_proba),
        "cold_start_group": compute_metrics(y_test_cold, cold_preds_proba)
    }
    
    # Save graph model
    joblib.dump(xgb_graph, "models/xgb_graph.pkl")
    with open(os.path.join(output_dir, "graph_model_metrics.json"), "w") as f:
        json.dump(graph_metrics, f, indent=2)
        
    # Compile comparison against baseline
    baseline_path = os.path.join(output_dir, "baseline_metrics.json")
    baseline_metrics = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_metrics = json.load(f)
            
    comparison = {
        "models": {
            "logistic_regression": baseline_metrics.get("logistic_regression", {}),
            "xgboost_baseline": baseline_metrics.get("xgboost_baseline", {}),
            "xgboost_with_graph": graph_metrics
        }
    }
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
        
    return comparison

if __name__ == "__main__":
    comp = train_and_compare_graph_model()
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY: BASELINE VS GRAPH PRIORS")
    print("=" * 80)
    print(f"{'Model':<25} | {'Slice':<12} | {'PR-AUC':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<8}")
    print("-" * 80)
    for model_name, data in comp["models"].items():
        for group in ["all_merchants", "cold_start_group"]:
            m = data.get(group, {})
            grp_name = "All" if group == "all_merchants" else "Cold-Start"
            print(f"{model_name:<25} | {grp_name:<12} | {m.get('pr_auc', 0):<8.4f} | {m.get('precision', 0):<10.4f} | {m.get('recall', 0):<8.4f} | {m.get('f1', 0):<8.4f}")
    print("=" * 80)
