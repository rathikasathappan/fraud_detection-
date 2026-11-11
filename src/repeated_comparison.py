"""Run a shared five-seed, ring-grouped comparison of tabular, graph, and MAML models."""

import argparse
import copy
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from src.graph_features import compute_temporal_graph_features
from src.meta_learning.maml import HAS_TORCH, MerchantMLP

try:
    import torch
except ImportError:
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
MAIN_SEEDS = tuple(range(42, 52))
MAML_SEEDS = tuple(range(42, 47))
MAML_CATEGORIES = ("Travel", "Fashion")
TABULAR_FEATURES = [
    "business_category", "business_age_days", "phone_verified", "email_verified",
    "document_verified", "transaction_count"
]
GRAPH_FEATURES = [
    "bank_share_count", "device_share_count", "ip_share_count",
    "connected_component_size", "neighbours_flagged_fraud",
    "louvain_community_fraud_rate"
]
MAML_FEATURES = [
    "business_age_days", "phone_verified", "email_verified", "document_verified",
    "transaction_count", *GRAPH_FEATURES
]
METRIC_NAMES = ["pr_auc", "roc_auc", "precision", "recall", "f1", "false_positive_rate"]


def ring_grouped_split(
    merchants: pd.DataFrame,
    rings_metadata: Dict[str, List[str]],
    seed: int,
    test_size: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Split whole fraud rings; merchants outside a named ring are individual groups."""
    merchant_to_ring = {
        merchant_id: ring_id
        for ring_id, merchant_ids in rings_metadata.items()
        if ring_id != "NONE"
        for merchant_id in merchant_ids
    }
    groups = merchants["merchant_id"].map(merchant_to_ring)
    groups = groups.where(groups.notna(), merchants["merchant_id"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_indices, test_indices = next(splitter.split(merchants, groups=groups))
    train_df = merchants.iloc[train_indices].copy()
    test_df = merchants.iloc[test_indices].copy()

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Ring-grouped split leaked {len(overlap)} group(s)")

    split_info = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "group_overlap": len(overlap)
    }
    return train_df, test_df, split_info


def ring_grouped_train_validation_test(
    merchants: pd.DataFrame,
    rings_metadata: Dict[str, List[str]],
    seed: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Create disjoint train, validation, and test partitions by ring group."""
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Train, validation, and test fractions must each be positive and sum to less than one")

    merchant_to_ring = {
        merchant_id: ring_id
        for ring_id, merchant_ids in rings_metadata.items()
        if ring_id != "NONE"
        for merchant_id in merchant_ids
    }
    groups = merchants["merchant_id"].map(merchant_to_ring)
    groups = groups.where(groups.notna(), merchants["merchant_id"])

    outer_splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_validation_indices, test_indices = next(outer_splitter.split(merchants, groups=groups))
    train_validation = merchants.iloc[train_validation_indices].copy()
    train_validation_groups = groups.iloc[train_validation_indices]
    relative_validation_fraction = validation_fraction / (1.0 - test_fraction)
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_fraction,
        random_state=seed + 100_003
    )
    train_indices, validation_indices = next(
        inner_splitter.split(train_validation, groups=train_validation_groups)
    )
    train_df = train_validation.iloc[train_indices].copy()
    validation_df = train_validation.iloc[validation_indices].copy()
    test_df = merchants.iloc[test_indices].copy()

    train_groups = set(groups.loc[train_df.index])
    validation_groups = set(groups.loc[validation_df.index])
    test_groups = set(groups.loc[test_df.index])
    overlap_count = len(
        (train_groups & validation_groups)
        | (train_groups & test_groups)
        | (validation_groups & test_groups)
    )
    if overlap_count:
        raise RuntimeError(f"Three-way ring-grouped split leaked {overlap_count} group(s)")

    split_info = {
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "test_groups": len(test_groups),
        "group_overlap": overlap_count
    }
    return train_df, validation_df, test_df, split_info


def compute_ring_link_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    relationships: pd.DataFrame,
    rings_metadata: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Measure entity links from test ring members to training merchants."""
    ring_member_ids = {
        merchant_id
        for ring_id, merchant_ids in rings_metadata.items()
        if ring_id != "NONE"
        for merchant_id in merchant_ids
    }
    test_ring_df = test_df.loc[test_df["merchant_id"].isin(ring_member_ids)]
    train_ids = set(train_df["merchant_id"])
    train_labels = train_df.set_index("merchant_id")["fraud_label"].to_dict()
    train_relationships = relationships.loc[relationships["merchant_id"].isin(train_ids)]
    entity_to_training_merchants = train_relationships.groupby("entity_id")["merchant_id"].apply(set).to_dict()
    relationships_by_merchant = relationships.groupby("merchant_id")["entity_id"].apply(set).to_dict()

    linked_count = 0
    fraud_link_count = 0
    for merchant_id in test_ring_df["merchant_id"]:
        entity_ids = relationships_by_merchant.get(merchant_id, set())
        peer_ids = set().union(*(entity_to_training_merchants.get(entity_id, set()) for entity_id in entity_ids))
        if peer_ids:
            linked_count += 1
        if any(train_labels.get(peer_id) == 1 for peer_id in peer_ids):
            fraud_link_count += 1

    denominator = len(test_ring_df)
    return {
        "test_ring_member_count": denominator,
        "linked_to_training_count": linked_count,
        "linked_to_training_fraud_count": fraud_link_count,
        "linked_to_training_share": linked_count / denominator if denominator else None,
        "linked_to_training_fraud_share": fraud_link_count / denominator if denominator else None
    }


def _main_model_features(model_name: str) -> List[str]:
    if model_name == "xgboost_graph":
        return TABULAR_FEATURES + GRAPH_FEATURES
    if model_name == "xgboost_tabular_no_business_age":
        return [feature for feature in TABULAR_FEATURES if feature != "business_age_days"]
    return list(TABULAR_FEATURES)


def _new_preprocessor(features: Sequence[str]) -> ColumnTransformer:
    categorical = [feature for feature in features if feature == "business_category"]
    numerical = [feature for feature in features if feature not in categorical]
    transformers = []
    if categorical:
        transformers.append((
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical
        ))
    if numerical:
        transformers.append(("num", StandardScaler(), numerical))
    return ColumnTransformer(transformers=transformers)


def _classifier_for_experiment(model_name: str, seed: int, positive_weight: float):
    if model_name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    graph_model = model_name == "xgboost_graph"
    return xgb.XGBClassifier(
        n_estimators=160 if graph_model else 120,
        max_depth=4,
        learning_rate=0.06 if graph_model else 0.08,
        scale_pos_weight=positive_weight,
        subsample=0.85 if graph_model else 0.8,
        colsample_bytree=0.85 if graph_model else 0.8,
        random_state=seed,
        eval_metric="aucpr",
        n_jobs=1,
        tree_method="hist"
    )


def _feature_importance_rows(
    model_name: str,
    model,
    preprocessor: ColumnTransformer,
    seed: int
) -> List[Dict[str, Any]]:
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "gain"
    else:
        values = np.abs(np.asarray(model.coef_, dtype=float)).reshape(-1)
        total = float(values.sum())
        if total:
            values = values / total
        method = "normalized_absolute_coefficient"
    return [
        {
            "seed": seed,
            "model_name": model_name,
            "feature": str(feature),
            "importance": float(value),
            "importance_method": method
        }
        for feature, value in zip(feature_names, values)
    ]


def _fit_main_model(
    model_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int
) -> Dict[str, Any]:
    features = _main_model_features(model_name)
    preprocessor = _new_preprocessor(features)
    x_train = preprocessor.fit_transform(train_df[features])
    x_validation = preprocessor.transform(validation_df[features])
    x_test = preprocessor.transform(test_df[features])
    y_train = train_df["fraud_label"].to_numpy(dtype=int)
    y_validation = validation_df["fraud_label"].to_numpy(dtype=int)
    y_test = test_df["fraud_label"].to_numpy(dtype=int)
    positive_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    model = _classifier_for_experiment(model_name, seed, positive_weight)
    model.fit(x_train, y_train)

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold, validation_f1 = choose_support_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    grouped_metrics = {}
    for group_name, group_df in {
        "all_test": test_df,
        "cold_start": test_df.loc[test_df["transaction_count"] == 0]
    }.items():
        positions = test_df.index.get_indexer(group_df.index)
        grouped_metrics[group_name] = compute_metrics(
            group_df["fraud_label"].to_numpy(dtype=int),
            test_probabilities[positions],
            threshold
        )

    return {
        "threshold": threshold,
        "validation_f1": validation_f1,
        "metrics": grouped_metrics,
        "feature_importance": _feature_importance_rows(model_name, model, preprocessor, seed)
    }


def _fit_tabular_xgboost(
    train_df: pd.DataFrame,
    seed: int,
    features: Sequence[str] = TABULAR_FEATURES,
    n_estimators: int = 120
):
    preprocessor = _new_preprocessor(features)
    x_train = preprocessor.fit_transform(train_df[list(features)])
    y_train = train_df["fraud_label"].to_numpy(dtype=int)
    positive_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=positive_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="aucpr",
        n_jobs=1,
        tree_method="hist"
    )
    model.fit(x_train, y_train)
    return model, preprocessor


def _query_group_metrics(
    query_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float
) -> Dict[str, Dict[str, Any]]:
    metrics = {}
    for group_name, group_df in {
        "held_out_category_query": query_df,
        "cold_start_query": query_df.loc[query_df["transaction_count"] == 0]
    }.items():
        positions = query_df.index.get_indexer(group_df.index)
        metrics[group_name] = compute_metrics(
            group_df["fraud_label"].to_numpy(dtype=int),
            probabilities[positions],
            threshold
        )
    return metrics


def _evaluate_tabular_baseline(
    model,
    preprocessor: ColumnTransformer,
    query_df: pd.DataFrame,
    features: Sequence[str],
    threshold: float
) -> Dict[str, Any]:
    probabilities = model.predict_proba(preprocessor.transform(query_df[list(features)]))[:, 1]
    return {
        "threshold": float(threshold),
        "metrics": _query_group_metrics(query_df, probabilities, threshold)
    }


def _fine_tune_xgboost_on_support(
    base_model,
    preprocessor: ColumnTransformer,
    support_df: pd.DataFrame,
    seed: int,
    features: Sequence[str] = TABULAR_FEATURES
):
    x_support = preprocessor.transform(support_df[list(features)])
    y_support = support_df["fraud_label"].to_numpy(dtype=int)
    positive_weight = float(np.sum(y_support == 0) / max(1, np.sum(y_support == 1)))
    fine_tuned = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.04,
        scale_pos_weight=positive_weight,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        eval_metric="aucpr",
        n_jobs=1,
        tree_method="hist"
    )
    fine_tuned.fit(x_support, y_support, xgb_model=base_model.get_booster())
    return fine_tuned


def _append_maml_metric_rows(
    rows: List[Dict[str, Any]],
    seed: int,
    category: str,
    model_name: str,
    support_shot: int,
    query_df: pd.DataFrame,
    evaluation: Dict[str, Any],
    threshold_source: str,
    support_count: int = 0,
    support_fraud_count: int = 0,
    support_f1: float = None
) -> None:
    for evaluation_group, metrics in evaluation["metrics"].items():
        rows.append({
            "seed": seed,
            "held_out_category": category,
            "model_name": model_name,
            "support_shot": support_shot,
            "evaluation_group": evaluation_group,
            "threshold": evaluation["threshold"],
            "threshold_source": threshold_source,
            "support_count": support_count,
            "support_fraud_count": support_fraud_count,
            "support_f1_at_threshold": support_f1,
            "held_out_category_in_pretraining": model_name == "xgboost_seen_category",
            "held_out_category_seen_in_support": model_name in {"xgboost_support_finetuned", "maml"},
            "support_examples_in_query": 0,
            "query_merchant_count": metrics["sample_count"],
            "query_fraud_cases": metrics["positive_count"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"]
        })


def _run_heldout_fairness(
    split_cache: Dict[int, Dict[str, Any]],
    seeds: Sequence[int],
    categories: Sequence[str],
    support_shots: Sequence[int],
    maml_epochs: int
) -> List[Dict[str, Any]]:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for held-out MAML fairness results")
    rows = []
    for seed in seeds:
        split = split_cache[seed]
        train_df = split["train_df"]
        validation_df = split["validation_df"]
        test_df = split["test_df"]

        for category in categories:
            query_df = test_df.loc[test_df["business_category"] == category].copy()
            train_category = train_df.loc[train_df["business_category"] == category]
            if query_df.empty or query_df["fraud_label"].nunique() < 2:
                raise ValueError(f"Seed {seed} has no two-class query for {category}")
            if train_category["fraud_label"].nunique() < 2:
                raise ValueError(f"Seed {seed} has no two-class support pool for {category}")

            seen_evaluation = _fit_main_model(
                "xgboost_tabular", train_df, validation_df, query_df, seed
            )
            seen_metrics = {
                "held_out_category_query": seen_evaluation["metrics"]["all_test"],
                "cold_start_query": seen_evaluation["metrics"]["cold_start"]
            }
            _append_maml_metric_rows(
                rows, seed, category, "xgboost_seen_category", 0, query_df,
                {"threshold": seen_evaluation["threshold"], "metrics": seen_metrics},
                "validation_all_categories"
            )

            zero_train = train_df.loc[train_df["business_category"] != category]
            zero_validation = validation_df.loc[validation_df["business_category"] != category]
            if zero_train.empty or zero_validation.empty:
                raise ValueError(f"Seed {seed} lacks seen-category data for the {category} zero-shot baseline")
            zero_model, zero_preprocessor = _fit_tabular_xgboost(zero_train, seed)
            zero_validation_probabilities = zero_model.predict_proba(
                zero_preprocessor.transform(zero_validation[TABULAR_FEATURES])
            )[:, 1]
            zero_threshold, _ = choose_support_threshold(
                zero_validation["fraud_label"].to_numpy(dtype=int),
                zero_validation_probabilities
            )
            zero_evaluation = _evaluate_tabular_baseline(
                zero_model, zero_preprocessor, query_df, TABULAR_FEATURES, zero_threshold
            )
            _append_maml_metric_rows(
                rows, seed, category, "xgboost_zero_shot", 0, query_df,
                zero_evaluation, "validation_seen_categories"
            )

            maml_scaler = StandardScaler().fit(zero_train[MAML_FEATURES])
            meta_model = _train_fomaml(
                zero_train, category, maml_scaler, seed, maml_epochs
            )
            for shot in support_shots:
                support_rng = np.random.default_rng(
                    seed * 1000 + shot * 17 + sum(ord(char) for char in category)
                )
                support_df = _select_support(train_category, shot, support_rng)
                support_ids = set(support_df["merchant_id"])
                query_ids = set(query_df["merchant_id"])
                if support_ids.intersection(query_ids):
                    raise RuntimeError("Support examples overlap the held-out query")

                fine_tuned_model = _fine_tune_xgboost_on_support(
                    zero_model, zero_preprocessor, support_df, seed
                )
                support_x = zero_preprocessor.transform(support_df[TABULAR_FEATURES])
                support_y = support_df["fraud_label"].to_numpy(dtype=int)
                fine_tuned_support_probabilities = fine_tuned_model.predict_proba(support_x)[:, 1]
                fine_tuned_threshold, fine_tuned_support_f1 = choose_support_threshold(
                    support_y, fine_tuned_support_probabilities
                )
                fine_tuned_evaluation = _evaluate_tabular_baseline(
                    fine_tuned_model, zero_preprocessor, query_df,
                    TABULAR_FEATURES, fine_tuned_threshold
                )
                _append_maml_metric_rows(
                    rows, seed, category, "xgboost_support_finetuned", shot, query_df,
                    fine_tuned_evaluation, "support_after_finetuning",
                    support_count=len(support_df),
                    support_fraud_count=int(support_y.sum()),
                    support_f1=fine_tuned_support_f1
                )

                maml_evaluation = _evaluate_maml_shot(
                    meta_model, train_df, query_df, maml_scaler, shot, seed,
                    support_df=support_df
                )
                _append_maml_metric_rows(
                    rows, seed, category, "maml", shot, query_df,
                    {"threshold": maml_evaluation["threshold"], "metrics": maml_evaluation["metrics"]},
                    "support_after_adaptation",
                    support_count=maml_evaluation["support_count"],
                    support_fraud_count=maml_evaluation["support_positive_count"],
                    support_f1=maml_evaluation["support_f1_at_selected_threshold"]
                )
    return rows


def _numeric_spread(values: Iterable[Any]) -> Dict[str, Any]:
    numeric_values = np.asarray([value for value in values if value is not None and pd.notna(value)], dtype=float)
    if not len(numeric_values):
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(numeric_values)),
        "std": float(np.std(numeric_values, ddof=1)) if len(numeric_values) > 1 else 0.0,
        "min": float(np.min(numeric_values)),
        "max": float(np.max(numeric_values))
    }


def _summarize_main_metrics(metrics_per_seed: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (model_name, evaluation_group), group in metrics_per_seed.groupby(["model_name", "evaluation_group"]):
        row = {
            "model_name": model_name,
            "evaluation_group": evaluation_group,
            "seed_count": int(group["seed"].nunique())
        }
        for field in [*METRIC_NAMES, "threshold", "validation_f1", "test_merchants", "test_fraud_cases"]:
            spread = _numeric_spread(group[field])
            for statistic, value in spread.items():
                row[f"{field}_{statistic}"] = value
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _summarize_maml_results(maml_results: pd.DataFrame) -> pd.DataFrame:
    group_fields = ["held_out_category", "model_name", "support_shot", "evaluation_group"]
    summary_rows = []
    for group_values, group in maml_results.groupby(group_fields, dropna=False):
        row = dict(zip(group_fields, group_values))
        row["seed_count"] = int(group["seed"].nunique())
        for field in [*METRIC_NAMES, "threshold", "support_count", "support_fraud_count", "support_f1_at_threshold"]:
            spread = _numeric_spread(group[field])
            for statistic, value in spread.items():
                row[f"{field}_{statistic}"] = value
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _build_paired_comparisons(
    metrics_by_seed: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]]
) -> pd.DataFrame:
    rows = []
    pairs = [
        ("graph_minus_tabular", "xgboost_graph", "xgboost_tabular"),
        ("graph_minus_tabular_no_business_age", "xgboost_graph", "xgboost_tabular_no_business_age"),
        ("tabular_with_age_minus_without", "xgboost_tabular", "xgboost_tabular_no_business_age")
    ]
    for comparison_name, model_a, model_b in pairs:
        for evaluation_group in ("all_test", "cold_start"):
            differences = []
            for seed in sorted(metrics_by_seed):
                score_a = metrics_by_seed[seed][model_a][evaluation_group]["pr_auc"]
                score_b = metrics_by_seed[seed][model_b][evaluation_group]["pr_auc"]
                if score_a is None or score_b is None:
                    continue
                difference = float(score_a - score_b)
                differences.append(difference)
                rows.append({
                    "comparison": comparison_name,
                    "evaluation_group": evaluation_group,
                    "seed": seed,
                    "model_a": model_a,
                    "model_b": model_b,
                    "pr_auc_model_a": score_a,
                    "pr_auc_model_b": score_b,
                    "pr_auc_difference": difference,
                    "model_a_wins": difference > 0
                })
            spread = _numeric_spread(differences)
            rows.append({
                "comparison": comparison_name,
                "evaluation_group": evaluation_group,
                "seed": "SUMMARY",
                "model_a": model_a,
                "model_b": model_b,
                "pr_auc_model_a": None,
                "pr_auc_model_b": None,
                "pr_auc_difference": spread["mean"],
                "difference_std": spread["std"],
                "difference_min": spread["min"],
                "difference_max": spread["max"],
                "model_a_wins": sum(value > 0 for value in differences),
                "seed_count": len(differences)
            })
    return pd.DataFrame(rows)


def _summarize_importance(feature_rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    per_seed = pd.DataFrame(feature_rows)
    summary_rows = []
    for (model_name, feature, method), group in per_seed.groupby(
        ["model_name", "feature", "importance_method"]
    ):
        spread = _numeric_spread(group["importance"])
        summary_rows.append({
            "seed": "SUMMARY",
            "model_name": model_name,
            "feature": feature,
            "importance": spread["mean"],
            "importance_std": spread["std"],
            "importance_method": method
        })
    return pd.concat([per_seed, pd.DataFrame(summary_rows)], ignore_index=True)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _format_mean_std(mean: Any, std: Any, digits: int = 4) -> str:
    if mean is None or pd.isna(mean):
        return "n/a"
    return f"{float(mean):.{digits}f} +/- {float(std or 0):.{digits}f}"


def _powerbi_metric_rows(
    main_summary: pd.DataFrame,
    maml_summary: pd.DataFrame
) -> List[Dict[str, Any]]:
    rows = []
    main_names = {
        "xgboost_tabular": "xgboost_baseline",
        "xgboost_graph": "xgboost_with_graph",
        "xgboost_tabular_no_business_age": "xgboost_without_business_age"
    }
    evaluation_names = {"all_test": "all_merchants", "cold_start": "cold_start_group"}
    for _, summary in main_summary.iterrows():
        rows.append({
            "model_name": main_names.get(summary["model_name"], summary["model_name"]),
            "evaluation_group": evaluation_names[summary["evaluation_group"]],
            "pr_auc": summary["pr_auc_mean"],
            "roc_auc": summary["roc_auc_mean"],
            "precision": summary["precision_mean"],
            "recall": summary["recall_mean"],
            "f1": summary["f1_mean"],
            "false_positive_rate": summary["false_positive_rate_mean"]
        })

    maml_models = {
        "xgboost_seen_category": "xgboost_seen_category",
        "xgboost_zero_shot": "xgboost_zero_shot",
        "xgboost_support_finetuned": "xgboost_support_finetuned",
        "maml": "maml"
    }
    for _, summary in maml_summary.iterrows():
        shot = int(summary["support_shot"] or 0)
        model_name = maml_models[summary["model_name"]]
        if shot:
            model_name = f"{model_name}_{shot}_shot"
        group_name = "cold_start" if summary["evaluation_group"] == "cold_start_query" else "held_out_category_query"
        rows.append({
            "model_name": model_name,
            "evaluation_group": f"{str(summary['held_out_category']).lower()}_{group_name}",
            "pr_auc": summary["pr_auc_mean"],
            "roc_auc": summary["roc_auc_mean"],
            "precision": summary["precision_mean"],
            "recall": summary["recall_mean"],
            "f1": summary["f1_mean"],
            "false_positive_rate": summary["false_positive_rate_mean"]
        })
    return rows


def _write_results_markdown(
    output_dir: str,
    main_summary: pd.DataFrame,
    paired: pd.DataFrame,
    ring_checks: pd.DataFrame,
    maml_summary: pd.DataFrame,
    importance: pd.DataFrame,
    main_seeds: Sequence[int],
    maml_seeds: Sequence[int],
    maml_categories: Sequence[str],
    powerbi_path: str
) -> str:
    main_rows = []
    for _, row in main_summary.iterrows():
        main_rows.append([
            row["model_name"], row["evaluation_group"],
            _format_mean_std(row["pr_auc_mean"], row["pr_auc_std"]),
            _format_mean_std(row["precision_mean"], row["precision_std"]),
            _format_mean_std(row["recall_mean"], row["recall_std"]),
            _format_mean_std(row["f1_mean"], row["f1_std"]),
            _format_mean_std(row["test_fraud_cases_mean"], row["test_fraud_cases_std"], 2)
        ])

    paired_summary = paired.loc[paired["seed"] == "SUMMARY"]
    paired_rows = []
    paired_findings = []
    for _, row in paired_summary.iterrows():
        paired_rows.append([
            row["comparison"], row["evaluation_group"],
            _format_mean_std(row["pr_auc_difference"], row["difference_std"]),
            row["model_a_wins"], row["seed_count"]
        ])
        mean = row["pr_auc_difference"]
        std = row["difference_std"]
        if mean is None or pd.isna(mean):
            continue
        if abs(mean) <= (std or 0):
            finding = "no clear difference"
        elif row["comparison"].startswith("graph_minus_tabular"):
            finding = "graph XGBoost has the higher mean PR-AUC"
        elif mean > 0:
            finding = "business age improves mean PR-AUC"
        else:
            finding = "omitting business age has the higher mean PR-AUC"
        paired_findings.append(f"- {row['comparison']} on {row['evaluation_group']}: {finding} (mean paired difference {mean:.4f}, SD {float(std or 0):.4f}).")

    ring_total = int(ring_checks["test_ring_member_count"].sum())
    link_total = int(ring_checks["linked_to_training_count"].sum())
    fraud_link_total = int(ring_checks["linked_to_training_fraud_count"].sum())
    pooled_link_share = link_total / ring_total if ring_total else None
    pooled_fraud_share = fraud_link_total / ring_total if ring_total else None
    ring_rows = [[
        ring_total,
        link_total,
        "n/a" if pooled_link_share is None else f"{pooled_link_share:.2%}",
        fraud_link_total,
        "n/a" if pooled_fraud_share is None else f"{pooled_fraud_share:.2%}",
        _format_mean_std(ring_checks["linked_to_training_fraud_share"].mean(), ring_checks["linked_to_training_fraud_share"].std(), 4)
    ]]

    maml_rows = []
    for _, row in maml_summary.iterrows():
        maml_rows.append([
            row["held_out_category"], row["model_name"], int(row["support_shot"] or 0),
            row["evaluation_group"], _format_mean_std(row["pr_auc_mean"], row["pr_auc_std"]),
            _format_mean_std(row["recall_mean"], row["recall_std"]),
            _format_mean_std(row["f1_mean"], row["f1_std"])
        ])

    maml_findings = []
    for category in sorted(maml_summary["held_out_category"].unique()):
        category_rows = maml_summary.loc[
            (maml_summary["held_out_category"] == category)
            & (maml_summary["evaluation_group"] == "held_out_category_query")
        ]
        zero_shot = category_rows.loc[category_rows["model_name"] == "xgboost_zero_shot"].iloc[0]
        for shot in sorted(int(value) for value in category_rows.loc[category_rows["model_name"] == "maml", "support_shot"].unique()):
            maml_row = category_rows.loc[
                (category_rows["model_name"] == "maml")
                & (category_rows["support_shot"] == shot)
            ].iloc[0]
            for baseline_name in ("xgboost_zero_shot", "xgboost_support_finetuned"):
                if baseline_name == "xgboost_zero_shot":
                    baseline_row = zero_shot
                else:
                    baseline_row = category_rows.loc[
                        (category_rows["model_name"] == baseline_name)
                        & (category_rows["support_shot"] == shot)
                    ].iloc[0]
                difference = float(maml_row["pr_auc_mean"] - baseline_row["pr_auc_mean"])
                spread = float(np.sqrt(
                    float(maml_row["pr_auc_std"] or 0) ** 2
                    + float(baseline_row["pr_auc_std"] or 0) ** 2
                ))
                if abs(difference) <= spread:
                    finding = "no clear difference"
                elif difference > 0:
                    finding = "MAML has the higher mean PR-AUC"
                else:
                    finding = "the XGBoost baseline has the higher mean PR-AUC"
                maml_findings.append(
                    f"- {category}, {shot}-shot MAML vs {baseline_name}: {finding} "
                    f"(mean gap {difference:.4f}; combined observed SD {spread:.4f})."
                )

    importance_summary = importance.loc[importance["seed"] == "SUMMARY"]
    importance_rows = []
    for model_name in sorted(importance_summary["model_name"].unique()):
        top = importance_summary.loc[importance_summary["model_name"] == model_name].nlargest(5, "importance")
        importance_rows.extend([
            [model_name, row["feature"], f"{row['importance']:.6f}", f"{float(row['importance_std'] or 0):.6f}"]
            for _, row in top.iterrows()
        ])

    age_rows = paired_summary.loc[paired_summary["comparison"] == "tabular_with_age_minus_without"]
    age_effects = {row["evaluation_group"]: row for _, row in age_rows.iterrows()}
    all_age = age_effects.get("all_test")
    cold_age = age_effects.get("cold_start")
    tabular_all = main_summary.loc[
        (main_summary["model_name"] == "xgboost_tabular")
        & (main_summary["evaluation_group"] == "all_test")
    ].iloc[0]
    no_age_all = main_summary.loc[
        (main_summary["model_name"] == "xgboost_tabular_no_business_age")
        & (main_summary["evaluation_group"] == "all_test")
    ].iloc[0]
    tabular_cold = main_summary.loc[
        (main_summary["model_name"] == "xgboost_tabular")
        & (main_summary["evaluation_group"] == "cold_start")
    ].iloc[0]
    no_age_cold = main_summary.loc[
        (main_summary["model_name"] == "xgboost_tabular_no_business_age")
        & (main_summary["evaluation_group"] == "cold_start")
    ].iloc[0]
    if all_age is None or all_age["pr_auc_difference"] is None:
        age_statement = "The business-age ablation could not be summarized from measured PR-AUC values."
    elif abs(all_age["pr_auc_difference"]) <= float(all_age["difference_std"] or 0):
        age_statement = (
            f"On all test merchants, tabular XGBoost PR-AUC was {tabular_all['pr_auc_mean']:.4f} with business age and "
            f"{no_age_all['pr_auc_mean']:.4f} without it. The paired change was {all_age['pr_auc_difference']:.4f} "
            f"(SD {float(all_age['difference_std'] or 0):.4f}); no clear difference is visible within the seed-to-seed spread."
        )
    else:
        age_statement = (
            f"The measured synthetic fraud signal depends substantially, but not exclusively, on business age. On all test merchants, "
            f"tabular XGBoost PR-AUC was {tabular_all['pr_auc_mean']:.4f} with age and {no_age_all['pr_auc_mean']:.4f} without it "
            f"(paired change {all_age['pr_auc_difference']:.4f}, SD {float(all_age['difference_std'] or 0):.4f}); "
            f"on cold-start merchants it was {tabular_cold['pr_auc_mean']:.4f} with age and {no_age_cold['pr_auc_mean']:.4f} without it "
            f"(paired change {cold_age['pr_auc_difference']:.4f}, SD {float(cold_age['difference_std'] or 0):.4f})."
        )

    fraud_statement = "The share of test ring members with a training fraud link was not measurable." \
        if pooled_fraud_share is None else (
            f"The share of test ring members with a training fraud link was {pooled_fraud_share:.2%}. "
            + ("This is low (near 0, but not zero); ring-grouped splitting keeps full rings out of training." if pooled_fraud_share < 0.05 else "")
        )

    lines = [
        "# Measured Model Comparison Results",
        "",
        f"Run folder: `{os.path.basename(output_dir)}`.",
        f"Main comparison seeds: {', '.join(map(str, main_seeds))}.",
        f"Held-out MAML seeds: {', '.join(map(str, maml_seeds))}; categories: {', '.join(maml_categories)}.",
        "Each main split uses disjoint ring-grouped train, validation, and test partitions. All main models use the same split, select a threshold by maximizing validation F1, and are evaluated only on test rows. Graph features are recomputed per seed using training labels only.",
        "",
        "## Main Comparison",
        "",
        _markdown_table(
            ["Model", "Test group", "PR-AUC mean +/- SD", "Precision", "Recall", "F1", "Fraud cases mean +/- SD"],
            main_rows
        ),
        "",
        "## Paired Differences",
        "",
        _markdown_table(
            ["Comparison", "Test group", "PR-AUC difference mean +/- SD", "Positive seeds", "Seeds"],
            paired_rows
        ),
        "",
        *paired_findings,
        "",
        "## Ring-Link Check",
        "",
        _markdown_table(
            ["Test ring-member occurrences", "Any training entity link", "Share", "Training fraud link", "Share", "Per-seed fraud-link share mean +/- SD"],
            ring_rows
        ),
        fraud_statement,
        "",
        "## MAML Fairness",
        "",
        "The standard tabular XGBoost baseline trained on the full training split, so it did see Travel merchants. The `xgboost_zero_shot` baseline and MAML meta-training exclude the held-out category; both receive only the listed training support examples for category adaptation. The support-fine-tuned XGBoost and MAML runs use the same sampled support rows. Query rows are disjoint from support rows. Zero-shot thresholds are chosen on validation rows from seen categories; fine-tuned XGBoost and MAML thresholds are chosen on support rows.",
        "",
        _markdown_table(
            ["Category", "Model", "Shots", "Query group", "PR-AUC mean +/- SD", "Recall", "F1"],
            maml_rows
        ),
        *maml_findings,
        "",
        "## Business-Age Ablation",
        "",
        age_statement,
        "",
        "## Feature Importance",
        "",
        "Importance is averaged across main-comparison seeds (normalized absolute coefficients for Logistic Regression; XGBoost gain-based `feature_importances_` for tree models).",
        "",
        _markdown_table(["Model", "Feature", "Mean importance", "SD"], importance_rows),
        "",
        f"The dashboard CSV was refreshed from this run: `{os.path.relpath(powerbi_path, os.getcwd())}`.",
        ""
    ]
    return "\n".join(lines)


def run_full_study(
    merchants_path: str = "data/synthetic/merchants.csv",
    relationships_path: str = "data/synthetic/relationships.csv",
    rings_path: str = "data/synthetic/rings_metadata.json",
    output_root: str = "results",
    main_seeds: Sequence[int] = MAIN_SEEDS,
    maml_seeds: Sequence[int] = MAML_SEEDS,
    maml_categories: Sequence[str] = MAML_CATEGORIES,
    support_shots: Sequence[int] = (5, 10, 20),
    maml_epochs: int = 20,
    powerbi_path: str = "data/powerbi_exports/model_metrics.csv"
) -> str:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for the measured MAML results; no simulated values are emitted")
    if len(main_seeds) != 10 or len(set(main_seeds)) != 10:
        raise ValueError("The main comparison requires exactly ten distinct seeds")
    if len(maml_seeds) < 5 or not set(maml_seeds).issubset(set(main_seeds)):
        raise ValueError("Use at least five MAML seeds drawn from the main seed set")
    if len(maml_categories) < 2:
        raise ValueError("At least two held-out categories are required")

    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.abspath(os.path.join(output_root, run_name))
    os.makedirs(output_dir, exist_ok=False)
    merchants = pd.read_csv(merchants_path)
    relationships = pd.read_csv(relationships_path)
    with open(rings_path, "r", encoding="utf-8") as rings_file:
        rings_metadata = json.load(rings_file)

    metrics_per_seed_rows = []
    feature_rows = []
    ring_check_rows = []
    split_cache = {}
    model_metrics_by_seed = {}
    main_model_names = [
        "logistic_regression", "xgboost_tabular", "xgboost_graph",
        "xgboost_tabular_no_business_age"
    ]

    for seed in main_seeds:
        train_raw, validation_raw, test_raw, split_info = ring_grouped_train_validation_test(
            merchants, rings_metadata, seed
        )
        train_ids = set(train_raw["merchant_id"])
        validation_ids = set(validation_raw["merchant_id"])
        test_ids = set(test_raw["merchant_id"])
        graph_df = compute_temporal_graph_features(
            merchants,
            relationships,
            train_merchant_ids=train_ids
        )
        train_df = graph_df.loc[graph_df["merchant_id"].isin(train_ids)].copy()
        validation_df = graph_df.loc[graph_df["merchant_id"].isin(validation_ids)].copy()
        test_df = graph_df.loc[graph_df["merchant_id"].isin(test_ids)].copy()
        ring_check = compute_ring_link_check(train_df, test_df, relationships, rings_metadata)
        ring_check_rows.append({"seed": int(seed), **ring_check})
        split_cache[int(seed)] = {
            "train_df": train_df,
            "validation_df": validation_df,
            "test_df": test_df,
            "split_info": split_info
        }
        model_metrics_by_seed[int(seed)] = {}

        for model_name in main_model_names:
            result = _fit_main_model(model_name, train_df, validation_df, test_df, int(seed))
            model_metrics_by_seed[int(seed)][model_name] = result["metrics"]
            feature_rows.extend(result["feature_importance"])
            for evaluation_group, metrics in result["metrics"].items():
                metrics_per_seed_rows.append({
                    "seed": int(seed),
                    "model_name": model_name,
                    "evaluation_group": evaluation_group,
                    "train_merchants": len(train_df),
                    "validation_merchants": len(validation_df),
                    "ring_group_overlap": split_info["group_overlap"],
                    "test_merchants": metrics["sample_count"],
                    "test_fraud_cases": metrics["positive_count"],
                    "threshold": result["threshold"],
                    "validation_f1": result["validation_f1"],
                    "test_ring_member_count": ring_check["test_ring_member_count"],
                    "linked_to_training_count": ring_check["linked_to_training_count"],
                    "linked_to_training_share": ring_check["linked_to_training_share"],
                    "linked_to_training_fraud_count": ring_check["linked_to_training_fraud_count"],
                    "linked_to_training_fraud_share": ring_check["linked_to_training_fraud_share"],
                    **metrics
                })
        logger.info(
            "Main seed %s complete: train=%s validation=%s test=%s",
            seed, len(train_df), len(validation_df), len(test_df)
        )

    metrics_per_seed = pd.DataFrame(metrics_per_seed_rows)
    main_summary = _summarize_main_metrics(metrics_per_seed)
    paired = _build_paired_comparisons(model_metrics_by_seed)
    feature_importance = _summarize_importance(feature_rows)
    ring_checks = pd.DataFrame(ring_check_rows)

    maml_results = pd.DataFrame(_run_heldout_fairness(
        split_cache,
        maml_seeds,
        maml_categories,
        support_shots,
        maml_epochs
    ))
    maml_summary = _summarize_maml_results(maml_results)
    powerbi_rows = _powerbi_metric_rows(main_summary, maml_summary)

    metrics_per_seed.to_csv(os.path.join(output_dir, "metrics_per_seed.csv"), index=False)
    main_summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    paired.to_csv(os.path.join(output_dir, "paired_comparison.csv"), index=False)
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    maml_results.to_csv(os.path.join(output_dir, "maml_results.csv"), index=False)
    pd.DataFrame(powerbi_rows).to_csv(powerbi_path, index=False)

    results_markdown = _write_results_markdown(
        output_dir,
        main_summary,
        paired,
        ring_checks,
        maml_summary,
        feature_importance,
        main_seeds,
        maml_seeds,
        maml_categories,
        powerbi_path
    )
    with open(os.path.join(output_dir, "RESULTS.md"), "w", encoding="utf-8") as results_file:
        results_file.write(results_markdown)

    print(f"Run artifacts: {output_dir}")
    print("\nMain comparison: mean +/- SD across seeds; test fraud counts are mean +/- SD.")
    print(main_summary[[
        "model_name", "evaluation_group", "pr_auc_mean", "pr_auc_std",
        "recall_mean", "recall_std", "f1_mean", "f1_std",
        "test_fraud_cases_mean", "test_fraud_cases_std"
    ]].to_string(index=False))
    print("\nPaired comparisons: PR-AUC differences, model A minus model B.")
    print(paired.loc[paired["seed"] == "SUMMARY", [
        "comparison", "evaluation_group", "pr_auc_difference",
        "difference_std", "model_a_wins", "seed_count"
    ]].to_string(index=False))
    print("\nRing-link check: test ring-member link shares pooled across main seeds.")
    ring_total = int(ring_checks["test_ring_member_count"].sum())
    linked_total = int(ring_checks["linked_to_training_count"].sum())
    fraud_link_total = int(ring_checks["linked_to_training_fraud_count"].sum())
    print({
        "test_ring_member_occurrences": ring_total,
        "linked_to_training": linked_total,
        "linked_to_training_share": linked_total / ring_total if ring_total else None,
        "linked_to_training_fraud": fraud_link_total,
        "linked_to_training_fraud_share": fraud_link_total / ring_total if ring_total else None
    })
    print("\nHeld-out fairness: mean +/- SD on the common query.")
    print(maml_summary.loc[maml_summary["evaluation_group"] == "held_out_category_query", [
        "held_out_category", "model_name", "support_shot", "pr_auc_mean",
        "pr_auc_std", "recall_mean", "recall_std", "f1_mean", "f1_std"
    ]].to_string(index=False))
    return output_dir


def compute_metrics(y_true: Iterable[int], probabilities: Iterable[float], threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(list(y_true), dtype=int)
    probabilities = np.asarray(list(probabilities), dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    labels = np.unique(y_true)
    has_both_classes = len(labels) == 2
    negative_count = int(np.sum(y_true == 0))

    return {
        "sample_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true == 1)),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4) if has_both_classes else None,
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4) if has_both_classes else None,
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "false_positive_rate": round(float(fp / negative_count), 4) if negative_count else None,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def choose_support_threshold(labels: Iterable[int], probabilities: Iterable[float]) -> Tuple[float, float]:
    """Choose the F1-maximizing threshold using support predictions only."""
    labels = np.asarray(list(labels), dtype=int)
    probabilities = np.asarray(list(probabilities), dtype=float)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))

    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0, -1.0)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        f1 = float(f1_score(labels, predictions, zero_division=0))
        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        candidate_key = (f1, precision, recall, float(threshold))
        if candidate_key > best_key:
            best_key = candidate_key
            best_threshold = float(threshold)
    return best_threshold, best_key[0]


def _evaluation_groups(query_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "held_out_category_query": query_df,
        "cold_start_query": query_df.loc[query_df["transaction_count"] == 0]
    }


def _fit_xgboost(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    features: Sequence[str],
    seed: int,
    graph_model: bool
) -> Dict[str, Any]:
    categorical = ["business_category"]
    numerical = [feature for feature in features if feature not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numerical)
        ]
    )
    x_train = preprocessor.fit_transform(train_df[list(features)])
    y_train = train_df["fraud_label"].to_numpy(dtype=int)
    x_query = preprocessor.transform(query_df[list(features)])
    y_query = query_df["fraud_label"].to_numpy(dtype=int)

    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    parameters = {
        "n_estimators": 160 if graph_model else 120,
        "max_depth": 4,
        "learning_rate": 0.06 if graph_model else 0.08,
        "scale_pos_weight": negative_count / max(1, positive_count),
        "subsample": 0.85 if graph_model else 0.8,
        "colsample_bytree": 0.85 if graph_model else 0.8,
        "random_state": seed,
        "eval_metric": "aucpr",
        "n_jobs": 1,
        "tree_method": "hist"
    }
    model = xgb.XGBClassifier(**parameters)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_query)[:, 1]
    grouped_metrics = {}
    for name, group in _evaluation_groups(query_df).items():
        positions = query_df.index.get_indexer(group.index)
        grouped_metrics[name] = compute_metrics(
            group["fraud_label"].to_numpy(dtype=int),
            probabilities[positions],
            threshold=0.5
        )

    feature_names = preprocessor.get_feature_names_out()
    importances = [
        {"feature": str(name), "importance": round(float(value), 8)}
        for name, value in zip(feature_names, model.feature_importances_)
    ]
    return {
        "threshold": 0.5,
        "metrics": grouped_metrics,
        "feature_importance": importances
    }


def _sample_episode(
    category_df: pd.DataFrame,
    rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = category_df["fraud_label"].to_numpy(dtype=int)
    positive_positions = np.flatnonzero(labels == 1)
    negative_positions = np.flatnonzero(labels == 0)
    support_positive_count = min(4, max(1, len(positive_positions) // 3))
    query_positive_count = min(4, len(positive_positions) - support_positive_count)
    support_negative_count = min(12, max(1, len(negative_positions) // 3))
    query_negative_count = min(16, len(negative_positions) - support_negative_count)
    if query_positive_count < 1 or query_negative_count < 1:
        return category_df.iloc[0:0], category_df.iloc[0:0]

    support_positive = rng.choice(positive_positions, support_positive_count, replace=False)
    remaining_positive = np.setdiff1d(positive_positions, support_positive)
    query_positive = rng.choice(remaining_positive, query_positive_count, replace=False)
    support_negative = rng.choice(negative_positions, support_negative_count, replace=False)
    remaining_negative = np.setdiff1d(negative_positions, support_negative)
    query_negative = rng.choice(remaining_negative, query_negative_count, replace=False)

    support_positions = np.concatenate((support_positive, support_negative))
    query_positions = np.concatenate((query_positive, query_negative))
    return category_df.iloc[support_positions], category_df.iloc[query_positions]


def _scaled_tensor(frame: pd.DataFrame, scaler: StandardScaler, features: Sequence[str]):
    return torch.tensor(scaler.transform(frame[list(features)]), dtype=torch.float32)


def _train_fomaml(
    train_df: pd.DataFrame,
    held_out_category: str,
    scaler: StandardScaler,
    seed: int,
    epochs: int
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    categories = sorted(set(train_df["business_category"]) - {held_out_category})
    meta_model = MerchantMLP(input_dim=len(MAML_FEATURES), hidden_dim=32)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.003)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        gradient_sums = [torch.zeros_like(parameter) for parameter in meta_model.parameters()]
        episode_count = 0
        rng = np.random.default_rng(seed * 1000 + epoch)
        for category in categories:
            category_df = train_df.loc[train_df["business_category"] == category].reset_index(drop=True)
            support_df, query_df = _sample_episode(category_df, rng)
            if support_df.empty or query_df.empty:
                continue

            task_model = MerchantMLP(input_dim=len(MAML_FEATURES), hidden_dim=32)
            task_model.load_state_dict(meta_model.state_dict())
            task_model.train()
            task_optimizer = torch.optim.SGD(task_model.parameters(), lr=0.05)
            support_x = _scaled_tensor(support_df, scaler, MAML_FEATURES)
            support_y = torch.tensor(support_df["fraud_label"].to_numpy(), dtype=torch.float32).unsqueeze(1)
            task_optimizer.zero_grad()
            support_loss = criterion(task_model(support_x), support_y)
            support_loss.backward()
            task_optimizer.step()

            query_x = _scaled_tensor(query_df, scaler, MAML_FEATURES)
            query_y = torch.tensor(query_df["fraud_label"].to_numpy(), dtype=torch.float32).unsqueeze(1)
            query_loss = criterion(task_model(query_x), query_y)
            task_gradients = torch.autograd.grad(query_loss, tuple(task_model.parameters()))
            for gradient_sum, task_gradient in zip(gradient_sums, task_gradients):
                gradient_sum.add_(task_gradient.detach())
            episode_count += 1

        if episode_count:
            meta_optimizer.zero_grad()
            for parameter, gradient_sum in zip(meta_model.parameters(), gradient_sums):
                parameter.grad = gradient_sum / episode_count
            meta_optimizer.step()

    return meta_model


def _select_support(
    training_category: pd.DataFrame,
    shot_count: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    fraud_rows = training_category.loc[training_category["fraud_label"] == 1]
    genuine_rows = training_category.loc[training_category["fraud_label"] == 0]
    fraud_count = max(1, shot_count // 5)
    genuine_count = shot_count - fraud_count
    if len(fraud_rows) < fraud_count or len(genuine_rows) < genuine_count:
        raise ValueError(f"Not enough training examples to construct a {shot_count}-shot support set")
    fraud_sample = fraud_rows.iloc[rng.choice(len(fraud_rows), fraud_count, replace=False)]
    genuine_sample = genuine_rows.iloc[rng.choice(len(genuine_rows), genuine_count, replace=False)]
    return pd.concat((fraud_sample, genuine_sample)).sample(frac=1, random_state=int(rng.integers(0, 2**31)))


def _evaluate_maml_shot(
    meta_model,
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    scaler: StandardScaler,
    shot_count: int,
    seed: int,
    support_df: pd.DataFrame = None
) -> Dict[str, Any]:
    training_category = train_df.loc[train_df["business_category"] == query_df["business_category"].iloc[0]]
    rng = np.random.default_rng(seed * 100 + shot_count)
    if support_df is None:
        support_df = _select_support(training_category, shot_count, rng)
    query_ids = set(query_df["merchant_id"])
    support_ids = set(support_df["merchant_id"])
    if support_ids.intersection(query_ids):
        raise RuntimeError("MAML support examples overlap the evaluation query")

    adapted_model = copy.deepcopy(meta_model)
    adapted_model.train()
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.05)
    support_x = _scaled_tensor(support_df, scaler, MAML_FEATURES)
    support_y = torch.tensor(support_df["fraud_label"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    positive_count = int(torch.sum(support_y).item())
    negative_count = len(support_y) - positive_count
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / max(1, positive_count)], dtype=torch.float32)
    )
    for _ in range(5):
        optimizer.zero_grad()
        criterion(adapted_model(support_x), support_y).backward()
        optimizer.step()

    adapted_model.eval()
    query_x = _scaled_tensor(query_df, scaler, MAML_FEATURES)
    with torch.no_grad():
        support_probabilities = torch.sigmoid(adapted_model(support_x)).squeeze(1).numpy()
        query_probabilities = torch.sigmoid(adapted_model(query_x)).squeeze(1).numpy()

    threshold, support_f1 = choose_support_threshold(support_y.squeeze(1).numpy(), support_probabilities)
    grouped_metrics = {}
    for group_name, group_df in _evaluation_groups(query_df).items():
        positions = query_df.index.get_indexer(group_df.index)
        grouped_metrics[group_name] = compute_metrics(
            group_df["fraud_label"].to_numpy(dtype=int),
            query_probabilities[positions],
            threshold
        )

    return {
        "threshold": round(threshold, 6),
        "support_count": len(support_df),
        "support_positive_count": positive_count,
        "support_f1_at_selected_threshold": round(support_f1, 4),
        "support_examples_in_query": 0,
        "metrics": grouped_metrics
    }


def _summarize_metric_rows(metric_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for metric_name in METRIC_NAMES:
        values = [row[metric_name] for row in metric_rows if row.get(metric_name) is not None]
        if not values:
            summary[metric_name] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
            continue
        summary[metric_name] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "n": len(values)
        }
    return summary


def _summarize_feature_importance(runs: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    summary = {}
    for model_name in ("xgboost_tabular", "xgboost_graph"):
        importance_by_run = [
            {item["feature"]: item["importance"] for item in run["models"][model_name]["feature_importance"]}
            for run in runs
        ]
        feature_names = sorted(set().union(*(importance.keys() for importance in importance_by_run)))
        rows = []
        for feature in feature_names:
            values = [importance.get(feature, 0.0) for importance in importance_by_run]
            rows.append({
                "feature": feature,
                "mean": round(float(np.mean(values)), 8),
                "std": round(float(np.std(values, ddof=1)), 8) if len(values) > 1 else 0.0
            })
        summary[model_name] = sorted(rows, key=lambda row: row["mean"], reverse=True)[:20]
    return summary


def summarize_runs(runs: Sequence[Dict[str, Any]], support_shots: Sequence[int]) -> Dict[str, Any]:
    summary = {}
    model_keys = ["xgboost_tabular", "xgboost_graph"]
    model_keys.extend(f"maml_{shot}_shot" for shot in support_shots)
    for model_key in model_keys:
        summary[model_key] = {}
        for group_name in ("held_out_category_query", "cold_start_query"):
            rows = []
            for run in runs:
                if model_key.startswith("maml_"):
                    shot = int(model_key.split("_")[1])
                    result = run["models"]["maml"]["shots"][str(shot)]
                else:
                    result = run["models"][model_key]
                rows.append(result["metrics"][group_name])
            summary[model_key][group_name] = _summarize_metric_rows(rows)
    return summary


def run_repeated_comparison(
    merchants_path: str = "data/synthetic/merchants.csv",
    relationships_path: str = "data/synthetic/relationships.csv",
    rings_path: str = "data/synthetic/rings_metadata.json",
    output_path: str = "data/processed/repeated_model_comparison.json",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    support_shots: Sequence[int] = (5, 10, 20),
    held_out_category: str = "Travel",
    maml_epochs: int = 20
) -> Dict[str, Any]:
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for the measured MAML comparison; no simulated results are produced")
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Exactly five distinct random seeds are required")
    if not support_shots or any(shot < 2 for shot in support_shots):
        raise ValueError("Support shot counts must be at least 2")

    merchants = pd.read_csv(merchants_path)
    relationships = pd.read_csv(relationships_path)
    with open(rings_path, "r", encoding="utf-8") as rings_file:
        rings_metadata = json.load(rings_file)

    runs = []
    for seed in seeds:
        train_base, test_base, split_info = ring_grouped_split(merchants, rings_metadata, seed)
        train_ids = set(train_base["merchant_id"])
        test_ids = set(test_base["merchant_id"])

        # Rebuild label-dependent graph priors for this split, not the legacy fixed split.
        graph_df = compute_temporal_graph_features(
            merchants,
            relationships,
            train_merchant_ids=train_ids
        )
        train_df = graph_df.loc[graph_df["merchant_id"].isin(train_ids)].copy()
        test_df = graph_df.loc[graph_df["merchant_id"].isin(test_ids)].copy()
        query_df = test_df.loc[test_df["business_category"] == held_out_category].copy()
        if query_df.empty or query_df["fraud_label"].nunique() < 2:
            raise ValueError(f"Seed {seed} has no two-class query for category {held_out_category}")
        training_category = train_df.loc[train_df["business_category"] == held_out_category]
        if training_category["fraud_label"].nunique() < 2:
            raise ValueError(f"Seed {seed} has no two-class support pool for {held_out_category}")

        split_info.update({
            "held_out_category": held_out_category,
            "query_rows": len(query_df),
            "query_positive_count": int(query_df["fraud_label"].sum())
        })
        tabular_result = _fit_xgboost(train_df, query_df, TABULAR_FEATURES, seed, graph_model=False)
        graph_result = _fit_xgboost(train_df, query_df, TABULAR_FEATURES + GRAPH_FEATURES, seed, graph_model=True)

        scaler = StandardScaler().fit(train_df[MAML_FEATURES])
        meta_model = _train_fomaml(train_df, held_out_category, scaler, seed, maml_epochs)
        maml_shots = {
            str(shot): _evaluate_maml_shot(meta_model, train_df, query_df, scaler, shot, seed)
            for shot in support_shots
        }

        runs.append({
            "seed": int(seed),
            "split": split_info,
            "models": {
                "xgboost_tabular": tabular_result,
                "xgboost_graph": graph_result,
                "maml": {"shots": maml_shots}
            }
        })
        logger.info("Completed seed %s: train=%s test=%s query=%s", seed, split_info["train_rows"], split_info["test_rows"], len(query_df))

    result = {
        "protocol": {
            "seeds": [int(seed) for seed in seeds],
            "split": "75/25 GroupShuffleSplit; named fraud-ring members share a group; ungrouped merchants are individual groups",
            "graph_features": "Recomputed separately per seed using only that split's training labels and temporal prior edges",
            "evaluation": f"All models use the same {held_out_category} test query per seed; cold_start_query is the zero-transaction subset",
            "xgboost_threshold": 0.5,
            "maml": "First-order MAML; held-out-category support comes from training only; threshold maximizes support-set F1; support rows are excluded from test query",
            "maml_epochs": maml_epochs
        },
        "runs": runs,
        "summary": summarize_runs(runs, support_shots),
        "feature_importance_mean_std": _summarize_feature_importance(runs)
    }
    output_file = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as report_file:
        json.dump(result, report_file, indent=2, allow_nan=False)
    result["output_path"] = output_file
    return result


def _print_summary(result: Dict[str, Any]) -> None:
    print(f"Five-seed results on {result['protocol']['evaluation']}")
    print(f"{'Model':<22} {'Group':<24} {'PR-AUC mean +/- SD':<22} {'Recall mean +/- SD':<22} {'F1 mean +/- SD':<22}")
    for model_key, groups in result["summary"].items():
        for group_name, metrics in groups.items():
            pr_auc = metrics["pr_auc"]
            recall = metrics["recall"]
            f1 = metrics["f1"]
            pr_text = "n/a" if pr_auc["mean"] is None else f"{pr_auc['mean']:.4f} +/- {pr_auc['std']:.4f}"
            recall_text = f"{recall['mean']:.4f} +/- {recall['std']:.4f}"
            f1_text = f"{f1['mean']:.4f} +/- {f1['std']:.4f}"
            print(f"{model_key:<22} {group_name:<24} {pr_text:<22} {recall_text:<22} {f1_text:<22}")
    print(f"Feature importance (top mean values) and per-seed metrics: {result['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-seeds", nargs=10, type=int, default=list(MAIN_SEEDS))
    parser.add_argument("--maml-seeds", nargs=5, type=int, default=list(MAML_SEEDS))
    parser.add_argument("--support-shots", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--maml-categories", nargs="+", default=list(MAML_CATEGORIES))
    parser.add_argument("--maml-epochs", type=int, default=20)
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    run_full_study(
        output_root=args.output_root,
        main_seeds=args.main_seeds,
        maml_seeds=args.maml_seeds,
        maml_categories=args.maml_categories,
        support_shots=args.support_shots,
        maml_epochs=args.maml_epochs
    )


if __name__ == "__main__":
    main()