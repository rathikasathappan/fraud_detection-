import numpy as np
import pandas as pd

from src.repeated_comparison import (
    _build_paired_comparisons,
    choose_support_threshold,
    compute_ring_link_check,
    ring_grouped_split,
    ring_grouped_train_validation_test,
)


def test_ring_grouped_split_keeps_ring_members_together():
    merchants = pd.DataFrame({
        "merchant_id": [f"M_{index}" for index in range(12)],
        "fraud_label": [0, 1] * 6
    })
    rings_metadata = {
        "ring-a": ["M_0", "M_1", "M_2"],
        "ring-b": ["M_3", "M_4", "M_5"]
    }

    train_df, test_df, split_info = ring_grouped_split(merchants, rings_metadata, seed=7)

    train_ids = set(train_df["merchant_id"])
    test_ids = set(test_df["merchant_id"])
    for ring_members in rings_metadata.values():
        assert not (set(ring_members) & train_ids and set(ring_members) & test_ids)
    assert split_info["group_overlap"] == 0


def test_support_threshold_maximizes_support_f1():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.8])

    threshold, support_f1 = choose_support_threshold(labels, probabilities)

    assert threshold == 0.45
    assert support_f1 == 1.0


def test_three_way_ring_grouped_split_has_no_overlap():
    merchants = pd.DataFrame({
        "merchant_id": [f"M_{index}" for index in range(40)],
        "fraud_label": [index % 2 for index in range(40)]
    })
    rings_metadata = {"ring-a": ["M_0", "M_1", "M_2"], "ring-b": ["M_3", "M_4", "M_5"]}

    train_df, validation_df, test_df, split_info = ring_grouped_train_validation_test(
        merchants,
        rings_metadata,
        seed=17
    )

    partitions = [set(frame["merchant_id"]) for frame in (train_df, validation_df, test_df)]
    assert sum(map(len, partitions)) == len(merchants)
    assert all(not partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3))
    for ring_members in rings_metadata.values():
        assert sum(bool(partition & set(ring_members)) for partition in partitions) == 1
    assert split_info["group_overlap"] == 0


def test_ring_link_check_counts_training_entity_and_fraud_links():
    train_df = pd.DataFrame([
        {"merchant_id": "TRAIN_FRAUD", "fraud_label": 1},
        {"merchant_id": "TRAIN_GOOD", "fraud_label": 0}
    ])
    test_df = pd.DataFrame([
        {"merchant_id": "TEST_RING_1", "fraud_label": 1},
        {"merchant_id": "TEST_RING_2", "fraud_label": 1},
        {"merchant_id": "TEST_OTHER", "fraud_label": 0}
    ])
    relationships = pd.DataFrame([
        {"merchant_id": "TRAIN_FRAUD", "entity_id": "BANK_SHARED"},
        {"merchant_id": "TRAIN_GOOD", "entity_id": "IP_SHARED"},
        {"merchant_id": "TEST_RING_1", "entity_id": "BANK_SHARED"},
        {"merchant_id": "TEST_RING_2", "entity_id": "IP_SHARED"},
        {"merchant_id": "TEST_OTHER", "entity_id": "BANK_SHARED"}
    ])

    link_check = compute_ring_link_check(
        train_df,
        test_df,
        relationships,
        {"ring-x": ["TEST_RING_1", "TEST_RING_2"]}
    )

    assert link_check["test_ring_member_count"] == 2
    assert link_check["linked_to_training_count"] == 2
    assert link_check["linked_to_training_fraud_count"] == 1
    assert link_check["linked_to_training_share"] == 1.0
    assert link_check["linked_to_training_fraud_share"] == 0.5


def test_paired_comparison_includes_graph_against_tabular_without_age():
    metrics_by_seed = {
        42: {
            "xgboost_graph": {
                "all_test": {"pr_auc": 0.7},
                "cold_start": {"pr_auc": 0.8}
            },
            "xgboost_tabular": {
                "all_test": {"pr_auc": 0.6},
                "cold_start": {"pr_auc": 0.7}
            },
            "xgboost_tabular_no_business_age": {
                "all_test": {"pr_auc": 0.5},
                "cold_start": {"pr_auc": 0.4}
            }
        }
    }

    paired = _build_paired_comparisons(metrics_by_seed)
    no_age_comparison = paired.loc[
        paired["comparison"] == "graph_minus_tabular_no_business_age"
    ]
    summary = no_age_comparison.loc[no_age_comparison["seed"] == "SUMMARY"].iloc[0]

    assert np.isclose(summary["pr_auc_difference"], 0.2)
    assert summary["model_a_wins"] == 1