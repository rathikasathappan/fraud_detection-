import pandas as pd

from src.generate_data import add_temporal_ring_bridges, generate_synthetic_dataset


def test_temporal_ring_bridge_links_later_member_to_earlier_labelled_fraud():
    merchants = pd.DataFrame([
        {
            "merchant_id": "M_RING_A_FRAUD",
            "business_category": "Travel",
            "registration_date": "2023-01-01 00:00:00",
            "bank_account_id": "BANK_SOURCE",
            "device_id": "DEVICE_SOURCE",
            "ip_address": "192.0.2.1",
            "fraud_label": 1
        },
        {
            "merchant_id": "M_RING_B_FIRST",
            "business_category": "Travel",
            "registration_date": "2023-02-01 00:00:00",
            "bank_account_id": "BANK_FIRST",
            "device_id": "DEVICE_FIRST",
            "ip_address": "192.0.2.2",
            "fraud_label": 0
        },
        {
            "merchant_id": "M_RING_B_LATER",
            "business_category": "Travel",
            "registration_date": "2023-02-02 00:00:00",
            "bank_account_id": "BANK_LATER",
            "device_id": "DEVICE_LATER",
            "ip_address": "192.0.2.3",
            "fraud_label": 1
        }
    ])
    relationships = pd.DataFrame([
        {"merchant_id": row.merchant_id, "entity_id": row.bank_account_id, "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT", "timestamp": row.registration_date}
        for row in merchants.itertuples()
    ] + [
        {"merchant_id": row.merchant_id, "entity_id": row.device_id, "entity_type": "device", "relationship": "USES_DEVICE", "timestamp": row.registration_date}
        for row in merchants.itertuples()
    ] + [
        {"merchant_id": row.merchant_id, "entity_id": row.ip_address, "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP", "timestamp": row.registration_date}
        for row in merchants.itertuples()
    ])
    rings = {
        "RING_A": ["M_RING_A_FRAUD"],
        "RING_B": ["M_RING_B_FIRST", "M_RING_B_LATER"]
    }

    bridges = add_temporal_ring_bridges(merchants, relationships, rings)

    later_bank = merchants.loc[merchants["merchant_id"] == "M_RING_B_LATER", "bank_account_id"].iloc[0]
    later_edge = relationships.loc[
        (relationships["merchant_id"] == "M_RING_B_LATER")
        & (relationships["entity_type"] == "bank_account"),
        "entity_id"
    ].iloc[0]
    assert bridges >= 1
    assert later_bank == "BANK_SOURCE"
    assert later_edge == "BANK_SOURCE"


def test_v2_generator_has_overlapping_ages_valid_rings_and_category_patterns(tmp_path):
    merchants, relationships, summary = generate_synthetic_dataset(output_dir=str(tmp_path))
    rings = pd.read_json(tmp_path / "rings_metadata.json", typ="series").to_dict()
    ring_members = {merchant_id for members in rings.values() for merchant_id in members}
    age_summary = summary["age_by_label"]
    category_cold_rates = summary["category_cold_start_rates"]

    assert len(merchants) == 5000
    assert len(relationships) == 15000
    assert ring_members == set(merchants.loc[merchants["fraud_label"].isin([0, 1]) & merchants["merchant_id"].isin(ring_members), "merchant_id"])
    assert age_summary[0]["q25"] < age_summary[1]["q75"]
    assert summary["cross_ring_bridge_count"] > 0
    assert max(category_cold_rates.values()) - min(category_cold_rates.values()) > 0.2
