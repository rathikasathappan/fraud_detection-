"""
Synthetic Merchant Onboarding Data Generator
Phase 1: Generates synthetic merchants and their entity relationships
simulating genuine onboarding, fraud rings, shared infrastructure noise,
and label noise.
"""

import os
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Set fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES = ["Electronics", "Fashion", "Digital Services", "Grocery", "Travel"]
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 6, 30)
CATEGORY_FRAUD_COLD_START = {
    "Electronics": 0.76,
    "Fashion": 0.88,
    "Digital Services": 0.58,
    "Grocery": 0.72,
    "Travel": 0.38
}
CATEGORY_GENUINE_COLD_START = {
    "Electronics": 0.44,
    "Fashion": 0.52,
    "Digital Services": 0.30,
    "Grocery": 0.40,
    "Travel": 0.22
}
CATEGORY_FRAUD_HISTORY_P = {
    "Electronics": 0.34,
    "Fashion": 0.24,
    "Digital Services": 0.15,
    "Grocery": 0.30,
    "Travel": 0.52
}
CATEGORY_GENUINE_HISTORY = {
    "Electronics": (4, 0.08),
    "Fashion": (3, 0.10),
    "Digital Services": (5, 0.06),
    "Grocery": (6, 0.09),
    "Travel": (3, 0.04)
}
CATEGORY_RING_BRIDGE_PROBABILITY = {
    "Electronics": 0.42,
    "Fashion": 0.30,
    "Digital Services": 0.60,
    "Grocery": 0.24,
    "Travel": 0.50
}
CATEGORY_RING_BRIDGE_ENTITY = {
    "Electronics": "ip_address",
    "Fashion": "device_id",
    "Digital Services": "device_id",
    "Grocery": "ip_address",
    "Travel": "bank_account_id"
}

def random_date(start: datetime, end: datetime) -> datetime:
    """Generate a random datetime between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86400)
    return start + timedelta(days=random_days, seconds=random_seconds)


def add_temporal_ring_bridges(
    merchants_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    rings_metadata: Dict[str, List[str]]
) -> int:
    """Link later ring cohorts to earlier labelled-fraud merchants across rings."""
    ring_start = {}
    for ring_id, merchant_ids in rings_metadata.items():
        ring_rows = merchants_df.loc[merchants_df["merchant_id"].isin(merchant_ids)]
        if not ring_rows.empty:
            ring_start[ring_id] = pd.to_datetime(ring_rows["registration_date"]).min()

    ordered_ring_ids = sorted(ring_start, key=ring_start.get)
    earlier_fraud_members = []
    bridge_count = 0
    field_relationship = {
        "bank_account_id": ("bank_account", "HAS_BANK_ACCOUNT"),
        "device_id": ("device", "USES_DEVICE"),
        "ip_address": ("ip_address", "OPERATES_FROM_IP")
    }

    for ring_id in ordered_ring_ids:
        ring_rows = merchants_df.loc[
            merchants_df["merchant_id"].isin(rings_metadata[ring_id])
        ].copy()
        ring_rows["_sort_date"] = pd.to_datetime(ring_rows["registration_date"])
        ring_rows = ring_rows.sort_values("_sort_date")
        for member_index, (_, member) in enumerate(ring_rows.iterrows()):
            eligible_sources = [
                source for source in earlier_fraud_members
                if source["_sort_date"] < member["_sort_date"]
            ]
            if not eligible_sources:
                continue
            category = member["business_category"]
            probability = CATEGORY_RING_BRIDGE_PROBABILITY.get(category, 0.35)
            if member_index != 1 and random.random() >= probability:
                continue

            source = random.choice(eligible_sources)
            entity_field = CATEGORY_RING_BRIDGE_ENTITY.get(category, "device_id")
            entity_type, relationship = field_relationship[entity_field]
            source_entity = source[entity_field]
            if member[entity_field] == source_entity:
                continue
            merchants_df.loc[
                merchants_df["merchant_id"] == member["merchant_id"], entity_field
            ] = source_entity
            edge_mask = (
                relationships_df["merchant_id"].eq(member["merchant_id"])
                & relationships_df["entity_type"].eq(entity_type)
            )
            relationships_df.loc[edge_mask, "entity_id"] = source_entity
            relationships_df.loc[edge_mask, "relationship"] = relationship
            bridge_count += 1

        earlier_fraud_members.extend(
            member.to_dict()
            for _, member in ring_rows.loc[ring_rows["fraud_label"] == 1].iterrows()
        )
    return bridge_count

def generate_synthetic_dataset(
    total_merchants: int = 5000,
    target_fraud_rate: float = 0.05,
    output_dir: str = "data/synthetic"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generates realistic merchant onboarding dataset with graph linkages and noise.
    
    Scenarios:
    1. Genuine merchants (~95%):
       - Mostly distinct bank accounts, devices, residential IPs
       - High verification rates (80-95%)
       - Noise: Shared office co-working IPs, shared family devices/banks
    2. Fraud rings (~4-5%):
       - Shared bank accounts across multiple merchant IDs
       - Shared device IDs (device farm / emulators)
       - Low to medium verification rates
       - Lower business age and high cold-start proportion
    3. Mimicking fraudsters (hard fraud):
       - Distinct bank/device, clean surface tabular profile
    4. Realistic noise & label noise:
       - 2% label noise (undetected fraud and false dispute labels)
       - Legitimate multi-tenant IP sharing (e.g. WeWork/tech hub)
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    target_fraud_count = int(total_merchants * target_fraud_rate)
    target_genuine_count = total_merchants - target_fraud_count
    
    # Pre-generate shared infrastructure pools for noise
    shared_office_ips = [f"198.51.100.{i}" for i in range(10, 25)]  # Co-working / office hubs
    shared_family_devices = [f"DEV_FAM_{i:04d}" for i in range(1, 30)] # Family tablets / shared laptops
    shared_family_banks = [f"BANK_FAM_{i:05d}" for i in range(1, 20)]  # Family / partnership accounts
    category_office_ip_share = {
        "Electronics": 0.06,
        "Fashion": 0.04,
        "Digital Services": 0.12,
        "Grocery": 0.16,
        "Travel": 0.05
    }
    category_family_device_share = {
        "Electronics": 0.03,
        "Fashion": 0.06,
        "Digital Services": 0.10,
        "Grocery": 0.05,
        "Travel": 0.02
    }
    
    # Tracking data structures
    merchants: List[Dict[str, Any]] = []
    rings_metadata: Dict[str, List[str]] = {}
    
    # -------------------------------------------------------------
    # 1. GENERATE FRAUD MERCHANTS (~220 base fraud profiles)
    # -------------------------------------------------------------
    # With ~1% dispute noise on genuine and ~8% undetected fraud,
    # final labeled fraud count lands right around 250 (5.00%).
    num_bank_rings = 14
    bank_ring_merchants_count = 70
    bank_ring_pool = [f"BANK_RING_{r:03d}" for r in range(num_bank_rings)]
    
    num_dev_rings = 14
    dev_ring_merchants_count = 70
    dev_ring_pool = [f"DEV_RING_{r:03d}" for r in range(num_dev_rings)]
    
    num_complex_rings = 9
    complex_ring_count = 45
    complex_bank_pool = [f"BANK_CPLX_{r:03d}" for r in range(num_complex_rings)]
    complex_dev_pool = [f"DEV_CPLX_{r:03d}" for r in range(num_complex_rings)]
    complex_ip_pool = [f"203.0.113.{r+40}" for r in range(num_complex_rings)]
    
    mimic_fraud_count = 35
        
    fraud_specs = []
    
    # Populate Scenario A: Bank Sharing Rings
    for i in range(bank_ring_merchants_count):
        ring_idx = i % num_bank_rings
        ring_id = f"RING_BANK_{ring_idx}"
        bank = bank_ring_pool[ring_idx]
        dev = f"DEV_FR_{i:05d}"
        ip = f"198.18.{random.randint(1, 50)}.{random.randint(1, 254)}"
        fraud_specs.append({
            "scenario": "fraud_ring_bank",
            "ring_id": ring_id,
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 1
        })

    # Populate Scenario B: Device Sharing Rings
    for i in range(dev_ring_merchants_count):
        ring_idx = i % num_dev_rings
        ring_id = f"RING_DEV_{ring_idx}"
        bank = f"BANK_FR_{i:05d}"
        dev = dev_ring_pool[ring_idx]
        ip = f"198.19.{random.randint(1, 50)}.{random.randint(1, 254)}"
        fraud_specs.append({
            "scenario": "fraud_ring_device",
            "ring_id": ring_id,
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 1
        })

    # Populate Scenario C: Complex Rings (Bank + Device + IP cluster)
    for i in range(complex_ring_count):
        ring_idx = i % num_complex_rings
        ring_id = f"RING_CPLX_{ring_idx}"
        bank = complex_bank_pool[ring_idx]
        dev = complex_dev_pool[ring_idx]
        ip = complex_ip_pool[ring_idx]
        fraud_specs.append({
            "scenario": "fraud_ring_complex",
            "ring_id": ring_id,
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 1
        })

    # Populate Scenario D: Hard Mimicking Fraudsters
    for i in range(mimic_fraud_count):
        ring_id = f"MIMIC_SOLO_{i:03d}"
        bank = f"BANK_MIMIC_{i:05d}"
        dev = f"DEV_MIMIC_{i:05d}"
        ip = f"100.64.{random.randint(1, 100)}.{random.randint(1, 254)}"
        fraud_specs.append({
            "scenario": "fraud_mimic",
            "ring_id": ring_id,
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 1
        })

    # -------------------------------------------------------------
    # 2. GENERATE GENUINE MERCHANTS (~4750 merchants)
    # -------------------------------------------------------------
    genuine_specs = []
    
    # 2a. Genuine merchants with legitimate shared office IP (noise ~400 merchants)
    office_sharing_count = 450
    for i in range(office_sharing_count):
        bank = f"BANK_GEN_OFF_{i:05d}"
        dev = f"DEV_GEN_OFF_{i:05d}"
        ip = random.choice(shared_office_ips) # Shared co-working IP
        genuine_specs.append({
            "scenario": "genuine_office_ip_noise",
            "ring_id": "NONE",
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 0
        })

    # 2b. Genuine merchants sharing family devices / family accounts (noise ~100 merchants)
    family_sharing_count = 120
    for i in range(family_sharing_count):
        fam_idx = i % len(shared_family_devices)
        dev = shared_family_devices[fam_idx]
        bank = shared_family_banks[fam_idx % len(shared_family_banks)] if random.random() < 0.3 else f"BANK_FAM_IND_{i:05d}"
        ip = f"172.16.{fam_idx+1}.{random.randint(2, 50)}"
        genuine_specs.append({
            "scenario": "genuine_family_sharing_noise",
            "ring_id": "NONE",
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 0
        })

    # 2c. Standard independent genuine merchants
    standard_genuine_count = total_merchants - len(fraud_specs) - len(genuine_specs)
    for i in range(standard_genuine_count):
        bank = f"BANK_GEN_{i:06d}"
        dev = f"DEV_GEN_{i:06d}"
        ip = f"{random.randint(11, 190)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        genuine_specs.append({
            "scenario": "genuine_standard",
            "ring_id": "NONE",
            "bank_account_id": bank,
            "device_id": dev,
            "ip_address": ip,
            "initial_fraud": 0
        })

    # Combine all candidate profiles
    all_profiles = fraud_specs + genuine_specs
    random.shuffle(all_profiles)

    # -------------------------------------------------------------
    # 3. ASSIGN ATTRIBUTES & DATES
    # -------------------------------------------------------------
    # Generate dates across the 18-month window
    # For fraud rings, group their registration dates within closer temporal proximity
    ring_anchor_dates = {}
    for ring_id in rings_metadata.keys():
        ring_anchor_dates[ring_id] = random_date(START_DATE, END_DATE - timedelta(days=60))

    merchant_rows = []
    relationships_rows = []

    for idx, p in enumerate(all_profiles):
        m_id = f"M_{idx+1:05d}"
        scenario = p["scenario"]
        initial_label = p["initial_fraud"]
        ring_id = p["ring_id"]

        # Determine registration date
        if ring_id != "NONE" and ring_id in ring_anchor_dates:
            # Fraud ring registrations cluster within 1 to 20 days
            reg_date = ring_anchor_dates[ring_id] + timedelta(
                days=random.randint(0, 20),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
        else:
            reg_date = random_date(START_DATE, END_DATE)

        # Business category distribution
        # Fraud often concentrates slightly more in Digital Services & Electronics
        if initial_label == 1:
            cat_weights = [0.35, 0.15, 0.35, 0.05, 0.10]
        else:
            cat_weights = [0.20, 0.25, 0.20, 0.25, 0.10]
        business_category = random.choices(CATEGORIES, weights=cat_weights, k=1)[0]

        bank_account_id = p["bank_account_id"]
        device_id = p["device_id"]
        ip_address = p["ip_address"]
        if scenario == "genuine_standard":
            if random.random() < category_office_ip_share[business_category]:
                ip_address = random.choice(shared_office_ips)
            if random.random() < category_family_device_share[business_category]:
                device_id = random.choice(shared_family_devices)

        # Fraud and genuine account ages overlap; some fraud uses aged fronts and
        # some legitimate shops are newly established.
        if scenario in ["fraud_ring_bank", "fraud_ring_device", "fraud_ring_complex"]:
            if random.random() < 0.30:
                business_age_days = int(np.clip(np.random.gamma(shape=2.1, scale=420), 90, 3650))
            else:
                business_age_days = int(np.clip(np.random.gamma(shape=1.7, scale=115), 1, 1800))
        elif scenario == "fraud_mimic":
            if random.random() < 0.45:
                business_age_days = int(np.clip(np.random.gamma(shape=1.8, scale=105), 1, 1200))
            else:
                business_age_days = int(np.clip(np.random.gamma(shape=2.0, scale=430), 30, 2500))
        else:
            if random.random() < 0.28:
                business_age_days = int(np.clip(np.random.gamma(shape=1.8, scale=42), 1, 240))
            else:
                business_age_days = int(np.clip(np.random.gamma(shape=2.1, scale=340), 1, 3650))

        # Verification rates
        if scenario in ["fraud_ring_bank", "fraud_ring_device", "fraud_ring_complex"]:
            phone_verified = 1 if random.random() < 0.40 else 0
            email_verified = 1 if random.random() < 0.50 else 0
            document_verified = 1 if random.random() < 0.25 else 0
        elif scenario == "fraud_mimic":
            # Hard fraudster mimics verified profile
            phone_verified = 1 if random.random() < 0.90 else 0
            email_verified = 1 if random.random() < 0.95 else 0
            document_verified = 1 if random.random() < 0.85 else 0
        else:
            # Genuine profile verification rates
            phone_verified = 1 if random.random() < 0.88 else 0
            email_verified = 1 if random.random() < 0.92 else 0
            document_verified = 1 if random.random() < 0.84 else 0

        # Transaction history differs by category as well as label.
        if initial_label == 1:
            if random.random() < CATEGORY_FRAUD_COLD_START[business_category]:
                transaction_count = 0
            else:
                transaction_count = int(np.random.geometric(p=CATEGORY_FRAUD_HISTORY_P[business_category]))
        else:
            if random.random() < CATEGORY_GENUINE_COLD_START[business_category]:
                transaction_count = 0
            else:
                n_trials, probability = CATEGORY_GENUINE_HISTORY[business_category]
                transaction_count = int(np.random.negative_binomial(n=n_trials, p=probability))

        # -------------------------------------------------------------
        # 4. LABEL NOISE SIMULATION (~1.5% - 2% label noise)
        # -------------------------------------------------------------
        # Some genuine merchants get chargeback/friendly fraud disputes (~1%)
        # Some fraudsters escape detection or are unflagged (~8%)
        final_label = initial_label
        if initial_label == 0:
            if random.random() < 0.010:  # ~47 genuine merchants mistakenly flagged
                final_label = 1
        else:
            if random.random() < 0.080:  # ~18 fraud merchants escape initial detection
                final_label = 0

        reg_date_str = reg_date.strftime("%Y-%m-%d %H:%M:%S")

        merchant_row = {
            "merchant_id": m_id,
            "business_category": business_category,
            "business_age_days": business_age_days,
            "registration_date": reg_date_str,
            "phone_verified": phone_verified,
            "email_verified": email_verified,
            "document_verified": document_verified,
            "bank_account_id": bank_account_id,
            "device_id": device_id,
            "ip_address": ip_address,
            "transaction_count": transaction_count,
            "fraud_label": final_label
        }
        merchant_rows.append(merchant_row)
        if ring_id != "NONE":
            rings_metadata.setdefault(ring_id, []).append(m_id)

        # Relationship rows
        # Bank Account Link
        relationships_rows.append({
            "merchant_id": m_id,
            "entity_id": bank_account_id,
            "entity_type": "bank_account",
            "relationship": "HAS_BANK_ACCOUNT",
            "timestamp": reg_date_str
        })
        # Device Link
        relationships_rows.append({
            "merchant_id": m_id,
            "entity_id": device_id,
            "entity_type": "device",
            "relationship": "USES_DEVICE",
            "timestamp": reg_date_str
        })
        # IP Link
        relationships_rows.append({
            "merchant_id": m_id,
            "entity_id": ip_address,
            "entity_type": "ip_address",
            "relationship": "OPERATES_FROM_IP",
            "timestamp": reg_date_str
        })

    df_merchants = pd.DataFrame(merchant_rows)
    df_relationships = pd.DataFrame(relationships_rows)

    cross_ring_bridge_count = add_temporal_ring_bridges(
        df_merchants,
        df_relationships,
        rings_metadata
    )

    # Sort merchants chronologically by registration_date
    df_merchants["dt"] = pd.to_datetime(df_merchants["registration_date"])
    df_merchants = df_merchants.sort_values(by="dt").reset_index(drop=True)
    df_merchants = df_merchants.drop(columns=["dt"])

    # Match relationships sort order with merchant order
    id_order_map = {m_id: i for i, m_id in enumerate(df_merchants["merchant_id"])}
    df_relationships["sort_idx"] = df_relationships["merchant_id"].map(id_order_map)
    df_relationships = df_relationships.sort_values(by=["sort_idx", "entity_type"]).reset_index(drop=True)
    df_relationships = df_relationships.drop(columns=["sort_idx"])

    # Output file paths
    merchants_path = os.path.join(output_dir, "merchants.csv")
    relationships_path = os.path.join(output_dir, "relationships.csv")
    meta_path = os.path.join(output_dir, "rings_metadata.json")

    df_merchants.to_csv(merchants_path, index=False)
    df_relationships.to_csv(relationships_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(rings_metadata, f, indent=2)

    actual_fraud_rate = df_merchants["fraud_label"].mean()
    age_by_label = df_merchants.groupby("fraud_label")["business_age_days"].agg(
        mean="mean",
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75)
    ).to_dict(orient="index")
    category_fraud_rates = df_merchants.groupby("business_category")["fraud_label"].mean().to_dict()
    category_cold_start_rates = df_merchants.assign(
        cold_start=df_merchants["transaction_count"].eq(0)
    ).groupby("business_category")["cold_start"].mean().to_dict()
    
    summary = {
        "total_merchants": len(df_merchants),
        "total_relationships": len(df_relationships),
        "fraud_count": int(df_merchants["fraud_label"].sum()),
        "fraud_rate": float(actual_fraud_rate),
        "cold_start_count": int((df_merchants["transaction_count"] == 0).sum()),
        "cold_start_percentage": float((df_merchants["transaction_count"] == 0).mean() * 100),
        "cross_ring_bridge_count": cross_ring_bridge_count,
        "ring_metadata_member_count": sum(len(members) for members in rings_metadata.values()),
        "age_by_label": age_by_label,
        "category_fraud_rates": category_fraud_rates,
        "category_cold_start_rates": category_cold_start_rates,
        "category_distribution": df_merchants["business_category"].value_counts().to_dict(),
        "missing_values": df_merchants.isnull().sum().to_dict(),
        "unique_banks": df_merchants["bank_account_id"].nunique(),
        "unique_devices": df_merchants["device_id"].nunique(),
        "unique_ips": df_merchants["ip_address"].nunique()
    }

    return df_merchants, df_relationships, summary

def run_eda(df_merchants: pd.DataFrame, df_relationships: pd.DataFrame, summary: Dict[str, Any]):
    """Prints comprehensive EDA summary as required by Phase 1."""
    print("=" * 70)
    print("PHASE 1 EDA: SYNTHETIC MERCHANT DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Merchants:        {summary['total_merchants']}")
    print(f"Total Relationships:    {summary['total_relationships']}")
    print(f"Fraud Count:            {summary['fraud_count']}")
    print(f"Fraud Rate:             {summary['fraud_rate']:.4f} ({summary['fraud_rate']*100:.2f}%)")
    print(f"Cold-Start Merchants:   {summary['cold_start_count']} ({summary['cold_start_percentage']:.1f}%)")
    print("-" * 70)
    print("Business Category Distribution:")
    for cat, count in summary['category_distribution'].items():
        pct = count / summary['total_merchants'] * 100
        print(f"  - {cat:18s}: {count:5d} ({pct:5.1f}%)")
    print("-" * 70)
    print("Entity Cardinality:")
    print(f"  - Unique Bank Accounts: {summary['unique_banks']}")
    print(f"  - Unique Device IDs:    {summary['unique_devices']}")
    print(f"  - Unique IP Addresses:  {summary['unique_ips']}")
    print("-" * 70)
    print("Missing Values Check:")
    has_missing = False
    for col, null_count in summary['missing_values'].items():
        if null_count > 0:
            print(f"  - {col}: {null_count} nulls")
            has_missing = True
    if not has_missing:
        print("  - All columns are complete (0 missing values).")
    print("=" * 70)

if __name__ == "__main__":
    df_m, df_r, summary = generate_synthetic_dataset()
    run_eda(df_m, df_r, summary)
