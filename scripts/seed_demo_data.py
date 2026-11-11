"""
seed_demo_data.py — Seed realistic synthetic demo data for FraudGuard.

Generates 500+ synthetic merchants spanning the last 90 days across all 5
business categories. Includes genuine merchants, bank-account rings, device
rings, multi-entity rings, and mimic solo fraudsters.

Idempotent: skips existing merchant IDs. Use --reset to clear first.
Test rows (data_source='test') excluded from all statistics.

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --reset
"""
import sys
import os
import argparse
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so src.* imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from src.db import (
    get_db_connection,
    insert_merchant,
    insert_relationships_bulk,
    insert_risk_assessment,
    insert_alert,
    insert_audit_log,
    get_merchant,
    lookup_entity_network,
)
from src.risk_engine import RiskEngine

# ---------------------------------------------------------------------------
# Constants — NEVER use real bank numbers or real IPs
# ---------------------------------------------------------------------------
# RFC 5737 documentation IP ranges only
_DOC_IPS = (
    [f"192.0.2.{i}" for i in range(10, 200)]
    + [f"198.51.100.{i}" for i in range(10, 200)]
    + [f"203.0.113.{i}" for i in range(10, 200)]
)

CATEGORIES = ["Electronics", "Digital Services", "Fashion", "Grocery", "Travel"]

# Shared entities for the fraud rings
RING_BANK_ID   = "BA_000028"      # shared bank for bank ring (5 merchants)
RING_DEVICE_ID = "DEV_000003"     # shared device for device ring (4 merchants)

# ---------------------------------------------------------------------------
# Preset anchor merchants (must exist with predictable IDs)
# ---------------------------------------------------------------------------
PRESET_ANCHORS = [
    # (merchant_id, bank_id, device_id, ip, category, is_fraud)
    ("M_00001",  "BA_UNIQUE_001", "DEV_UNIQUE_001", "198.51.100.1",   "Grocery",         False),
    ("M_00013",  RING_BANK_ID,    "DEV_UNIQUE_013", "192.0.2.13",     "Travel",          True),
    ("M_00022",  "BA_UNIQUE_022", RING_DEVICE_ID,   "198.51.100.22",  "Electronics",     True),
    ("M_00029",  "BA_UNIQUE_029", "DEV_UNIQUE_029", "192.0.2.29",     "Fashion",         True),
    ("M_00119",  RING_BANK_ID,    RING_DEVICE_ID,   "203.0.113.119",  "Digital Services",True),
]

# Bank-ring peers — share RING_BANK_ID with M_00013
BANK_RING_PEERS = [
    ("M_00014", RING_BANK_ID, "DEV_RING_014", "192.0.2.14", "Travel",      True),
    ("M_00015", RING_BANK_ID, "DEV_RING_015", "192.0.2.15", "Travel",      True),
    ("M_00016", RING_BANK_ID, "DEV_RING_016", "192.0.2.16", "Electronics", True),
    ("M_00017", RING_BANK_ID, "DEV_RING_017", "192.0.2.17", "Fashion",     True),
]

# Device-ring peers — share RING_DEVICE_ID with M_00022
DEVICE_RING_PEERS = [
    ("M_00023", "BA_RING_023", RING_DEVICE_ID, "198.51.100.23", "Electronics",     True),
    ("M_00024", "BA_RING_024", RING_DEVICE_ID, "198.51.100.24", "Digital Services",True),
    ("M_00025", "BA_RING_025", RING_DEVICE_ID, "198.51.100.25", "Grocery",         True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_bank():
    return f"BA_{random.randint(100000, 999999)}"


def _rand_device():
    return f"DEV_{uuid.uuid4().hex[:8].upper()}"


def _rand_date(days_back_max: int = 90) -> str:
    delta = timedelta(
        days=random.randint(0, days_back_max),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return (datetime.now() - delta).strftime("%Y-%m-%d %H:%M:%S")


def _seed_one(
    engine: RiskEngine,
    merchant_id: str,
    bank_id: str,
    device_id: str,
    ip: str,
    category: str,
    reg_date: str,
    business_age_days=None,
    phone_v=None,
    email_v=None,
    doc_v=None,
    txn_count: int = 0,
    fraud_label=None,
    data_source: str = "synthetic",
) -> bool:
    """Insert one merchant + assessment + relationships. Skip if already in DB."""
    if get_merchant(merchant_id):
        print(f"  SKIP  {merchant_id} (already exists)")
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    merchant_record = {
        "merchant_id":        merchant_id,
        "business_category":  category,
        "business_age_days":  business_age_days,
        "registration_date":  reg_date,
        "phone_verified":     phone_v,
        "email_verified":     email_v,
        "document_verified":  doc_v,
        "bank_account_id":    bank_id,
        "device_id":          device_id,
        "ip_address":         ip,
        "transaction_count":  txn_count,
        "fraud_label":        fraud_label,
        "data_source":        data_source,
    }

    # Graph signals — look up peers already in DB before inserting
    graph_evidence = lookup_entity_network(
        bank_id=bank_id,
        device_id=device_id,
        ip_address=ip,
        exclude_merchant_id=merchant_id,
        limit=20,
    )
    bsc = graph_evidence.get("bank_share_count", 0)
    dsc = graph_evidence.get("device_share_count", 0)
    fn  = graph_evidence.get("neighbours_flagged_fraud", 0)

    merchant_record.update({
        "bank_share_count":              bsc,
        "device_share_count":            dsc,
        "ip_share_count":                graph_evidence.get("ip_share_count", 0),
        "connected_component_size":      max(1, bsc + dsc),
        "neighbours_flagged_fraud":      fn,
        "louvain_community_fraud_rate":  0.85 if fn > 0 else (0.15 if bsc > 1 else 0.0),
    })

    assessment = engine.score_merchant(merchant_record, graph_evidence)
    assessment["assessment_id"] = f"ASM_{uuid.uuid4().hex[:8].upper()}"
    assessment["evaluated_at"]  = now_str

    # Persist
    insert_merchant(merchant_record)
    insert_relationships_bulk([
        {"merchant_id": merchant_id, "entity_id": bank_id,   "entity_type": "bank_account",
         "relationship": "HAS_BANK_ACCOUNT",   "timestamp": now_str},
        {"merchant_id": merchant_id, "entity_id": device_id, "entity_type": "device",
         "relationship": "USES_DEVICE",        "timestamp": now_str},
        {"merchant_id": merchant_id, "entity_id": ip,        "entity_type": "ip_address",
         "relationship": "OPERATES_FROM_IP",   "timestamp": now_str},
    ])
    insert_risk_assessment(assessment)

    if assessment["decision"] in ("Hold", "Review"):
        severity = "CRITICAL" if assessment["decision"] == "Hold" else "HIGH"
        insert_alert({
            "alert_id":   f"ALT_{uuid.uuid4().hex[:8].upper()}",
            "merchant_id": merchant_id,
            "severity":    severity,
            "alert_type":  "COLD_START_FRAUD_DETECTED",
            "details":     "; ".join(assessment.get("reasons", [])[:3]),
            "status":      "OPEN",
            "created_at":  now_str,
        })

    print(
        f"  SEED  {merchant_id}: {assessment['decision']:7s}  "
        f"score={assessment['risk_score']:.3f}  "
        f"bank_peers={bsc}  fraud_peers={fn}"
    )
    return True


def _reset_seed_merchants():
    """Delete M_000xx, M_DEMO_ and M_RING_ merchants (synthetic seed only)."""
    print("Resetting seed merchants …")
    conn = get_db_connection()
    cur  = conn.cursor()
    for pattern in ("M_000%%", "M_DEMO_%%", "M_RING_%%"):
        cur.execute("SELECT merchant_id FROM merchants WHERE merchant_id LIKE %s", (pattern,))
        ids = [r[0] for r in cur.fetchall()]
        if not ids:
            continue
        fmt = ",".join(["%s"] * len(ids))
        cur.execute(f"DELETE FROM audit_log WHERE merchant_id IN ({fmt})", ids)
        cur.execute(f"DELETE FROM investigation_cases WHERE merchant_id IN ({fmt})", ids)
        cur.execute(f"DELETE FROM alerts WHERE merchant_id IN ({fmt})", ids)
        cur.execute(f"DELETE FROM risk_assessments WHERE merchant_id IN ({fmt})", ids)
        cur.execute(f"DELETE FROM relationships WHERE merchant_id IN ({fmt})", ids)
        cur.execute(f"DELETE FROM merchants WHERE merchant_id IN ({fmt})", ids)
        conn.commit()
        print(f"  Deleted {len(ids)} merchants matching {pattern}")
    cur.close()
    conn.close()
    print("Reset complete.\n")


# ---------------------------------------------------------------------------
# Main seed function
# ---------------------------------------------------------------------------

def seed(reset: bool = False) -> int:
    if reset:
        _reset_seed_merchants()

    engine  = RiskEngine()
    seeded  = 0

    # --- 1. Preset anchor merchants (required for Fraud Network presets) ----
    print("\n=== Preset anchor merchants ===")
    for mid, bank, dev, ip, cat, is_fraud in PRESET_ANCHORS:
        age = random.randint(30, 500)
        ok  = _seed_one(
            engine, mid, bank, dev, ip, cat, _rand_date(80),
            business_age_days=age,
            phone_v=1, email_v=1, doc_v=1,
            fraud_label=1 if is_fraud else 0,
        )
        seeded += int(ok)

    # --- 2. Bank-ring peers ------------------------------------------------
    print("\n=== Bank-ring peers (share bank with M_00013) ===")
    for mid, bank, dev, ip, cat, is_fraud in BANK_RING_PEERS:
        ok = _seed_one(
            engine, mid, bank, dev, ip, cat, _rand_date(80),
            business_age_days=random.randint(5, 60),
            phone_v=0, email_v=0, doc_v=0,
            fraud_label=1 if is_fraud else 0,
        )
        seeded += int(ok)

    # --- 3. Device-ring peers ----------------------------------------------
    print("\n=== Device-ring peers (share device with M_00022) ===")
    for mid, bank, dev, ip, cat, is_fraud in DEVICE_RING_PEERS:
        ok = _seed_one(
            engine, mid, bank, dev, ip, cat, _rand_date(80),
            business_age_days=random.randint(5, 60),
            phone_v=0, email_v=None, doc_v=0,
            fraud_label=1 if is_fraud else 0,
        )
        seeded += int(ok)

    # --- 4. Mimic solo extra peer (links to M_00029 via IP) ----------------
    print("\n=== Mimic solo peer ===")
    ok = _seed_one(
        engine, "M_00030", "BA_MIMIC_030", "DEV_MIMIC_030",
        "192.0.2.29",   # same IP as M_00029 → IP peer
        "Fashion", _rand_date(70),
        business_age_days=10, phone_v=1, email_v=0, doc_v=0,
        fraud_label=1,
    )
    seeded += int(ok)

    # --- 5. 500 random synthetic merchants ---------------------------------
    print("\n=== 500 random synthetic merchants ===")
    # pools of shared entities to simulate realistic sharing
    shared_banks   = [_rand_bank()   for _ in range(40)]
    shared_devices = [_rand_device() for _ in range(25)]

    for i in range(500):
        mid = f"M_DEMO_{i + 1:04d}"
        if get_merchant(mid):
            continue
        cat = random.choice(CATEGORIES)
        reg = _rand_date(90)
        roll = random.random()
        # 15 % share a bank, 10 % share a device, 75 % fully isolated
        if roll < 0.15:
            bank = random.choice(shared_banks)
            dev  = _rand_device()
        elif roll < 0.25:
            bank = _rand_bank()
            dev  = random.choice(shared_devices)
        else:
            bank = _rand_bank()
            dev  = _rand_device()
        ip       = random.choice(_DOC_IPS)
        age      = random.randint(0, 1200) if random.random() > 0.15 else None
        phone_v  = random.choice([0, 1, None])
        email_v  = random.choice([0, 1, None])
        doc_v    = random.choice([0, 1, None])
        is_fraud = roll < 0.12       # ~12 % fraud rate among random batch
        ok = _seed_one(
            engine, mid, bank, dev, ip, cat, reg,
            business_age_days=age,
            phone_v=phone_v, email_v=email_v, doc_v=doc_v,
            fraud_label=1 if is_fraud else 0,
        )
        seeded += int(ok)

    print(f"\n✓ Seeded {seeded} new merchants total.")
    return seeded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed FraudGuard demo data")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing M_000xx / M_DEMO_ merchants before seeding",
    )
    args = parser.parse_args()
    seed(reset=args.reset)

