"""Backfill risk assessments for existing non-test merchants and sync training split flags."""

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.db import (
    get_db_connection,
    insert_alert,
    insert_risk_assessment,
    lookup_entity_network,
)
from src.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


def update_in_training_set_flags(train_csv_path: Optional[str] = None) -> int:
    """
    Populates in_training_set (1 = train, 0 = held-out test/other)
    from the canonical ring-grouped train/test split.
    """
    if train_csv_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        train_csv_path = str(base_dir / "data" / "processed" / "train_tabular.csv")

    if not os.path.exists(train_csv_path):
        logger.warning("Train split reference file not found at %s", train_csv_path)
        return 0

    import pandas as pd
    train_df = pd.read_csv(train_csv_path, usecols=["merchant_id"])
    train_ids = list(set(train_df["merchant_id"].dropna()))

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Reset all to 0
        cursor.execute("UPDATE merchants SET in_training_set = 0")
        
        # Batch update train IDs to 1
        chunk_size = 1000
        updated = 0
        for i in range(0, len(train_ids), chunk_size):
            chunk = train_ids[i:i + chunk_size]
            fmt = ",".join(["%s"] * len(chunk))
            cursor.execute(f"UPDATE merchants SET in_training_set = 1 WHERE merchant_id IN ({fmt})", tuple(chunk))
            updated += cursor.rowcount
        conn.commit()
        logger.info("Updated in_training_set = 1 for %d merchants.", updated)
        return updated
    finally:
        cursor.close()
        conn.close()


def _fetch_unassessed_merchants(batch_size: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT m.*
            FROM merchants m
            WHERE m.data_source <> 'test'
              AND NOT EXISTS (
                  SELECT 1 FROM risk_assessments ra
                  WHERE ra.merchant_id = m.merchant_id
              )
            ORDER BY m.registration_date, m.merchant_id
            LIMIT %s
            """,
            (batch_size,),
        )
        return cursor.fetchall() or []
    finally:
        cursor.close()
        conn.close()


def _sync_hold_alerts_and_cases() -> int:
    """Ensures every Hold decision has an alert and every alert has an investigation case."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Find latest Hold assessments without alerts
        cursor.execute("""
            SELECT m.merchant_id, m.transaction_count, ra.reasons, ra.evaluated_at
            FROM merchants m
            JOIN risk_assessments ra ON ra.merchant_id = m.merchant_id
            WHERE m.data_source <> 'test'
              AND ra.decision = 'Hold'
              AND NOT EXISTS (
                  SELECT 1 FROM risk_assessments newer
                  WHERE newer.merchant_id = ra.merchant_id
                    AND (newer.evaluated_at > ra.evaluated_at
                         OR (newer.evaluated_at = ra.evaluated_at
                             AND newer.assessment_id > ra.assessment_id))
              )
              AND NOT EXISTS (
                  SELECT 1 FROM alerts a WHERE a.merchant_id = m.merchant_id
              )
            ORDER BY ra.evaluated_at, m.merchant_id
        """)
        missing_holds = cursor.fetchall() or []
        for hold in missing_holds:
            _create_missing_hold_alert(hold)

        # Ensure every alert has an investigation case
        cursor.execute("""
            SELECT a.alert_id, a.merchant_id, a.created_at
            FROM alerts a
            WHERE NOT EXISTS (
                SELECT 1 FROM investigation_cases ic WHERE ic.alert_id = a.alert_id
            )
        """)
        missing_cases = cursor.fetchall() or []
        for c in missing_cases:
            cursor.execute("""
                INSERT INTO investigation_cases (
                    case_id, alert_id, merchant_id, investigator_id,
                    status, decision, notes, created_at, updated_at
                ) VALUES (%s, %s, %s, NULL, 'Open', NULL, NULL, %s, %s)
                ON DUPLICATE KEY UPDATE alert_id = VALUES(alert_id)
            """, (
                f"CASE_BF_{uuid.uuid4().hex[:16].upper()}",
                c["alert_id"],
                c["merchant_id"],
                c["created_at"],
                c["created_at"]
            ))
        conn.commit()
        return len(missing_holds) + len(missing_cases)
    finally:
        cursor.close()
        conn.close()


def _assessment_counts() -> Dict[str, int]:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT 
                COUNT(*) AS eligible_merchants,
                SUM(CASE WHEN m.data_source = 'synthetic' THEN 1 ELSE 0 END) AS synthetic_merchants,
                COUNT(DISTINCT CASE WHEN ra.assessment_id IS NOT NULL THEN m.merchant_id END) AS assessed_merchants,
                COUNT(DISTINCT CASE WHEN ra.assessment_id IS NOT NULL AND m.data_source = 'synthetic' THEN m.merchant_id END) AS synthetic_assessed,
                COUNT(ra.assessment_id) AS assessment_rows,
                SUM(CASE WHEN m.in_training_set = 1 THEN 1 ELSE 0 END) AS in_training_set_merchants
            FROM merchants m
            LEFT JOIN risk_assessments ra ON ra.merchant_id = m.merchant_id
            WHERE m.data_source <> 'test'
        """)
        stats = cursor.fetchone() or {}

        cursor.execute("SELECT COUNT(*) AS test_count FROM merchants WHERE data_source = 'test'")
        stats["test_merchants_excluded"] = cursor.fetchone()["test_count"]

        cursor.execute("SELECT COUNT(*) AS alert_count FROM alerts")
        stats["total_alerts"] = cursor.fetchone()["alert_count"]

        cursor.execute("SELECT COUNT(*) AS case_count FROM investigation_cases")
        stats["open_investigation_cases"] = cursor.fetchone()["case_count"]

        return {key: int(stats.get(key) or 0) for key in stats}
    finally:
        cursor.close()
        conn.close()


def _create_missing_hold_alert(hold: Dict[str, Any]) -> None:
    reasons: Optional[Any] = hold.get("reasons")
    if isinstance(reasons, bytes):
        reasons = reasons.decode("utf-8")
    if isinstance(reasons, str):
        try:
            reasons = json.loads(reasons)
        except json.JSONDecodeError:
            reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = []

    transaction_count = int(hold.get("transaction_count") or 0)
    alert = {
        "alert_id": f"ALT_BF_{uuid.uuid4().hex[:20].upper()}",
        "merchant_id": hold["merchant_id"],
        "severity": "CRITICAL",
        "alert_type": "COLD_START_FRAUD_DETECTED" if transaction_count == 0 else "HIGH_RISK_ONBOARDING",
        "details": "; ".join(str(reason) for reason in reasons),
        "status": "OPEN",
        "created_at": hold["evaluated_at"],
    }
    if not insert_alert(alert):
        raise RuntimeError(f"Could not create Hold alert for {hold['merchant_id']}")


def run_backfill(batch_size: int = 100, engine: Optional[RiskEngine] = None) -> Dict[str, int]:
    """Score pending merchants, populate in_training_set flags, and sync alerts & cases."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    # 1. Update in_training_set column
    update_in_training_set_flags()

    # 2. Score unassessed merchants
    risk_engine = engine or RiskEngine()
    inserted = 0
    while True:
        merchants = _fetch_unassessed_merchants(batch_size)
        if not merchants:
            break

        for merchant in merchants:
            graph_evidence = lookup_entity_network(
                bank_id=merchant["bank_account_id"],
                device_id=merchant["device_id"],
                ip_address=merchant["ip_address"],
                exclude_merchant_id=merchant["merchant_id"],
                limit=20,
            )
            merchant["bank_share_count"] = graph_evidence.get("bank_share_count", 0)
            merchant["device_share_count"] = graph_evidence.get("device_share_count", 0)
            merchant["ip_share_count"] = graph_evidence.get("ip_share_count", 0)
            merchant["connected_component_size"] = max(
                1, merchant["bank_share_count"] + merchant["device_share_count"]
            )
            merchant["neighbours_flagged_fraud"] = graph_evidence.get("neighbours_flagged_fraud", 0)
            merchant["louvain_community_fraud_rate"] = (
                0.85 if merchant["neighbours_flagged_fraud"] > 0
                else 0.15 if merchant["bank_share_count"] > 1
                else 0.0
            )

            assessment = risk_engine.score_merchant(merchant, graph_evidence)
            assessment["assessment_id"] = f"ASM_BF_{uuid.uuid4().hex[:20].upper()}"
            assessment["evaluated_at"] = merchant["registration_date"]
            if not insert_risk_assessment(assessment):
                raise RuntimeError(f"Could not save assessment for {merchant['merchant_id']}")
            inserted += 1

    # 3. Create missing alerts and open cases
    synced_items = _sync_hold_alerts_and_cases()

    counts = _assessment_counts()
    counts["newly_inserted_assessments"] = inserted
    counts["synced_alerts_and_cases"] = synced_items
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    counts = run_backfill(batch_size=args.batch_size)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
