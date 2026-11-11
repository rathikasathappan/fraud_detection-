"""
New Merchant Data Ingestion & Batch Scoring Pipeline
Implements the contract defined in project_spec.json:
1. Validates schema, data completeness & reject invalid rows.
2. Imputes missing optional fields with neutral defaults.
3. Dynamically updates graph with new entities.
4. Scores cold-start risk with models/xgb_graph.pkl.
5. Persists assessments & alerts to MySQL.
6. Refreshes Power BI export tables.
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.db import (
    init_db, insert_merchant, insert_relationships_bulk,
    insert_risk_assessment, insert_alert, get_db_connection
)
from src.risk_engine import RiskEngine
from src.export_for_powerbi import export_all_for_powerbi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "merchant_id", "business_category", "registration_date",
    "bank_account_id", "device_id", "ip_address"
]

OPTIONAL_DEFAULTS = {
    "business_age_days": None,
    "phone_verified": None,
    "email_verified": None,
    "document_verified": None,
    "transaction_count": 0,
    "fraud_label": None
}

COMPLETENESS_FIELDS = [
    "business_category", "bank_account_id", "device_id", "ip_address",
    "business_age_days", "phone_verified", "email_verified", "document_verified"
]

def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Validates CSV schema against required contract, rejects invalid rows,
    and returns valid rows, rejected rows, and data completeness percentage.
    """
    missing_req = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_req:
        raise ValueError(f"Uploaded CSV is missing mandatory columns: {missing_req}")

    available_fields = [field for field in COMPLETENESS_FIELDS if field in df.columns]
    total_cells = df.shape[0] * len(COMPLETENESS_FIELDS)
    non_null_cells = df[available_fields].notnull().sum().sum()
    completeness_pct = round((non_null_cells / max(1, total_cells)) * 100, 2)

    # Validate row-level constraints
    valid_mask = (
        df["merchant_id"].notnull() &
        df["bank_account_id"].notnull() &
        df["device_id"].notnull() &
        df["ip_address"].notnull() &
        ~df["merchant_id"].duplicated()
    )

    valid_df = df[valid_mask].copy()
    rejected_df = df[~valid_mask].copy()

    # Preserve missing risk inputs for RiskEngine neutral imputation.
    for col, default_val in OPTIONAL_DEFAULTS.items():
        if col not in valid_df.columns:
            valid_df[col] = pd.Series([default_val] * len(valid_df), index=valid_df.index, dtype=object)
        elif default_val is None:
            valid_df[col] = valid_df[col].astype(object).where(valid_df[col].notnull(), None)
        else:
            valid_df[col] = valid_df[col].fillna(default_val)

    return valid_df, rejected_df, completeness_pct

def process_incoming_file(
    file_path: str,
    data_source: str = "company"
) -> Dict[str, Any]:
    """Processes a single batch CSV file through the ingestion & scoring pipeline."""
    logger.info(f"Starting ingestion for file: {file_path}")
    raw_df = pd.read_csv(file_path)

    valid_df, rejected_df, completeness = validate_dataframe(raw_df)
    logger.info(f"Data Completeness: {completeness}%. Valid: {len(valid_df)}, Rejected: {len(rejected_df)}")

    if len(valid_df) == 0:
        return {"status": "error", "message": "No valid rows found in file", "completeness": completeness}

    init_db()
    engine = RiskEngine()
    scored_assessments = []

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    for _, row in valid_df.iterrows():
        m_dict = row.to_dict()
        m_id = m_dict["merchant_id"]
        bank = m_dict["bank_account_id"]
        dev = m_dict["device_id"]
        ip = m_dict["ip_address"]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Dynamic graph degree lookup from existing database entities
        cursor.execute("SELECT COUNT(DISTINCT merchant_id) as cnt FROM relationships WHERE entity_id = %s", (bank,))
        b_cnt = cursor.fetchone()["cnt"] or 0

        cursor.execute("SELECT COUNT(DISTINCT merchant_id) as cnt FROM relationships WHERE entity_id = %s", (dev,))
        d_cnt = cursor.fetchone()["cnt"] or 0

        cursor.execute("SELECT COUNT(DISTINCT merchant_id) as cnt FROM relationships WHERE entity_id = %s", (ip,))
        ip_cnt = cursor.fetchone()["cnt"] or 0

        cursor.execute("""
            SELECT COUNT(DISTINCT m.merchant_id) as cnt
            FROM relationships r
            JOIN merchants m ON r.merchant_id = m.merchant_id
            WHERE r.entity_id IN (%s, %s) AND m.fraud_label = 1
        """, (bank, dev))
        fraud_cnt = cursor.fetchone()["cnt"] or 0

        m_dict["bank_share_count"] = b_cnt
        m_dict["device_share_count"] = d_cnt
        m_dict["ip_share_count"] = ip_cnt
        m_dict["connected_component_size"] = max(1, b_cnt + d_cnt)
        m_dict["neighbours_flagged_fraud"] = fraud_cnt
        m_dict["louvain_community_fraud_rate"] = 0.85 if fraud_cnt > 0 else (0.15 if b_cnt > 1 else 0.0)
        m_dict["data_source"] = data_source

        # Score with risk engine
        assessment = engine.score_merchant(m_dict)
        assessment["assessment_id"] = f"ASM_BAT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{m_id}"
        assessment["evaluated_at"] = now_str
        scored_assessments.append(assessment)

        # Persist merchant & links
        insert_merchant(m_dict)
        rels = [
            {"merchant_id": m_id, "entity_id": bank, "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT", "timestamp": now_str},
            {"merchant_id": m_id, "entity_id": dev, "entity_type": "device", "relationship": "USES_DEVICE", "timestamp": now_str},
            {"merchant_id": m_id, "entity_id": ip, "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP", "timestamp": now_str}
        ]
        insert_relationships_bulk(rels)
        insert_risk_assessment(assessment)

        # Generate alert if High or Hold
        if assessment["decision"] in ["Hold", "Review"]:
            sev = "CRITICAL" if assessment["decision"] == "Hold" else "HIGH"
            insert_alert({
                "alert_id": f"ALT_BAT_{m_id}",
                "merchant_id": m_id,
                "severity": sev,
                "alert_type": "COMPANY_BATCH_ONBOARDING_ALERT",
                "details": "; ".join(assessment["reasons"]),
                "status": "OPEN",
                "created_at": now_str
            })

    cursor.close()
    conn.close()

    # Refresh Power BI Exports
    export_all_for_powerbi()

    return {
        "status": "success",
        "processed_file": file_path,
        "rows_valid": len(valid_df),
        "rows_rejected": len(rejected_df),
        "data_completeness": f"{completeness}%",
        "high_risk_flagged": sum(1 for a in scored_assessments if a["decision"] == "Hold"),
        "under_review": sum(1 for a in scored_assessments if a["decision"] == "Review"),
        "auto_approved": sum(1 for a in scored_assessments if a["decision"] == "Approve")
    }

def process_all_incoming(incoming_dir: str = "data/incoming"):
    """Scans data/incoming/ and processes all pending CSVs."""
    csv_files = glob.glob(os.path.join(incoming_dir, "*.csv"))
    if not csv_files:
        logger.info(f"No pending CSV files in {incoming_dir}")
        return
    for f in csv_files:
        res = process_incoming_file(f)
        logger.info(f"Completed {f}: {res}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_incoming_file(sys.argv[1])
    else:
        process_all_incoming()

