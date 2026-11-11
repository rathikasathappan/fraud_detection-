"""
Database Interface & Connection Manager
Connects to MySQL database with credentials and provides query helpers.
Gracefully handles tables creation, merchant onboarding persistence,
risk assessments, graph edges, and alert logging.
"""

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import Error, pooling

# Load environment variables from .env in project root
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_USER = (os.getenv("DB_USER") or "").strip()
if not DB_USER:
    raise RuntimeError("DB_USER must be configured in the project .env file")
if DB_USER.lower() == "root":
    raise RuntimeError("DB_USER must be a dedicated non-root MySQL account")

DB_CONFIG = {
    "host": os.getenv("DB_HOST") or os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT") or os.getenv("MYSQL_PORT", 3306)),
    "user": DB_USER,
    "password": os.getenv("DB_PASSWORD") or os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("DB_NAME") or os.getenv("MYSQL_DATABASE", "merchant_fraud_db")
}

def get_server_connection():
    """Connects to MySQL server (without selecting specific DB)."""
    cfg = DB_CONFIG.copy()
    cfg.pop("database", None)
    return mysql.connector.connect(**cfg)

def get_db_connection():
    """Connects to the merchant_fraud_db database."""
    return mysql.connector.connect(**DB_CONFIG)

def init_db(schema_file: str = "database/schema.sql"):
    """
    Ensures database and tables exist according to database/schema.sql.
    Gracefully handles both root and dedicated application users.
    """
    try:
        # Try direct connection to target DB first
        try:
            conn = get_db_connection()
            # Migrate existing installations for test-source and missing optional inputs.
            try:
                alter_cur = conn.cursor()
                alter_statements = [
                    "ALTER TABLE merchants MODIFY COLUMN data_source ENUM('synthetic', 'company', 'api', 'test') NOT NULL DEFAULT 'synthetic'",
                    "ALTER TABLE merchants MODIFY COLUMN business_age_days INT NULL DEFAULT NULL",
                    "ALTER TABLE merchants MODIFY COLUMN phone_verified TINYINT(1) NULL DEFAULT NULL",
                    "ALTER TABLE merchants MODIFY COLUMN email_verified TINYINT(1) NULL DEFAULT NULL",
                    "ALTER TABLE merchants MODIFY COLUMN document_verified TINYINT(1) NULL DEFAULT NULL",
                    "ALTER TABLE merchants ADD COLUMN in_training_set TINYINT(1) NOT NULL DEFAULT 0",
                    "ALTER TABLE risk_assessments ADD COLUMN data_completeness VARCHAR(16) NULL DEFAULT NULL"
                ]
                for statement in alter_statements:
                    try:
                        alter_cur.execute(statement)
                    except Exception:
                        pass
                conn.commit()
                alter_cur.close()
            except Exception:
                pass
        except Error:
            # Fall back to server connection to create database if permitted
            srv_conn = get_server_connection()
            srv_cursor = srv_conn.cursor()
            srv_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']};")
            srv_cursor.close()
            srv_conn.close()
            conn = get_db_connection()

        cursor = conn.cursor()
        if os.path.exists(schema_file):
            with open(schema_file, "r") as f:
                schema_sql = f.read()
            
            # Split statements and strip SQL comments
            raw_statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
            for raw_stmt in raw_statements:
                # Remove comment lines
                lines = [line for line in raw_stmt.splitlines() if not line.strip().startswith("--")]
                clean_stmt = "\n".join(lines).strip()
                if not clean_stmt:
                    continue
                # Skip CREATE DATABASE or USE statements
                upper_stmt = clean_stmt.upper()
                if upper_stmt.startswith("USE ") or upper_stmt.startswith("CREATE DATABASE"):
                    continue
                cursor.execute(clean_stmt)
            conn.commit()
            logger.info("Database schema initialized successfully.")
            cursor.execute("""
                INSERT IGNORE INTO investigation_cases (
                    case_id, alert_id, merchant_id, investigator_id,
                    status, decision, notes, created_at, updated_at
                )
                SELECT CONCAT('CASE_', MD5(a.alert_id)), a.alert_id, a.merchant_id, NULL,
                       'Open', NULL, NULL, a.created_at, a.created_at
                FROM alerts a
                LEFT JOIN investigation_cases c ON c.alert_id = a.alert_id
                WHERE c.alert_id IS NULL
            """)
            conn.commit()
        cursor.close()
        conn.close()

        # Seed default admin from .env
        ensure_default_admin()
        return True
    except Error as e:
        logger.error(f"Error initializing MySQL database: {e}")
        return False

def insert_merchant(m: Dict[str, Any]) -> bool:
    """Inserts or updates a merchant record."""
    query = """
    INSERT INTO merchants (
        merchant_id, business_category, business_age_days, registration_date,
        phone_verified, email_verified, document_verified,
        bank_account_id, device_id, ip_address, transaction_count,
        fraud_label, data_source
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        business_category = VALUES(business_category),
        business_age_days = VALUES(business_age_days),
        phone_verified = VALUES(phone_verified),
        email_verified = VALUES(email_verified),
        document_verified = VALUES(document_verified),
        bank_account_id = VALUES(bank_account_id),
        device_id = VALUES(device_id),
        ip_address = VALUES(ip_address),
        transaction_count = VALUES(transaction_count),
        fraud_label = COALESCE(VALUES(fraud_label), fraud_label),
        data_source = VALUES(data_source);
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, (
            m["merchant_id"],
            m["business_category"],
            None if m.get("business_age_days") is None or str(m.get("business_age_days")).strip() == "" else int(m["business_age_days"]),
            m["registration_date"],
            None if m.get("phone_verified") is None or str(m.get("phone_verified")).strip() == "" else int(m["phone_verified"]),
            None if m.get("email_verified") is None or str(m.get("email_verified")).strip() == "" else int(m["email_verified"]),
            None if m.get("document_verified") is None or str(m.get("document_verified")).strip() == "" else int(m["document_verified"]),
            m["bank_account_id"],
            m["device_id"],
            m["ip_address"],
            int(m.get("transaction_count", 0)),
            m.get("fraud_label"),
            m.get("data_source", "synthetic")
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        logger.error(f"Failed to insert merchant {m.get('merchant_id')}: {e}")
        return False

def insert_relationships_bulk(rels: List[Dict[str, Any]]) -> bool:
    """Inserts entity relationships in bulk."""
    if not rels:
        return True
    query = """
    INSERT INTO relationships (merchant_id, entity_id, entity_type, relationship, timestamp)
    VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        data = [
            (r["merchant_id"], r["entity_id"], r["entity_type"], r["relationship"], r["timestamp"])
            for r in rels
        ]
        cursor.executemany(query, data)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        logger.error(f"Failed to bulk insert relationships: {e}")
        return False

def insert_risk_assessment(assessment: Dict[str, Any]) -> bool:
    """Saves risk assessment results."""
    query = """
    INSERT INTO risk_assessments (
        assessment_id, merchant_id, risk_score, decision, reasons, graph_evidence, model_version, data_completeness, evaluated_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        risk_score = VALUES(risk_score),
        decision = VALUES(decision),
        reasons = VALUES(reasons),
        graph_evidence = VALUES(graph_evidence),
        model_version = VALUES(model_version),
        data_completeness = VALUES(data_completeness),
        evaluated_at = VALUES(evaluated_at);
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        data_comp = assessment.get("data_completeness")
        if isinstance(data_comp, (int, float)):
            data_comp = f"{data_comp:.1f}%"
        cursor.execute(query, (
            assessment["assessment_id"],
            assessment["merchant_id"],
            float(assessment["risk_score"]),
            assessment["decision"],
            json.dumps(assessment.get("reasons", [])),
            json.dumps(assessment.get("graph_evidence", {})),
            assessment.get("model_version", "v1.0.0"),
            data_comp,
            assessment.get("evaluated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        logger.error(f"Failed to insert risk assessment for {assessment.get('merchant_id')}: {e}")
        return False

def insert_alert(alert: Dict[str, Any]) -> bool:
    """Logs an operational security alert."""
    query = """
    INSERT INTO alerts (alert_id, merchant_id, severity, alert_type, details, status, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, (
            alert["alert_id"],
            alert["merchant_id"],
            alert.get("severity", "HIGH"),
            alert.get("alert_type", "SUSPECTED_FRAUD_RING"),
            alert.get("details", ""),
            alert.get("status", "OPEN"),
            alert.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ))
        cursor.execute("""
            INSERT INTO investigation_cases (
                case_id, alert_id, merchant_id, investigator_id,
                status, decision, notes, created_at, updated_at
            ) VALUES (%s, %s, %s, NULL, 'Open', NULL, NULL, %s, %s)
            ON DUPLICATE KEY UPDATE alert_id = VALUES(alert_id)
        """, (
            f"CASE_{uuid.uuid4().hex[:16].upper()}",
            alert["alert_id"],
            alert["merchant_id"],
            alert.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            alert.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Error as e:
        logger.error(f"Failed to insert alert for {alert.get('merchant_id')}: {e}")
        return False

def get_merchant(merchant_id: str) -> Optional[Dict[str, Any]]:
    """Fetches single merchant details."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM merchants WHERE merchant_id = %s", (merchant_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except Error as e:
        logger.error(f"Error fetching merchant {merchant_id}: {e}")
        return None

def get_merchant_risk(merchant_id: str) -> Optional[Dict[str, Any]]:
    """Fetches latest risk assessment for a merchant."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM risk_assessments 
            WHERE merchant_id = %s 
            ORDER BY evaluated_at DESC LIMIT 1
        """, (merchant_id,))
        row = cursor.fetchone()
        if row and isinstance(row.get("reasons"), str):
            row["reasons"] = json.loads(row["reasons"])
        if row and isinstance(row.get("graph_evidence"), str):
            row["graph_evidence"] = json.loads(row["graph_evidence"])
        cursor.close()
        conn.close()
        return row
    except Error as e:
        logger.error(f"Error fetching risk assessment for {merchant_id}: {e}")
        return None

def get_merchant_graph(merchant_id: str) -> Dict[str, Any]:
    """
    Fetches the 1-hop and 2-hop connected network neighborhood for a merchant.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 1-hop: entities linked to this merchant
        cursor.execute("""
            SELECT entity_id, entity_type, relationship, timestamp 
            FROM relationships 
            WHERE merchant_id = %s
        """, (merchant_id,))
        entities = cursor.fetchall()
        
        # 2-hop: other merchants sharing any of these entities
        entity_ids = [e["entity_id"] for e in entities]
        shared_merchants = []
        if entity_ids:
            format_strings = ','.join(['%s'] * len(entity_ids))
            cursor.execute(f"""
                SELECT r.entity_id, r.merchant_id, m.fraud_label, m.business_category, m.registration_date
                FROM relationships r
                JOIN merchants m ON r.merchant_id = m.merchant_id
                WHERE r.entity_id IN ({format_strings}) AND r.merchant_id != %s
            """, (*entity_ids, merchant_id))
            shared_merchants = cursor.fetchall()

        cursor.close()
        conn.close()
        return {
            "merchant_id": merchant_id,
            "connected_entities": entities,
            "shared_merchants": shared_merchants
        }
    except Error as e:
        logger.error(f"Error fetching graph for {merchant_id}: {e}")
        return {"merchant_id": merchant_id, "connected_entities": [], "shared_merchants": []}

def lookup_entity_network(
    bank_id: str,
    device_id: str,
    ip_address: str,
    exclude_merchant_id: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Looks up the real network connections and peer merchants across bank, device, and IP entities.
    Returns connected entities and shared peer merchants (capped at limit).
    """
    connected_entities = [
        {"entity_id": bank_id, "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT"},
        {"entity_id": device_id, "entity_type": "device", "relationship": "USES_DEVICE"},
        {"entity_id": ip_address, "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP"}
    ]
    
    shared_merchants = []
    bank_peers = set()
    device_peers = set()
    ip_peers = set()
    bank_fraud_peers = set()
    device_fraud_peers = set()
    flagged_peers = set()

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        entity_ids = [bank_id, device_id, ip_address]
        format_strings = ','.join(['%s'] * len(entity_ids))
        
        query = f"""
            SELECT DISTINCT r.entity_id, r.entity_type, r.merchant_id, 
                   m.fraud_label, m.business_category, m.registration_date
            FROM relationships r
            JOIN merchants m ON r.merchant_id = m.merchant_id
                 WHERE r.entity_id IN ({format_strings}) AND m.data_source != 'test'
        """
        params = list(entity_ids)
        if exclude_merchant_id:
            query += " AND r.merchant_id != %s"
            params.append(exclude_merchant_id)
            
        query += " ORDER BY m.fraud_label DESC, r.timestamp DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, tuple(params))
        shared_merchants = cursor.fetchall() or []

        # Count per-entity degree and fraud neighbor stats
        for p in shared_merchants:
            is_fraud = (p.get("fraud_label") == 1)
            if p.get("entity_id") == bank_id:
                bank_peers.add(p["merchant_id"])
                if is_fraud:
                    bank_fraud_peers.add(p["merchant_id"])
            if p.get("entity_id") == device_id:
                device_peers.add(p["merchant_id"])
                if is_fraud:
                    device_fraud_peers.add(p["merchant_id"])
            if p.get("entity_id") == ip_address:
                ip_peers.add(p["merchant_id"])
            if is_fraud:
                flagged_peers.add(p["merchant_id"])

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error querying entity network: {e}")

    return {
        "connected_entities": connected_entities,
        "shared_merchants": shared_merchants[:limit],
        "bank_share_count": len(set(bank_peers)),
        "bank_fraud_peers": len(bank_fraud_peers),
        "device_share_count": len(set(device_peers)),
        "device_fraud_peers": len(device_fraud_peers),
        "ip_share_count": len(set(ip_peers)),
        "neighbours_flagged_fraud": len(flagged_peers)
    }

def get_test_merchants_count() -> int:
    """Returns the count of temporary merchants registered with data_source = 'test'."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM merchants WHERE data_source = 'test'")
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return int(row[0]) if row else 0
    except Exception as e:
        logger.error(f"Error checking test merchants count: {e}")
        return 0

def delete_test_merchants(user_id: Optional[int] = None, username: Optional[str] = None) -> int:
    """
    Cleans up all temporary test registrations (data_source = 'test')
    along with their relationships, risk assessments, alerts, cases, and audit logs.
    Leaves the 5,000 baseline dataset untouched.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT merchant_id FROM merchants WHERE data_source = 'test'")
        test_ids = [r["merchant_id"] for r in cursor.fetchall()]

        if not test_ids:
            cursor.close()
            conn.close()
            return 0

        fmt = ','.join(['%s'] * len(test_ids))
        cursor.execute(f"DELETE FROM audit_log WHERE merchant_id IN ({fmt})", tuple(test_ids))
        cursor.execute(f"DELETE FROM investigation_cases WHERE merchant_id IN ({fmt})", tuple(test_ids))
        cursor.execute(f"DELETE FROM alerts WHERE merchant_id IN ({fmt})", tuple(test_ids))
        cursor.execute(f"DELETE FROM risk_assessments WHERE merchant_id IN ({fmt})", tuple(test_ids))
        cursor.execute(f"DELETE FROM relationships WHERE merchant_id IN ({fmt})", tuple(test_ids))
        cursor.execute("DELETE FROM merchants WHERE data_source = 'test'")
        deleted_count = cursor.rowcount
        conn.commit()

        if user_id and username:
            try:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("""
                    INSERT INTO audit_log (user_id, username, action, merchant_id, created_at)
                    VALUES (%s, %s, %s, NULL, %s)
                """, (user_id, username, f"Cleaned up {deleted_count} temporary test merchant records", now))
                conn.commit()
            except Exception as audit_err:
                logger.warning(f"Failed to record audit log for test cleanup: {audit_err}")

        cursor.close()
        conn.close()
        logger.info(f"Cleaned up {deleted_count} test merchants from MySQL database.")
        return deleted_count
    except Exception as e:
        logger.error(f"Error deleting test merchants: {e}")
        return 0

def get_alerts(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches operational alerts."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        if status:
            cursor.execute("SELECT * FROM alerts WHERE status = %s ORDER BY created_at DESC LIMIT %s", (status, limit))
        else:
            cursor.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Error as e:
        logger.error(f"Error fetching alerts: {e}")
        return []


def get_investigation_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 200
) -> Optional[List[Dict[str, Any]]]:
    """Fetch alerts with their investigation case state for the review page."""
    conditions = []
    params = []
    if status:
        conditions.append("a.status = %s")
        params.append(status)
    if severity:
        conditions.append("a.severity = %s")
        params.append(severity)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"""
            SELECT a.alert_id, a.merchant_id, a.severity, a.alert_type, a.details,
                   a.status AS alert_status, a.created_at,
                   c.case_id, c.investigator_id, c.status AS case_status,
                   c.decision AS case_decision, c.notes AS case_notes,
                   c.updated_at AS case_updated_at
            FROM alerts a
            LEFT JOIN investigation_cases c ON c.alert_id = a.alert_id
            {where_clause}
            ORDER BY a.created_at DESC
            LIMIT %s
        """, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as exc:
        logger.error("Error fetching investigation alerts: %s", exc)
        return None


def get_dashboard_summary(date_from: str, date_to: str, category: Optional[str] = None) -> Dict[str, Any]:
    """Return read-only dashboard aggregates for a filtered, non-test merchant cohort."""
    conditions = ["m.data_source <> 'test'", "DATE(m.registration_date) BETWEEN %s AND %s"]
    params: List[Any] = [date_from, date_to]
    if category:
        conditions.append("m.business_category = %s")
        params.append(category)
    merchant_filter = " AND ".join(conditions)
    latest = """
        SELECT ra.*
        FROM risk_assessments ra
        JOIN (
            SELECT merchant_id, MAX(evaluated_at) AS evaluated_at
            FROM risk_assessments
            GROUP BY merchant_id
        ) latest_ra ON latest_ra.merchant_id = ra.merchant_id
                  AND latest_ra.evaluated_at = ra.evaluated_at
    """

    def fetch_all(cursor, query: str, query_params: tuple = ()):
        cursor.execute(query, query_params)
        return cursor.fetchall()

    def fetch_one(cursor, query: str, query_params: tuple = ()):
        cursor.execute(query, query_params)
        return cursor.fetchone() or {}

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cohort = f"FROM merchants m LEFT JOIN ({latest}) lr ON lr.merchant_id = m.merchant_id WHERE {merchant_filter}"
        kpis = fetch_one(cursor, f"""
            SELECT COUNT(*) AS total_merchants,
                   COALESCE(SUM(m.transaction_count = 0), 0) AS cold_start_count,
                   COALESCE(SUM(lr.decision = 'Hold'), 0) AS hold_count,
                   COALESCE(AVG(lr.risk_score), 0) AS average_risk_score,
                   (SELECT COUNT(*) FROM alerts a JOIN merchants am ON am.merchant_id = a.merchant_id
                    WHERE a.status = 'OPEN' AND am.data_source <> 'test'
                      AND DATE(am.registration_date) BETWEEN %s AND %s
                      {"AND am.business_category = %s" if category else ""}) AS open_alerts
            {cohort}
        """, tuple(params + [date_from, date_to] + ([category] if category else [])))
        period_start = datetime.strptime(date_from, "%Y-%m-%d").date()
        period_end = datetime.strptime(date_to, "%Y-%m-%d").date()
        period_length = (period_end - period_start).days + 1
        previous_end = period_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=period_length - 1)
        previous_conditions = ["m.data_source <> 'test'", "DATE(m.registration_date) BETWEEN %s AND %s"]
        previous_params: List[Any] = [previous_start.isoformat(), previous_end.isoformat()]
        if category:
            previous_conditions.append("m.business_category = %s")
            previous_params.append(category)
        previous_filter = " AND ".join(previous_conditions)
        previous_kpis = fetch_one(cursor, f"""
            SELECT COUNT(*) AS total_merchants,
                   COALESCE(SUM(m.transaction_count = 0), 0) AS cold_start_count,
                   COALESCE(SUM(lr.decision = 'Hold'), 0) AS hold_count,
                   COALESCE(AVG(lr.risk_score), 0) AS average_risk_score,
                   (SELECT COUNT(*) FROM alerts a JOIN merchants am ON am.merchant_id = a.merchant_id
                    WHERE a.status = 'OPEN' AND am.data_source <> 'test'
                      AND DATE(am.registration_date) BETWEEN %s AND %s
                      {"AND am.business_category = %s" if category else ""}) AS open_alerts
            FROM merchants m LEFT JOIN ({latest}) lr ON lr.merchant_id = m.merchant_id
            WHERE {previous_filter}
        """, tuple(previous_params + previous_params + ([category] if category else [])))
        decision_split = fetch_all(cursor, f"""
            SELECT COALESCE(lr.decision, 'Unassessed') AS decision, COUNT(*) AS count
            {cohort} GROUP BY COALESCE(lr.decision, 'Unassessed')
        """, tuple(params))
        daily = fetch_all(cursor, f"""
            SELECT DATE_FORMAT(m.registration_date, '%Y-%m-%d') AS day,
                   COALESCE(lr.decision, 'Unassessed') AS decision, COUNT(*) AS count
            {cohort} GROUP BY DATE_FORMAT(m.registration_date, '%Y-%m-%d'),
                              COALESCE(lr.decision, 'Unassessed')
            ORDER BY DATE_FORMAT(m.registration_date, '%Y-%m-%d')
        """, tuple(params))
        categories = fetch_all(cursor, f"""
            SELECT m.business_category AS category,
                   COALESCE(AVG(lr.risk_score), 0) AS average_risk_score,
                   COALESCE(SUM(lr.decision = 'Hold') / NULLIF(COUNT(*), 0), 0) AS hold_rate,
                   COUNT(*) AS merchant_count
            {cohort} GROUP BY m.business_category ORDER BY m.business_category
        """, tuple(params))
        histogram = fetch_all(cursor, f"""
            SELECT LEAST(FLOOR(lr.risk_score * 10), 9) AS bucket, COUNT(*) AS count
            {cohort} AND lr.risk_score IS NOT NULL
            GROUP BY LEAST(FLOOR(lr.risk_score * 10), 9) ORDER BY bucket
        """, tuple(params))
        alerts = fetch_all(cursor, f"""
            SELECT a.severity, a.status, COUNT(*) AS count
            FROM alerts a JOIN merchants m ON m.merchant_id = a.merchant_id
            WHERE {merchant_filter}
            GROUP BY a.severity, a.status ORDER BY a.severity, a.status
        """, tuple(params))
        shared_entities = []
        for entity_type in ("bank_account", "device", "ip_address"):
            shared_entities.extend(fetch_all(cursor, f"""
                SELECT r.entity_type, r.entity_id, COUNT(DISTINCT r.merchant_id) AS shared_size,
                       SUBSTRING_INDEX(
                           GROUP_CONCAT(DISTINCT m.merchant_id ORDER BY m.merchant_id SEPARATOR ','),
                           ',', 6
                       ) AS merchant_ids
                FROM relationships r JOIN merchants m ON m.merchant_id = r.merchant_id
                WHERE {merchant_filter} AND r.entity_type = %s
                GROUP BY r.entity_type, r.entity_id
                ORDER BY shared_size DESC, r.entity_id LIMIT 10
            """, tuple(params + [entity_type])))
        latest_high_risk = fetch_all(cursor, f"""
            SELECT m.merchant_id, m.business_category AS category, lr.risk_score,
                   lr.decision, lr.evaluated_at
            {cohort} AND lr.decision = 'Hold'
            ORDER BY lr.evaluated_at DESC LIMIT 10
        """, tuple(params))
        category_options = fetch_all(cursor, """
            SELECT DISTINCT business_category AS category FROM merchants
            WHERE data_source <> 'test' ORDER BY business_category
        """)
        return {
            "kpis": kpis,
            "previous_kpis": previous_kpis,
            "decisions": decision_split,
            "daily_registrations": daily,
            "categories": categories,
            "risk_histogram": histogram,
            "alerts": alerts,
            "shared_entities": shared_entities,
            "latest_high_risk": latest_high_risk,
            "category_options": category_options,
        }
    finally:
        cursor.close()
        conn.close()


def get_investigation_case(case_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single investigation case for read-only case access."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM investigation_cases WHERE case_id = %s", (case_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except Exception as exc:
        logger.error("Error fetching investigation case %s: %s", case_id, exc)
        return None


def update_investigation_case(
    case_id: str,
    status: str,
    decision: Optional[str],
    notes: Optional[str],
    user_id: int,
    username: str
) -> Optional[Dict[str, Any]]:
    """Update a case and append its audit entry atomically."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT * FROM investigation_cases WHERE case_id = %s FOR UPDATE",
            (case_id,)
        )
        case = cursor.fetchone()
        if not case:
            conn.rollback()
            return None

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE investigation_cases
            SET investigator_id = %s, status = %s, decision = %s,
                notes = %s, updated_at = %s
            WHERE case_id = %s
        """, (user_id, status, decision, notes, now, case_id))
        action = (
            f"Updated investigation case {case_id}: status={status}; "
            f"decision={decision or 'unset'}; notes updated"
        )
        cursor.execute("""
            INSERT INTO audit_log (user_id, username, action, merchant_id, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, username, action, case["merchant_id"], now))
        conn.commit()

        case.update({
            "investigator_id": user_id,
            "status": status,
            "decision": decision,
            "notes": notes,
            "updated_at": now
        })
        return case
    except Exception:
        conn.rollback()
        logger.exception("Error updating investigation case %s", case_id)
        raise
    finally:
        cursor.close()
        conn.close()

# -------------------------------------------------------------
# Audit Log & Performance Summaries
# -------------------------------------------------------------

def insert_audit_log(user_id: int, username: str, action: str, merchant_id: Optional[str] = None) -> bool:
    """Inserts a platform or investigation action into audit_log table."""
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (user_id, username, action, merchant_id, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, username, action, merchant_id, now))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error inserting audit log: {e}")
        return False


def get_audit_logs(limit: int = 100, merchant_id: Optional[str] = None, user_id: Optional[int] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves chronological audit log entries with optional search and filters."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        conditions = []
        params = []
        if merchant_id:
            conditions.append("a.merchant_id = %s")
            params.append(merchant_id)
        if user_id:
            conditions.append("a.user_id = %s")
            params.append(user_id)
        if search:
            conditions.append("(a.action LIKE %s OR a.username LIKE %s OR a.merchant_id LIKE %s)")
            wildcard = f"%{search}%"
            params.extend([wildcard, wildcard, wildcard])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT a.audit_id, a.user_id, a.username, a.action, a.merchant_id,
                   a.created_at,
                   COALESCE(u.role, 'System') as user_role
            FROM audit_log a
            LEFT JOIN users u ON a.user_id = u.id
            {where_clause}
            ORDER BY a.created_at DESC
            LIMIT %s
        """
        params.append(limit)
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        for r in rows:
            if r.get("created_at") is not None:
                r["created_at"] = str(r["created_at"])
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error fetching audit logs: {e}")
        return []


def get_sample_cluster_merchants() -> List[Dict[str, Any]]:
    """Returns curated merchant IDs representing various fraud rings and legitimate profiles."""
    return [
        {
            "id": "M_00013",
            "type": "Bank Ring",
            "badge": "danger",
            "description": "5 merchants sharing bank account BANK_0028 with flagged peers"
        },
        {
            "id": "M_00022",
            "type": "Device Ring",
            "badge": "warning",
            "description": "5 merchants linked through shared hardware DEV_0003"
        },
        {
            "id": "M_00119",
            "type": "Complex Ring",
            "badge": "danger",
            "description": "Coordinated multi-entity ring spanning shared bank, device & office IP"
        },
        {
            "id": "M_00029",
            "type": "Mimic Solo",
            "badge": "warning",
            "description": "Solo mimic fraudster attempting to spoof legitimate transaction attributes"
        },
        {
            "id": "M_00001",
            "type": "Genuine Merchant",
            "badge": "success",
            "description": "Legitimate cold-start merchant with clean graph topology (Approve)"
        }
    ]


def get_model_performance_summary() -> Dict[str, Any]:
    """Loads model evaluation comparisons and meta-learning benchmark metrics."""
    base_dir = Path(__file__).resolve().parent.parent
    comparison_file = base_dir / "data" / "processed" / "model_comparison.json"
    meta_file = base_dir / "data" / "processed" / "meta_learning_results.json"
    metrics_csv = base_dir / "data" / "powerbi_exports" / "model_metrics.csv"

    models_data = {}
    if comparison_file.exists():
        try:
            with open(comparison_file, "r") as f:
                models_data = json.load(f).get("models", {})
        except Exception as exc:
            logger.warning(f"Could not load model comparison json: {exc}")

    meta_data = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load meta-learning results json: {exc}")

    metrics_table = []
    if metrics_csv.exists():
        try:
            import pandas as pd
            df_m = pd.read_csv(metrics_csv)
            metrics_table = df_m.to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"Could not load model_metrics.csv: {exc}")

    return {
        "models": models_data,
        "meta_learning": meta_data,
        "metrics_table": metrics_table
    }

# -------------------------------------------------------------
# User Management & Authentication (RBAC)
# -------------------------------------------------------------

def create_user(username: str, email: str, password: str, role: str = "Analyst") -> Optional[Dict[str, Any]]:
    """
    Creates a new user with a werkzeug-hashed password and assigned RBAC role.
    Allowed roles: 'Admin', 'Investigator', 'Analyst'
    """
    valid_roles = ["Admin", "Investigator", "Analyst"]
    if role not in valid_roles:
        role = "Analyst"

    password_hash = generate_password_hash(password)
    query = """
    INSERT INTO users (username, email, password_hash, role, is_active)
    VALUES (%s, %s, %s, %s, 1)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (username, email, password_hash, role))
        conn.commit()
        user_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return {"id": user_id, "username": username, "email": email, "role": role}
    except Error as e:
        logger.error(f"Failed to create user {username}: {e}")
        return None

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves user by username."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND is_active = 1", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Error as e:
        logger.error(f"Error retrieving user {username}: {e}")
        return None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves user by primary ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username, email, role, is_active, last_login, created_at FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        return user
    except Error as e:
        logger.error(f"Error retrieving user ID {user_id}: {e}")
        return None

def authenticate_user(username: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates password hash for a username.
    Returns (user_dict, None) on success, or (None, failure_reason) on failure.
    Failure reasons: 'user_not_found', 'inactive_account', 'hash_mismatch', 'database_error'.
    Updates last_login timestamp upon successful authentication.
    """
    clean_username = str(username).strip().strip("'\"")
    clean_password = str(password).strip().strip("'\"")

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (clean_username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            return None, "user_not_found"

        if not user.get("is_active", 0):
            return None, "inactive_account"

        if not check_password_hash(user["password_hash"], clean_password):
            return None, "hash_mismatch"

        # Update last login
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET last_login = %s WHERE id = %s", (now_str, user["id"]))
        conn.commit()
        cursor.close()
        conn.close()

        # Return safe user dictionary without password hash
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }, None

    except Exception as e:
        logger.error(f"Database error during authentication: {e}")
        return None, f"database_error: {str(e)}"

def ensure_default_admin():
    """
    Seeds default admin user using values loaded from .env.
    Never duplicates if admin username already exists.
    """
    admin_user = (os.getenv("ADMIN_USERNAME") or "admin").strip().strip("'\"")
    admin_email = (os.getenv("ADMIN_EMAIL") or "admin@fraudshield.internal").strip().strip("'\"")
    admin_pass = (os.getenv("ADMIN_PASSWORD") or "").strip().strip("'\"")

    if not admin_pass:
        logger.warning("No ADMIN_PASSWORD defined in .env; skipping default admin generation.")
        return False

    existing = get_user_by_username(admin_user)
    if existing:
        logger.info(f"Default admin '{admin_user}' already exists in database.")
        return True

    logger.info(f"Seeding default Admin user '{admin_user}' from .env configuration...")
    created = create_user(
        username=admin_user,
        email=admin_email,
        password=admin_pass,
        role="Admin"
    )
    if created:
        logger.info("Default Admin user successfully seeded.")
        return True
    return False

def populate_database_from_synthetic(data_dir: str = "data/synthetic") -> Dict[str, int]:
    """Loads generated synthetic CSV data into MySQL tables in batches."""
    import pandas as pd
    merchants_file = os.path.join(data_dir, "merchants.csv")
    rels_file = os.path.join(data_dir, "relationships.csv")
    
    if not os.path.exists(merchants_file) or not os.path.exists(rels_file):
        raise FileNotFoundError("Synthetic dataset CSVs not found. Run generate_data.py first.")

    df_m = pd.read_csv(merchants_file)
    df_r = pd.read_csv(rels_file)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Bulk insert merchants
    logger.info(f"Loading {len(df_m)} merchants into MySQL...")
    m_query = """
    INSERT INTO merchants (
        merchant_id, business_category, business_age_days, registration_date,
        phone_verified, email_verified, document_verified,
        bank_account_id, device_id, ip_address, transaction_count,
        fraud_label, data_source
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'synthetic')
    ON DUPLICATE KEY UPDATE fraud_label = VALUES(fraud_label);
    """
    m_tuples = [
        (
            row["merchant_id"], row["business_category"], int(row["business_age_days"]),
            str(row["registration_date"]), int(row["phone_verified"]), int(row["email_verified"]),
            int(row["document_verified"]), str(row["bank_account_id"]), str(row["device_id"]),
            str(row["ip_address"]), int(row["transaction_count"]), int(row["fraud_label"])
        )
        for _, row in df_m.iterrows()
    ]
    # Execute in chunks of 1000
    chunk_size = 1000
    for i in range(0, len(m_tuples), chunk_size):
        cursor.executemany(m_query, m_tuples[i:i+chunk_size])
        conn.commit()

    logger.info(f"Loading {len(df_r)} relationships into MySQL...")
    r_query = """
    INSERT INTO relationships (merchant_id, entity_id, entity_type, relationship, timestamp)
    VALUES (%s, %s, %s, %s, %s);
    """
    r_tuples = [
        (row["merchant_id"], str(row["entity_id"]), str(row["entity_type"]), str(row["relationship"]), str(row["timestamp"]))
        for _, row in df_r.iterrows()
    ]
    # Clear old relationships if populating fresh synthetic
    cursor.execute("TRUNCATE TABLE relationships;")
    conn.commit()
    for i in range(0, len(r_tuples), chunk_size):
        cursor.executemany(r_query, r_tuples[i:i+chunk_size])
        conn.commit()

    cursor.close()
    conn.close()
    logger.info("Successfully populated MySQL database from synthetic CSVs.")
    return {"merchants_loaded": len(df_m), "relationships_loaded": len(df_r)}
