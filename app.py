
"""
Merchant Fraud Detection - Flask Application & REST API
Phase 5: Production-ready backend supporting cold-start merchant onboarding,
real-time graph prior scoring, MySQL persistence, and data ingestion.
"""

import os
import socket
import io
import uuid
import logging
import time
import threading
from collections import deque
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, render_template_string, session, redirect, url_for, Response
from flask_wtf.csrf import CSRFProtect

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from src.db import (
    init_db, insert_merchant, insert_relationships_bulk,
    insert_risk_assessment, insert_alert,
    get_merchant, get_merchant_risk, get_merchant_graph, get_alerts,
    get_db_connection, authenticate_user, get_user_by_username,
    lookup_entity_network, delete_test_merchants, get_test_merchants_count,
    get_investigation_alerts, get_investigation_case, update_investigation_case,
    insert_audit_log, get_audit_logs, get_model_performance_summary, get_sample_cluster_merchants,
    get_dashboard_summary
)
from src.retrain import retrain_model_pipeline
from src.risk_engine import RiskEngine
from src.export_for_powerbi import export_all_for_powerbi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _required_secret_key():
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY must be configured in .env before starting the application")
    return secret_key


def load_model_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "data", "powerbi_exports", "model_metrics.csv")
    try:
        metrics_df = pd.read_csv(metrics_path)
    except (OSError, ValueError) as exc:
        logger.warning("Could not load model metrics from %s: %s", metrics_path, exc)
        return [], "Model metrics could not be loaded."

    required_columns = {"model_name", "evaluation_group", "pr_auc"}
    if not required_columns.issubset(metrics_df.columns):
        logger.warning("Model metrics CSV is missing required columns: %s", sorted(required_columns))
        return [], "Model metrics file is missing required columns."

    metrics = []
    for row in metrics_df.dropna(subset=["model_name", "evaluation_group", "pr_auc"]).to_dict("records"):
        try:
            pr_auc = float(row["pr_auc"])
        except (TypeError, ValueError):
            continue
        metrics.append({
            "model_name": str(row["model_name"]),
            "evaluation_group": str(row["evaluation_group"]),
            "pr_auc": f"{pr_auc:.4f}"
        })
    return metrics, None


def load_dataset_stats():
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "synthetic", "merchants.csv")
    try:
        dataset_df = pd.read_csv(dataset_path, usecols=["transaction_count", "fraud_label"])
    except (OSError, ValueError) as exc:
        logger.warning("Could not load synthetic dataset stats from %s: %s", dataset_path, exc)
        return None, "Dataset statistics could not be loaded."
    if dataset_df.empty:
        return None, "No dataset records are available."
    return {
        "total_merchants": len(dataset_df),
        "cold_start_percentage": float(dataset_df["transaction_count"].eq(0).mean() * 100),
        "fraud_percentage": float(dataset_df["fraud_label"].mean() * 100)
    }, None


app = Flask(__name__)
app.secret_key = _required_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
csrf = CSRFProtect(app)
risk_engine = None
DEMO_RATE_LIMIT = 10
DEMO_RATE_WINDOW_SECONDS = 60
DEMO_CLIENT_LIMIT = 4096
_DEMO_REQUESTS: Dict[str, deque] = {}
_DEMO_REQUESTS_LOCK = threading.Lock()
_DEMO_RATE_CHECKS = 0

DEMO_SCENARIOS = {
    "genuine": {
        "business_category": "Grocery",
        "business_age_days": 420,
        "phone_verified": 1,
        "email_verified": 1,
        "document_verified": 1,
        "transaction_count": 0,
        "bank_share_count": 0,
        "device_share_count": 0,
        "ip_share_count": 0,
        "connected_component_size": 1,
        "neighbours_flagged_fraud": 0,
        "louvain_community_fraud_rate": 0.0,
    },
    "shared_device": {
        "business_category": "Digital Services",
        "business_age_days": 90,
        "phone_verified": 1,
        "email_verified": 1,
        "document_verified": 1,
        "transaction_count": 0,
        "bank_share_count": 0,
        "device_share_count": 3,
        "ip_share_count": 0,
        "connected_component_size": 4,
        "neighbours_flagged_fraud": 0,
        "louvain_community_fraud_rate": 0.15,
    },
    "fraud_ring": {
        "business_category": "Electronics",
        "business_age_days": 10,
        "phone_verified": 0,
        "email_verified": 1,
        "document_verified": 0,
        "transaction_count": 0,
        "bank_share_count": 5,
        "device_share_count": 4,
        "ip_share_count": 2,
        "connected_component_size": 8,
        "neighbours_flagged_fraud": 3,
        "louvain_community_fraud_rate": 0.85,
    },
}


def get_engine():
    global risk_engine
    if risk_engine is None:
        risk_engine = RiskEngine()
    return risk_engine


def load_landing_metrics():
    """Return aggregate metrics from the simulated sample and saved model results."""
    dataset_stats, _ = load_dataset_stats()
    model_metrics, _ = load_model_metrics()
    metrics = []
    if dataset_stats:
        metrics.extend([
            {"label": "Simulated merchant profiles", "value": dataset_stats["total_merchants"], "suffix": ""},
            {"label": "Cold-start profiles", "value": dataset_stats["cold_start_percentage"], "suffix": "%"},
            {"label": "Fraud prevalence in sample", "value": dataset_stats["fraud_percentage"], "suffix": "%"},
        ])
    graph_cold_start = next(
        (
            row for row in model_metrics
            if row["model_name"] == "xgboost_with_graph"
            and row["evaluation_group"] == "cold_start_group"
        ),
        None
    )
    if graph_cold_start:
        metrics.append({
            "label": "Cold-start PR-AUC",
            "value": float(graph_cold_start["pr_auc"]),
            "suffix": "",
        })
    return metrics


def measure_database_response_ms():
    """Measure a read-only database round-trip for the landing-page status chip."""
    connection = None
    cursor = None
    started_at = time.perf_counter()
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return round((time.perf_counter() - started_at) * 1000)
    except Exception as exc:
        logger.warning("Landing page database response time is unavailable: %s", exc)
        return None
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception as exc:
                logger.warning("Landing page database cursor could not be closed: %s", exc)
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                logger.warning("Landing page database connection could not be closed: %s", exc)


def check_demo_rate_limit(client_key: str):
    """Apply a process-local rolling-window limit without persisting client data."""
    global _DEMO_RATE_CHECKS
    now = time.monotonic()
    cutoff = now - DEMO_RATE_WINDOW_SECONDS
    with _DEMO_REQUESTS_LOCK:
        _DEMO_RATE_CHECKS += 1
        if _DEMO_RATE_CHECKS % 128 == 0:
            expired_keys = [
                key for key, requests in _DEMO_REQUESTS.items()
                if not requests or requests[-1] <= cutoff
            ]
            for key in expired_keys:
                _DEMO_REQUESTS.pop(key, None)

        requests = _DEMO_REQUESTS.get(client_key)
        if requests is None:
            if len(_DEMO_REQUESTS) >= DEMO_CLIENT_LIMIT:
                _DEMO_REQUESTS.pop(next(iter(_DEMO_REQUESTS)))
            requests = deque()
            _DEMO_REQUESTS[client_key] = requests
        while requests and requests[0] <= cutoff:
            requests.popleft()
        if len(requests) >= DEMO_RATE_LIMIT:
            retry_after = max(1, int(DEMO_RATE_WINDOW_SECONDS - (now - requests[0])))
            return False, retry_after
        requests.append(now)
        return True, None


def find_available_port(start_port: int, attempts: int = 50) -> int:
    """Return the first free localhost port at or above start_port."""
    if not 1 <= start_port <= 65535:
        raise ValueError("APP_PORT must be between 1 and 65535")
    for port in range(start_port, min(start_port + attempts, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free application port found at or above {start_port}")


FAILED_LOGIN_ATTEMPTS = {}
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 5


def check_rate_limit(username: str) -> Tuple[bool, Optional[int]]:
    """Returns (is_locked, remaining_seconds)."""
    clean_user = username.lower().strip()
    record = FAILED_LOGIN_ATTEMPTS.get(clean_user)
    if not record:
        return False, None
    locked_until = record.get("locked_until")
    if locked_until and datetime.now() < locked_until:
        remaining_secs = int((locked_until - datetime.now()).total_seconds())
        return True, remaining_secs
    if locked_until and datetime.now() >= locked_until:
        FAILED_LOGIN_ATTEMPTS.pop(clean_user, None)
    return False, None


def record_failed_attempt(username: str):
    """Increments failed attempt count and triggers lock if threshold exceeded."""
    clean_user = username.lower().strip()
    record = FAILED_LOGIN_ATTEMPTS.setdefault(clean_user, {"count": 0, "locked_until": None})
    record["count"] += 1
    if record["count"] >= MAX_FAILED_ATTEMPTS:
        record["locked_until"] = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)
        logger.warning("[SECURITY] Login rate limit triggered for user '%s'.", clean_user)


def reset_failed_attempts(username: str):
    """Clears failed login tracking upon successful authentication."""
    FAILED_LOGIN_ATTEMPTS.pop(username.lower().strip(), None)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required", "message": "Please log in to continue"}), 401
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def role_required(*allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user" not in session:
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Authentication required"}), 401
                return redirect(url_for("login", next=request.path))
            user_role = session["user"].get("role")
            if user_role not in allowed_roles:
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Forbidden", "message": f"Requires one of roles: {list(allowed_roles)}"}), 403
                return render_template_string(
                    "<h2>403 - Forbidden</h2><p>Role {{ role }} cannot access this page.</p>",
                    role=user_role
                ), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# -------------------------------------------------------------
# Authentication Routes (Session Login & Logout)
# -------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Renders login UI (GET) and processes secure authentication (POST).
    Uses werkzeug password verification against MySQL users table.
    Enforces failed-login rate limiting and logs failure diagnostics.
    """
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("dashboard"))
        return render_template("login.html", next_url=request.args.get("next", ""))

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        return render_template(
            "login.html",
            error="Please provide both username and password.",
            next_url=request.args.get("next", "")
        ), 400

    # Rate limiting check (5 failures -> lock for 5 minutes)
    is_locked, remaining_secs = check_rate_limit(username)
    if is_locked:
        rem_mins = max(1, (remaining_secs + 59) // 60)
        logger.warning(f"[AUTH RATE LIMIT] Login attempt blocked for user '{username}': temporarily locked ({remaining_secs}s remaining).")
        return render_template(
            "login.html",
            error=f"Account is temporarily locked due to {MAX_FAILED_ATTEMPTS} failed attempts. Please try again in {rem_mins} minute(s).",
            next_url=request.args.get("next", "")
        ), 429

    user, failure_reason = authenticate_user(username, password)
    if not user:
        record_failed_attempt(username)
        # Log the real failure reason to terminal WITHOUT printing password
        logger.warning(f"[AUTH FAILURE] Login failed for username '{username}'. Diagnostics: {failure_reason}")
        return render_template(
            "login.html",
            error="Invalid username or password. Please verify credentials.",
            next_url=request.args.get("next", "")
        ), 401

    # On success: reset failure tracking
    reset_failed_attempts(username)
    logger.info(f"[AUTH SUCCESS] User '{username}' ({user['role']}) signed in successfully.")

    session["user"] = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"]
    }
    
    next_url = request.args.get("next", "")
    next_target = urlparse(next_url)
    if (
        next_target.path.startswith("/")
        and not next_target.path.startswith("//")
        and not next_target.scheme
        and not next_target.netloc
        and "\\" not in next_target.path
    ):
        return redirect(next_url)
    return redirect(url_for("dashboard"))

@app.route("/logout", methods=["GET", "POST"])
def logout():
    """Clears user session and redirects to login."""
    session.clear()
    return redirect(url_for("login"))

@app.route("/api/auth/me", methods=["GET"])
def current_user():
    """Returns currently authenticated session user profile."""
    if "user" in session:
        return jsonify({"authenticated": True, "user": session["user"]}), 200
    return jsonify({"authenticated": False, "user": None}), 401

@app.route("/", methods=["GET"])
def landing_page():
    """Serve the public product landing page regardless of authentication state."""
    return render_template(
        "landing.html",
        landing_metrics=load_landing_metrics(),
        database_response_ms=measure_database_response_ms(),
    )


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    """Renders authenticated web dashboard."""
    return render_template("dashboard.html", page_key="dashboard")


@app.route("/home", methods=["GET"])
@login_required
def home():
    """Backward-compatible dashboard route used by older links."""
    return redirect(url_for("dashboard"))


@app.route("/api/dashboard/summary", methods=["GET"])
@login_required
def dashboard_summary():
    """Return filtered, read-only MySQL aggregates for the analytics dashboard."""
    today = datetime.now().date()
    default_start = today - timedelta(days=29)
    date_from = request.args.get("date_from", default_start.isoformat())
    date_to = request.args.get("date_to", today.isoformat())
    category = request.args.get("category", "").strip() or None
    try:
        parsed_from = datetime.strptime(date_from, "%Y-%m-%d").date()
        parsed_to = datetime.strptime(date_to, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Dates must use YYYY-MM-DD format"}), 400
    if parsed_from > parsed_to:
        return jsonify({"error": "date_from must be on or before date_to"}), 400
    try:
        summary = get_dashboard_summary(parsed_from.isoformat(), parsed_to.isoformat(), category)
    except Exception:
        logger.exception("Dashboard summary query failed")
        return jsonify({"error": "Dashboard data could not be loaded"}), 500
    summary["filters"] = {
        "date_from": parsed_from.isoformat(),
        "date_to": parsed_to.isoformat(),
        "category": category or ""
    }
    return jsonify(summary), 200


@app.route("/onboarding", methods=["GET"])
@login_required
@role_required("Admin", "Investigator")
def onboarding():
    return render_template("onboarding.html", page_key="onboarding")


@app.route("/risk", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def risk_lookup():
    merchant_id = request.args.get("merchant_id", "").strip()
    if merchant_id:
        return redirect(url_for("risk_assessment", merchant_id=merchant_id))
    return render_template("risk_assessment.html", lookup_mode=True, page_key="risk")


@app.route("/risk/<merchant_id>", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def risk_assessment(merchant_id: str):
    assessment = get_merchant_risk(merchant_id)
    if not assessment:
        return render_template(
            "risk_assessment.html",
            merchant_id=merchant_id,
            assessment=None,
            error=f"No risk assessment was found for merchant {merchant_id}.",
            page_key="risk"
        ), 404
    evidence = assessment.get("graph_evidence") or {}
    evidence["connected_entities"] = (evidence.get("connected_entities") or [])[:20]
    evidence["shared_merchants"] = (evidence.get("shared_merchants") or [])[:20]
    return render_template(
        "risk_assessment.html",
        merchant_id=merchant_id,
        assessment=assessment,
        error=None,
        page_key="risk"
    )


@app.route("/network", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def network_lookup():
    merchant_id = request.args.get("merchant_id", "").strip()
    if merchant_id:
        return redirect(url_for("network_explorer", merchant_id=merchant_id))
    return render_template("network.html", lookup_mode=True, page_key="network")


@app.route("/network/<merchant_id>", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def network_explorer(merchant_id: str):
    merchant = get_merchant(merchant_id)
    return render_template(
        "network.html",
        lookup_mode=False,
        merchant_id=merchant_id,
        merchant=merchant,
        page_key="network"
    )


@app.route("/alerts", methods=["GET"])
@login_required
@role_required("Admin", "Investigator")
def alerts_page():
    statuses = {"OPEN", "ACKNOWLEDGED", "RESOLVED"}
    severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    status = request.args.get("status", "").strip().upper() or None
    severity = request.args.get("severity", "").strip().upper() or None
    error = None
    if status and status not in statuses:
        error = "Choose a valid alert status."
    if severity and severity not in severities:
        error = "Choose a valid alert severity."
    alerts = None if error else get_investigation_alerts(status=status, severity=severity)
    if alerts is None and error is None:
        error = "Alerts could not be loaded. Please try again."
    return render_template(
        "alerts.html",
        alerts=alerts or [],
        error=error,
        selected_status=status or "",
        selected_severity=severity or "",
        page_key="alerts"
    ), 400 if status and status not in statuses or severity and severity not in severities else 200


@app.route("/upload", methods=["GET"])
@login_required
@role_required("Admin")
def upload_page():
    return render_template("upload.html", page_key="upload")


@app.route("/models", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def model_performance_page():
    summary = get_model_performance_summary()
    return render_template(
        "model_performance.html",
        models=summary.get("models", {}),
        meta_learning=summary.get("meta_learning", {}),
        metrics_table=summary.get("metrics_table", []),
        page_key="models"
    )


@app.route("/audit-log", methods=["GET"])
@login_required
@role_required("Admin", "Investigator")
def audit_log_page():
    search = request.args.get("search", "").strip() or None
    merchant_id = request.args.get("merchant_id", "").strip() or None
    logs = get_audit_logs(limit=150, merchant_id=merchant_id, search=search)
    test_count = get_test_merchants_count()
    return render_template(
        "audit_log.html",
        logs=logs,
        search_term=search or "",
        merchant_id=merchant_id or "",
        test_merchants_count=test_count,
        page_key="audit_log"
    )


@app.route("/api/investigation-cases/<case_id>", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def get_case(case_id: str):
    case = get_investigation_case(case_id)
    if not case:
        return jsonify({"error": f"Investigation case {case_id} was not found"}), 404
    return jsonify(case), 200


@app.route("/api/investigation-cases/<case_id>", methods=["PATCH"])
@login_required
@role_required("Admin", "Investigator")
def patch_case(case_id: str):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    allowed_statuses = {"Open", "In review", "Closed"}
    allowed_decisions = {"Confirm fraud", "False positive", "Needs more info"}
    status = data.get("status")
    decision = data.get("decision") or None
    notes = data.get("notes", "")
    if status not in allowed_statuses:
        return jsonify({"error": "status must be Open, In review, or Closed"}), 400
    if decision is not None and decision not in allowed_decisions:
        return jsonify({"error": "decision must be Confirm fraud, False positive, or Needs more info"}), 400
    if notes is not None and not isinstance(notes, str):
        return jsonify({"error": "notes must be text"}), 400
    if notes and len(notes) > 10000:
        return jsonify({"error": "notes must be 10000 characters or fewer"}), 400

    user = session["user"]
    try:
        case = update_investigation_case(
            case_id=case_id,
            status=status,
            decision=decision,
            notes=notes or "",
            user_id=user["id"],
            username=user["username"]
        )
    except Exception:
        logger.exception("Investigation case update failed")
        return jsonify({"error": "Investigation case could not be updated"}), 500
    if not case:
        return jsonify({"error": f"Investigation case {case_id} was not found"}), 404
    return jsonify({"case": case}), 200

@app.route("/api/health", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "merchant_fraud_detection",
        "model_loaded": get_engine().model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/merchant/register", methods=["POST"])
@login_required
@role_required("Admin", "Investigator")
def register_merchant():
    """
    POST /api/merchant/register
    Registers a new merchant, links entities into graph,
    computes graph features, scores risk, and persists to MySQL.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    # Required field validation — ip_address is optional (auto-detected if absent)
    required = ["business_category", "bank_account_id", "device_id"]
    missing = [f for f in required if f not in data or not str(data.get(f, "")).strip()]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Resolve IP address: use provided value or fall back to request IP
    provided_ip = (data.get("ip_address") or "").strip()
    if not provided_ip:
        forwarded = request.headers.get("X-Forwarded-For", "")
        provided_ip = forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "127.0.0.1")

    merchant_id = data.get("merchant_id", f"M_{uuid.uuid4().hex[:8].upper()}")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_source = data.get("data_source")
    if data_source is None:
        data_source = "company" if data.get("is_company", False) else "api"
    if data_source not in {"api", "company", "test"}:
        return jsonify({"error": "data_source must be 'api', 'company', or 'test'"}), 400

    def optional_integer(field):
        value = data.get(field)
        return None if value is None or str(value).strip() == "" else int(value)

    merchant_record = {
        "merchant_id": merchant_id,
        "business_category": data["business_category"],
        "business_age_days": optional_integer("business_age_days"),
        "registration_date": data.get("registration_date", now_str),
        "phone_verified": optional_integer("phone_verified"),
        "email_verified": optional_integer("email_verified"),
        "document_verified": optional_integer("document_verified"),
        "bank_account_id": data["bank_account_id"],
        "device_id": data["device_id"],
        "ip_address": provided_ip,
        "transaction_count": int(data.get("transaction_count", 0)),
        "fraud_label": data.get("fraud_label", None),
        "data_source": data_source
    }

    # Look up prior peers before inserting this merchant into the graph.
    graph_evidence = lookup_entity_network(
        bank_id=data["bank_account_id"],
        device_id=data["device_id"],
        ip_address=provided_ip,
        exclude_merchant_id=merchant_id,
        limit=20
    )
    bank_share_cnt = graph_evidence.get("bank_share_count", 0)
    device_share_cnt = graph_evidence.get("device_share_count", 0)
    ip_share_cnt = graph_evidence.get("ip_share_count", 0)
    fraud_neighbour_cnt = graph_evidence.get("neighbours_flagged_fraud", 0)

    # Enrich with graph signals
    merchant_record["bank_share_count"] = bank_share_cnt
    merchant_record["device_share_count"] = device_share_cnt
    merchant_record["ip_share_count"] = ip_share_cnt
    merchant_record["connected_component_size"] = max(1, bank_share_cnt + device_share_cnt)
    merchant_record["neighbours_flagged_fraud"] = fraud_neighbour_cnt
    merchant_record["louvain_community_fraud_rate"] = 0.85 if fraud_neighbour_cnt > 0 else (0.15 if bank_share_cnt > 1 else 0.0)

    # 2. Score with Risk Engine
    engine = get_engine()
    assessment = engine.score_merchant(merchant_record, graph_evidence)
    assessment["assessment_id"] = f"ASM_{uuid.uuid4().hex[:8].upper()}"
    assessment["evaluated_at"] = now_str

    # 3. Save Merchant to Database
    insert_merchant(merchant_record)

    # 4. Save Relationships to Database
    rels = [
        {"merchant_id": merchant_id, "entity_id": data["bank_account_id"], "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT", "timestamp": now_str},
        {"merchant_id": merchant_id, "entity_id": data["device_id"], "entity_type": "device", "relationship": "USES_DEVICE", "timestamp": now_str},
        {"merchant_id": merchant_id, "entity_id": provided_ip, "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP", "timestamp": now_str}
    ]
    insert_relationships_bulk(rels)

    # 5. Save Assessment
    insert_risk_assessment(assessment)

    # 6. Raise Alert if High/Critical Risk
    if assessment["decision"] in ["Hold", "Review"]:
        severity = "CRITICAL" if assessment["decision"] == "Hold" else "HIGH"
        alert = {
            "alert_id": f"ALT_{uuid.uuid4().hex[:8].upper()}",
            "merchant_id": merchant_id,
            "severity": severity,
            "alert_type": "COLD_START_FRAUD_DETECTED" if merchant_record["transaction_count"] == 0 else "HIGH_RISK_ONBOARDING",
            "details": "; ".join(assessment["reasons"]),
            "status": "OPEN",
            "created_at": now_str
        }
        insert_alert(alert)

    # 7. Record Audit Log for Merchant Onboarding
    user = session.get("user")
    if user:
        insert_audit_log(
            user_id=user["id"],
            username=user["username"],
            action=f"Onboarded merchant: Decision={assessment['decision']}, Risk Score={assessment['risk_score']:.3f}",
            merchant_id=merchant_id
        )

    return jsonify(assessment), 201

@app.route("/api/fraud/predict", methods=["POST"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def predict_fraud():
    """
    POST /api/fraud/predict
    Evaluates risk score without registering a permanent record.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    engine = get_engine()
    assessment = engine.score_merchant(data)
    return jsonify(assessment), 200


@app.route("/api/demo/predict", methods=["POST"])
@csrf.exempt
def demo_predict():
    """Score one fixed simulated profile without authentication or persistence."""
    if request.content_length is not None and request.content_length > 1024:
        return jsonify({"error": "Demo request is too large"}), 413
    if not request.is_json:
        return jsonify({"error": "Send a JSON scenario"}), 415

    data = request.get_json(silent=True)
    scenario = data.get("scenario") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or set(data) != {"scenario"}
        or not isinstance(scenario, str)
        or scenario not in DEMO_SCENARIOS
    ):
        return jsonify({
            "error": "Choose one of: genuine, shared_device, fraud_ring"
        }), 400

    allowed, retry_after = check_demo_rate_limit(request.remote_addr or "unknown")
    if not allowed:
        response = jsonify({"error": "Demo limit reached. Try again shortly."})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    try:
        assessment = get_engine().score_merchant(dict(DEMO_SCENARIOS[scenario]))
    except Exception:
        logger.exception("Public simulated demo assessment failed")
        return jsonify({"error": "The demo is temporarily unavailable"}), 503

    reasons = assessment.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    return jsonify({
        "decision": assessment.get("decision"),
        "risk_score": assessment.get("risk_score"),
        "reasons": [str(reason) for reason in reasons[:3]],
    }), 200

@app.route("/api/risk/<merchant_id>", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def get_risk_by_merchant(merchant_id: str):
    """GET /api/risk/<id>"""
    assessment = get_merchant_risk(merchant_id)
    if not assessment:
        return jsonify({"error": f"No assessment found for merchant {merchant_id}"}), 404
    return jsonify(assessment), 200

@app.route("/api/graph/<merchant_id>", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def get_graph_by_merchant(merchant_id: str):
    """GET /api/graph/<id>"""
    graph_data = get_merchant_graph(merchant_id)
    return jsonify(graph_data), 200

@app.route("/api/alerts", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def get_recent_alerts():
    """GET /api/alerts"""
    status = request.args.get("status")
    limit = int(request.args.get("limit", 50))
    alerts_list = get_alerts(status=status, limit=limit)
    return jsonify({"alerts": alerts_list, "count": len(alerts_list)}), 200

@app.route("/api/data/upload", methods=["POST"])
@login_required
@role_required("Admin", "Investigator")
def upload_csv():
    """
    POST /api/data/upload
    Ingests batch CSV, validates completeness, scores rows, and exports for Power BI.
    """
    if "merchants_csv" not in request.files:
        return jsonify({"error": "Missing file field 'merchants_csv'"}), 400
        
    file = request.files["merchants_csv"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        df_upload = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse CSV file: {str(exc)}"}), 400

    required = ["merchant_id", "business_category", "registration_date", "bank_account_id", "device_id", "ip_address"]
    missing = [c for c in required if c not in df_upload.columns]
    if missing:
        return jsonify({"error": f"CSV missing required columns: {missing}"}), 400

    optional_fields = ["business_age_days", "phone_verified", "email_verified", "document_verified"]
    for field in optional_fields:
        if field not in df_upload.columns:
            df_upload[field] = pd.Series([None] * len(df_upload), index=df_upload.index, dtype=object)
        else:
            df_upload[field] = df_upload[field].astype(object).where(df_upload[field].notna(), None)
    if "transaction_count" not in df_upload.columns:
        df_upload["transaction_count"] = 0
    else:
        df_upload["transaction_count"] = df_upload["transaction_count"].fillna(0)

    completeness_fields = ["business_category", "bank_account_id", "device_id", "ip_address"] + optional_fields
    populated_cells = sum(
        int(df_upload[field].notna().sum()) for field in completeness_fields
    )
    data_completeness = round(populated_cells / max(1, len(df_upload) * len(completeness_fields)) * 100, 1)

    engine = get_engine()
    results = []
    for _, row in df_upload.iterrows():
        m_dict = row.to_dict()
        assessment = engine.score_merchant(m_dict)
        results.append(assessment)

    # Refresh Power BI exports
    try:
        export_all_for_powerbi()
    except Exception as exp_err:
        logger.warning(f"Power BI export refresh notice: {exp_err}")

    # Record in Audit Log
    user = session.get("user")
    if user:
        insert_audit_log(
            user_id=user["id"],
            username=user["username"],
            action=f"Batch uploaded & scored {len(df_upload)} merchants (Completeness: {data_completeness}%)"
        )

    return jsonify({
        "status": "success",
        "rows_processed": len(df_upload),
        "data_completeness": f"{data_completeness}%",
        "sample_assessments": results[:5]
    }), 200


@app.route("/api/sample-csv", methods=["GET"])
@login_required
def download_sample_csv():
    """Generates a downloadable sample CSV template for batch onboarding."""
    sample_records = [
        {
            "merchant_id": "M_BATCH_SAMPLE_01",
            "business_category": "Electronics",
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bank_account_id": "BANK_SAMPLE_01",
            "device_id": "DEV_SAMPLE_01",
            "ip_address": "192.168.1.101",
            "business_age_days": 180,
            "phone_verified": 1,
            "email_verified": 1,
            "document_verified": 1,
            "transaction_count": 0
        },
        {
            "merchant_id": "M_BATCH_SAMPLE_02",
            "business_category": "Fashion",
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bank_account_id": "BANK_0028",
            "device_id": "DEV_SAMPLE_02",
            "ip_address": "192.168.1.102",
            "business_age_days": 45,
            "phone_verified": 1,
            "email_verified": 0,
            "document_verified": 0,
            "transaction_count": 0
        },
        {
            "merchant_id": "M_BATCH_SAMPLE_03",
            "business_category": "Grocery",
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bank_account_id": "BANK_SAMPLE_03",
            "device_id": "DEV_0003",
            "ip_address": "192.168.1.103",
            "business_age_days": 365,
            "phone_verified": 1,
            "email_verified": 1,
            "document_verified": 1,
            "transaction_count": 0
        }
    ]
    df_sample = pd.DataFrame(sample_records)
    csv_buffer = io.StringIO()
    df_sample.to_csv(csv_buffer, index=False)
    csv_output = csv_buffer.getvalue()

    return Response(
        csv_output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=merchant_upload_template.csv"}
    )


@app.route("/api/audit-logs", methods=["GET"])
@login_required
@role_required("Admin", "Investigator")
def api_audit_logs():
    """GET /api/audit-logs — JSON by default; CSV when ?format=csv"""
    limit = int(request.args.get("limit", 100))
    search = request.args.get("search")
    merchant_id = request.args.get("merchant_id")
    fmt = request.args.get("format", "json").lower()
    logs = get_audit_logs(limit=limit, merchant_id=merchant_id, search=search)
    if fmt == "csv":
        import csv as _csv
        buf = io.StringIO()
        fieldnames = ["audit_id", "user_id", "username", "action", "merchant_id", "created_at", "user_role"]
        writer = _csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in logs:
            writer.writerow({k: ("" if row.get(k) is None else str(row.get(k, ""))) for k in fieldnames})
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment;filename=audit_log.csv"})
    return jsonify({"logs": logs, "count": len(logs)}), 200


@app.route("/api/retrain", methods=["POST"])
@login_required
@role_required("Admin")
def trigger_retraining():
    """
    POST /api/retrain
    Executes model retraining pipeline, saves versioned model artifact,
    and logs the maintenance event in the audit trail.
    """
    user = session["user"]
    try:
        results = retrain_model_pipeline()
        insert_audit_log(
            user_id=user["id"],
            username=user["username"],
            action="Triggered model retraining pipeline: XGBoost with graph priors updated"
        )
        return jsonify({
            "status": "success",
            "message": "Model retraining pipeline completed successfully and artifacts archived.",
            "results": results
        }), 200
    except Exception as e:
        logger.exception("Error during model retraining pipeline")
        return jsonify({"error": f"Model retraining failed: {str(e)}"}), 500


@app.route("/api/test/cleanup", methods=["POST"])
@login_required
@role_required("Admin")
def cleanup_test_records():
    """
    POST /api/test/cleanup
    Safely purges temporary test registrations (data_source = 'test') without touching baseline 5,000 merchants.
    """
    user = session["user"]
    try:
        deleted = delete_test_merchants(user_id=user["id"], username=user["username"])
        return jsonify({
            "status": "success",
            "deleted_count": deleted,
            "message": f"Successfully purged {deleted} temporary test merchant records."
        }), 200
    except Exception as e:
        logger.exception("Error deleting test merchants")
        return jsonify({"error": f"Test data cleanup failed: {str(e)}"}), 500

@app.route("/api/my-ip", methods=["GET"])
@login_required
def my_ip():
    """Returns the caller's IP address (honours X-Forwarded-For)."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "127.0.0.1")
    return jsonify({"ip": ip}), 200


@app.route("/api/export/merchants.csv", methods=["GET"])
@login_required
@role_required("Admin", "Investigator")
def export_merchants_csv():
    """Power BI: flat merchant+assessment export, excluding test rows."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT m.merchant_id, m.business_category, m.business_age_days,
                   m.registration_date, m.phone_verified, m.email_verified,
                   m.document_verified, m.transaction_count, m.fraud_label,
                   m.data_source, ra.risk_score, ra.decision, ra.evaluated_at
            FROM merchants m
            LEFT JOIN (
                SELECT ra1.*
                FROM risk_assessments ra1
                JOIN (SELECT merchant_id, MAX(evaluated_at) AS evaluated_at
                      FROM risk_assessments GROUP BY merchant_id) latest
                  ON ra1.merchant_id = latest.merchant_id AND ra1.evaluated_at = latest.evaluated_at
            ) ra ON ra.merchant_id = m.merchant_id
            WHERE m.data_source <> 'test'
            ORDER BY m.registration_date DESC
            LIMIT 50000
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        import csv as _csv
        buf = io.StringIO()
        if rows:
            writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else str(v)) for k, v in r.items()})
        insert_audit_log(user_id=session["user"]["id"], username=session["user"]["username"],
                         action="Exported merchants.csv for Power BI")
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment;filename=merchants.csv"})
    except Exception:
        logger.exception("merchants.csv export failed")
        return jsonify({"error": "Export failed"}), 500


@app.route("/api/export/alerts.csv", methods=["GET"])
@login_required
@role_required("Admin", "Investigator")
def export_alerts_csv():
    """Power BI: flat alerts export, excluding test-sourced merchants."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT a.alert_id, a.merchant_id, a.severity, a.alert_type,
                   a.status, a.details, a.created_at,
                   ic.case_id, ic.status AS case_status, ic.decision AS case_decision
            FROM alerts a
            JOIN merchants m ON m.merchant_id = a.merchant_id
            LEFT JOIN investigation_cases ic ON ic.alert_id = a.alert_id
            WHERE m.data_source <> 'test'
            ORDER BY a.created_at DESC
            LIMIT 50000
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        import csv as _csv
        buf = io.StringIO()
        if rows:
            writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else str(v)) for k, v in r.items()})
        insert_audit_log(user_id=session["user"]["id"], username=session["user"]["username"],
                         action="Exported alerts.csv for Power BI")
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment;filename=alerts.csv"})
    except Exception:
        logger.exception("alerts.csv export failed")
        return jsonify({"error": "Export failed"}), 500


@app.route("/api/export/decisions.csv", methods=["GET"])
@login_required
@role_required("Admin", "Investigator", "Analyst")
def export_decisions_csv():
    """Power BI: risk decision history export, excluding test rows."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT ra.assessment_id, ra.merchant_id, ra.risk_score, ra.decision,
                   ra.evaluated_at, ra.reasons, ra.data_completeness,
                   m.business_category, m.data_source
            FROM risk_assessments ra
            JOIN merchants m ON m.merchant_id = ra.merchant_id
            WHERE m.data_source <> 'test'
            ORDER BY ra.evaluated_at DESC
            LIMIT 50000
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        import csv as _csv
        buf = io.StringIO()
        if rows:
            writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ("" if v is None else str(v)) for k, v in r.items()})
        insert_audit_log(user_id=session["user"]["id"], username=session["user"]["username"],
                         action="Exported decisions.csv for Power BI")
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment;filename=decisions.csv"})
    except Exception:
        logger.exception("decisions.csv export failed")
        return jsonify({"error": "Export failed"}), 500


if __name__ == "__main__":
    init_db()
    preferred_port = int(os.getenv("APP_PORT", "5000"))
    app_port = find_available_port(preferred_port)
    if app_port != preferred_port:
        logger.warning(
            "Port %s is occupied; starting FraudGuard on port %s instead.",
            preferred_port,
            app_port,
        )
    logger.info("FraudGuard is available at http://127.0.0.1:%s", app_port)
    app.run(host="127.0.0.1", port=app_port, debug=False)