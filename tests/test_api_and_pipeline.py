"""
Unit & Integration Tests for Fraud Detection API, Database, and Pipeline
Tests API endpoints, graph priors, missing optional columns, and error handling.
"""

import os
import json
import io
import re
import pytest
import pandas as pd
import app as app_module
from app import app
from src.process_new_data import validate_dataframe, process_incoming_file

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user"] = {"id": 1, "username": "admin", "email": "admin@fraudshield.internal", "role": "Admin"}
        yield client

@pytest.fixture
def registration_setup(monkeypatch):
    network_by_bank = {}
    persisted_merchants = []
    lookup_calls = []

    def lookup_network(bank_id, device_id, ip_address, exclude_merchant_id=None, limit=20):
        lookup_calls.append((bank_id, device_id, ip_address, exclude_merchant_id, limit))
        return network_by_bank.get(bank_id, {
            "connected_entities": [
                {"entity_id": bank_id, "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT"},
                {"entity_id": device_id, "entity_type": "device", "relationship": "USES_DEVICE"},
                {"entity_id": ip_address, "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP"}
            ],
            "shared_merchants": [],
            "bank_share_count": 0,
            "bank_fraud_peers": 0,
            "device_share_count": 0,
            "device_fraud_peers": 0,
            "ip_share_count": 0,
            "neighbours_flagged_fraud": 0
        })

    monkeypatch.setattr(app_module, "lookup_entity_network", lookup_network)
    monkeypatch.setattr(app_module, "insert_merchant", lambda record: persisted_merchants.append(record.copy()) or True)
    monkeypatch.setattr(app_module, "insert_relationships_bulk", lambda relationships: True)
    monkeypatch.setattr(app_module, "insert_risk_assessment", lambda assessment: True)
    monkeypatch.setattr(app_module, "insert_alert", lambda alert: True)
    return network_by_bank, persisted_merchants, lookup_calls


def genuine_registration(merchant_id, bank_account_id, device_id, ip_address, **overrides):
    payload = {
        "merchant_id": merchant_id,
        "business_category": "Grocery",
        "bank_account_id": bank_account_id,
        "device_id": device_id,
        "ip_address": ip_address,
        "transaction_count": 0,
        "business_age_days": 400,
        "phone_verified": 1,
        "email_verified": 1,
        "document_verified": 1,
        "data_source": "test"
    }
    payload.update(overrides)
    return payload

def test_health_check(client):
    """Verifies API health check."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_dashboard_renders_analytics_controls_charts_and_states(client):
    response = client.get("/dashboard")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Merchant risk analytics" in html
    assert 'href="/onboarding"' in html
    assert 'id="dateFrom"' in html and 'id="dateTo"' in html
    assert 'id="categoryFilter"' in html
    assert "chart.js@4.4.7" in html
    for chart in ["decision", "daily", "category", "histogram", "alerts", "shared"]:
        assert f'id="{chart}State"' in html
        assert f'id="{chart}Visual"' in html
    assert "No data matches these filters." in html
    assert "This chart could not be rendered." in html
    assert "fg-theme-change" in html
    assert "Latest high-risk merchants" in html
    assert "encodeURIComponent(merchant.merchant_id)" in html


def dashboard_summary_payload():
    return {
        "kpis": {"total_merchants": 2, "cold_start_count": 1, "hold_count": 1, "average_risk_score": 0.8, "open_alerts": 1},
        "previous_kpis": {"total_merchants": 1, "cold_start_count": 1, "hold_count": 0, "average_risk_score": 0.5, "open_alerts": 0},
        "decisions": [{"decision": "Hold", "count": 1}, {"decision": "Approve", "count": 1}],
        "daily_registrations": [{"day": "2026-11-06", "decision": "Hold", "count": 1}],
        "categories": [{"category": "Grocery", "average_risk_score": 0.8, "hold_rate": 0.5, "merchant_count": 2}],
        "risk_histogram": [{"bucket": 8, "count": 1}],
        "alerts": [{"severity": "HIGH", "status": "OPEN", "count": 1}],
        "shared_entities": [{"entity_type": "device", "entity_id": "DEVICE_SHARED", "shared_size": 2}],
        "latest_high_risk": [{"merchant_id": "M_HOLD", "category": "Grocery", "risk_score": 0.8, "decision": "Hold", "evaluated_at": "2026-11-06 12:00:00"}],
        "category_options": [{"category": "Grocery"}]
    }


def test_dashboard_summary_requires_login():
    with app.test_client() as unauthenticated_client:
        response = unauthenticated_client.get("/api/dashboard/summary")
    assert response.status_code == 401


def test_dashboard_summary_returns_aggregates_and_forwards_filters(client, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "get_dashboard_summary", lambda *args: calls.append(args) or dashboard_summary_payload())

    response = client.get("/api/dashboard/summary?date_from=2026-10-01&date_to=2026-10-31&category=Grocery")
    data = response.get_json()

    assert response.status_code == 200
    assert calls == [("2026-10-01", "2026-10-31", "Grocery")]
    assert data["kpis"]["total_merchants"] == 2
    assert data["decisions"][0]["decision"] == "Hold"
    assert data["daily_registrations"]
    assert data["categories"]
    assert data["risk_histogram"]
    assert data["alerts"]
    assert data["shared_entities"]
    assert data["latest_high_risk"][0]["merchant_id"] == "M_HOLD"
    assert data["filters"] == {"date_from": "2026-10-01", "date_to": "2026-10-31", "category": "Grocery"}


def test_dashboard_summary_sql_excludes_test_merchants(monkeypatch):
    import src.db as db_module

    queries = []

    class FakeCursor:
        def execute(self, query, params=()):
            queries.append(query)

        def fetchone(self):
            return {}

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConnection:
        def cursor(self, **kwargs):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(db_module, "get_db_connection", lambda: FakeConnection())
    db_module.get_dashboard_summary("2026-10-01", "2026-10-31", "Grocery")

    merchant_queries = [query for query in queries if "FROM merchants m" in query or "JOIN merchants m" in query]
    assert merchant_queries
    assert all("m.data_source <> 'test'" in query for query in merchant_queries)
    assert any("am.data_source <> 'test'" in query for query in queries)
    daily_query = next(query for query in queries if "AS day" in query)
    date_expression = "DATE_FORMAT(m.registration_date, '%Y-%m-%d')"
    assert f"GROUP BY {date_expression}" in daily_query
    shared_query = next(query for query in queries if "AS shared_size" in query)
    assert "GROUP_CONCAT(DISTINCT m.merchant_id" in shared_query
    assert "HAVING COUNT(DISTINCT r.merchant_id) > 1" not in shared_query


def test_onboarding_page_has_fields_validation_and_result_states(client):
    response = client.get("/onboarding")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    for field in [
        "business_category", "bank_account_id", "device_id", "ip_address",
        "transaction_count", "business_age_days", "phone_verified",
        "email_verified", "document_verified"
    ]:
        assert f'id="{field}"' in html
    assert "X-CSRFToken" in html
    assert "resultLoading" in html
    assert "resultError" in html
    assert "View full assessment" in html


def test_risk_detail_shows_assessment_and_graph_evidence(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_merchant_risk", lambda merchant_id: {
        "merchant_id": merchant_id,
        "risk_score": 0.81,
        "decision": "Hold",
        "model_version": "test-model",
        "evaluated_at": "2026-11-06 12:00:00",
        "reasons": ["Bank account is shared with 2 other merchants, 1 flagged as fraud", "New business"],
        "graph_evidence": {
            "bank_share_count": 2,
            "device_share_count": 1,
            "ip_share_count": 3,
            "flagged_peer_merchants": 1,
            "community_fraud_risk": 0.4,
            "connected_entities": [{"entity_type": "bank_account", "entity_id": "BANK_SHARED", "relationship": "HAS_BANK_ACCOUNT"}],
            "shared_merchants": [{"merchant_id": "M_FLAGGED", "entity_id": "BANK_SHARED", "entity_type": "bank_account", "fraud_label": 1}]
        }
    })

    response = client.get("/risk/M_RISK_TEST")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "M_RISK_TEST" in html
    assert "test-model" in html
    assert "2026-11-06 12:00:00" in html
    assert "Shared bank" in html and "Shared device" in html and "Shared IP" in html
    assert "BANK_SHARED" in html and "M_FLAGGED" in html
    assert "Open in Fraud Network" in html
    assert "disabled" in html
    assert 'href="/network/M_RISK_TEST"' in html


def test_unknown_risk_page_shows_not_found_state(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_merchant_risk", lambda merchant_id: None)

    response = client.get("/risk/MISSING_ASSESSMENT")
    html = response.get_data(as_text=True)

    assert response.status_code == 404
    assert "No risk assessment was found" in html
    assert "Find another merchant" in html


def test_network_page_has_visualization_and_unknown_graph_states(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_merchant", lambda merchant_id: None)

    response = client.get("/network/M_UNKNOWN_GRAPH")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "vis-network" in html
    assert "networkLoading" in html
    assert "networkEmpty" in html
    assert "networkError" in html
    assert "Select a node to view details" in html


def test_alert_page_filters_and_empty_error_states(client, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "get_investigation_alerts", lambda **kwargs: calls.append(kwargs) or [])

    response = client.get("/alerts?status=OPEN&severity=HIGH")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert calls == [{"status": "OPEN", "severity": "HIGH"}]
    assert "No alerts match these filters" in html
    assert 'name="status"' in html
    assert 'name="severity"' in html

    monkeypatch.setattr(app_module, "get_investigation_alerts", lambda **kwargs: None)
    error_response = client.get("/alerts")
    assert "Alerts could not be loaded" in error_response.get_data(as_text=True)


def test_case_update_validates_fields_and_returns_updated_case(client, monkeypatch):
    updates = []

    def update_case(**kwargs):
        updates.append(kwargs)
        return {"case_id": kwargs["case_id"], "merchant_id": "M_CASE", "status": kwargs["status"], "decision": kwargs["decision"], "notes": kwargs["notes"]}

    monkeypatch.setattr(app_module, "update_investigation_case", update_case)

    invalid_status = client.patch("/api/investigation-cases/CASE_1", json={"status": "Working"})
    invalid_decision = client.patch("/api/investigation-cases/CASE_1", json={"status": "Open", "decision": "Fraud"})
    invalid_notes = client.patch("/api/investigation-cases/CASE_1", json={"status": "Open", "notes": 123})
    assert invalid_status.status_code == 400
    assert invalid_decision.status_code == 400
    assert invalid_notes.status_code == 400
    assert updates == []

    updated = client.patch("/api/investigation-cases/CASE_1", json={
        "status": "In review",
        "decision": "Needs more info",
        "notes": "Request supporting documents"
    })
    assert updated.status_code == 200
    assert updated.get_json()["case"]["status"] == "In review"
    assert updates[0]["username"] == "admin"


def test_analyst_can_read_case_but_cannot_update(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_investigation_case", lambda case_id: {"case_id": case_id, "merchant_id": "M_CASE", "status": "Open"})
    with app.test_client() as analyst_client:
        with analyst_client.session_transaction() as sess:
            sess["user"] = {"id": 17, "username": "analyst_viewer", "role": "Analyst"}
        read_response = analyst_client.get("/api/investigation-cases/CASE_READ")
        update_response = analyst_client.patch("/api/investigation-cases/CASE_READ", json={"status": "Closed"})

    assert read_response.status_code == 200
    assert update_response.status_code == 403


def test_case_update_writes_audit_entry_atomically(monkeypatch):
    import src.db as db_module

    executed = []
    case = {"case_id": "CASE_AUDIT", "merchant_id": "M_AUDIT", "status": "Open", "decision": None, "notes": None}

    class FakeCursor:
        def execute(self, query, params):
            executed.append((query, params))

        def fetchone(self):
            return case.copy()

        def close(self):
            pass

    class FakeConnection:
        committed = False
        rolled_back = False

        def cursor(self, **kwargs):
            return FakeCursor()

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def close(self):
            pass

    connection = FakeConnection()
    monkeypatch.setattr(db_module, "get_db_connection", lambda: connection)

    updated = db_module.update_investigation_case(
        "CASE_AUDIT", "Closed", "Confirm fraud", "Verified linked account", 22, "investigator"
    )

    audit_query, audit_params = next(
        (query, params) for query, params in executed if "INSERT INTO audit_log" in query
    )
    assert "merchant_id" in audit_query
    assert audit_params[0:2] == (22, "investigator")
    assert audit_params[3] == "M_AUDIT"
    assert updated["status"] == "Closed"
    assert connection.committed is True
    assert connection.rolled_back is False


def test_json_post_requires_csrf_header_from_page(client):
    csrf_enabled = app.config.get("WTF_CSRF_ENABLED", False)
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        page = client.get("/dashboard").get_data(as_text=True)
        token_match = re.search(r'name="csrf-token" content="([^"]+)"', page)
        assert token_match is not None
        csrf_token = token_match.group(1)
        payload = {
            "business_category": "Grocery",
            "business_age_days": 400,
            "phone_verified": 1,
            "email_verified": 1,
            "document_verified": 1,
            "transaction_count": 0,
            "bank_share_count": 0,
            "device_share_count": 0,
            "ip_share_count": 0,
            "connected_component_size": 1,
            "neighbours_flagged_fraud": 0,
            "louvain_community_fraud_rate": 0.0
        }

        rejected = client.post("/api/fraud/predict", json=payload)
        accepted = client.post(
            "/api/fraud/predict",
            json=payload,
            headers={"X-CSRFToken": csrf_token}
        )

        assert rejected.status_code == 400
        assert accepted.status_code == 200
    finally:
        app.config["WTF_CSRF_ENABLED"] = csrf_enabled

def test_predict_fraud_direct(client):
    """Tests POST /api/fraud/predict scoring."""
    payload = {
        "business_category": "Electronics",
        "business_age_days": 10,
        "phone_verified": 0,
        "email_verified": 1,
        "document_verified": 0,
        "transaction_count": 0,  # Cold-start merchant
        "bank_share_count": 5,   # High graph degree
        "device_share_count": 4,
        "ip_share_count": 2,
        "connected_component_size": 8,
        "neighbours_flagged_fraud": 3,
        "louvain_community_fraud_rate": 0.85
    }
    res = client.post("/api/fraud/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "risk_score" in data
    assert "decision" in data
    assert "reasons" in data
    assert data["decision"] in ["Hold", "Review"]
    assert data["risk_score"] > 0.40


def test_public_demo_predict_uses_fixed_synthetic_input_without_persistence(monkeypatch):
    scored_profiles = []
    persistence_functions = (
        "insert_merchant",
        "insert_relationships_bulk",
        "insert_risk_assessment",
        "insert_alert",
        "insert_audit_log",
        "export_all_for_powerbi",
        "get_db_connection",
    )

    class DemoEngine:
        def score_merchant(self, profile):
            scored_profiles.append(profile)
            return {
                "decision": "Review",
                "risk_score": 0.61,
                "reasons": ["Shared device connection", "Limited operating history"],
                "merchant_id": "SHOULD_NOT_BE_RETURNED",
            }

    def fail_if_persisted(*args, **kwargs):
        raise AssertionError("The public demo must not persist assessment data")

    app_module._DEMO_REQUESTS.clear()
    monkeypatch.setattr(app_module, "get_engine", lambda: DemoEngine())
    for function_name in persistence_functions:
        monkeypatch.setattr(app_module, function_name, fail_if_persisted)

    with app.test_client() as public_client:
        response = public_client.post("/api/demo/predict", json={"scenario": "shared_device"})

    payload = response.get_json()
    assert response.status_code == 200
    assert payload == {
        "decision": "Review",
        "risk_score": 0.61,
        "reasons": ["Shared device connection", "Limited operating history"],
    }
    assert len(scored_profiles) == 1
    assert scored_profiles[0]["device_share_count"] == 3
    assert "merchant_id" not in scored_profiles[0]
    assert "SHOULD_NOT_BE_RETURNED" not in response.get_data(as_text=True)


def test_landing_metrics_use_aggregate_and_saved_results_and_hide_missing(monkeypatch):
    monkeypatch.setattr(app_module, "load_dataset_stats", lambda: ({
        "total_merchants": 123,
        "cold_start_percentage": 42.5,
        "fraud_percentage": 6.25,
    }, None))
    monkeypatch.setattr(app_module, "load_model_metrics", lambda: ([
        {"model_name": "xgboost_with_graph", "evaluation_group": "all_merchants", "pr_auc": "0.6500"},
        {"model_name": "xgboost_with_graph", "evaluation_group": "cold_start_group", "pr_auc": "0.6746"},
    ], None))

    metrics = app_module.load_landing_metrics()

    assert [metric["label"] for metric in metrics] == [
        "Simulated merchant profiles",
        "Cold-start profiles",
        "Fraud prevalence in sample",
        "Cold-start PR-AUC",
    ]
    assert [metric["value"] for metric in metrics] == [123, 42.5, 6.25, 0.6746]

    monkeypatch.setattr(app_module, "load_dataset_stats", lambda: (None, "unavailable"))
    monkeypatch.setattr(app_module, "load_model_metrics", lambda: ([], "unavailable"))
    assert app_module.load_landing_metrics() == []


def test_public_demo_predict_validates_scenario_and_is_csrf_exempt(monkeypatch):
    class DemoEngine:
        def score_merchant(self, profile):
            return {"decision": "Hold", "risk_score": 0.8, "reasons": ["Simulated signal"]}

    csrf_enabled = app.config.get("WTF_CSRF_ENABLED", True)
    app.config["WTF_CSRF_ENABLED"] = True
    app_module._DEMO_REQUESTS.clear()
    monkeypatch.setattr(app_module, "get_engine", lambda: DemoEngine())
    try:
        with app.test_client() as public_client:
            missing = public_client.post("/api/demo/predict", json={})
            extra = public_client.post("/api/demo/predict", json={"scenario": "genuine", "merchant_id": "M_123"})
            unhashable = public_client.post("/api/demo/predict", json={"scenario": []})
            malformed = public_client.post(
                "/api/demo/predict",
                data="{",
                content_type="application/json",
            )
            accepted = public_client.post("/api/demo/predict", json={"scenario": "fraud_ring"})
    finally:
        app.config["WTF_CSRF_ENABLED"] = csrf_enabled

    assert missing.status_code == 400
    assert extra.status_code == 400
    assert unhashable.status_code == 400
    assert malformed.status_code == 400
    assert accepted.status_code == 200


def test_public_demo_predict_is_rate_limited(monkeypatch):
    class DemoEngine:
        def score_merchant(self, profile):
            return {"decision": "Approve", "risk_score": 0.12, "reasons": []}

    app_module._DEMO_REQUESTS.clear()
    monkeypatch.setattr(app_module, "get_engine", lambda: DemoEngine())
    with app.test_client() as public_client:
        responses = [
            public_client.post("/api/demo/predict", json={"scenario": "genuine"})
            for _ in range(app_module.DEMO_RATE_LIMIT)
        ]
        limited = public_client.post("/api/demo/predict", json={"scenario": "genuine"})

    assert all(response.status_code == 200 for response in responses)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_cold_start_genuine_scoring(client):
    """Verifies that cold-start genuine merchant with clean graph is Approved."""
    payload = {
        "business_category": "Grocery",
        "business_age_days": 400,
        "phone_verified": 1,
        "email_verified": 1,
        "document_verified": 1,
        "transaction_count": 0,  # Cold-start
        "bank_share_count": 0,   # Clean graph
        "device_share_count": 0,
        "ip_share_count": 0,
        "connected_component_size": 1,
        "neighbours_flagged_fraud": 0,
        "louvain_community_fraud_rate": 0.0
    }
    res = client.post("/api/fraud/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["decision"] == "Approve"
    assert data["risk_score"] < 0.35

def test_validation_missing_optional_columns():
    """
    Spec Test 1: Upload a CSV with missing optional columns
    and confirm it still passes validation with default neutral imputations.
    """
    df = pd.DataFrame([{
        "merchant_id": "M_TEST_001",
        "business_category": "Fashion",
        "registration_date": "2024-05-01 10:00:00",
        "bank_account_id": "BANK_TEST_001",
        "device_id": "DEV_TEST_001",
        "ip_address": "192.0.2.1"
        # Optional columns omitted
    }])
    valid_df, rejected_df, completeness = validate_dataframe(df)
    assert len(valid_df) == 1
    assert len(rejected_df) == 0
    assert "transaction_count" in valid_df.columns
    assert valid_df["transaction_count"].iloc[0] == 0
    for field in ["business_age_days", "phone_verified", "email_verified", "document_verified"]:
        assert valid_df[field].iloc[0] is None
    assert completeness == 50.0

def test_validation_wrong_column_name():
    """
    Spec Test 3: Upload a CSV with a wrong column name and confirm clear error message.
    """
    bad_df = pd.DataFrame([{
        "merchant_id": "M_BAD_001",
        "wrong_category_col": "Fashion",
        "registration_date": "2024-05-01 10:00:00",
        "bank_account_id": "BANK_BAD_001"
    }])
    with pytest.raises(ValueError) as excinfo:
        validate_dataframe(bad_df)
    assert "missing mandatory columns" in str(excinfo.value).lower()


def test_registration_unique_genuine_merchant_is_approved(client, registration_setup):
    _, persisted_merchants, lookup_calls = registration_setup
    payload = genuine_registration("M_TEST_CLEAN", "BANK_TEST_CLEAN", "DEV_TEST_CLEAN", "192.0.2.10")

    response = client.post("/api/merchant/register", json=payload)

    assert response.status_code == 201
    assessment = response.get_json()
    assert assessment["decision"] == "Approve"
    assert lookup_calls[0][0:3] == (payload["bank_account_id"], payload["device_id"], payload["ip_address"])
    assert lookup_calls[0][4] == 20
    assert persisted_merchants[0]["data_source"] == "test"


def test_registration_flagged_shared_bank_scores_higher(client, registration_setup):
    network_by_bank, _, _ = registration_setup
    clean_payload = genuine_registration("M_TEST_BASE", "BANK_TEST_BASE", "DEV_TEST_BASE", "192.0.2.11")
    flagged_payload = genuine_registration("M_TEST_SHARED", "BANK_TEST_SHARED", "DEV_TEST_SHARED", "192.0.2.12")
    network_by_bank["BANK_TEST_SHARED"] = {
        "connected_entities": [
            {"entity_id": "BANK_TEST_SHARED", "entity_type": "bank_account", "relationship": "HAS_BANK_ACCOUNT"},
            {"entity_id": "DEV_TEST_SHARED", "entity_type": "device", "relationship": "USES_DEVICE"},
            {"entity_id": "192.0.2.12", "entity_type": "ip_address", "relationship": "OPERATES_FROM_IP"}
        ],
        "shared_merchants": [{
            "entity_id": "BANK_TEST_SHARED",
            "entity_type": "bank_account",
            "merchant_id": "M_FLAGGED_PEER",
            "fraud_label": 1,
            "business_category": "Grocery",
            "registration_date": "2024-01-01 00:00:00"
        }],
        "bank_share_count": 1,
        "bank_fraud_peers": 1,
        "device_share_count": 0,
        "device_fraud_peers": 0,
        "ip_share_count": 0,
        "neighbours_flagged_fraud": 1
    }

    clean_response = client.post("/api/merchant/register", json=clean_payload)
    flagged_response = client.post("/api/merchant/register", json=flagged_payload)

    assert clean_response.status_code == 201
    assert flagged_response.status_code == 201
    clean_assessment = clean_response.get_json()
    flagged_assessment = flagged_response.get_json()
    assert clean_assessment["decision"] == "Approve"
    assert flagged_assessment["risk_score"] > clean_assessment["risk_score"]
    evidence = flagged_assessment["graph_evidence"]
    assert evidence["connected_entities"][0]["entity_id"] == "BANK_TEST_SHARED"
    assert evidence["shared_merchants"][0]["merchant_id"] == "M_FLAGGED_PEER"
    assert len(evidence["shared_merchants"]) <= 20
    assert flagged_assessment["reasons"][0] == "Bank account is shared with 1 other merchant, 1 flagged as fraud"


def test_registration_missing_optional_fields_is_not_held(client, registration_setup):
    _, persisted_merchants, _ = registration_setup
    payload = genuine_registration(
        "M_TEST_MISSING",
        "BANK_TEST_MISSING",
        "DEV_TEST_MISSING",
        "192.0.2.13",
        business_age_days=None,
        phone_verified=None,
        email_verified=None,
        document_verified=None
    )

    response = client.post("/api/merchant/register", json=payload)

    assert response.status_code == 201
    assessment = response.get_json()
    assert assessment["decision"] != "Hold"
    assert assessment["data_completeness"] == "50.0%"
    assert assessment["missing_fields"] == [
        "business_age_days", "document_verified", "email_verified", "phone_verified"
    ]
    assert persisted_merchants[0]["business_age_days"] is None
    assert persisted_merchants[0]["phone_verified"] is None
    assert not any("unverified" in reason.lower() or "operating age" in reason.lower() for reason in assessment["reasons"])


def test_csv_upload_preserves_missing_optional_fields(client, monkeypatch):
    scored_records = []

    class CaptureEngine:
        def score_merchant(self, record):
            scored_records.append(record)
            return {"decision": "Approve", "risk_score": 0.1}

    monkeypatch.setattr(app_module, "get_engine", lambda: CaptureEngine())
    monkeypatch.setattr(app_module, "export_all_for_powerbi", lambda: {})
    csv_content = (
        "merchant_id,business_category,registration_date,bank_account_id,device_id,ip_address\n"
        "M_BATCH_TEST,Grocery,2024-05-01 10:00:00,BANK_BATCH_TEST,DEV_BATCH_TEST,192.0.2.14\n"
    )

    response = client.post(
        "/api/data/upload",
        data={"merchants_csv": (io.BytesIO(csv_content.encode("utf-8")), "merchants.csv")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert response.get_json()["data_completeness"] == "50.0%"
    assert len(scored_records) == 1
    for field in ["business_age_days", "phone_verified", "email_verified", "document_verified"]:
        assert scored_records[0][field] is None


def test_export_statistics_exclude_test_source_rows(tmp_path):
    from src.export_for_powerbi import export_all_for_powerbi

    processed_dir = tmp_path / "processed"
    synthetic_dir = tmp_path / "synthetic"
    output_dir = tmp_path / "exports"
    processed_dir.mkdir()
    synthetic_dir.mkdir()
    pd.DataFrame([
        {"merchant_id": "M_REAL", "registration_date": "2024-01-01", "data_source": "synthetic"},
        {"merchant_id": "M_TEST", "registration_date": "2024-01-01", "data_source": "test"}
    ]).to_csv(processed_dir / "merchants_with_graph.csv", index=False)

    result = export_all_for_powerbi(
        output_dir=str(output_dir),
        processed_dir=str(processed_dir),
        synthetic_dir=str(synthetic_dir)
    )

    exported_merchants = pd.read_csv(output_dir / "merchants_scored.csv")
    assert result["merchants_scored_rows"] == 1
    assert exported_merchants["merchant_id"].tolist() == ["M_REAL"]


def test_inserting_alert_creates_open_investigation_case(monkeypatch):
    import src.db as db_module

    executed = []

    class FakeCursor:
        def execute(self, query, params):
            executed.append((query, params))

        def close(self):
            pass

    class FakeConnection:
        committed = False

        def cursor(self):
            return FakeCursor()

        def commit(self):
            self.committed = True

        def close(self):
            pass

    connection = FakeConnection()
    monkeypatch.setattr(db_module, "get_db_connection", lambda: connection)
    alert = {
        "alert_id": "ALT_CASE_TEST",
        "merchant_id": "M_CASE_TEST",
        "severity": "HIGH",
        "alert_type": "TEST",
        "details": "test alert",
        "created_at": "2026-11-06 12:00:00"
    }

    assert db_module.insert_alert(alert) is True
    case_query, case_params = next(
        (query, params) for query, params in executed
        if "INSERT INTO investigation_cases" in query
    )
    assert "'Open'" in case_query
    assert case_params[1:3] == ("ALT_CASE_TEST", "M_CASE_TEST")
    assert connection.committed is True


def test_upload_page_renders_successfully(client):
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Batch Merchant Ingestion" in response.data
    assert b"merchants_csv" in response.data


def test_model_performance_page_renders_successfully(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert b"Model Performance" in response.data
    assert b"Model Architecture Comparison" in response.data
    assert b"Model-Agnostic Meta-Learning" in response.data


def test_audit_log_page_renders_successfully(client, monkeypatch):
    sample_logs = [
        {"audit_id": 1, "user_id": 1, "username": "admin", "action": "Test Action", "merchant_id": "M_TEST_01", "created_at": "2026-11-10 12:00:00", "user_role": "Admin"}
    ]
    monkeypatch.setattr(app_module, "get_audit_logs", lambda **kwargs: sample_logs)
    monkeypatch.setattr(app_module, "get_test_merchants_count", lambda: 3)
    response = client.get("/audit-log")
    assert response.status_code == 200
    assert b"Audit Log" in response.data
    assert b"Test Action" in response.data
    assert b"3 Test Registrations Active" in response.data


def test_api_sample_csv_download(client):
    response = client.get("/api/sample-csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"merchant_id,business_category,registration_date" in response.data


def test_api_audit_logs_json(client, monkeypatch):
    monkeypatch.setattr(app_module, "get_audit_logs", lambda **kwargs: [
        {"audit_id": 1, "username": "admin", "action": "Logged in", "merchant_id": None}
    ])
    response = client.get("/api/audit-logs")
    assert response.status_code == 200
    data = response.get_json()
    assert "logs" in data
    assert len(data["logs"]) == 1
    assert data["logs"][0]["action"] == "Logged in"


def test_api_test_cleanup(client, monkeypatch):
    monkeypatch.setattr(app_module, "delete_test_merchants", lambda **kwargs: 5)
    response = client.post("/api/test/cleanup", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["deleted_count"] == 5


def test_api_retrain_endpoint(client, monkeypatch):
    monkeypatch.setattr(app_module, "retrain_model_pipeline", lambda: {"status": "trained", "pr_auc": 0.65})
    monkeypatch.setattr(app_module, "insert_audit_log", lambda **kwargs: True)
    response = client.post("/api/retrain", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
