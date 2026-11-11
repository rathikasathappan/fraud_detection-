"""
Unit & Integration Tests for Authentication, RBAC, Rate Limiting & Security Basics
Verifies:
1. Login with hashed passwords from .env
2. Protected routes & logout
3. Detailed terminal failure logging & generic user messaging
4. Rate limiting (locks after 5 failed attempts)
5. RBAC role enforcement across /api/* endpoints
6. CSRF protection on form POSTs
7. Non-root database user enforcement
"""

import os
import re
import pytest
from flask import url_for
from dotenv import load_dotenv
import app as app_module
from app import app, reset_failed_attempts, _required_secret_key, find_available_port
from src.db import create_user, authenticate_user, get_user_by_username, ensure_default_admin

load_dotenv()

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client

def test_unauthenticated_dashboard_redirects(client):
    """Protected routes must redirect unauthenticated users to /login."""
    res = client.get("/dashboard")
    assert res.status_code == 302
    assert "/login?next=/dashboard" in res.headers["Location"]

def test_login_page_renders(client):
    """GET /login must render the login form with CSRF token input."""
    res = client.get("/login?next=%2Fdashboard")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Sign In to Platform" in html
    assert 'name="username"' in html
    assert 'name="password"' in html
    assert 'id="themeToggle"' in html
    assert 'action="/login?next=/dashboard"' in html


def test_public_landing_page_has_original_product_sections(monkeypatch):
    monkeypatch.setattr(app_module, "measure_database_response_ms", lambda: None)
    monkeypatch.setattr(app_module, "load_landing_metrics", lambda: [
        {"label": "Simulated merchant profiles", "value": 120, "suffix": ""},
        {"label": "Cold-start PR-AUC", "value": 0.6746, "suffix": ""},
    ])
    with app.test_client() as client:
        response = client.get("/")

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Cold-start merchant fraud detection" in html
    assert "Detect risky merchants" in html
    assert "Graph priors" in html
    assert "Explainable reasons" in html
    assert "Few-shot adaptation" in html
    assert "See it in action" in html
    assert "All examples and metrics use simulated data." in html
    assert 'href="/login"' in html
    assert "app-sidebar" not in html
    assert "bootstrap" not in html.lower()
    assert "cdn.jsdelivr.net" not in html
    assert "DB_PASSWORD" not in html
    assert "M_TEST_" not in html


def test_landing_hero_media_is_served_with_range_support(client):
    video_response = client.get("/static/video/hero.mp4")
    poster_response = client.get("/static/img/hero-poster.jpg")

    assert video_response.status_code == 200
    assert video_response.mimetype == "video/mp4"
    assert poster_response.status_code == 200
    assert poster_response.mimetype == "image/jpeg"

    range_response = client.get(
        "/static/video/hero.mp4",
        headers={"Range": "bytes=0-31"},
    )
    assert range_response.status_code == 206
    assert range_response.headers["Content-Range"].startswith("bytes 0-31/")


def test_authenticated_visitors_still_start_at_public_landing(client, monkeypatch):
    with client.session_transaction() as sess:
        sess["user"] = {"id": 5, "username": "layout_user", "role": "Admin"}

    monkeypatch.setattr(app_module, "measure_database_response_ms", lambda: None)
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Cold-start merchant fraud detection" in html
    assert "Login" in html
    assert "layout_user" not in html
    assert 'href="/dashboard"' not in html


@pytest.mark.parametrize(
    ("role", "visible_links", "hidden_links"),
    [
        ("Analyst", ["Dashboard", "Risk Assessment", "Fraud Network", "Model Performance"], ["Onboarding", "Alerts", "Upload"]),
        ("Investigator", ["Dashboard", "Onboarding", "Risk Assessment", "Fraud Network", "Alerts", "Model Performance"], ["Upload"]),
        ("Admin", ["Dashboard", "Onboarding", "Risk Assessment", "Fraud Network", "Alerts", "Upload", "Model Performance"], [])
    ]
)
def test_base_navigation_is_filtered_by_role(role, visible_links, hidden_links):
    with app.test_client() as role_client:
        with role_client.session_transaction() as sess:
            sess["user"] = {"id": 5, "username": "layout_user", "role": role}

        response = role_client.get("/dashboard")
        html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "layout_user" in html
    assert role in html
    for label in visible_links:
        assert f"<span>{label}</span>" in html
    for label in hidden_links:
        assert f"<span>{label}</span>" not in html

def test_invalid_login_fails(client):
    """POST /login with wrong password returns 401 and generic error message."""
    reset_failed_attempts("admin")
    res = client.post("/login", data={
        "username": "admin",
        "password": "wrong_dummy_password_123"
    })
    assert res.status_code == 401
    html = res.get_data(as_text=True)
    assert "Invalid username or password" in html

def test_valid_admin_login_and_session(client):
    """POST /login with valid admin credentials creates session and redirects to /."""
    admin_user = (os.getenv("ADMIN_USERNAME") or "admin").strip().strip("'\"")
    admin_pass = (os.getenv("ADMIN_PASSWORD") or "").strip().strip("'\"")
    
    ensure_default_admin()
    reset_failed_attempts(admin_user)

    res = client.post("/login", data={
        "username": admin_user,
        "password": admin_pass
    }, follow_redirects=True)
    
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Sign Out" in html
    assert admin_user in html

    # Verify session profile via /api/auth/me
    auth_res = client.get("/api/auth/me")
    assert auth_res.status_code == 200
    data = auth_res.get_json()
    assert data["authenticated"] is True
    assert data["user"]["username"] == admin_user
    assert data["user"]["role"] == "Admin"


def test_login_preserves_protected_destination_and_landing_is_still_public(client, monkeypatch):
    monkeypatch.setattr(app_module, "measure_database_response_ms", lambda: None)
    monkeypatch.setattr(app_module, "authenticate_user", lambda username, password: (
        {"id": 7, "username": username, "email": "analyst@example.test", "role": "Analyst"},
        None
    ))

    response = client.post("/login?next=/dashboard", data={"username": "analyst", "password": "valid"})
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")

    landing = client.get("/")
    assert landing.status_code == 200
    assert "Login" in landing.get_data(as_text=True)


def test_login_rejects_external_next_destination(client, monkeypatch):
    monkeypatch.setattr(app_module, "authenticate_user", lambda username, password: (
        {"id": 7, "username": username, "email": "analyst@example.test", "role": "Analyst"},
        None
    ))

    response = client.post("/login?next=%2F%2Fexample.com", data={"username": "analyst", "password": "valid"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_logout_clears_session(client):
    """GET /logout clears session and redirects to /login."""
    admin_user = (os.getenv("ADMIN_USERNAME") or "admin").strip().strip("'\"")
    admin_pass = (os.getenv("ADMIN_PASSWORD") or "").strip().strip("'\"")

    reset_failed_attempts(admin_user)
    client.post("/login", data={"username": admin_user, "password": admin_pass})
    
    res = client.get("/logout", follow_redirects=True)
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Sign In to Platform" in html
    assert "See the connections before fraud happens." in html
    assert "login-flow" in html

    auth_res = client.get("/api/auth/me")
    assert auth_res.status_code == 401
    assert auth_res.get_json()["authenticated"] is False

def test_failed_login_rate_limiting(client):
    """Account locks for 5 minutes after 5 consecutive failed login attempts (HTTP 429)."""
    test_user = "rate_limit_test_user"
    reset_failed_attempts(test_user)

    for _ in range(5):
        res = client.post("/login", data={"username": test_user, "password": "wrong_password"})
        assert res.status_code == 401

    # 6th attempt must be blocked by rate limiter
    locked_res = client.post("/login", data={"username": test_user, "password": "wrong_password"})
    assert locked_res.status_code == 429
    html = locked_res.get_data(as_text=True)
    assert "Account is temporarily locked" in html

def test_api_endpoints_require_login_and_role():
    """All /api/* endpoints require authenticated session and enforce RBAC roles."""
    with app.test_client() as unauth_client:
        # Unauthenticated calls must return 401
        assert unauth_client.get("/api/alerts").status_code == 401
        assert unauth_client.get("/api/health").status_code == 401
        assert unauth_client.post("/api/fraud/predict", json={}).status_code == 401

    # Role enforcement test: Analyst cannot access Admin-only upload route
    with app.test_client() as analyst_client:
        with analyst_client.session_transaction() as sess:
            sess["user"] = {"id": 99, "username": "analyst1", "role": "Analyst"}
        
        # Analyst can access read-only risk lookup
        res_ok = analyst_client.get("/api/alerts")
        assert res_ok.status_code == 200

        # Analyst is forbidden from admin upload
        res_forbidden = analyst_client.post("/api/data/upload")
        assert res_forbidden.status_code == 403
        assert "Forbidden" in res_forbidden.get_json()["error"]


def test_every_route_requires_authentication_or_redirect():
    csrf_enabled = app.config.get("WTF_CSRF_ENABLED", False)
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        route_requests = []
        with app.test_request_context():
            for rule in app.url_map.iter_rules():
                if rule.endpoint in {"static", "demo_predict"}:
                    continue
                path = url_for(rule.endpoint, **{argument: "test" for argument in rule.arguments})
                if path in {"/", "/login", "/health"}:
                    continue
                for method in rule.methods.intersection({"GET", "POST", "PUT", "PATCH", "DELETE"}):
                    route_requests.append((method, path))

        with app.test_client() as unauthenticated_client:
            for method, path in route_requests:
                response = unauthenticated_client.open(path, method=method)
                assert response.status_code in {401, 301, 302, 303, 307, 308}, (
                    f"{method} {path} returned {response.status_code} without a session"
                )
    finally:
        app.config["WTF_CSRF_ENABLED"] = csrf_enabled


def test_onboarding_and_risk_pages_enforce_roles():
    with app.test_client() as anonymous_client:
        assert anonymous_client.get("/onboarding").status_code == 302
        assert anonymous_client.get("/risk").status_code == 302
        assert anonymous_client.get("/risk/M_UNKNOWN").status_code == 302

    with app.test_client() as analyst_client:
        with analyst_client.session_transaction() as sess:
            sess["user"] = {"id": 12, "username": "analyst", "role": "Analyst"}
        assert analyst_client.get("/onboarding").status_code == 403
        assert analyst_client.get("/risk").status_code == 200

    with app.test_client() as investigator_client:
        with investigator_client.session_transaction() as sess:
            sess["user"] = {"id": 13, "username": "investigator", "role": "Investigator"}
        assert investigator_client.get("/onboarding").status_code == 200
        assert investigator_client.get("/risk").status_code == 200

    with app.test_client() as admin_client:
        with admin_client.session_transaction() as sess:
            sess["user"] = {"id": 14, "username": "admin_layout", "role": "Admin"}
        assert admin_client.get("/onboarding").status_code == 200


def test_network_and_alert_pages_enforce_roles(monkeypatch):
    monkeypatch.setattr(app_module, "get_merchant", lambda merchant_id: None)
    monkeypatch.setattr(app_module, "get_investigation_alerts", lambda **kwargs: [])
    with app.test_client() as anonymous_client:
        assert anonymous_client.get("/network").status_code == 302
        assert anonymous_client.get("/network/M_NETWORK").status_code == 302
        assert anonymous_client.get("/alerts").status_code == 302

    with app.test_client() as analyst_client:
        with analyst_client.session_transaction() as sess:
            sess["user"] = {"id": 15, "username": "read_only", "role": "Analyst"}
        assert analyst_client.get("/network").status_code == 200
        assert analyst_client.get("/network/M_NETWORK").status_code == 200
        assert analyst_client.get("/alerts").status_code == 403
        assert analyst_client.get("/api/investigation-cases/CASE_READ").status_code in {404, 200}

    with app.test_client() as investigator_client:
        with investigator_client.session_transaction() as sess:
            sess["user"] = {"id": 16, "username": "reviewer", "role": "Investigator"}
        assert investigator_client.get("/alerts").status_code == 200


def test_app_requires_secret_key(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY must be configured"):
        _required_secret_key()


def test_app_startup_port_selection_skips_busy_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_socket:
        occupied_socket.bind(("127.0.0.1", 0))
        occupied_socket.listen(1)
        occupied_port = occupied_socket.getsockname()[1]
        selected_port = find_available_port(occupied_port, attempts=10)

    assert selected_port >= occupied_port
    assert selected_port != occupied_port

def test_csrf_protection_on_post_forms():
    """Confirms CSRF protection is active: rejecting missing tokens and accepting valid tokens."""
    app.config["WTF_CSRF_ENABLED"] = True
    with app.test_client() as csrf_client:
        # Missing CSRF token -> 400 Bad Request
        missing_res = csrf_client.post("/login", data={"username": "admin", "password": "any"})
        assert missing_res.status_code == 400

        # Retrieve valid CSRF token from login page
        get_res = csrf_client.get("/login")
        html = get_res.get_data(as_text=True)
        token_match = re.search(r'name="csrf_token" value="([^"]+)"', html)
        assert token_match is not None
        csrf_token = token_match.group(1)

        # POST with CSRF token is evaluated (returns 401 for bad pass, not 400 CSRF error)
        valid_csrf_res = csrf_client.post("/login", data={
            "username": "admin",
            "password": "wrong_password_test",
            "csrf_token": csrf_token
        })
        assert valid_csrf_res.status_code == 401
    app.config["WTF_CSRF_ENABLED"] = False

def test_db_user_not_root():
    """Confirms that the configured MySQL database user in .env is not root."""
    db_user = (os.getenv("DB_USER") or "").lower()
    assert db_user != "root", "Database user must not be root for application security."
    assert len(db_user) > 0, "Database user must be configured in .env."
