"""
Admin User Reset Utility
Deletes and recreates the default Admin user from .env credentials,
ensuring a freshly generated Werkzeug password hash with stripped whitespace/quotes.
Never logs or prints plain-text credentials.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector

# Ensure root .env is loaded
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

def reset_admin_user() -> bool:
    admin_user = os.getenv("ADMIN_USERNAME", "admin").strip().strip("'\"")
    admin_email = os.getenv("ADMIN_EMAIL", "admin@fraudshield.internal").strip().strip("'\"")
    admin_pass = os.getenv("ADMIN_PASSWORD", "").strip().strip("'\"")

    if not admin_pass:
        print("[ERROR] ADMIN_PASSWORD is not configured in .env")
        return False

    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = int(os.getenv("DB_PORT", 3306))
    db_user = os.getenv("DB_USER", "fraud_app")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "merchant_fraud_db")

    try:
        conn = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        # Delete existing admin user if present
        cursor.execute("DELETE FROM users WHERE username = %s", (admin_user,))
        deleted_count = cursor.rowcount
        print(f"[INFO] Cleared {deleted_count} existing record(s) for user '{admin_user}'")

        # Generate fresh secure hash
        new_hash = generate_password_hash(admin_pass)

        # Insert fresh admin record
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, role, is_active)
            VALUES (%s, %s, %s, 'Admin', 1)
            """,
            (admin_user, admin_email, new_hash)
        )
        conn.commit()
        print(f"[SUCCESS] Admin user '{admin_user}' created successfully with role 'Admin' and is_active=1")

        # Verify hash matches
        cursor.execute("SELECT password_hash, is_active FROM users WHERE username = %s", (admin_user,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and check_password_hash(row[0], admin_pass):
            print("[SUCCESS] Hash verification confirmed: password matches .env value.")
            return True
        else:
            print("[ERROR] Hash verification failed.")
            return False

    except Exception as e:
        print(f"[ERROR] Database error during admin reset: {e}")
        return False

if __name__ == "__main__":
    success = reset_admin_user()
    sys.exit(0 if success else 1)

