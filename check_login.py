import os
from dotenv import load_dotenv
import mysql.connector
from werkzeug.security import check_password_hash

load_dotenv()
print("ADMIN_USERNAME from .env:", repr(os.getenv("ADMIN_USERNAME")))
print("ADMIN_PASSWORD length:", len(os.getenv("ADMIN_PASSWORD") or ""))

conn = mysql.connector.connect(
    host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT", 3306)),
    user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)
cur = conn.cursor()
cur.execute("SELECT username, password_hash, is_active FROM users")
rows = cur.fetchall()
print("Users found:", len(rows))
for username, pw_hash, active in rows:
    ok = check_password_hash(pw_hash, os.getenv("ADMIN_PASSWORD") or "")
    print(repr(username), "| active:", active, "| password matches .env:", ok)