"""
Local Administrator Account Bootstrap Utility for Cotton Leaf AI.

Creates or promotes a local administrator user in SQLite/PostgreSQL cleanly.
"""

import argparse
import getpass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.auth import hash_password
from backend.app.database import SessionLocal, init_db
from backend.app.models import User


def ensure_admin_user(username: str, email: str, password: str) -> None:
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            user.role = "admin"
            user.is_active = True
            if password:
                user.password_hash = hash_password(password)
            db.commit()
            print(f"[ADMIN] Promoted existing user '{username}' (id={user.id}) to role='admin'.")
        else:
            user = User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                role="admin",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"[ADMIN] Created new administrator '{username}' (id={user.id}).")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Create or promote a local administrator account")
    parser.add_argument("--username", type=str, default="admin")
    parser.add_argument("--email", type=str, default="admin@cotton.local")
    parser.add_argument("--password", type=str, default=None)
    args = parser.parse_args()

    password = args.password
    if not password:
        password = getpass.getpass(f"Enter password for admin user '{args.username}': ")
    if len(password) < 6:
        raise ValueError("Admin password must be at least 6 characters.")

    ensure_admin_user(args.username, args.email, password)


if __name__ == "__main__":
    main()
