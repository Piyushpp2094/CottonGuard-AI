"""
CottonGuard AI — Quick Backend & Database Data Inspector
Displays users, login roles, and recorded prediction data from cotton_leaf_ai.db.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[1] / "cotton_leaf_ai.db"

def inspect_data():
    if not DB_PATH.is_file():
        print(f"[ERROR] Database file not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("=" * 80)
    print("                  COTTONGUARD AI — BACKEND DATABASE INSPECTOR")
    print(f" Database: {DB_PATH.name} ({DB_PATH.stat().st_size:,} bytes)")
    print("=" * 80)

    # 1. Inspect Users Table
    print("\n--- 1. REGISTERED USERS & ROLES ---")
    cursor.execute("""
        SELECT u.id, u.username, u.email, u.role, u.is_active, u.created_at,
               COUNT(p.id) as total_scans
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.id ASC;
    """)
    users = cursor.fetchall()
    print(f"Total Users: {len(users)}\n")
    print(f"{'ID':<4} | {'Username':<15} | {'Role':<14} | {'Total Scans':<11} | {'Created At':<20} | {'Email'}")
    print("-" * 80)
    for u in users:
        created = str(u['created_at'])[:19] if u['created_at'] else "N/A"
        print(f"{u['id']:<4} | {u['username']:<15} | {u['role']:<14} | {u['total_scans']:<11} | {created:<20} | {u['email'] or 'N/A'}")

    # 2. Inspect Recent Predictions Data
    print("\n\n--- 2. RECENT PREDICTION & DIAGNOSTIC DATA (Last 10 Records) ---")
    cursor.execute("""
        SELECT p.id, u.username, p.predicted_class, p.confidence, p.severity_level, 
               p.image_filename, p.created_at
        FROM predictions p
        LEFT JOIN users u ON p.user_id = u.id
        ORDER BY p.id DESC
        LIMIT 10;
    """)
    preds = cursor.fetchall()
    
    cursor.execute("SELECT COUNT(*) FROM predictions;")
    total_preds = cursor.fetchone()[0]
    print(f"Total Scans in Database: {total_preds}\n")
    print(f"{'ID':<4} | {'User':<12} | {'Predicted Disease':<22} | {'Confidence':<10} | {'Severity':<10} | {'Timestamp'}")
    print("-" * 80)
    for p in preds:
        conf_pct = f"{round(p['confidence'] * 100, 1)}%" if p['confidence'] is not None else "N/A"
        created = str(p['created_at'])[:19] if p['created_at'] else "N/A"
        user_name = p['username'] or "anonymous"
        print(f"{p['id']:<4} | {user_name:<12} | {p['predicted_class']:<22} | {conf_pct:<10} | {p['severity_level'] or 'N/A':<10} | {created}")

    # 3. Overall Statistics
    print("\n\n--- 3. CONDITION BREAKDOWN ---")
    cursor.execute("""
        SELECT predicted_class, COUNT(*) as count 
        FROM predictions 
        GROUP BY predicted_class 
        ORDER BY count DESC;
    """)
    for row in cursor.fetchall():
        print(f"  • {row['predicted_class']:<25}: {row['count']} scans")

    conn.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect_data()
