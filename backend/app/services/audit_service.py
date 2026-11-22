"""
Centralized audit trail logger for security, diagnostic scans, and admin operations.
"""

from typing import Optional
from sqlalchemy.orm import Session

from backend.app.models import AuditLog, User


def record_audit_event(
    db: Session,
    action: str,
    resource: str,
    details: Optional[str] = None,
    user: Optional[User] = None,
    commit: bool = False,
) -> AuditLog:
    """Appends an immutable audit event entry to the database session."""
    entry = AuditLog(
        user_id=user.id if user else None,
        username=user.username if user else "system",
        action=action,
        resource=resource,
        details=details,
    )
    db.add(entry)
    if commit:
        db.commit()
    return entry
