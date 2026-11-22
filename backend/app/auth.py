"""
Authentication, Password Hashing, JWT Access/Refresh Rotation, and Revocation Utilities.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.models import RefreshToken, RevokedToken, User

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt (cost factor 12)."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT access token with unique jti and expiration."""
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update(
        {
            "exp": expire,
            "jti": uuid.uuid4().hex,
            "type": "access",
        }
    )
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user: User, db: Session, expires_delta: Optional[timedelta] = None) -> Tuple[str, RefreshToken]:
    """Generates and persists a refresh JWT token for rotation and revocation."""
    settings = get_settings()
    jti = uuid.uuid4().hex
    expire = datetime.utcnow() + (
        expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    payload = {
        "sub": str(user.id),
        "jti": jti,
        "type": "refresh",
        "exp": expire,
    }
    encoded = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    record = RefreshToken(
        user_id=user.id,
        jti=jti,
        expires_at=expire,
        revoked=False,
    )
    db.add(record)
    return encoded, record


def decode_token(token: str) -> dict:
    """Decodes and verifies JWT signature and expiry."""
    settings = get_settings()
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


def is_jti_revoked(db: Session, jti: Optional[str]) -> bool:
    if not jti:
        return False
    return db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None


def revoke_access_token(db: Session, token: str) -> None:
    """Adds an access token's jti to the revocation table until expiry."""
    try:
        payload = decode_token(token)
        jti = payload.get("jti")
        exp_ts = payload.get("exp")
        if jti and not is_jti_revoked(db, jti):
            exp_dt = datetime.utcfromtimestamp(exp_ts) if exp_ts else datetime.utcnow()
            db.add(RevokedToken(jti=jti, expires_at=exp_dt))
    except jwt.PyJWTError:
        pass


def extract_bearer_or_cookie_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials],
) -> Optional[str]:
    if credentials and credentials.credentials:
        return credentials.credentials
    return request.cookies.get("cottonguard_access_token")


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Validates Bearer or HttpOnly cookie access JWT, enforces revocation & account status,
    and returns the authenticated User.
    """
    token = extract_bearer_or_cookie_token(request, credentials)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token has expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception

    try:
        payload = decode_token(token)
        if payload.get("type", "access") != "access":
            raise credentials_exception
        if is_jti_revoked(db, payload.get("jti")):
            raise credentials_exception
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except (jwt.PyJWTError, ValueError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    if not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact an administrator.",
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency restricting access to administrator role users."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required for this operation.",
        )
    return current_user
