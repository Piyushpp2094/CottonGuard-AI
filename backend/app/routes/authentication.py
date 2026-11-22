"""
Authentication API routes: Registration, Login, Token Refresh Rotation, Logout, and Profile.
"""

from datetime import datetime
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    extract_bearer_or_cookie_token,
    get_current_user,
    hash_password,
    revoke_access_token,
    security,
    verify_password,
)
from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.models import RefreshToken, User
from backend.app.schemas import (
    LogoutRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from backend.app.services.audit_service import record_audit_event

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key="cottonguard_access_token",
        value=access_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="cottonguard_refresh_token",
        value=refresh_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        path="/api/auth",
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    """Registers a new farmer user account."""
    if db.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this username already exists.",
        )

    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    # Enforce farmer role on public registration; admin roles are promoted via CLI or existing admin
    user = User(
        username=user_in.username,
        email=user_in.email,
        password_hash=hash_password(user_in.password),
        role="farmer",
    )
    db.add(user)
    db.flush()
    record_audit_event(
        db,
        action="USER_REGISTER",
        resource=f"user:{user.id}",
        details=f"Registered farmer '{user.username}'",
        user=user,
    )
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(login_in: UserLogin, response: Response, db: Session = Depends(get_db)):
    """Authenticates user and issues rotated JWT access & refresh tokens."""
    user = (
        db.query(User)
        .filter(
            (User.username == login_in.username_or_email)
            | (User.email == login_in.username_or_email)
        )
        .first()
    )

    if not user or not verify_password(login_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator.",
        )

    settings = get_settings()
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role}
    )
    refresh_token, _ = create_refresh_token(user=user, db=db)
    record_audit_event(
        db,
        action="USER_LOGIN",
        resource=f"user:{user.id}",
        details=f"Role '{user.role}' signed in",
        user=user,
        commit=True,
    )
    _set_auth_cookies(response, access_token, refresh_token)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "token_type": "bearer",
        "user": user,
    }


@router.post("/refresh", response_model=TokenResponse)
def refresh_tokens(
    request: Request,
    response: Response,
    payload_in: Optional[RefreshTokenRequest] = None,
    db: Session = Depends(get_db),
):
    """Rotates a valid refresh token and issues a fresh access/refresh token pair."""
    raw_refresh = (
        (payload_in.refresh_token if payload_in else None)
        or request.cookies.get("cottonguard_refresh_token")
    )
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing.",
        )

    try:
        decoded = decode_token(raw_refresh)
        if decoded.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type.")
        jti = decoded.get("jti")
        user_id = int(decoded.get("sub"))
    except (jwt.PyJWTError, ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    token_row = db.query(RefreshToken).filter(RefreshToken.jti == jti).first()
    if not token_row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not recognized.")

    if token_row.revoked:
        # Refresh token reuse detected -> revoke all active refresh tokens for this user
        db.query(RefreshToken).filter(
            RefreshToken.user_id == token_row.user_id,
            RefreshToken.revoked == False,  # noqa: E712
        ).update({"revoked": True})
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token already used or revoked. Please sign in again.",
        )

    if token_row.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has expired.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not getattr(user, "is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account inactive or missing.")

    new_refresh, new_row = create_refresh_token(user=user, db=db)
    token_row.revoked = True
    token_row.replaced_by_jti = new_row.jti

    new_access = create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role}
    )
    record_audit_event(
        db,
        action="TOKEN_REFRESH",
        resource=f"user:{user.id}",
        details="Rotated refresh and access tokens",
        user=user,
        commit=True,
    )
    _set_auth_cookies(response, new_access, new_refresh)
    settings = get_settings()
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "token_type": "bearer",
        "user": user,
    }


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    payload_in: Optional[LogoutRequest] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revokes current access token and optional refresh token, clearing auth cookies."""
    access_token = extract_bearer_or_cookie_token(request, credentials)
    if access_token:
        revoke_access_token(db, access_token)

    raw_refresh = (
        (payload_in.refresh_token if payload_in else None)
        or request.cookies.get("cottonguard_refresh_token")
    )
    if raw_refresh:
        try:
            decoded = decode_token(raw_refresh)
            jti = decoded.get("jti")
            if jti:
                row = db.query(RefreshToken).filter(RefreshToken.jti == jti).first()
                if row:
                    row.revoked = True
        except jwt.PyJWTError:
            pass

    record_audit_event(
        db,
        action="USER_LOGOUT",
        resource=f"user:{current_user.id}",
        details="Signed out and revoked session token",
        user=current_user,
        commit=True,
    )
    response.delete_cookie("cottonguard_access_token", path="/")
    response.delete_cookie("cottonguard_refresh_token", path="/api/auth")
    return {"status": "logged_out"}


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns the authenticated user's profile."""
    return current_user
