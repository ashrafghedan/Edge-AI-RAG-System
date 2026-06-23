from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

from .constants import AUTH_TOKEN_TTL_DAYS


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str, *, salt: str | None = None) -> str:
    chosen_salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), chosen_salt.encode('utf-8'), 200_000)
    return f'{chosen_salt}${derived.hex()}'


def verify_password(password: str, stored_value: str) -> bool:
    try:
        salt, expected_hash = stored_value.split('$', 1)
    except ValueError:
        return False
    actual = hash_password(password, salt=salt).split('$', 1)[1]
    return hmac.compare_digest(actual, expected_hash)


def issue_auth_token() -> tuple[str, str, datetime]:
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
    expires_at = utc_now() + timedelta(days=AUTH_TOKEN_TTL_DAYS)
    return raw_token, token_hash, expires_at


def hash_auth_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
