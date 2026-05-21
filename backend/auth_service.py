"""Simple session auth for research demo (not production-grade)."""

import json
import secrets
from pathlib import Path

from config import USERS_PATH

_sessions: dict[str, str] = {}


def _load_users() -> dict[str, str]:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text('{"demo": "demo123"}', encoding="utf-8")
    return json.loads(USERS_PATH.read_text(encoding="utf-8"))


def _save_users(users: dict[str, str]) -> None:
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def register(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = password
    _save_users(users)
    return True, "Account created."


def login(username: str, password: str) -> tuple[bool, str, str | None]:
    username = username.strip().lower()
    users = _load_users()
    if username not in users or users[username] != password:
        return False, "Invalid username or password.", None
    token = secrets.token_urlsafe(32)
    _sessions[token] = username
    return True, "Login successful.", token


def logout(token: str) -> None:
    _sessions.pop(token, None)


def get_user(token: str | None) -> str | None:
    if not token:
        return None
    return _sessions.get(token)
