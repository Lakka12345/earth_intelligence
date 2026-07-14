"""
Agent 4 — Credential Store.

Wraps the OS keychain via the `keyring` library (locked-in decision:
most secure of the options considered -- credentials never touch a
plaintext file). Only called when the user explicitly opts to persist
credentials for a source; otherwise credentials live only in memory
for the current run and are discarded when the process exits.

Install: pip install keyring
"""

from dataclasses import dataclass
from typing import Optional

try:
    import keyring
    _KEYRING_AVAILABLE = True
except ImportError:
    _KEYRING_AVAILABLE = False

_SERVICE_NAME = "earth_intel_agent4"


@dataclass
class StoredCredentials:
    username: str = ""
    password: str = ""
    api_key: Optional[str] = None
    token: Optional[str] = None
    session_token: Optional[str] = None
    refresh_token: Optional[str] = None
    bearer_token: Optional[str] = None


def keyring_available() -> bool:
    if not _KEYRING_AVAILABLE:
        return False
    try:
        keyring.get_keyring()
        return True
    except Exception:
        return False


def save_credentials(source_id: str, username: str, password: str) -> bool:
    return save_provider_credentials(source_id, username=username, password=password)


def save_provider_credentials(
    source_id: str,
    username: str = "",
    password: str = "",
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    session_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> bool:
    if not keyring_available():
        print("[Credential Store] keyring is not available on this system -- cannot persist credentials. "
              "Install with: pip install keyring")
        return False
    try:
        values = {
            "username": username,
            "password": password,
            "api_key": api_key,
            "token": token,
            "session_token": session_token,
            "refresh_token": refresh_token,
            "bearer_token": bearer_token,
        }
        for field, value in values.items():
            if value:
                keyring.set_password(_SERVICE_NAME, f"{source_id}::{field}", value)
        return True
    except Exception as exc:
        print(f"[Credential Store] Failed to save credentials for {source_id}: {exc}")
        return False


def load_credentials(source_id: str) -> Optional[StoredCredentials]:
    return load_provider_credentials(source_id)


def load_provider_credentials(source_id: str) -> Optional[StoredCredentials]:
    if not keyring_available():
        return None
    try:
        username = keyring.get_password(_SERVICE_NAME, f"{source_id}::username")
        password = keyring.get_password(_SERVICE_NAME, f"{source_id}::password")
        api_key = keyring.get_password(_SERVICE_NAME, f"{source_id}::api_key")
        token = keyring.get_password(_SERVICE_NAME, f"{source_id}::token")
        session_token = keyring.get_password(_SERVICE_NAME, f"{source_id}::session_token")
        refresh_token = keyring.get_password(_SERVICE_NAME, f"{source_id}::refresh_token")
        bearer_token = keyring.get_password(_SERVICE_NAME, f"{source_id}::bearer_token")
        if (username and password) or api_key or token or session_token or refresh_token or bearer_token:
            return StoredCredentials(
                username=username or "",
                password=password or "",
                api_key=api_key,
                token=token,
                session_token=session_token,
                refresh_token=refresh_token,
                bearer_token=bearer_token,
            )
        return None
    except Exception as exc:
        print(f"[Credential Store] Failed to load credentials for {source_id}: {exc}")
        return None


def delete_credentials(source_id: str) -> bool:
    if not keyring_available():
        return False
    try:
        for field in ("username", "password", "api_key", "token", "session_token", "refresh_token", "bearer_token"):
            try:
                keyring.delete_password(_SERVICE_NAME, f"{source_id}::{field}")
            except keyring.errors.PasswordDeleteError:
                pass  # nothing stored for this field -- fine
        return True
    except Exception as exc:
        print(f"[Credential Store] Failed to delete credentials for {source_id}: {exc}")
        return False
