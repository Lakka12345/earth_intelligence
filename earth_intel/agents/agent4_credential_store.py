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
    username: str
    password: str


def keyring_available() -> bool:
    if not _KEYRING_AVAILABLE:
        return False
    try:
        keyring.get_keyring()
        return True
    except Exception:
        return False


def save_credentials(source_id: str, username: str, password: str) -> bool:
    if not keyring_available():
        print("[Credential Store] keyring is not available on this system -- cannot persist credentials. "
              "Install with: pip install keyring")
        return False
    try:
        keyring.set_password(_SERVICE_NAME, f"{source_id}::username", username)
        keyring.set_password(_SERVICE_NAME, f"{source_id}::password", password)
        return True
    except Exception as exc:
        print(f"[Credential Store] Failed to save credentials for {source_id}: {exc}")
        return False


def load_credentials(source_id: str) -> Optional[StoredCredentials]:
    if not keyring_available():
        return None
    try:
        username = keyring.get_password(_SERVICE_NAME, f"{source_id}::username")
        password = keyring.get_password(_SERVICE_NAME, f"{source_id}::password")
        if username and password:
            return StoredCredentials(username=username, password=password)
        return None
    except Exception as exc:
        print(f"[Credential Store] Failed to load credentials for {source_id}: {exc}")
        return None


def delete_credentials(source_id: str) -> bool:
    if not keyring_available():
        return False
    try:
        for field in ("username", "password"):
            try:
                keyring.delete_password(_SERVICE_NAME, f"{source_id}::{field}")
            except keyring.errors.PasswordDeleteError:
                pass  # nothing stored for this field -- fine
        return True
    except Exception as exc:
        print(f"[Credential Store] Failed to delete credentials for {source_id}: {exc}")
        return False
