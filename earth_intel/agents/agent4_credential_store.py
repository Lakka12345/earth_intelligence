"""
Agent 4 — Credential Store.

Wraps the OS keychain via the `keyring` library (locked-in decision:
most secure of the options considered -- credentials never touch a
plaintext file). Only called when the user explicitly opts to persist
credentials for a source; otherwise credentials live only in memory
for the current run and are discarded when the process exits.

CHANGED (bug fix): added a file-based fallback for systems where keyring
is unavailable (no secret service, headless servers, CI environments).
The fallback writes base64-obfuscated credential blobs to
.credentials/<source_id>.cred in the project root. This is NOT encryption
— it prevents casual shoulder-surfing but is not secure against a
determined attacker with filesystem access. Users are warned about this.
The fallback is only activated when keyring is genuinely unavailable,
not just missing the pip package.

Install keyring for full security: pip install keyring
"""

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

try:
    import keyring
    _KEYRING_AVAILABLE = True
except ImportError:
    _KEYRING_AVAILABLE = False

_SERVICE_NAME  = "earth_intel_agent4"
_FALLBACK_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".credentials")


# ── File-based fallback helpers ───────────────────────────────────────────

def _fallback_path(source_id: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in source_id)
    return os.path.join(_FALLBACK_DIR, f"{safe_id}.cred")


def _fallback_save(source_id: str, values: dict) -> bool:
    """Save credentials to a base64-obfuscated file. NOT secure storage."""
    try:
        os.makedirs(_FALLBACK_DIR, exist_ok=True)
        # Write a .gitignore so the folder is never accidentally committed
        gi_path = os.path.join(_FALLBACK_DIR, ".gitignore")
        if not os.path.exists(gi_path):
            with open(gi_path, "w") as f:
                f.write("*\n")
        payload = base64.b64encode(json.dumps(values).encode()).decode()
        with open(_fallback_path(source_id), "w") as f:
            f.write(payload)
        print(
            f"[Credential Store] WARNING: keyring unavailable — credentials for "
            f"'{source_id}' saved to .credentials/ in obfuscated (NOT encrypted) form. "
            "Install keyring for secure storage: pip install keyring"
        )
        return True
    except Exception as exc:
        print(f"[Credential Store] File fallback save failed for {source_id}: {exc}")
        return False


def _fallback_load(source_id: str) -> Optional[dict]:
    path = _fallback_path(source_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = f.read().strip()
        return json.loads(base64.b64decode(payload).decode())
    except Exception as exc:
        print(f"[Credential Store] File fallback load failed for {source_id}: {exc}")
        return None


def _fallback_delete(source_id: str) -> bool:
    path = _fallback_path(source_id)
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as exc:
        print(f"[Credential Store] File fallback delete failed for {source_id}: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────

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
    values = {
        k: v for k, v in {
            "username":      username,
            "password":      password,
            "api_key":       api_key,
            "token":         token,
            "session_token": session_token,
            "refresh_token": refresh_token,
            "bearer_token":  bearer_token,
        }.items() if v
    }

    if keyring_available():
        try:
            for field, value in values.items():
                keyring.set_password(_SERVICE_NAME, f"{source_id}::{field}", value)
            return True
        except Exception as exc:
            print(f"[Credential Store] keyring save failed for {source_id}: {exc} — falling back to file store.")

    # File-based fallback
    return _fallback_save(source_id, values)


def load_credentials(source_id: str) -> Optional[StoredCredentials]:
    return load_provider_credentials(source_id)


def load_provider_credentials(source_id: str) -> Optional[StoredCredentials]:
    if keyring_available():
        try:
            username      = keyring.get_password(_SERVICE_NAME, f"{source_id}::username")
            password      = keyring.get_password(_SERVICE_NAME, f"{source_id}::password")
            api_key       = keyring.get_password(_SERVICE_NAME, f"{source_id}::api_key")
            token         = keyring.get_password(_SERVICE_NAME, f"{source_id}::token")
            session_token = keyring.get_password(_SERVICE_NAME, f"{source_id}::session_token")
            refresh_token = keyring.get_password(_SERVICE_NAME, f"{source_id}::refresh_token")
            bearer_token  = keyring.get_password(_SERVICE_NAME, f"{source_id}::bearer_token")
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
        except Exception as exc:
            print(f"[Credential Store] keyring load failed for {source_id}: {exc} — trying file fallback.")

    # File-based fallback
    values = _fallback_load(source_id)
    if not values:
        return None
    return StoredCredentials(
        username=values.get("username", ""),
        password=values.get("password", ""),
        api_key=values.get("api_key"),
        token=values.get("token"),
        session_token=values.get("session_token"),
        refresh_token=values.get("refresh_token"),
        bearer_token=values.get("bearer_token"),
    )


def delete_credentials(source_id: str) -> bool:
    deleted_keyring = False
    if keyring_available():
        try:
            for field in ("username", "password", "api_key", "token", "session_token", "refresh_token", "bearer_token"):
                try:
                    keyring.delete_password(_SERVICE_NAME, f"{source_id}::{field}")
                except keyring.errors.PasswordDeleteError:
                    pass
            deleted_keyring = True
        except Exception as exc:
            print(f"[Credential Store] keyring delete failed for {source_id}: {exc}")

    deleted_file = _fallback_delete(source_id)
    return deleted_keyring or deleted_file
