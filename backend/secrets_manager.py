from __future__ import annotations

import base64
import hashlib
import os
from typing import Dict

from cryptography.fernet import Fernet


def _resolve_master_key() -> bytes:
    raw = str(os.getenv("SECRETS_MASTER_KEY", "")).strip()
    if raw:
        key_bytes = raw.encode("utf-8")
        try:
            Fernet(key_bytes)
            return key_bytes
        except Exception:
            # allow passphrase style value and derive fernet key
            digest = hashlib.sha256(key_bytes).digest()
            return base64.urlsafe_b64encode(digest)
    # strict-only live path should set SECRETS_MASTER_KEY explicitly.
    dev_seed = str(os.getenv("SECRETS_DEV_SEED", "strict-dev-seed")).encode("utf-8")
    digest = hashlib.sha256(dev_seed).digest()
    return base64.urlsafe_b64encode(digest)


class SecretsManager:
    def __init__(self):
        key = _resolve_master_key()
        self._fernet = Fernet(key)

    def encrypt(self, plain: str) -> str:
        raw = str(plain or "").encode("utf-8")
        return self._fernet.encrypt(raw).decode("utf-8")

    def decrypt(self, cipher_text: str) -> str:
        raw = self._fernet.decrypt(str(cipher_text).encode("utf-8"))
        return raw.decode("utf-8")

    def encrypt_bitget(self, api_key: str, api_secret: str, passphrase: str) -> Dict[str, str]:
        return {
            "api_key_enc": self.encrypt(api_key),
            "api_secret_enc": self.encrypt(api_secret),
            "passphrase_enc": self.encrypt(passphrase),
        }

    def decrypt_bitget(self, row: Dict[str, str]) -> Dict[str, str]:
        return {
            "api_key": self.decrypt(str(row.get("api_key_enc") or "")),
            "api_secret": self.decrypt(str(row.get("api_secret_enc") or "")),
            "passphrase": self.decrypt(str(row.get("passphrase_enc") or "")),
        }
