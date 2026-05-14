"""Google OAuth and signed app-session helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(f"{raw}{padding}")


@dataclass(frozen=True)
class GoogleOAuthConfig:
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: list[str]
    auth_url: str = "https://accounts.google.com/o/oauth2/v2/auth"
    token_url: str = "https://oauth2.googleapis.com/token"
    userinfo_url: str = "https://openidconnect.googleapis.com/v1/userinfo"

    @classmethod
    def from_env(cls) -> Optional["GoogleOAuthConfig"]:
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
        scopes_raw = os.environ.get(
            "GOOGLE_SCOPES",
            "openid email profile https://www.googleapis.com/auth/drive.file",
        ).strip()

        if not client_id or not client_secret or not redirect_uri:
            return None

        scopes = [part.strip() for part in scopes_raw.replace(",", " ").split() if part.strip()]
        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scopes=scopes,
        )

    def build_authorize_url(
        self,
        *,
        state: str,
        prompt: str = "consent",
        access_type: str = "offline",
        include_granted_scopes: bool = True,
    ) -> str:
        query = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": " ".join(self.scopes),
                "state": state,
                "prompt": prompt,
                "access_type": access_type,
                "include_granted_scopes": "true" if include_granted_scopes else "false",
            }
        )
        return f"{self.auth_url}?{query}"

    def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        body = urlencode(
            {
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        request = Request(
            self.token_url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        body = urlencode(
            {
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        request = Request(
            self.token_url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_userinfo(self, access_token: str) -> Dict[str, Any]:
        request = Request(
            self.userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))


class AppAuthService:
    """Issues and verifies signed session/state tokens without extra dependencies."""

    def __init__(self, app_secret: str, session_ttl_seconds: int = 60 * 60 * 24 * 14):
        if not app_secret:
            raise ValueError("app_secret is required")
        self._secret = app_secret.encode("utf-8")
        self._session_ttl_seconds = max(int(session_ttl_seconds), 60)

    @classmethod
    def from_env(cls) -> Optional["AppAuthService"]:
        secret = os.environ.get("CORTEX_AUTH_SECRET", "").strip()
        if not secret:
            return None
        ttl = int(os.environ.get("CORTEX_AUTH_TTL_SECONDS", str(60 * 60 * 24 * 14)).strip() or str(60 * 60 * 24 * 14))
        return cls(app_secret=secret, session_ttl_seconds=ttl)

    def _sign(self, payload: Dict[str, Any]) -> str:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(self._secret, body, hashlib.sha256).digest()
        return f"{_b64url_encode(body)}.{_b64url_encode(signature)}"

    def _verify(self, token: str) -> Dict[str, Any]:
        if "." not in token:
            raise ValueError("invalid_token_format")

        payload_part, signature_part = token.split(".", 1)
        body = _b64url_decode(payload_part)
        signature = _b64url_decode(signature_part)
        expected = hmac.new(self._secret, body, hashlib.sha256).digest()

        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid_signature")

        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid_payload")
        return payload

    def issue_session_token(self, claims: Dict[str, Any]) -> str:
        now = int(time.time())
        payload = {
            "kind": "cortex_session",
            "provider": "google",
            "iat": now,
            "exp": now + self._session_ttl_seconds,
            **dict(claims or {}),
        }
        return self._sign(payload)

    def verify_session_token(self, token: str) -> Dict[str, Any]:
        payload = self._verify(token)
        if payload.get("kind") != "cortex_session":
            raise ValueError("invalid_session_kind")
        if int(payload.get("exp", 0) or 0) < int(time.time()):
            raise ValueError("session_expired")
        return payload

    def issue_state_token(self, payload: Dict[str, Any], ttl_seconds: int = 900) -> str:
        now = int(time.time())
        body = {
            "kind": "oauth_state",
            "iat": now,
            "exp": now + max(int(ttl_seconds), 60),
            **dict(payload or {}),
        }
        return self._sign(body)

    def verify_state_token(self, token: str) -> Dict[str, Any]:
        payload = self._verify(token)
        if payload.get("kind") != "oauth_state":
            raise ValueError("invalid_state_kind")
        if int(payload.get("exp", 0) or 0) < int(time.time()):
            raise ValueError("state_expired")
        return payload

    @staticmethod
    def extract_bearer_token(auth_header: str) -> str:
        if not auth_header or not auth_header.startswith("Bearer "):
            return ""
        return auth_header[7:].strip()

    @staticmethod
    def normalize_scopes(scopes: Iterable[str] | str | None) -> list[str]:
        if scopes is None:
            return []
        if isinstance(scopes, str):
            return [part.strip() for part in scopes.replace(",", " ").split() if part.strip()]
        normalized: list[str] = []
        for scope in scopes:
            value = str(scope or "").strip()
            if value:
                normalized.append(value)
        return normalized
