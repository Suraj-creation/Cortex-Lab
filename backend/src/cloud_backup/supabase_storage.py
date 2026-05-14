"""Supabase Storage backup upload client."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from urllib.parse import quote
from urllib.request import Request, urlopen


class SupabaseStorageBackupClient:
    """Uploads backup bundles to a private Supabase Storage bucket."""

    def __init__(
        self,
        *,
        supabase_url: str,
        service_role_key: str,
        bucket: str = "cortex-backups",
    ):
        self.supabase_url = supabase_url.rstrip("/")
        self.service_role_key = service_role_key.strip()
        self.bucket = bucket.strip() or "cortex-backups"

    @property
    def configured(self) -> bool:
        return bool(self.supabase_url and self.service_role_key)

    def upload_backup(
        self,
        *,
        bundle_path: str,
        object_path: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.configured:
            raise ValueError("Supabase Storage is not configured")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(bundle_path)

        encoded_path = "/".join(quote(part, safe="") for part in object_path.split("/"))
        endpoint = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{encoded_path}"
        with open(bundle_path, "rb") as handle:
            payload = handle.read()

        request = Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.service_role_key}",
                "apikey": self.service_role_key,
                "Content-Type": "application/zip",
                "Content-Length": str(len(payload)),
                "x-upsert": "true",
                "x-metadata": json.dumps(metadata, separators=(",", ":"), sort_keys=True),
            },
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
        response_payload = json.loads(raw) if raw else {}
        return {
            "bucket": self.bucket,
            "object_path": object_path,
            "response": response_payload,
        }
