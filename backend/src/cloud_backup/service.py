"""Local-first backup bundle generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency for remote backup.
    psycopg = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: str) -> Iterable[str]:
    for current_root, _dirs, files in os.walk(root):
        for filename in files:
            yield os.path.join(current_root, filename)


class BackupWorkspaceInspector:
    """Inspects the local Cortex workspace and packages a restore bundle."""

    def __init__(self, data_root: str):
        self.data_root = os.path.abspath(data_root)

    def inspect(self) -> Dict[str, Any]:
        duckdb_path = os.path.join(self.data_root, "cortex.duckdb")
        graph_path = os.path.join(self.data_root, "graph", "knowledge_graph.json")
        wiki_dir = os.path.join(self.data_root, "wiki", "pages")
        conversations_dir = os.path.join(self.data_root, "conversations")

        return {
            "captured_at": _utc_now_iso(),
            "data_root": self.data_root,
            "duckdb": self._describe_file(duckdb_path),
            "graph_file": self._describe_file(graph_path),
            "ambient_config": self._describe_file(os.path.join(self.data_root, "ambient_config.json")),
            "wiki_pages": self._count_files(wiki_dir, suffixes=(".md",)),
            "wiki_metadata_files": self._count_files(wiki_dir, suffixes=(".json",)),
            "conversation_records": self._count_files(conversations_dir, suffixes=(".json",)),
        }

    def build_bundle(self, *, output_dir: str, user_snapshot: Dict[str, Any] | None = None) -> str:
        os.makedirs(output_dir, exist_ok=True)

        manifest = {
            "version": "2026-05-14",
            "created_at": _utc_now_iso(),
            "local_workspace": self.inspect(),
            "user_snapshot": dict(user_snapshot or {}),
        }

        bundle_name = f"cortex-backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.zip"
        bundle_path = os.path.join(output_dir, bundle_name)

        with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            for relative in self._default_relpaths():
                absolute = os.path.join(self.data_root, relative)
                if os.path.isfile(absolute):
                    archive.write(absolute, arcname=relative)
                elif os.path.isdir(absolute):
                    for file_path in _iter_files(absolute):
                        arcname = os.path.relpath(file_path, self.data_root).replace("\\", "/")
                        archive.write(file_path, arcname=arcname)

        return bundle_path

    def build_temp_bundle(self, user_snapshot: Dict[str, Any] | None = None) -> str:
        temp_dir = tempfile.mkdtemp(prefix="cortex-backup-")
        return self.build_bundle(output_dir=temp_dir, user_snapshot=user_snapshot)

    @staticmethod
    def _describe_file(path: str) -> Dict[str, Any]:
        exists = os.path.exists(path)
        if not exists:
            return {
                "path": path,
                "exists": False,
                "size_bytes": 0,
                "sha256": "",
            }
        return {
            "path": path,
            "exists": True,
            "size_bytes": os.path.getsize(path),
            "sha256": _sha256_file(path),
        }

    @staticmethod
    def _count_files(root: str, *, suffixes: tuple[str, ...]) -> int:
        if not os.path.isdir(root):
            return 0
        count = 0
        for file_path in _iter_files(root):
            if file_path.endswith(suffixes):
                count += 1
        return count

    @staticmethod
    def _default_relpaths() -> List[str]:
        return [
            "cortex.duckdb",
            "ambient_config.json",
            "graph",
            "wiki",
            "vectors",
            "conversations",
            "chronicle",
            "deep_apps",
            "sessions",
            "pageindex",
        ]


class CloudBackupCoordinator:
    """Coordinates local bundle creation and optional remote persistence."""

    def __init__(
        self,
        *,
        data_root: str,
        history_path: str,
        postgres_dsn: str = "",
        inspector: BackupWorkspaceInspector | None = None,
        google_oauth_config: Any | None = None,
        google_refresh_token_resolver: Callable[[Dict[str, Any]], str] | None = None,
        google_drive_client: Any | None = None,
        supabase_storage_client: Any | None = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.history_path = os.path.abspath(history_path)
        self.postgres_dsn = postgres_dsn.strip()
        self.inspector = inspector or BackupWorkspaceInspector(self.data_root)
        self.google_oauth_config = google_oauth_config
        self.google_refresh_token_resolver = google_refresh_token_resolver
        self.google_drive_client = google_drive_client
        self.supabase_storage_client = supabase_storage_client
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

    def status(self) -> Dict[str, Any]:
        last_backup_at = None
        history = self.list_history(limit=1)
        if history:
            last_backup_at = history[0].get("created_at")
        return {
            "enabled": True,
            "supabase_postgres_configured": bool(self.postgres_dsn),
            "supabase_storage_configured": self._supabase_storage_configured(),
            "google_drive_configured": self._google_drive_configured(),
            "last_backup_at": last_backup_at,
            "workspace": self.inspector.inspect(),
        }

    def create_backup(
        self,
        user_claims: Dict[str, Any],
        client_snapshot: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        backup_id = f"backup-{uuid.uuid4().hex[:12]}"
        bundle_path = self.inspector.build_temp_bundle(user_snapshot=client_snapshot or {})
        with ZipFile(bundle_path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

        with open(bundle_path, "rb") as handle:
            bundle_bytes = handle.read()

        sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        size_bytes = len(bundle_bytes)
        google_drive = self._write_google_drive_backup(
            user_claims=user_claims,
            bundle_path=bundle_path,
            backup_id=backup_id,
            manifest=manifest,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        google_drive_uploaded = bool(google_drive.get("uploaded"))
        supabase_storage = self._write_supabase_storage_backup(
            user_claims=user_claims,
            bundle_path=bundle_path,
            backup_id=backup_id,
            manifest=manifest,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        supabase_storage_uploaded = bool(supabase_storage.get("uploaded"))
        supabase_remote_written = self._write_remote_record(
            backup_id=backup_id,
            user_claims=user_claims,
            manifest=manifest,
            bundle_bytes=bundle_bytes,
            sha256=sha256,
            size_bytes=size_bytes,
            google_drive=google_drive,
            supabase_storage=supabase_storage,
        )
        record = {
            "backup_id": backup_id,
            "created_at": _utc_now_iso(),
            "user_sub": str(user_claims.get("sub", "")),
            "email": str(user_claims.get("email", "")),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "remote_written": bool(supabase_remote_written or google_drive_uploaded or supabase_storage_uploaded),
            "supabase_remote_written": supabase_remote_written,
            "supabase_storage_uploaded": supabase_storage_uploaded,
            "supabase_storage": supabase_storage,
            "google_drive_uploaded": google_drive_uploaded,
            "google_drive": google_drive,
            "manifest": manifest,
        }
        self._append_history(record)
        return {
            "status": "created",
            "backup_id": backup_id,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "remote_written": bool(supabase_remote_written or google_drive_uploaded or supabase_storage_uploaded),
            "supabase_remote_written": supabase_remote_written,
            "supabase_storage_uploaded": supabase_storage_uploaded,
            "supabase_storage": supabase_storage,
            "google_drive_uploaded": google_drive_uploaded,
            "google_drive": google_drive,
            "manifest": manifest,
        }

    def list_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_path):
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.history_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return rows[: max(limit, 1)]

    def _append_history(self, record: Dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _write_remote_record(
        self,
        *,
        backup_id: str,
        user_claims: Dict[str, Any],
        manifest: Dict[str, Any],
        bundle_bytes: bytes,
        sha256: str,
        size_bytes: int,
        google_drive: Optional[Dict[str, Any]] = None,
        supabase_storage: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.postgres_dsn or psycopg is None:
            return False

        with psycopg.connect(self.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cortex_profiles (
                        user_sub TEXT PRIMARY KEY,
                        email TEXT NOT NULL,
                        display_name TEXT,
                        avatar_url TEXT,
                        provider TEXT NOT NULL DEFAULT 'google',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO cortex_profiles (
                        user_sub,
                        email,
                        display_name,
                        avatar_url,
                        provider,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_sub) DO UPDATE SET
                        email = EXCLUDED.email,
                        display_name = EXCLUDED.display_name,
                        avatar_url = EXCLUDED.avatar_url,
                        provider = EXCLUDED.provider,
                        updated_at = NOW()
                    """,
                    (
                        str(user_claims.get("sub", "")),
                        str(user_claims.get("email", "")),
                        str(user_claims.get("name", "")),
                        str(user_claims.get("picture", "")),
                        str(user_claims.get("provider", "google")),
                    ),
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cortex_backup_snapshots (
                        backup_id TEXT PRIMARY KEY,
                        user_sub TEXT NOT NULL REFERENCES cortex_profiles(user_sub) ON DELETE CASCADE,
                        email TEXT NOT NULL,
                        provider TEXT NOT NULL DEFAULT 'google',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        size_bytes BIGINT NOT NULL,
                        sha256 TEXT NOT NULL,
                        manifest JSONB NOT NULL,
                        bundle BYTEA NOT NULL,
                        google_drive JSONB NOT NULL DEFAULT '{}'::jsonb,
                        supabase_storage JSONB NOT NULL DEFAULT '{}'::jsonb
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE cortex_backup_snapshots
                    ADD COLUMN IF NOT EXISTS google_drive JSONB NOT NULL DEFAULT '{}'::jsonb
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE cortex_backup_snapshots
                    ADD COLUMN IF NOT EXISTS supabase_storage JSONB NOT NULL DEFAULT '{}'::jsonb
                    """
                )
                cur.execute(
                    """
                    INSERT INTO cortex_backup_snapshots (
                        backup_id,
                        user_sub,
                        email,
                        provider,
                        size_bytes,
                        sha256,
                        manifest,
                        bundle,
                        google_drive,
                        supabase_storage
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (backup_id) DO NOTHING
                    """,
                    (
                        backup_id,
                        str(user_claims.get("sub", "")),
                        str(user_claims.get("email", "")),
                        str(user_claims.get("provider", "google")),
                        int(size_bytes),
                        sha256,
                        json.dumps(manifest),
                        bundle_bytes,
                        json.dumps(google_drive or {}),
                        json.dumps(supabase_storage or {}),
                    ),
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cortex_backup_files (
                        file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        backup_id TEXT NOT NULL REFERENCES cortex_backup_snapshots(backup_id) ON DELETE CASCADE,
                        user_sub TEXT NOT NULL REFERENCES cortex_profiles(user_sub) ON DELETE CASCADE,
                        provider TEXT NOT NULL,
                        external_file_id TEXT,
                        folder_id TEXT,
                        web_view_link TEXT,
                        sha256 TEXT,
                        size_bytes BIGINT,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                if google_drive and google_drive.get("uploaded"):
                    cur.execute(
                        """
                        INSERT INTO cortex_backup_files (
                            backup_id,
                            user_sub,
                            provider,
                            external_file_id,
                            folder_id,
                            web_view_link,
                            sha256,
                            size_bytes,
                            metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            backup_id,
                            str(user_claims.get("sub", "")),
                            "google_drive",
                            str(google_drive.get("file_id", "")),
                            str(google_drive.get("folder_id", "")),
                            str(google_drive.get("web_view_link", "")),
                            sha256,
                            int(size_bytes),
                            json.dumps(google_drive),
                        ),
                    )
                if supabase_storage and supabase_storage.get("uploaded"):
                    cur.execute(
                        """
                        INSERT INTO cortex_backup_files (
                            backup_id,
                            user_sub,
                            provider,
                            external_file_id,
                            folder_id,
                            web_view_link,
                            sha256,
                            size_bytes,
                            metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            backup_id,
                            str(user_claims.get("sub", "")),
                            "supabase_storage",
                            str(supabase_storage.get("object_path", "")),
                            str(supabase_storage.get("bucket", "")),
                            "",
                            sha256,
                            int(size_bytes),
                            json.dumps(supabase_storage),
                        ),
                    )
            conn.commit()
        return True

    def _supabase_storage_configured(self) -> bool:
        return bool(self.supabase_storage_client is not None)

    @staticmethod
    def _safe_object_segment(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
        return cleaned.strip("-") or "anonymous"

    def _write_supabase_storage_backup(
        self,
        *,
        user_claims: Dict[str, Any],
        bundle_path: str,
        backup_id: str,
        manifest: Dict[str, Any],
        sha256: str,
        size_bytes: int,
    ) -> Dict[str, Any]:
        if not self._supabase_storage_configured():
            return {"uploaded": False, "reason": "not_configured"}

        try:
            user_segment = self._safe_object_segment(str(user_claims.get("sub", "")))
            object_path = f"{user_segment}/{backup_id}.zip"
            upload = self.supabase_storage_client.upload_backup(
                bundle_path=bundle_path,
                object_path=object_path,
                metadata={
                    "backup_id": backup_id,
                    "user_sub": str(user_claims.get("sub", "")),
                    "email": str(user_claims.get("email", "")),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "created_at": manifest.get("created_at"),
                },
            )
            return {
                "uploaded": True,
                **dict(upload or {}),
            }
        except Exception as exc:
            return {
                "uploaded": False,
                "reason": "upload_failed",
                "error": str(exc),
            }

    def _google_drive_configured(self) -> bool:
        return bool(
            self.google_oauth_config is not None
            and self.google_refresh_token_resolver is not None
            and self.google_drive_client is not None
        )

    def _write_google_drive_backup(
        self,
        *,
        user_claims: Dict[str, Any],
        bundle_path: str,
        backup_id: str,
        manifest: Dict[str, Any],
        sha256: str,
        size_bytes: int,
    ) -> Dict[str, Any]:
        if not self._google_drive_configured():
            return {"uploaded": False, "reason": "not_configured"}

        try:
            refresh_token = self.google_refresh_token_resolver(user_claims)
            if not refresh_token:
                return {"uploaded": False, "reason": "missing_refresh_token"}

            token_payload = self.google_oauth_config.refresh_access_token(refresh_token)
            access_token = str(token_payload.get("access_token", "")).strip()
            if not access_token:
                return {"uploaded": False, "reason": "missing_access_token"}

            filename = os.path.basename(bundle_path)
            upload = self.google_drive_client.upload_backup(
                access_token=access_token,
                bundle_path=bundle_path,
                filename=filename,
                metadata={
                    "backup_id": backup_id,
                    "user_sub": str(user_claims.get("sub", "")),
                    "email": str(user_claims.get("email", "")),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "created_at": manifest.get("created_at"),
                },
            )
            return {
                "uploaded": True,
                **dict(upload or {}),
            }
        except Exception as exc:
            return {
                "uploaded": False,
                "reason": "upload_failed",
                "error": str(exc),
            }
