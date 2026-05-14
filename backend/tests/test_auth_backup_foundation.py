"""Regression tests for auth and cloud-backup foundation services."""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from zipfile import ZipFile
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import server as backend_server
from src.auth.service import AppAuthService, GoogleOAuthConfig
from src.cloud_backup.service import BackupWorkspaceInspector


def test_app_auth_service_round_trips_session_token():
    service = AppAuthService(app_secret="test-secret", session_ttl_seconds=3600)

    token = service.issue_session_token(
        {
            "sub": "google-oauth2|user-123",
            "email": "eva@example.com",
            "name": "Eva Cortex",
            "picture": "https://example.com/avatar.png",
        }
    )

    claims = service.verify_session_token(token)

    assert claims["sub"] == "google-oauth2|user-123"
    assert claims["email"] == "eva@example.com"
    assert claims["name"] == "Eva Cortex"
    assert claims["provider"] == "google"


def test_google_oauth_config_builds_expected_authorize_url():
    cfg = GoogleOAuthConfig(
        client_id="google-client-id",
        client_secret="google-client-secret",
        redirect_uri="https://cortex.example.com/api/auth/google/callback",
        scopes=["openid", "email", "profile", "https://www.googleapis.com/auth/drive.file"],
    )

    url = cfg.build_authorize_url(
        state="signed-state",
        prompt="consent",
        access_type="offline",
        include_granted_scopes=True,
    )

    assert "accounts.google.com/o/oauth2/v2/auth" in url
    assert "client_id=google-client-id" in url
    assert "state=signed-state" in url
    assert "access_type=offline" in url
    assert "include_granted_scopes=true" in url


def test_backup_workspace_inspector_builds_bundle_with_manifest(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "graph").mkdir(parents=True)
    (data_root / "wiki" / "pages").mkdir(parents=True)
    (data_root / "chronicle").mkdir(parents=True)

    (data_root / "cortex.duckdb").write_bytes(b"duckdb-bytes")
    (data_root / "graph" / "knowledge_graph.json").write_text(
        json.dumps({"nodes": [{"id": "eva"}], "links": []}),
        encoding="utf-8",
    )
    (data_root / "wiki" / "pages" / "eva.md").write_text(
        "# Eva\n\nRemembers what matters.",
        encoding="utf-8",
    )
    (data_root / "ambient_config.json").write_text(
        json.dumps({"assistant_name": "Eva"}),
        encoding="utf-8",
    )

    inspector = BackupWorkspaceInspector(str(data_root))
    bundle_path = inspector.build_bundle(
        output_dir=str(tmp_path),
        user_snapshot={
            "device": "android",
            "conversation_count": 4,
        },
    )

    assert os.path.exists(bundle_path)

    with ZipFile(bundle_path, "r") as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "cortex.duckdb" in names
        assert "graph/knowledge_graph.json" in names
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["local_workspace"]["duckdb"]["exists"] is True
    assert manifest["local_workspace"]["wiki_pages"] == 1
    assert manifest["local_workspace"]["graph_file"]["exists"] is True
    assert manifest["user_snapshot"]["device"] == "android"
    assert manifest["user_snapshot"]["conversation_count"] == 4


class _FakeBackupCoordinator:
    def __init__(self):
        self.calls = []

    def status(self):
        return {
            "enabled": True,
            "supabase_postgres_configured": True,
            "google_drive_configured": False,
            "last_backup_at": None,
        }

    def create_backup(self, user_claims, client_snapshot=None):
        self.calls.append(
            {
                "user": user_claims,
                "client_snapshot": client_snapshot or {},
            }
        )
        return {
            "status": "created",
            "backup_id": "backup-123",
            "manifest": {
                "user_snapshot": client_snapshot or {},
            },
        }


def _auth_client(monkeypatch):
    service = AppAuthService(app_secret="test-secret", session_ttl_seconds=3600)
    backup = _FakeBackupCoordinator()

    @asynccontextmanager
    async def _no_lifespan(_app):
        yield

    monkeypatch.setattr(backend_server.app.router, "lifespan_context", _no_lifespan)
    monkeypatch.setattr(backend_server, "_app_auth_service", service)
    monkeypatch.setattr(backend_server, "_google_oauth_config", None)
    monkeypatch.setattr(backend_server, "_cloud_backup_coordinator", backup)
    client = TestClient(backend_server.app)
    return client, service, backup


def test_auth_status_reports_google_configuration(monkeypatch):
    client, _service, _backup = _auth_client(monkeypatch)

    with client:
        response = client.get("/api/auth/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["google"]["configured"] is False
    assert payload["backup"]["supabase_postgres_configured"] is True


def test_auth_me_accepts_signed_bearer_token(monkeypatch):
    client, service, _backup = _auth_client(monkeypatch)
    token = service.issue_session_token(
        {
            "sub": "google-oauth2|me-123",
            "email": "me@example.com",
            "name": "Me",
        }
    )

    with client:
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is True
    assert payload["user"]["email"] == "me@example.com"


def test_backup_run_requires_user_and_forwards_client_snapshot(monkeypatch):
    client, service, backup = _auth_client(monkeypatch)
    token = service.issue_session_token(
        {
            "sub": "google-oauth2|backup-123",
            "email": "backup@example.com",
            "name": "Backup User",
        }
    )

    with client:
        response = client.post(
            "/api/backup/run",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "client_snapshot": {
                    "platform": "android",
                    "conversation_count": 12,
                }
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "created"
    assert backup.calls[0]["user"]["email"] == "backup@example.com"
    assert backup.calls[0]["client_snapshot"]["platform"] == "android"


def test_cloud_backup_uploads_google_drive_when_refresh_token_available(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "ambient_config.json").write_text(
        json.dumps({"assistant_name": "Eva"}),
        encoding="utf-8",
    )

    class _FakeGoogleOAuth:
        def refresh_access_token(self, refresh_token):
            assert refresh_token == "refresh-token-1"
            return {"access_token": "access-token-1"}

    class _FakeDriveClient:
        def __init__(self):
            self.uploads = []

        def upload_backup(self, *, access_token, bundle_path, filename, metadata):
            self.uploads.append(
                {
                    "access_token": access_token,
                    "bundle_path": bundle_path,
                    "filename": filename,
                    "metadata": metadata,
                }
            )
            assert os.path.exists(bundle_path)
            return {
                "file_id": "drive-file-123",
                "folder_id": "drive-folder-123",
                "web_view_link": "https://drive.google.com/file/d/drive-file-123/view",
            }

    from src.cloud_backup.service import CloudBackupCoordinator

    drive_client = _FakeDriveClient()
    coordinator = CloudBackupCoordinator(
        data_root=str(data_root),
        history_path=str(tmp_path / "history.jsonl"),
        google_oauth_config=_FakeGoogleOAuth(),
        google_refresh_token_resolver=lambda claims: "refresh-token-1",
        google_drive_client=drive_client,
    )

    result = coordinator.create_backup(
        {
            "sub": "google-oauth2|drive-user",
            "email": "drive@example.com",
            "provider": "google",
        },
        client_snapshot={"platform": "android"},
    )

    assert result["google_drive_uploaded"] is True
    assert result["google_drive"]["file_id"] == "drive-file-123"
    assert drive_client.uploads[0]["access_token"] == "access-token-1"
    assert drive_client.uploads[0]["metadata"]["user_sub"] == "google-oauth2|drive-user"


def test_cloud_backup_uploads_supabase_storage_when_configured(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "ambient_config.json").write_text(
        json.dumps({"assistant_name": "Eva"}),
        encoding="utf-8",
    )

    class _FakeSupabaseStorageClient:
        def __init__(self):
            self.uploads = []

        def upload_backup(self, *, bundle_path, object_path, metadata):
            self.uploads.append(
                {
                    "bundle_path": bundle_path,
                    "object_path": object_path,
                    "metadata": metadata,
                }
            )
            assert os.path.exists(bundle_path)
            return {
                "bucket": "cortex-backups",
                "object_path": object_path,
            }

    from src.cloud_backup.service import CloudBackupCoordinator

    storage_client = _FakeSupabaseStorageClient()
    coordinator = CloudBackupCoordinator(
        data_root=str(data_root),
        history_path=str(tmp_path / "history.jsonl"),
        supabase_storage_client=storage_client,
    )

    result = coordinator.create_backup(
        {
            "sub": "google-oauth2|storage-user",
            "email": "storage@example.com",
            "provider": "google",
        },
        client_snapshot={"platform": "android"},
    )

    assert result["supabase_storage_uploaded"] is True
    assert result["supabase_storage"]["bucket"] == "cortex-backups"
    assert storage_client.uploads[0]["metadata"]["user_sub"] == "google-oauth2|storage-user"
    assert storage_client.uploads[0]["object_path"].startswith("google-oauth2-storage-user/")


def test_hybrid_request_provider_falls_back_to_gemini_when_local_unavailable(monkeypatch):
    monkeypatch.setattr(
        backend_server,
        "_llm_provider_availability",
        lambda: {"local": False, "gemma_local": False, "gemini": True},
    )

    assert (
        backend_server._resolve_effective_request_llm_provider(
            "gemma_local",
            allow_cloud_fallback=True,
            mode="hybrid",
        )
        == "gemini"
    )

    with pytest.raises(ValueError, match="local_unavailable"):
        backend_server._resolve_effective_request_llm_provider(
            "gemma_local",
            allow_cloud_fallback=False,
            mode="hybrid",
        )


def test_supabase_schema_covers_local_first_auth_backup_and_realtime():
    schema_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "infra",
        "supabase",
        "cortex_backup_schema.sql",
    )
    with open(schema_path, "r", encoding="utf-8") as handle:
        schema = handle.read().lower()

    for table_name in [
        "cortex_profiles",
        "cortex_devices",
        "cortex_sync_cursors",
        "cortex_memory_events",
        "cortex_backup_snapshots",
        "cortex_backup_files",
        "cortex_realtime_events",
    ]:
        assert f"create table if not exists public.{table_name}" in schema

    assert "alter table public.cortex_backup_snapshots enable row level security" in schema
    assert "publication supabase_realtime" in schema


def test_supabase_database_url_composition_escapes_special_password(monkeypatch):
    monkeypatch.delenv("SUPABASE_DATABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SUPABASE_DB_HOST", "db.example.supabase.co")
    monkeypatch.setenv("SUPABASE_DB_PORT", "5432")
    monkeypatch.setenv("SUPABASE_DB_NAME", "postgres")
    monkeypatch.setenv("SUPABASE_DB_USER", "postgres")
    monkeypatch.setenv("SUPABASE_DB_PASSWORD", "p@ss/word!")

    composed = backend_server._compose_supabase_database_url()
    parsed = urlparse(composed)

    assert parsed.hostname == "db.example.supabase.co"
    assert parsed.path == "/postgres"
    assert "p%40ss%2Fword%21" in composed
