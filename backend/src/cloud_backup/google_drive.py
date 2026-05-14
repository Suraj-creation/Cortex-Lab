"""Google Drive backup upload client."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


class GoogleDriveBackupClient:
    """Uploads Cortex backup bundles into a user-owned Google Drive folder."""

    def __init__(
        self,
        *,
        root_folder_name: str = "Cortex Lab Backups",
        opener: Callable[..., Any] = urlopen,
    ):
        self.root_folder_name = root_folder_name.strip() or "Cortex Lab Backups"
        self._opener = opener

    def upload_backup(
        self,
        *,
        access_token: str,
        bundle_path: str,
        filename: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not access_token.strip():
            raise ValueError("Google Drive access token is required")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(bundle_path)

        root_id = self._ensure_folder(access_token, self.root_folder_name)
        user_label = (
            str(metadata.get("email") or "").strip()
            or str(metadata.get("user_sub") or "").strip()
            or "anonymous"
        )
        user_folder_id = self._ensure_folder(access_token, user_label, parent_id=root_id)

        result = self._upload_multipart(
            access_token=access_token,
            bundle_path=bundle_path,
            filename=filename,
            parent_id=user_folder_id,
            metadata=metadata,
        )
        return {
            "file_id": result.get("id", ""),
            "folder_id": user_folder_id,
            "root_folder_id": root_id,
            "web_view_link": result.get("webViewLink", ""),
            "web_content_link": result.get("webContentLink", ""),
        }

    def _ensure_folder(
        self,
        access_token: str,
        name: str,
        parent_id: Optional[str] = None,
    ) -> str:
        query = (
            "mimeType='application/vnd.google-apps.folder' "
            f"and name='{_escape_drive_query_value(name)}' "
            "and trashed=false"
        )
        if parent_id:
            query += f" and '{_escape_drive_query_value(parent_id)}' in parents"

        search_url = (
            "https://www.googleapis.com/drive/v3/files?"
            + urlencode(
                {
                    "q": query,
                    "fields": "files(id,name)",
                    "pageSize": "1",
                }
            )
        )
        payload = self._request_json("GET", search_url, access_token=access_token)
        files = payload.get("files") if isinstance(payload, dict) else None
        if isinstance(files, list) and files:
            folder_id = str(files[0].get("id", "")).strip()
            if folder_id:
                return folder_id

        body: Dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            body["parents"] = [parent_id]

        created = self._request_json(
            "POST",
            "https://www.googleapis.com/drive/v3/files?fields=id,name",
            access_token=access_token,
            payload=body,
        )
        folder_id = str(created.get("id", "")).strip()
        if not folder_id:
            raise RuntimeError("Google Drive folder creation did not return an id")
        return folder_id

    def _upload_multipart(
        self,
        *,
        access_token: str,
        bundle_path: str,
        filename: str,
        parent_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        mime_type = mimetypes.guess_type(filename)[0] or "application/zip"
        drive_metadata = {
            "name": filename,
            "parents": [parent_id],
            "mimeType": mime_type,
            "description": json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        }
        with open(bundle_path, "rb") as handle:
            file_bytes = handle.read()

        boundary = f"cortex-{uuid.uuid4().hex}"
        delimiter = f"--{boundary}\r\n".encode("utf-8")
        close_delimiter = f"--{boundary}--\r\n".encode("utf-8")
        body = b"".join(
            [
                delimiter,
                b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
                json.dumps(drive_metadata, separators=(",", ":")).encode("utf-8"),
                b"\r\n",
                delimiter,
                f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
                file_bytes,
                b"\r\n",
                close_delimiter,
            ]
        )
        request = Request(
            "https://www.googleapis.com/upload/drive/v3/files?"
            + urlencode(
                {
                    "uploadType": "multipart",
                    "fields": "id,name,webViewLink,webContentLink,parents",
                }
            ),
            data=body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )
        with self._opener(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        access_token: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        request = Request(url, data=data, headers=headers, method=method)
        with self._opener(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}
