"""
JSONL Session Persistence — Tree semantics aligned to pi-mono SessionManager.

Features:
- Append-only entries with parent pointers.
- Leaf pointer for branching/replay.
- Compaction-aware context reconstruction.
- Deferred persistence until first assistant response (or compaction).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


SESSION_VERSION = 3


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionEntry:
    entry_id: str
    entry_type: str  # session | message | compaction | custom | label | branch_summary
    parent_id: str | None
    data: dict[str, Any]
    timestamp: str = field(default_factory=_utcnow)


@dataclass
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    compacted_message_count: int
    timestamp: str = field(default_factory=_utcnow)


class SessionPersistence:
    """Append-only JSONL session tree with compaction-aware replay."""

    def __init__(
        self,
        session_id: str,
        agent_id: str,
        base_dir: str = "data/sessions",
        parent_id: str | None = None,
        title: str = "",
    ):
        self.session_id = session_id
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.title = title or f"Session {session_id[:8]}"
        self._base_dir = Path(base_dir) / agent_id
        self._file_path = self._base_dir / f"{session_id}.jsonl"

        self._entries: list[SessionEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._labels_by_id: dict[str, str] = {}
        self._leaf_id: str | None = None

        self._lock = Lock()
        self._persisted = False

    # ── Constructors ─────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        agent_id: str,
        base_dir: str = "data/sessions",
        title: str = "",
        parent_id: str | None = None,
    ) -> "SessionPersistence":
        session_id = str(uuid.uuid4())
        session = cls(
            session_id=session_id,
            agent_id=agent_id,
            base_dir=base_dir,
            parent_id=parent_id,
            title=title,
        )
        header = SessionEntry(
            entry_id=str(uuid.uuid4()),
            entry_type="session",
            parent_id=None,
            data={
                "type": "session",
                "version": SESSION_VERSION,
                "id": session_id,
                "parentId": parent_id,
                "title": session.title,
                "createdAt": _utcnow(),
            },
        )
        session._entries.append(header)
        return session

    @classmethod
    def open(
        cls,
        session_id: str,
        agent_id: str,
        base_dir: str = "data/sessions",
    ) -> "SessionPersistence":
        session = cls(session_id=session_id, agent_id=agent_id, base_dir=base_dir)
        file_path = session._file_path
        if not file_path.exists():
            return session

        previous_id: str | None = None
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line:
                    continue
                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                entry_id = str(raw.get("entry_id") or raw.get("id") or str(uuid.uuid4()))
                entry_type = str(raw.get("entry_type") or raw.get("type") or "message")
                parent_id = raw.get("parentId")
                if parent_id is None and entry_type != "session":
                    # Backward compatibility with older linear JSONL files.
                    parent_id = previous_id

                entry = SessionEntry(
                    entry_id=entry_id,
                    entry_type=entry_type,
                    parent_id=parent_id,
                    data=dict(raw),
                    timestamp=str(raw.get("timestamp") or raw.get("createdAt") or _utcnow()),
                )
                session._entries.append(entry)
                if entry_type != "session":
                    session._by_id[entry_id] = entry
                    session._leaf_id = entry_id
                    previous_id = entry_id

                if entry_type == "label":
                    target_id = str(raw.get("targetId") or "")
                    label = raw.get("label")
                    if target_id:
                        if isinstance(label, str) and label.strip():
                            session._labels_by_id[target_id] = label.strip()
                        elif target_id in session._labels_by_id:
                            session._labels_by_id.pop(target_id, None)

        session._persisted = True

        if session._entries:
            header = session._entries[0].data
            session.parent_id = header.get("parentId")
            session.title = str(header.get("title") or session.title)

        return session

    # ── Branch / Leaf Management ─────────────────────────────────────────

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    def branch(self, branch_from_id: str) -> None:
        with self._lock:
            if branch_from_id not in self._by_id:
                raise KeyError(f"Entry not found: {branch_from_id}")
            self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        with self._lock:
            self._leaf_id = None

    def append_label_change(self, target_id: str, label: str | None) -> SessionEntry:
        with self._lock:
            if target_id not in self._by_id:
                raise KeyError(f"Entry not found: {target_id}")
            entry = self._append_entry_locked(
                entry_type="label",
                data={
                    "type": "label",
                    "targetId": target_id,
                    "label": label,
                    "timestamp": _utcnow(),
                },
            )
            if isinstance(label, str) and label.strip():
                self._labels_by_id[target_id] = label.strip()
            else:
                self._labels_by_id.pop(target_id, None)
            return entry

    def get_branch(self, leaf_id: str | None = None) -> list[SessionEntry]:
        with self._lock:
            return self._build_path_locked(leaf_id=leaf_id)

    # ── Append Operations ────────────────────────────────────────────────

    def append_message(
        self,
        role: str,
        content: str,
        tool_use: list[dict] | None = None,
        tool_call_id: str = "",
        is_error: bool = False,
        source: str = "",
    ) -> SessionEntry:
        entry_data: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": content,
            "timestamp": _utcnow(),
        }
        if tool_use:
            entry_data["tool_use"] = tool_use
        if tool_call_id:
            entry_data["tool_call_id"] = tool_call_id
        if is_error:
            entry_data["is_error"] = True
        if source:
            entry_data["source"] = source

        with self._lock:
            return self._append_entry_locked("message", entry_data)

    def append_compaction(self, summary: str) -> CompactionResult:
        with self._lock:
            path = self._build_path_locked()
            path_messages = [e for e in path if e.entry_type == "message"]

            if path_messages:
                keep_window = 24
                first_kept = path_messages[max(0, len(path_messages) - keep_window)].entry_id
            else:
                first_kept = self._leaf_id or ""

            entry = self._append_entry_locked(
                "compaction",
                {
                    "type": "compaction",
                    "summary": summary,
                    "firstKeptEntryId": first_kept,
                    "timestamp": _utcnow(),
                },
            )

            return CompactionResult(
                summary=summary,
                first_kept_entry_id=str(entry.data.get("firstKeptEntryId") or ""),
                compacted_message_count=len(path_messages),
            )

    def append_custom(
        self,
        source: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionEntry:
        entry_data: dict[str, Any] = {
            "type": "custom_message",
            "role": "custom",
            "source": source,
            "content": content,
            "timestamp": _utcnow(),
        }
        if metadata:
            entry_data["metadata"] = metadata

        with self._lock:
            return self._append_entry_locked("custom", entry_data)

    # ── Context Reconstruction (pi-mono semantics) ───────────────────────

    def build_session_context(self, leaf_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            path = self._build_path_locked(leaf_id=leaf_id)

        if not path:
            return []

        compaction_idx = -1
        compaction_entry: SessionEntry | None = None
        for idx, entry in enumerate(path):
            if entry.entry_type == "compaction":
                compaction_idx = idx
                compaction_entry = entry

        messages: list[dict[str, Any]] = []

        def append_payload(entry: SessionEntry) -> None:
            if entry.entry_type == "message":
                payload = self._entry_to_message_payload(entry)
                if payload:
                    messages.append(payload)
            elif entry.entry_type == "custom":
                payload = self._entry_to_message_payload(entry)
                if payload:
                    messages.append(payload)

        if compaction_entry is not None:
            summary = str(compaction_entry.data.get("summary") or "").strip()
            if summary:
                messages.append({
                    "role": "system",
                    "content": f"[Session Summary]\n{summary}",
                })

            first_kept = str(compaction_entry.data.get("firstKeptEntryId") or "")
            found_first_kept = not first_kept

            for idx in range(0, compaction_idx):
                candidate = path[idx]
                if candidate.entry_id == first_kept:
                    found_first_kept = True
                if found_first_kept:
                    append_payload(candidate)

            for idx in range(compaction_idx + 1, len(path)):
                append_payload(path[idx])
        else:
            for entry in path:
                append_payload(entry)

        return messages

    def get_messages_before_boundary(self) -> list[dict[str, Any]]:
        with self._lock:
            path = self._build_path_locked()

        last_compaction_idx = -1
        for i, entry in enumerate(path):
            if entry.entry_type == "compaction":
                last_compaction_idx = i

        if last_compaction_idx >= 0:
            candidates = path[:last_compaction_idx]
        else:
            candidates = path

        result: list[dict[str, Any]] = []
        for entry in candidates:
            if entry.entry_type in {"message", "custom"}:
                payload = self._entry_to_message_payload(entry)
                if payload:
                    result.append(payload)
        return result

    # ── Utilities ────────────────────────────────────────────────────────

    def get_all_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._serialize_entry(entry) for entry in self._entries]

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._by_id.get(entry_id)
            if not entry:
                return None
            return self._serialize_entry(entry)

    def strip_last_assistant(self) -> bool:
        """Strip last assistant leaf message for retry semantics."""
        with self._lock:
            if not self._leaf_id:
                return False

            leaf = self._by_id.get(self._leaf_id)
            if not leaf or leaf.entry_type != "message":
                return False

            if str(leaf.data.get("role", "")).lower() != "assistant":
                return False

            leaf_id = leaf.entry_id
            self._entries = [e for e in self._entries if e.entry_id != leaf_id]
            self._by_id.pop(leaf_id, None)
            self._leaf_id = leaf.parent_id

            # Persisted files are append-only. To avoid rewriting on every retry,
            # mark as not persisted so the next assistant write rewrites the file.
            self._persisted = False
            return True

    def fork(self, title: str = "") -> "SessionPersistence":
        return SessionPersistence.create(
            agent_id=self.agent_id,
            base_dir=str(self._base_dir.parent),
            title=title or f"Fork of {self.title}",
            parent_id=self.session_id,
        )

    @property
    def message_count(self) -> int:
        with self._lock:
            return sum(1 for e in self._entries if e.entry_type in {"message", "custom"})

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _append_entry_locked(self, entry_type: str, data: dict[str, Any]) -> SessionEntry:
        entry = SessionEntry(
            entry_id=str(uuid.uuid4()),
            entry_type=entry_type,
            parent_id=self._leaf_id,
            data=data,
            timestamp=str(data.get("timestamp") or _utcnow()),
        )
        self._entries.append(entry)
        self._by_id[entry.entry_id] = entry
        self._leaf_id = entry.entry_id
        self._persist_entry_locked(entry)
        return entry

    def _build_path_locked(self, leaf_id: str | None = None) -> list[SessionEntry]:
        cursor = leaf_id if leaf_id is not None else self._leaf_id
        if cursor is None:
            return []

        path_rev: list[SessionEntry] = []
        visited: set[str] = set()
        while cursor:
            if cursor in visited:
                break
            visited.add(cursor)

            current = self._by_id.get(cursor)
            if current is None:
                break
            path_rev.append(current)
            cursor = current.parent_id

        path_rev.reverse()
        return path_rev

    def _entry_to_message_payload(self, entry: SessionEntry) -> dict[str, Any] | None:
        payload = dict(entry.data)
        payload.pop("type", None)
        payload.setdefault("timestamp", entry.timestamp)

        role = str(payload.get("role", "")).strip().lower()
        if entry.entry_type in {"message", "custom"} and role:
            payload["role"] = role
            payload.setdefault("content", "")
            return payload

        return None

    def _serialize_entry(self, entry: SessionEntry) -> dict[str, Any]:
        payload = dict(entry.data)
        payload.setdefault("type", entry.entry_type)
        payload["entry_id"] = entry.entry_id
        payload["entry_type"] = entry.entry_type
        payload["parentId"] = entry.parent_id
        payload["timestamp"] = entry.timestamp
        return payload

    def _persist_entry_locked(self, entry: SessionEntry) -> None:
        """Persist append-only JSONL with deferred creation semantics."""
        should_flush_full = False
        if not self._persisted:
            role = str(entry.data.get("role", "")).strip().lower()
            if role != "assistant" and entry.entry_type != "compaction":
                return
            self._persisted = True
            should_flush_full = True

        self._base_dir.mkdir(parents=True, exist_ok=True)

        if should_flush_full:
            with open(self._file_path, "w", encoding="utf-8") as f:
                for item in self._entries:
                    f.write(json.dumps(self._serialize_entry(item), ensure_ascii=False) + "\n")
            return

        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._serialize_entry(entry), ensure_ascii=False) + "\n")

    def to_jsonl(self) -> str:
        with self._lock:
            lines = [json.dumps(self._serialize_entry(e), ensure_ascii=False) for e in self._entries]
        return "\n".join(lines)
