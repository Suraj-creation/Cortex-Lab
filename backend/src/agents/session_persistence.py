"""
JSONL Session Persistence — Pi-Mono SessionManager Pattern.
Source reference: pi-mono/packages/coding-agent/src/core/session-manager.ts

Append-only JSONL with session tree (fork/branch) and compaction boundaries.
buildSessionContext() walks leaf→root applying compaction boundaries.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional


SESSION_VERSION = 3


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionEntry:
    entry_id: str
    entry_type: str  # "session" | "message" | "compaction" | "custom"
    data: dict[str, Any]
    timestamp: str = field(default_factory=_utcnow)


@dataclass
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    compacted_message_count: int
    timestamp: str = field(default_factory=_utcnow)


class SessionPersistence:
    """
    JSONL append-only session store.
    Maps to pi-mono's SessionManager:
    - create/open/forkFrom
    - appendMessage/appendCompaction/appendCustomEntry
    - buildSessionContext (compaction-aware)
    """

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
        self._lock = Lock()
        self._persisted = False

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
    def open(cls, session_id: str, agent_id: str, base_dir: str = "data/sessions") -> "SessionPersistence":
        session = cls(session_id=session_id, agent_id=agent_id, base_dir=base_dir)
        file_path = session._file_path
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    session._entries.append(SessionEntry(
                        entry_id=raw.get("entry_id", str(uuid.uuid4())),
                        entry_type=raw.get("entry_type", raw.get("type", "message")),
                        data=raw,
                        timestamp=raw.get("timestamp", ""),
                    ))
            session._persisted = True
            header = session._entries[0].data if session._entries else {}
            session.parent_id = header.get("parentId")
            session.title = header.get("title", "")
        return session

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

        entry = SessionEntry(
            entry_id=str(uuid.uuid4()),
            entry_type="message",
            data=entry_data,
        )
        with self._lock:
            self._entries.append(entry)
            self._persist_entry(entry)
        return entry

    def append_compaction(self, summary: str) -> CompactionResult:
        kept_id = self._entries[-1].entry_id if self._entries else ""
        msg_count = sum(1 for e in self._entries if e.entry_type == "message")

        entry_data = {
            "type": "compaction",
            "summary": summary,
            "firstKeptEntryId": kept_id,
            "timestamp": _utcnow(),
        }
        entry = SessionEntry(
            entry_id=str(uuid.uuid4()),
            entry_type="compaction",
            data=entry_data,
        )
        with self._lock:
            self._entries.append(entry)
            self._persist_entry(entry)

        return CompactionResult(
            summary=summary,
            first_kept_entry_id=kept_id,
            compacted_message_count=msg_count,
        )

    def append_custom(self, source: str, content: str, metadata: dict[str, Any] | None = None) -> SessionEntry:
        entry_data: dict[str, Any] = {
            "type": "message",
            "role": "custom",
            "source": source,
            "content": content,
            "timestamp": _utcnow(),
        }
        if metadata:
            entry_data["metadata"] = metadata

        entry = SessionEntry(
            entry_id=str(uuid.uuid4()),
            entry_type="message",
            data=entry_data,
        )
        with self._lock:
            self._entries.append(entry)
            self._persist_entry(entry)
        return entry

    def build_session_context(self) -> list[dict[str, Any]]:
        """
        Pi-mono pattern (session-manager.ts buildSessionContext):
        1. Find latest compaction entry
        2. Return: [compaction.summary as system msg] + [messages after boundary]
        3. If no compaction: return all messages
        """
        with self._lock:
            entries = list(self._entries)

        last_compaction = None
        last_compaction_idx = -1
        for i, entry in enumerate(entries):
            if entry.entry_type == "compaction":
                last_compaction = entry
                last_compaction_idx = i

        messages: list[dict[str, Any]] = []

        if last_compaction:
            summary = last_compaction.data.get("summary", "")
            if summary:
                messages.append({
                    "role": "system",
                    "content": f"[Session Summary]\n{summary}",
                })
            for entry in entries[last_compaction_idx + 1:]:
                if entry.entry_type == "message":
                    messages.append(entry.data)
        else:
            for entry in entries:
                if entry.entry_type == "message":
                    messages.append(entry.data)

        return messages

    def get_all_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e.data for e in self._entries]

    def get_messages_before_boundary(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._entries)

        last_compaction_idx = -1
        for i, entry in enumerate(entries):
            if entry.entry_type == "compaction":
                last_compaction_idx = i

        if last_compaction_idx >= 0:
            return [e.data for e in entries[:last_compaction_idx] if e.entry_type == "message"]
        return [e.data for e in entries if e.entry_type == "message"]

    def strip_last_assistant(self) -> bool:
        with self._lock:
            for i in range(len(self._entries) - 1, -1, -1):
                if (self._entries[i].entry_type == "message"
                        and self._entries[i].data.get("role") == "assistant"):
                    self._entries.pop(i)
                    return True
        return False

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
            return sum(1 for e in self._entries if e.entry_type == "message")

    def _persist_entry(self, entry: SessionEntry) -> None:
        """Pi-mono: _persist defers file creation until first assistant message."""
        if not self._persisted:
            role = entry.data.get("role", "")
            if role != "assistant" and entry.entry_type != "compaction":
                return
            self._persisted = True
            self._base_dir.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, "w", encoding="utf-8") as f:
                for e in self._entries:
                    line = json.dumps({**e.data, "entry_id": e.entry_id}, ensure_ascii=False)
                    f.write(line + "\n")
            return

        with open(self._file_path, "a", encoding="utf-8") as f:
            line = json.dumps({**entry.data, "entry_id": entry.entry_id}, ensure_ascii=False)
            f.write(line + "\n")

    def to_jsonl(self) -> str:
        with self._lock:
            lines = []
            for e in self._entries:
                lines.append(json.dumps({**e.data, "entry_id": e.entry_id}, ensure_ascii=False))
            return "\n".join(lines)
