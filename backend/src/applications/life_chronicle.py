"""Phase 1 Life Chronicle passive-mode scaffolding service."""

from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso(value: str) -> datetime:
    normalized = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


class LifeChronicleService:
    """Passive life chronicle capture buffer with explicit consent controls."""

    def __init__(self, data_dir: str = "data/chronicle", buffer_seconds: int = 180) -> None:
        self._base_dir = Path(data_dir)
        self._moments_dir = self._base_dir / "moments"
        self._albums_dir = self._base_dir / "albums"
        self._people_dir = self._base_dir / "people_appearances"
        self._timeline_dir = self._base_dir / "timeline"

        self._moments_dir.mkdir(parents=True, exist_ok=True)
        self._albums_dir.mkdir(parents=True, exist_ok=True)
        self._people_dir.mkdir(parents=True, exist_ok=True)
        self._timeline_dir.mkdir(parents=True, exist_ok=True)

        self._buffer_seconds = max(int(buffer_seconds), 30)
        self._passive_enabled = False
        self._consent_granted_at = ""
        self._consent_actor = ""
        self._saved_count = 0
        self._buffer: deque[dict[str, Any]] = deque()
        self._lock = Lock()

    def status(self) -> dict[str, Any]:
        with self._lock:
            buffer_size = len(self._buffer)
            oldest = self._buffer[0]["timestamp"] if self._buffer else ""
            newest = self._buffer[-1]["timestamp"] if self._buffer else ""
            enabled = self._passive_enabled
            consent_at = self._consent_granted_at
            consent_actor = self._consent_actor
            saved = self._saved_count

        return {
            "service": "life_chronicle",
            "passive_mode_enabled": enabled,
            "consent_granted_at": consent_at,
            "consent_actor": consent_actor,
            "buffer_seconds": self._buffer_seconds,
            "buffer_entries": buffer_size,
            "buffer_oldest": oldest,
            "buffer_newest": newest,
            "saved_moments": saved,
            "updated_at": _utc_now_iso(),
        }

    def enable_passive_mode(self, *, consent_actor: str = "user") -> dict[str, Any]:
        actor = _safe_str(consent_actor) or "user"
        with self._lock:
            self._passive_enabled = True
            self._consent_granted_at = _utc_now_iso()
            self._consent_actor = actor
        return self.status()

    def disable_passive_mode(self, *, reason: str = "") -> dict[str, Any]:
        with self._lock:
            self._passive_enabled = False
            self._buffer.clear()
            if reason:
                self._consent_actor = f"{self._consent_actor or 'user'} (disabled: {reason})"
        return self.status()

    def add_passive_observation(
        self,
        *,
        note: str = "",
        location: dict[str, Any] | None = None,
        people_present: list[str] | None = None,
        tags: list[str] | None = None,
        media_ref: str = "",
        source: str = "passive_notification",
        emotion_hint: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            if not self._passive_enabled:
                raise RuntimeError("Passive mode is disabled")

            now = _utc_now()
            observation = {
                "observation_id": f"obs-{uuid.uuid4().hex[:12]}",
                "timestamp": now.isoformat(),
                "note": _safe_str(note),
                "location": dict(location or {}),
                "people_present": [p for p in (people_present or []) if _safe_str(p)],
                "tags": [t for t in (tags or []) if _safe_str(t)],
                "media_ref": _safe_str(media_ref),
                "source": _safe_str(source) or "passive_notification",
                "emotion_hint": _safe_str(emotion_hint),
            }
            self._buffer.append(observation)
            self._prune_locked(now)
            buffered = len(self._buffer)

        return {
            "status": "buffered",
            "observation_id": observation["observation_id"],
            "buffer_entries": buffered,
            "passive_mode_enabled": True,
        }

    def save_recent_window(
        self,
        *,
        title: str = "",
        window_seconds: int = 180,
        retrieval_hint: str = "",
        life_domain: str = "everyday",
    ) -> dict[str, Any]:
        window = max(int(window_seconds), 10)
        now = _utc_now()

        with self._lock:
            self._prune_locked(now)
            cutoff = now - timedelta(seconds=window)
            entries = [entry for entry in self._buffer if _parse_iso(entry["timestamp"]) >= cutoff]

        if not entries:
            return {
                "status": "skipped",
                "reason": "no_buffered_entries",
                "window_seconds": window,
            }

        people = self._dedupe([person for entry in entries for person in entry.get("people_present", [])])
        tags = self._dedupe([tag for entry in entries for tag in entry.get("tags", [])])
        media_refs = [entry.get("media_ref", "") for entry in entries if _safe_str(entry.get("media_ref"))]

        location = self._resolve_location(entries)
        tone = self._resolve_tone(entries)
        quote_candidates = [
            _safe_str(entry.get("note"))
            for entry in entries
            if len(_safe_str(entry.get("note"))) > 20
        ]
        key_quotes = quote_candidates[:3]

        moment_id = f"moment-{uuid.uuid4().hex[:12]}"
        narrative = self._build_narrative(entries, location_name=_safe_str(location.get("name")), tone=tone)

        moment = {
            "memory_id": moment_id,
            "type": "captured_moment",
            "capture_mode": "passive",
            "timestamp": now.isoformat(),
            "location": location,
            "people_present": people,
            "duration_seconds": window,
            "media": {
                "video_path": "",
                "thumbnail_paths": [],
                "audio_transcript": "",
                "observation_refs": [entry.get("observation_id", "") for entry in entries],
                "media_refs": media_refs,
            },
            "narrative": narrative,
            "key_quotes": key_quotes,
            "emotional_tone": tone,
            "life_domain": _safe_str(life_domain) or "everyday",
            "importance_score": self._importance_score(entries),
            "tags": tags,
            "retrieval_hint": _safe_str(retrieval_hint) or self._default_retrieval_hint(location, tags, people),
            "title": _safe_str(title) or f"Saved passive window {now.strftime('%Y-%m-%d %H:%M')}",
            "observation_count": len(entries),
        }

        file_path = self._persist_moment(moment)
        self._append_timeline(moment)
        self._update_people_indexes(moment)

        with self._lock:
            self._saved_count += 1

        return {
            "status": "saved",
            "memory_id": moment_id,
            "file_path": str(file_path),
            "observation_count": len(entries),
            "window_seconds": window,
            "moment": moment,
        }

    def list_moments(self, *, limit: int = 50, tag: str = "") -> list[dict[str, Any]]:
        timeline_path = self._timeline_dir / "timeline.jsonl"
        if not timeline_path.exists():
            return []

        with open(timeline_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        rows: list[dict[str, Any]] = []
        normalized_tag = _safe_str(tag).lower()
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if normalized_tag:
                tags = [str(t).lower() for t in row.get("tags", [])]
                if normalized_tag not in tags:
                    continue
            rows.append(row)
            if len(rows) >= max(limit, 1):
                break
        return rows

    def get_moment(self, memory_id: str) -> dict[str, Any] | None:
        target = _safe_str(memory_id)
        if not target:
            return None

        for date_dir in sorted(self._moments_dir.glob("*/*/*"), reverse=True):
            path = date_dir / f"{target}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        return None

    def heartbeat(self) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            self._prune_locked(now)
            enabled = self._passive_enabled
            entries = len(self._buffer)
        return {
            "status": "ok",
            "passive_mode_enabled": enabled,
            "buffer_entries": entries,
            "timestamp": now.isoformat(),
        }

    def _prune_locked(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._buffer_seconds)
        while self._buffer:
            oldest = self._buffer[0]
            try:
                oldest_ts = _parse_iso(oldest.get("timestamp", ""))
            except Exception:
                self._buffer.popleft()
                continue
            if oldest_ts < cutoff:
                self._buffer.popleft()
            else:
                break

    def _persist_moment(self, moment: dict[str, Any]) -> Path:
        timestamp = _parse_iso(moment["timestamp"])
        relative_dir = Path(str(timestamp.year)) / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        target_dir = self._moments_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        file_path = target_dir / f"{moment['memory_id']}.json"
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(moment, handle, ensure_ascii=True, indent=2)
        return file_path

    def _append_timeline(self, moment: dict[str, Any]) -> None:
        timeline_path = self._timeline_dir / "timeline.jsonl"
        with open(timeline_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(moment, ensure_ascii=True, default=str) + "\n")

    def _update_people_indexes(self, moment: dict[str, Any]) -> None:
        for person in moment.get("people_present", []):
            normalized = _safe_str(person).lower().replace(" ", "_")
            if not normalized:
                continue
            path = self._people_dir / f"{normalized}.json"
            payload = {
                "person": person,
                "moments": [],
            }
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    payload = {"person": person, "moments": []}

            payload.setdefault("moments", []).append(
                {
                    "memory_id": moment.get("memory_id", ""),
                    "timestamp": moment.get("timestamp", ""),
                    "title": moment.get("title", ""),
                }
            )

            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            normalized = _safe_str(value)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _resolve_location(entries: list[dict[str, Any]]) -> dict[str, Any]:
        for entry in reversed(entries):
            location = entry.get("location")
            if isinstance(location, dict) and location:
                return dict(location)
        return {"name": "", "coordinates": {}}

    @staticmethod
    def _resolve_tone(entries: list[dict[str, Any]]) -> str:
        hints = [
            _safe_str(entry.get("emotion_hint")).lower()
            for entry in entries
            if _safe_str(entry.get("emotion_hint"))
        ]
        if not hints:
            return "neutral"
        counts = {}
        for hint in hints:
            counts[hint] = counts.get(hint, 0) + 1
        return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]

    @staticmethod
    def _importance_score(entries: list[dict[str, Any]]) -> float:
        base = 0.35
        if len(entries) >= 5:
            base += 0.2
        if any(_safe_str(entry.get("people_present")) for entry in entries):
            base += 0.15
        if any(_safe_str(entry.get("emotion_hint")) for entry in entries):
            base += 0.15
        if any(_safe_str(entry.get("note")) for entry in entries):
            base += 0.1
        return round(min(base, 0.95), 2)

    @staticmethod
    def _build_narrative(entries: list[dict[str, Any]], *, location_name: str, tone: str) -> str:
        notes = [
            _safe_str(entry.get("note"))
            for entry in entries
            if _safe_str(entry.get("note"))
        ]
        excerpt = " ".join(notes[:3])
        location_text = f" at {location_name}" if location_name else ""
        if not excerpt:
            excerpt = "A passive moment was saved from recent contextual observations."
        return (
            f"This moment was captured passively{location_text}. "
            f"The overall tone felt {tone or 'neutral'}. "
            f"Notable context: {excerpt[:500]}"
        )

    @staticmethod
    def _default_retrieval_hint(location: dict[str, Any], tags: list[str], people: list[str]) -> str:
        if people:
            return f"Moment with {', '.join(people[:2])}"
        if tags:
            return f"Moment tagged {', '.join(tags[:2])}"
        location_name = _safe_str(location.get("name"))
        if location_name:
            return f"Moment captured near {location_name}"
        return "Recent passive capture"


life_chronicle_service = LifeChronicleService()
