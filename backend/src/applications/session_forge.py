"""Phase 1 Session Memory Forge scaffolding service."""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
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


def _coerce_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return _parse_iso(value)
        except Exception:
            return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _extract_topics_from_text(text: str) -> list[str]:
    stop = {
        "about",
        "after",
        "again",
        "because",
        "before",
        "being",
        "could",
        "doing",
        "from",
        "have",
        "into",
        "just",
        "more",
        "need",
        "over",
        "really",
        "should",
        "some",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "thing",
        "think",
        "with",
        "would",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", text.lower())
    counts = Counter(w for w in words if w not in stop)
    return [topic for topic, _ in counts.most_common(5)]


class SessionMemoryForgeService:
    """Turns recent session material into structured artifacts."""

    _ARTIFACT_FILES = {
        "thought_objects": "thought_objects.jsonl",
        "decision_records": "decision_records.jsonl",
        "open_loops": "open_loops.jsonl",
        "gap_signals": "gap_signals.jsonl",
        "belief_evolution": "belief_evolution.jsonl",
        "structured_summaries": "structured_summaries.jsonl",
        "runs": "runs.jsonl",
    }

    def __init__(self, data_dir: str = "data/deep_apps/session_forge") -> None:
        self._base_dir = Path(data_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def status(self) -> dict[str, Any]:
        return {
            "service": "session_memory_forge",
            "base_dir": str(self._base_dir),
            "artifacts": {
                key: self._count_lines(self._path(key))
                for key in self._ARTIFACT_FILES
                if key != "runs"
            },
            "runs": self._count_lines(self._path("runs")),
            "updated_at": _utc_now_iso(),
        }

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._read_jsonl(self._path("runs"), limit=max(limit, 1))

    def list_artifacts(
        self,
        artifact_type: str,
        *,
        limit: int = 50,
        session_id: str = "",
    ) -> list[dict[str, Any]]:
        if artifact_type not in self._ARTIFACT_FILES or artifact_type == "runs":
            raise ValueError(f"Unsupported artifact type: {artifact_type}")

        rows = self._read_jsonl(self._path(artifact_type), limit=max(limit, 1) * 3)
        if session_id:
            sid = _safe_str(session_id)
            rows = [row for row in rows if _safe_str(row.get("source_session")) == sid]
        return rows[: max(limit, 1)]

    def run_crystallizer(
        self,
        *,
        session_id: str = "",
        payload: dict[str, Any] | None = None,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        sid = _safe_str(session_id) or _safe_str((payload or {}).get("session_id"))
        if not sid:
            recent_ids = self._recent_session_ids(limit=1)
            sid = recent_ids[0] if recent_ids else ""

        if not sid:
            result = {
                "status": "skipped",
                "reason": "no_session_available",
                "trigger": trigger,
                "session_id": "",
                "generated": {"thought_objects": 0, "decision_records": 0, "open_loops": 0},
            }
            self._record_run("crystallizer", result)
            return result

        turns = self._session_turns(sid)
        if not turns:
            result = {
                "status": "skipped",
                "reason": "no_session_turns",
                "trigger": trigger,
                "session_id": sid,
                "generated": {"thought_objects": 0, "decision_records": 0, "open_loops": 0},
            }
            self._record_run("crystallizer", result)
            return result

        user_turns = [turn for turn in turns if _safe_str(turn.get("role")).lower() == "user"]
        thought_objects: list[dict[str, Any]] = []
        decision_records: list[dict[str, Any]] = []
        open_loops: list[dict[str, Any]] = []

        for idx, turn in enumerate(user_turns):
            text = _safe_str(turn.get("content"))
            if len(text) < 24:
                continue
            lower = text.lower()
            topics = _extract_topics_from_text(text)

            if self._is_thought_candidate(lower):
                thought_objects.append(
                    {
                        "thought_id": f"thought-{uuid.uuid4().hex[:12]}",
                        "category": self._thought_category(lower),
                        "domain": topics[0] if topics else "general",
                        "core_claim": text[:320],
                        "evidence_quality": "user_stated_explicit",
                        "confidence": self._heuristic_confidence(lower),
                        "source_session": sid,
                        "timestamp": _safe_str(turn.get("timestamp")) or _utc_now_iso(),
                        "related_entities": list(topics[:3]),
                        "emotional_tone": self._emotional_tone(lower),
                        "follow_up_flag": "?" in text,
                        "follow_up_question": self._follow_up_question(text),
                    }
                )

            if self._is_decision_candidate(lower):
                decision_records.append(
                    {
                        "decision_id": f"decision-{uuid.uuid4().hex[:12]}",
                        "source_session": sid,
                        "timestamp": _safe_str(turn.get("timestamp")) or _utc_now_iso(),
                        "decision_text": text[:320],
                        "status": "proposed",
                        "confidence": self._heuristic_confidence(lower),
                        "related_entities": list(topics[:4]),
                    }
                )

            if "?" in text:
                open_loops.append(
                    {
                        "loop_id": f"loop-{uuid.uuid4().hex[:12]}",
                        "source_session": sid,
                        "created_at": _safe_str(turn.get("timestamp")) or _utc_now_iso(),
                        "question": text[:320],
                        "ttl_days": 7,
                        "status": "open",
                        "priority": "high" if len(text) > 120 else "normal",
                    }
                )

            if idx >= 80:
                break

        for row in thought_objects:
            self._append_jsonl(self._path("thought_objects"), row)
        for row in decision_records:
            self._append_jsonl(self._path("decision_records"), row)
        for row in open_loops:
            self._append_jsonl(self._path("open_loops"), row)

        result = {
            "status": "ok",
            "trigger": trigger,
            "session_id": sid,
            "generated": {
                "thought_objects": len(thought_objects),
                "decision_records": len(decision_records),
                "open_loops": len(open_loops),
            },
        }
        self._record_run("crystallizer", result)
        return result

    def run_summary_forge(
        self,
        *,
        session_id: str = "",
        window_days: int = 14,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        candidate_ids = [_safe_str(session_id)] if session_id else self._recent_session_ids(limit=8)
        candidate_ids = [sid for sid in candidate_ids if sid]

        created: list[dict[str, Any]] = []
        cutoff = _utc_now() - timedelta(days=max(window_days, 1))

        for sid in candidate_ids:
            turns = self._session_turns(sid)
            if not turns:
                continue

            latest_turn_time = self._latest_turn_datetime(turns)
            if latest_turn_time and latest_turn_time < cutoff:
                continue

            thoughts = self.list_artifacts("thought_objects", limit=40, session_id=sid)
            decisions = self.list_artifacts("decision_records", limit=40, session_id=sid)
            loops = self.list_artifacts("open_loops", limit=40, session_id=sid)

            if not thoughts and not decisions:
                crystallized = self.run_crystallizer(session_id=sid, trigger="summary_prefetch")
                if crystallized.get("status") == "ok":
                    thoughts = self.list_artifacts("thought_objects", limit=40, session_id=sid)
                    decisions = self.list_artifacts("decision_records", limit=40, session_id=sid)
                    loops = self.list_artifacts("open_loops", limit=40, session_id=sid)

            quotes = [
                _safe_str(t.get("core_claim"))
                for t in thoughts[:3]
                if _safe_str(t.get("core_claim"))
            ]
            if not quotes:
                quotes = [
                    _safe_str(t.get("content"))
                    for t in turns
                    if _safe_str(t.get("role")).lower() == "user" and len(_safe_str(t.get("content"))) > 40
                ][:3]

            topic_counter: Counter[str] = Counter()
            for thought in thoughts:
                topic_counter.update([_safe_str(thought.get("domain"))])
                topic_counter.update([_safe_str(topic) for topic in thought.get("related_entities", [])])
            top_topics = [topic for topic, _ in topic_counter.most_common(5) if topic]

            narrative = self._build_narrative_summary(sid, thoughts, decisions, loops, top_topics)
            next_prompt = self._next_chapter_prompt(loops, top_topics)

            summary = {
                "summary_id": f"summary-{uuid.uuid4().hex[:12]}",
                "source_session": sid,
                "created_at": _utc_now_iso(),
                "narrative_summary": narrative,
                "structured_summary": {
                    "entities": top_topics,
                    "decisions": [d.get("decision_text", "") for d in decisions[:6]],
                    "open_loops": [l.get("question", "") for l in loops[:6]],
                    "status": "open" if loops else "settled",
                },
                "key_quotes": quotes,
                "next_chapter_prompt": next_prompt,
            }
            self._append_jsonl(self._path("structured_summaries"), summary)
            created.append(summary)

        result = {
            "status": "ok",
            "trigger": trigger,
            "session_id": _safe_str(session_id),
            "created": len(created),
            "summaries": created[:5],
        }
        self._record_run("summary_forge", result)
        return result

    def run_gap_mapper(
        self,
        *,
        lookback_days: int = 7,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        from src.wiki.wiki_store import WikiStore

        warnings: list[str] = []
        try:
            wiki = WikiStore.get_instance()
            pages = wiki.list_pages(limit=200)
        except Exception as e:
            pages = []
            warnings.append(f"wiki_unavailable: {e}")

        topic_importance: dict[str, float] = {}
        for page in pages:
            page_topics = list(page.get("topics") or [])
            if not page_topics:
                page_topics = [_safe_str(page.get("title"))]
            weight = 0.95 if len(page.get("claim_ids", [])) >= 5 else 0.8
            for topic in page_topics:
                normalized = _safe_str(topic).lower()
                if not normalized:
                    continue
                topic_importance[normalized] = max(weight, topic_importance.get(normalized, 0.0))

        try:
            recent_topic_counts = self._recent_topic_counts(days=lookback_days)
        except Exception as e:
            recent_topic_counts = Counter()
            warnings.append(f"recent_topics_unavailable: {e}")

        total_recent = max(sum(recent_topic_counts.values()), 1)

        signals: list[dict[str, Any]] = []
        for topic, importance in topic_importance.items():
            attention = recent_topic_counts.get(topic, 0) / total_recent
            if attention >= 0.03:
                continue
            days = max(lookback_days, 1)
            severity = "high" if attention < 0.01 else "medium"
            signal = {
                "gap_id": f"gap-{uuid.uuid4().hex[:12]}",
                "gap_type": "stated_priority_vs_attention",
                "entity": topic,
                "stated_importance": round(float(importance), 2),
                "recent_attention_score": round(float(attention), 4),
                "gap_duration_days": days,
                "severity": severity,
                "suggested_question": f"It's been {days} days with little focus on {topic}. What's blocking progress?",
                "route_to": "presence_agent_idle_queue",
                "created_at": _utc_now_iso(),
            }
            self._append_jsonl(self._path("gap_signals"), signal)
            signals.append(signal)
            if len(signals) >= 25:
                break

        result = {
            "status": "ok",
            "trigger": trigger,
            "generated": len(signals),
            "signals": signals[:8],
        }
        if warnings:
            result["warnings"] = warnings
        self._record_run("gap_mapper", result)
        return result

    def run_belief_detector(
        self,
        *,
        lookback_days: int = 30,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        store = self._metadata_store()
        rows = list(store.get_belief_deltas(limit=200) or [])
        cutoff = _utc_now() - timedelta(days=max(lookback_days, 1))

        records: list[dict[str, Any]] = []
        for row in rows:
            detected_raw = _safe_str(row.get("detected_at"))
            try:
                detected_at = _parse_iso(detected_raw)
            except Exception:
                continue
            if detected_at < cutoff:
                continue

            record = {
                "belief_evolution_id": f"belief-evo-{uuid.uuid4().hex[:12]}",
                "topic": _safe_str(row.get("topic")),
                "old_position": _safe_str(row.get("old_belief_text")),
                "new_position": _safe_str(row.get("new_belief_text")),
                "change_type": _safe_str(row.get("change_type")) or "new_belief",
                "confidence": float(row.get("confidence") or 0.0),
                "detected_at": detected_at.isoformat(),
            }
            self._append_jsonl(self._path("belief_evolution"), record)
            records.append(record)

        result = {
            "status": "ok",
            "trigger": trigger,
            "generated": len(records),
            "records": records[:10],
        }
        self._record_run("belief_detector", result)
        return result

    def _metadata_store(self):
        from src.engine import rag_engine

        store = getattr(rag_engine, "metadata_store", None)
        if store is None:
            raise RuntimeError("Metadata store is not initialized")
        return store

    def _path(self, artifact_type: str) -> Path:
        return self._base_dir / self._ARTIFACT_FILES[artifact_type]

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True, default=str)
        with self._lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _read_jsonl(self, path: Path, *, limit: int = 50) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with self._lock:
            with open(path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        rows: list[dict[str, Any]] = []
        for line in reversed(lines[-max(limit, 1):]):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        with self._lock:
            with open(path, "r", encoding="utf-8") as handle:
                return sum(1 for _ in handle)

    def _record_run(self, run_type: str, payload: dict[str, Any]) -> None:
        entry = {
            "run_id": f"forge-run-{uuid.uuid4().hex[:12]}",
            "run_type": run_type,
            "timestamp": _utc_now_iso(),
            "payload": payload,
        }
        self._append_jsonl(self._path("runs"), entry)

    def _session_turns(self, session_id: str) -> list[dict[str, Any]]:
        store = self._metadata_store()
        turns = list(store.get_conversation(session_id) or [])
        return [dict(turn) for turn in turns]

    def _recent_session_ids(self, limit: int = 10) -> list[str]:
        store = self._metadata_store()

        conn = getattr(store, "conn", None)
        use_duckdb = bool(getattr(store, "_use_duckdb", False))
        if conn is not None and use_duckdb:
            try:
                rows = conn.execute(
                    """
                    SELECT session_id, MAX(timestamp) AS ts
                    FROM conversations
                    WHERE session_id != ''
                    GROUP BY session_id
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    [max(limit, 1)],
                ).fetchall()
                return [str(row[0]) for row in rows if str(row[0]).strip()]
            except Exception:
                pass

        memories = list(store.get_all_memories(limit=2000) or [])
        session_latest: dict[str, datetime] = {}
        for memory in memories:
            sid = _safe_str(getattr(memory, "session_id", ""))
            if not sid:
                continue
            ts = _coerce_utc_datetime(getattr(memory, "timestamp", None)) or _utc_now()
            if sid not in session_latest or ts > session_latest[sid]:
                session_latest[sid] = ts

        ordered = sorted(session_latest.items(), key=lambda item: item[1], reverse=True)
        return [sid for sid, _ in ordered[: max(limit, 1)]]

    def _recent_topic_counts(self, *, days: int) -> Counter[str]:
        store = self._metadata_store()
        cutoff = _utc_now() - timedelta(days=max(days, 1))
        memories = list(store.get_all_memories(limit=4000) or [])

        counts: Counter[str] = Counter()
        for memory in memories:
            ts = _coerce_utc_datetime(getattr(memory, "timestamp", None))
            if ts is not None and ts < cutoff:
                continue

            topics = getattr(memory, "topics", []) or []
            if not topics:
                topics = _extract_topics_from_text(_safe_str(getattr(memory, "content", "")))
            for topic in topics:
                normalized = _safe_str(topic).lower()
                if normalized:
                    counts[normalized] += 1

        return counts

    @staticmethod
    def _is_thought_candidate(text: str) -> bool:
        markers = (
            "i think",
            "i feel",
            "i realized",
            "i've been",
            "i have been",
            "i want",
            "i need",
            "i worry",
            "i am afraid",
        )
        return any(marker in text for marker in markers) or len(text) >= 120

    @staticmethod
    def _is_decision_candidate(text: str) -> bool:
        markers = (
            "i decided",
            "i will",
            "i'm going to",
            "lets",
            "let's",
            "i should",
            "plan is",
            "next step",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _thought_category(text: str) -> str:
        if "i realized" in text or "i've been" in text or "i have been" in text:
            return "self_insight"
        if "i feel" in text or "afraid" in text or "worry" in text:
            return "emotional_pattern"
        if "i decided" in text or "i will" in text:
            return "decision_intent"
        return "general_reflection"

    @staticmethod
    def _emotional_tone(text: str) -> str:
        if "afraid" in text or "anxious" in text or "worry" in text:
            return "vulnerable_honest"
        if "excited" in text or "grateful" in text:
            return "energized"
        if "frustrated" in text or "stuck" in text:
            return "frustrated"
        return "neutral_reflective"

    @staticmethod
    def _heuristic_confidence(text: str) -> float:
        score = 0.7
        if any(token in text for token in ("i realized", "i decided", "i will")):
            score += 0.15
        if "?" in text:
            score -= 0.1
        return round(max(0.3, min(score, 0.98)), 2)

    @staticmethod
    def _follow_up_question(text: str) -> str:
        if "?" in text:
            return text[:240]
        return "What changed since this session that affects this thought?"

    @staticmethod
    def _latest_turn_datetime(turns: list[dict[str, Any]]) -> datetime | None:
        values: list[datetime] = []
        for turn in turns:
            raw = _safe_str(turn.get("timestamp"))
            if not raw:
                continue
            try:
                values.append(_parse_iso(raw))
            except Exception:
                continue
        return max(values) if values else None

    @staticmethod
    def _build_narrative_summary(
        session_id: str,
        thoughts: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        loops: list[dict[str, Any]],
        topics: list[str],
    ) -> str:
        topic_text = ", ".join(topics[:3]) if topics else "several personal priorities"
        return (
            f"Session {session_id[:8]} focused on {topic_text}. "
            f"The forge extracted {len(thoughts)} thought objects and {len(decisions)} decision records. "
            f"There are {len(loops)} open loops that should be revisited in an idle window."
        )

    @staticmethod
    def _next_chapter_prompt(loops: list[dict[str, Any]], topics: list[str]) -> str:
        if loops:
            question = _safe_str(loops[0].get("question"))
            return question or "Which open loop deserves closure this week?"
        if topics:
            return f"What meaningful progress happened this week in {topics[0]}?"
        return "What changed in your priorities since this session?"


session_memory_forge_service = SessionMemoryForgeService()
