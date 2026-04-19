"""Wiki compaction utilities for canonical page hygiene.

Phase alignment:
- Agentic-Rag-Wiki.md §4.4 (compaction operation)
- Agentic-Rag-Wiki.md §Phase 4 (macro compaction + snapshotting)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.wiki.wiki_store import WikiStore


class WikiCompactor:
    """Perform deterministic wiki section and macro compaction passes."""

    def __init__(self, data_dir: str = "data/wiki"):
        self.data_dir = Path(data_dir)
        self.compaction_dir = self.data_dir / "compaction"
        self.compaction_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_store = WikiStore.get_instance(data_dir=data_dir)

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for raw in lines:
            line = raw.rstrip()
            normalized = re.sub(r"\s+", " ", line.strip().lower())
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(line)
        return result

    @staticmethod
    def _compress_text(text: str, *, max_tokens: int) -> str:
        max_chars = max(120, int(max_tokens) * 4)

        lines = [line for line in text.splitlines() if line.strip()]
        deduped = WikiCompactor._dedupe_lines(lines)
        merged = "\n".join(deduped).strip()

        if len(merged) <= max_chars:
            return merged

        sentences = re.split(r"(?<=[.!?])\s+", merged)
        compact_parts: list[str] = []
        current_len = 0
        for sentence in sentences:
            candidate = sentence.strip()
            if not candidate:
                continue
            proposed = current_len + len(candidate) + (1 if compact_parts else 0)
            if proposed > max_chars:
                break
            compact_parts.append(candidate)
            current_len = proposed

        if not compact_parts:
            return merged[:max_chars].rstrip()

        compacted = " ".join(compact_parts).strip()
        if len(compacted) < int(max_chars * 0.35):
            # Fall back to line-based clipping if sentence splitting was too aggressive.
            return merged[:max_chars].rstrip()
        return compacted

    @staticmethod
    def _extract_section(content: str, section: str) -> str | None:
        escaped = re.escape(section.strip())
        pattern = re.compile(
            rf"(?:^|\n)##\s+{escaped}\s*\n(.*?)(?=\n##\s+|\Z)",
            re.DOTALL,
        )
        match = pattern.search(content)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _list_sections(content: str) -> list[str]:
        matches = re.findall(r"^##\s+(.+?)\s*$", content or "", flags=re.MULTILINE)
        return [m.strip() for m in matches if m.strip()]

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_latest(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.compaction_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def compact_section(
        self,
        page_id: str,
        section: str,
        *,
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        page = self.wiki_store.get_page(page_id)
        if page is None:
            raise KeyError(f"Page not found: {page_id}")

        original_section = self._extract_section(page.content, section)
        if original_section is None:
            result = {
                "page_id": page_id,
                "section": section,
                "compacted": False,
                "reason": "section_not_found",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            self._append_jsonl(self.compaction_dir / "runs.jsonl", result)
            self._write_latest("latest_run.json", result)
            return result

        compacted = self._compress_text(original_section, max_tokens=max_tokens)
        changed = compacted.strip() != original_section.strip()

        if changed:
            self.wiki_store.patch_page(
                page_id=page_id,
                section=section,
                content=compacted,
                operation="replace",
            )

        before_chars = len(original_section)
        after_chars = len(compacted)
        reduction = (before_chars - after_chars) / max(before_chars, 1)

        result = {
            "page_id": page_id,
            "section": section,
            "compacted": changed,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "reduction_ratio": round(max(reduction, 0.0), 3),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        self._append_jsonl(self.compaction_dir / "runs.jsonl", result)
        self._write_latest("latest_run.json", result)
        return result

    def compact_page(
        self,
        page_id: str,
        *,
        max_tokens_per_section: int = 500,
        section_limit: int = 3,
    ) -> dict[str, Any]:
        page = self.wiki_store.get_page(page_id)
        if page is None:
            raise KeyError(f"Page not found: {page_id}")

        sections = self._list_sections(page.content)
        if not sections:
            sections = ["Summary"]

        selected_sections = sections[: max(int(section_limit), 1)]
        section_runs: list[dict[str, Any]] = []
        for section in selected_sections:
            try:
                section_runs.append(
                    self.compact_section(
                        page_id,
                        section,
                        max_tokens=max_tokens_per_section,
                    )
                )
            except Exception as exc:
                section_runs.append(
                    {
                        "page_id": page_id,
                        "section": section,
                        "compacted": False,
                        "reason": "error",
                        "error": str(exc),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        compacted_count = sum(1 for item in section_runs if item.get("compacted"))
        avg_reduction = (
            sum(float(item.get("reduction_ratio", 0.0)) for item in section_runs) / max(len(section_runs), 1)
        )

        return {
            "page_id": page_id,
            "sections_checked": len(section_runs),
            "sections_compacted": compacted_count,
            "avg_reduction_ratio": round(avg_reduction, 3),
            "section_runs": section_runs,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def compact_all(
        self,
        *,
        limit: int = 60,
        max_tokens_per_section: int = 500,
        section_limit: int = 2,
    ) -> dict[str, Any]:
        pages = self.wiki_store.list_pages(limit=max(int(limit), 1))
        page_runs: list[dict[str, Any]] = []
        for page in pages:
            page_id = str(page.get("id", "")).strip()
            if not page_id:
                continue
            try:
                page_runs.append(
                    self.compact_page(
                        page_id,
                        max_tokens_per_section=max_tokens_per_section,
                        section_limit=section_limit,
                    )
                )
            except Exception as exc:
                page_runs.append(
                    {
                        "page_id": page_id,
                        "sections_checked": 0,
                        "sections_compacted": 0,
                        "avg_reduction_ratio": 0.0,
                        "error": str(exc),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        compacted_sections = sum(int(item.get("sections_compacted", 0)) for item in page_runs)
        checked_sections = sum(int(item.get("sections_checked", 0)) for item in page_runs)

        summary = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "pages_checked": len(page_runs),
            "sections_checked": checked_sections,
            "sections_compacted": compacted_sections,
            "page_runs": page_runs[:80],
        }

        self._append_jsonl(self.compaction_dir / "summary.jsonl", summary)
        self._write_latest("latest_summary.json", summary)
        return summary

    def latest_summary(self) -> dict[str, Any]:
        path = self.compaction_dir / "latest_summary.json"
        if not path.exists():
            return {
                "checked_at": None,
                "pages_checked": 0,
                "sections_checked": 0,
                "sections_compacted": 0,
                "page_runs": [],
            }
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
