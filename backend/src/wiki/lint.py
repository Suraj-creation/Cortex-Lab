"""Wiki linting for knowledge hygiene and contradiction surfacing.

Phase alignment:
- Agentic-Rag-Wiki.md §4.3 (lint operation)
- Agentic-Rag-Wiki.md §Phase 4 (scheduled lint jobs)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.wiki.claim_store import ClaimStore
from src.wiki.wiki_store import WikiStore


class WikiLinter:
    """Run deterministic wiki hygiene checks and persist reports."""

    def __init__(self, data_dir: str = "data/wiki"):
        self.data_dir = Path(data_dir)
        self.lint_dir = self.data_dir / "lint"
        self.lint_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_store = WikiStore.get_instance(data_dir=data_dir)
        self.claim_store = ClaimStore.get_instance(data_dir=str(self.data_dir / "claims"))

    @staticmethod
    def _parse_iso(value: str) -> datetime | None:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None

    @staticmethod
    def _dedupe_preserve(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_latest(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.lint_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def lint_page(
        self,
        page_id: str,
        *,
        stale_days: int = 90,
        min_confidence: float = 0.45,
    ) -> dict[str, Any]:
        page = self.wiki_store.get_page(page_id)
        if page is None:
            raise KeyError(f"Page not found: {page_id}")

        issues: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        content = str(page.content or "")
        headings = re.findall(r"^##\s+.+$", content, flags=re.MULTILINE)
        if not headings:
            issues.append(
                {
                    "code": "missing_sections",
                    "severity": "medium",
                    "message": "Wiki page has no section headings.",
                }
            )

        normalized_lines: list[str] = []
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip().lower())
            if len(line) < 16:
                continue
            normalized_lines.append(line)

        duplicate_count = 0
        seen_lines: set[str] = set()
        for line in normalized_lines:
            if line in seen_lines:
                duplicate_count += 1
            else:
                seen_lines.add(line)
        if duplicate_count >= 3:
            issues.append(
                {
                    "code": "redundant_content",
                    "severity": "low",
                    "message": f"Detected repeated lines ({duplicate_count}).",
                }
            )

        updated_at = self._parse_iso(getattr(page, "updated_at", ""))
        if updated_at and (now - updated_at) > timedelta(days=max(stale_days, 1)):
            issues.append(
                {
                    "code": "stale_page",
                    "severity": "medium",
                    "message": (
                        f"Page older than {stale_days} days "
                        f"(last_update={updated_at.isoformat()})."
                    ),
                }
            )

        missing_claim_ids: list[str] = []
        low_conf_claim_ids: list[str] = []
        contradicted_claim_ids: list[str] = []

        for claim_id in list(getattr(page, "claim_ids", []) or []):
            claim = self.claim_store.get_claim(str(claim_id))
            if claim is None:
                missing_claim_ids.append(str(claim_id))
                continue
            if float(claim.confidence) < float(min_confidence):
                low_conf_claim_ids.append(claim.id)
            if list(claim.contradiction_ids or []):
                contradicted_claim_ids.append(claim.id)

        if missing_claim_ids:
            issues.append(
                {
                    "code": "orphan_claim_links",
                    "severity": "high",
                    "message": f"Page links {len(missing_claim_ids)} missing claims.",
                    "claim_ids": self._dedupe_preserve(missing_claim_ids)[:20],
                }
            )

        if low_conf_claim_ids:
            issues.append(
                {
                    "code": "low_confidence_claims",
                    "severity": "medium",
                    "message": f"Page references {len(low_conf_claim_ids)} low-confidence claims.",
                    "claim_ids": self._dedupe_preserve(low_conf_claim_ids)[:20],
                }
            )

        if contradicted_claim_ids:
            issues.append(
                {
                    "code": "contradicted_claims",
                    "severity": "high",
                    "message": (
                        f"Page references {len(contradicted_claim_ids)} "
                        "claims marked as contradicted."
                    ),
                    "claim_ids": self._dedupe_preserve(contradicted_claim_ids)[:20],
                }
            )

        severity_weights = {
            "high": 0.35,
            "medium": 0.2,
            "low": 0.08,
        }
        penalty = 0.0
        for issue in issues:
            penalty += severity_weights.get(str(issue.get("severity", "low")), 0.08)

        hygiene_score = max(0.0, min(1.0, 1.0 - penalty))

        report = {
            "page_id": page.id,
            "title": page.title,
            "checked_at": now.isoformat(),
            "status": "clean" if not issues else "issues_detected",
            "hygiene_score": round(hygiene_score, 3),
            "issue_count": len(issues),
            "issues": issues,
        }

        self._append_jsonl(self.lint_dir / "reports.jsonl", report)
        self._write_latest("latest_page_report.json", report)
        return report

    def lint_all(
        self,
        *,
        limit: int = 200,
        stale_days: int = 90,
        min_confidence: float = 0.45,
    ) -> dict[str, Any]:
        pages = self.wiki_store.list_pages(limit=max(int(limit), 1))
        reports: list[dict[str, Any]] = []

        for page in pages:
            page_id = str(page.get("id", "")).strip()
            if not page_id:
                continue
            try:
                reports.append(
                    self.lint_page(
                        page_id,
                        stale_days=stale_days,
                        min_confidence=min_confidence,
                    )
                )
            except Exception as exc:
                reports.append(
                    {
                        "page_id": page_id,
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                        "status": "error",
                        "hygiene_score": 0.0,
                        "issue_count": 1,
                        "issues": [
                            {
                                "code": "lint_error",
                                "severity": "high",
                                "message": str(exc),
                            }
                        ],
                    }
                )

        issue_total = sum(int(item.get("issue_count", 0)) for item in reports)
        dirty_pages = sum(1 for item in reports if item.get("status") == "issues_detected")
        avg_score = (
            sum(float(item.get("hygiene_score", 0.0)) for item in reports) / max(len(reports), 1)
        )

        summary = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "pages_checked": len(reports),
            "dirty_pages": dirty_pages,
            "issue_total": issue_total,
            "avg_hygiene_score": round(avg_score, 3),
            "reports": reports[:100],
        }

        self._append_jsonl(self.lint_dir / "summary.jsonl", summary)
        self._write_latest("latest_summary.json", summary)
        return summary

    def latest_summary(self) -> dict[str, Any]:
        path = self.lint_dir / "latest_summary.json"
        if not path.exists():
            return {
                "checked_at": None,
                "pages_checked": 0,
                "dirty_pages": 0,
                "issue_total": 0,
                "avg_hygiene_score": 0.0,
                "reports": [],
            }
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
