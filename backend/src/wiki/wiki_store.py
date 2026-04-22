"""
Wiki Store — Personal knowledge wiki backed by markdown files.
Architecture: Agentic-RAG-Architecture.md §5.4 (Wiki Memory Plane)

Each wiki page is a markdown file with frontmatter metadata.
Pages are organized by topic and maintained by the Wiki Agent.
The Wiki Agent patches pages when new claims are extracted from memories.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional


@dataclass
class WikiPage:
    id: str
    title: str
    content: str
    topics: list[str]
    claim_ids: list[str]
    created_at: str
    updated_at: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "topics": self.topics,
            "claim_ids": self.claim_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }


class WikiStore:
    """
    Local-first wiki store backed by markdown files + JSON index.
    
    File layout:
      data/wiki/pages/
        {page_id}.md          # Markdown content
        {page_id}.meta.json   # Metadata (topics, claims, versions)
      data/wiki/index.json    # Topic → page_id index
    """

    _instance: Optional["WikiStore"] = None

    def __init__(self, data_dir: str = "data/wiki"):
        self._data_dir = Path(data_dir)
        self._pages_dir = self._data_dir / "pages"
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        self._pages: dict[str, WikiPage] = {}
        self._topic_index: dict[str, list[str]] = {}
        self._lock = Lock()
        self._load()

    @classmethod
    def get_instance(cls, data_dir: str = "data/wiki") -> "WikiStore":
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def create_page(
        self,
        title: str,
        content: str,
        topics: list[str] | None = None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        page_id = f"wiki-{uuid.uuid4().hex[:12]}"
        topics = topics or []

        page = WikiPage(
            id=page_id,
            title=title,
            content=content,
            topics=topics,
            claim_ids=[],
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._pages[page_id] = page
            for topic in topics:
                self._topic_index.setdefault(topic, []).append(page_id)
            self._persist_page(page)
            self._save_index()

        return page_id

    def get_page(self, page_id: str) -> WikiPage | None:
        with self._lock:
            return self._pages.get(page_id)

    def search(
        self,
        query: str,
        include_claims: bool = True,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []

        with self._lock:
            for page in self._pages.values():
                score = 0.0
                if query_lower in page.title.lower():
                    score += 0.6
                if query_lower in page.content.lower():
                    score += 0.3
                if any(query_lower in t.lower() for t in page.topics):
                    score += 0.1

                if score > 0:
                    entry = page.to_dict()
                    entry["search_score"] = round(score, 2)
                    if not include_claims:
                        entry.pop("claim_ids", None)
                    results.append(entry)

        results.sort(key=lambda x: x["search_score"], reverse=True)
        return results[:limit]

    def search_by_topic(self, topic: str) -> list[dict[str, Any]]:
        with self._lock:
            page_ids = self._topic_index.get(topic, [])
            return [self._pages[pid].to_dict() for pid in page_ids if pid in self._pages]

    def patch_page(
        self,
        page_id: str,
        section: str,
        content: str,
        operation: str = "append",
    ) -> None:
        with self._lock:
            page = self._pages.get(page_id)
            if not page:
                raise KeyError(f"Page not found: {page_id}")

            if operation == "replace":
                pattern = re.compile(
                    rf"(## {re.escape(section)}\n)(.*?)(?=\n## |\Z)",
                    re.DOTALL,
                )
                if pattern.search(page.content):
                    page.content = pattern.sub(rf"\g<1>{content}\n", page.content)
                else:
                    page.content += f"\n\n## {section}\n{content}\n"
            elif operation == "prepend":
                pattern = re.compile(
                    rf"(## {re.escape(section)}\n)",
                    re.DOTALL,
                )
                if pattern.search(page.content):
                    page.content = pattern.sub(rf"\g<1>{content}\n", page.content)
                else:
                    page.content += f"\n\n## {section}\n{content}\n"
            else:
                section_header = f"## {section}"
                if section_header in page.content:
                    next_section = re.search(
                        rf"\n## (?!{re.escape(section)})",
                        page.content[page.content.index(section_header):]
                    )
                    if next_section:
                        insert_pos = page.content.index(section_header) + next_section.start()
                        page.content = (
                            page.content[:insert_pos] + f"\n{content}" + page.content[insert_pos:]
                        )
                    else:
                        page.content += f"\n{content}"
                else:
                    page.content += f"\n\n## {section}\n{content}\n"

            page.version += 1
            page.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist_page(page)

    def link_claim(self, page_id: str, claim_id: str) -> None:
        with self._lock:
            page = self._pages.get(page_id)
            if page and claim_id not in page.claim_ids:
                page.claim_ids.append(claim_id)
                page.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist_page(page)

    def list_pages(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            pages = sorted(
                self._pages.values(),
                key=lambda p: p.updated_at,
                reverse=True,
            )
            return [p.to_dict() for p in pages[:limit]]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_claims = sum(len(p.claim_ids) for p in self._pages.values())
            return {
                "total_pages": len(self._pages),
                "total_topics": len(self._topic_index),
                "total_linked_claims": total_claims,
            }

    def _persist_page(self, page: WikiPage) -> None:
        md_path = self._pages_dir / f"{page.id}.md"
        meta_path = self._pages_dir / f"{page.id}.meta.json"

        frontmatter = f"""---
id: {page.id}
title: {page.title}
topics: {json.dumps(page.topics)}
version: {page.version}
updated_at: {page.updated_at}
---

"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(frontmatter + page.content)

        meta = {
            "id": page.id,
            "title": page.title,
            "topics": page.topics,
            "claim_ids": page.claim_ids,
            "created_at": page.created_at,
            "updated_at": page.updated_at,
            "version": page.version,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _save_index(self) -> None:
        index_path = self._data_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self._topic_index, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        for meta_path in self._pages_dir.glob("*.meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                md_path = self._pages_dir / f"{meta['id']}.md"
                content = ""
                if md_path.exists():
                    with open(md_path, "r", encoding="utf-8") as f:
                        raw = f.read()
                    end_fm = raw.find("---", 4)
                    content = raw[end_fm + 4:].strip() if end_fm > 0 else raw

                page = WikiPage(
                    id=meta["id"],
                    title=meta["title"],
                    content=content,
                    topics=meta.get("topics", []),
                    claim_ids=meta.get("claim_ids", []),
                    created_at=meta.get("created_at", ""),
                    updated_at=meta.get("updated_at", ""),
                    version=meta.get("version", 1),
                )
                self._pages[page.id] = page
                for topic in page.topics:
                    self._topic_index.setdefault(topic, []).append(page.id)
            except (json.JSONDecodeError, KeyError):
                continue

        index_path = self._data_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    saved_index = json.load(f)
                for topic, ids in saved_index.items():
                    self._topic_index.setdefault(topic, []).extend(
                        pid for pid in ids if pid not in self._topic_index.get(topic, [])
                    )
            except (json.JSONDecodeError, KeyError):
                pass
