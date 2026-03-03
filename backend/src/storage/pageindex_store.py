"""
PageIndex Document Store — Manages documents indexed via PageIndex API.

This module bridges Cortex Lab's local architecture with PageIndex's
cloud-based reasoning retrieval. It handles:
  - Document upload with privacy guards
  - Document status tracking and polling
  - Tree-based and chat-based retrieval
  - Local doc_id mapping (file_hash → PageIndex doc_id)
  - Usage tracking for cost control
  - Graceful degradation on API failure

Privacy: Only non-sensitive, user-consented documents are uploaded.
Personal memories, chat history, voice transcripts NEVER leave the device.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from pageindex import PageIndexClient
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False
    PageIndexClient = None


class PageIndexStore:
    """Interface between Cortex Lab and PageIndex cloud API."""

    def __init__(self, api_key: str, data_dir: str = "data/pageindex",
                 config: Optional[Dict] = None):
        if not PAGEINDEX_AVAILABLE:
            raise ImportError(
                "pageindex SDK not installed. Run: pip install pageindex"
            )

        self.api_key = api_key
        self.data_dir = data_dir
        self.config = config or {}
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "uploads"), exist_ok=True)

        # Initialize SDK client
        self.client = PageIndexClient(api_key=api_key)

        # Local mapping: file_hash → {doc_id, filename, uploaded_at, pages, status}
        self.doc_mapping: Dict[str, Dict[str, Any]] = {}
        self._load_mapping()

        # Usage tracking
        self._usage: Dict[str, int] = {"queries": 0, "pages": 0, "month": ""}
        self._load_usage()

        # Connection test
        self._connected = False
        self._test_connection()

    # ─── Connection ──────────────────────────────────────────────────

    def _test_connection(self):
        """Verify API key and connectivity."""
        try:
            result = self.client.list_documents(limit=1)
            self._connected = True
            doc_count = result.get("total", 0)
            print(f"    ✓ PageIndex connected ({doc_count} documents)")
        except Exception as e:
            self._connected = False
            print(f"    ⚠ PageIndex connection failed: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_documents(self) -> bool:
        return bool(self.doc_mapping)

    # ─── Document Upload ─────────────────────────────────────────────

    def upload_document(self, file_path: str,
                        filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a PDF to PageIndex for tree-based indexing.

        Args:
            file_path: Path to the PDF file
            filename: Display name (defaults to basename)

        Returns:
            Dict with doc_id, status, and metadata

        Raises:
            ValueError: If file is not a PDF or fails privacy checks
            ConnectionError: If API is unavailable
        """
        if not self._connected:
            raise ConnectionError("PageIndex API not connected")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file type
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".pdf",):
            raise ValueError(
                f"PageIndex only supports PDF files. Got: {ext}"
            )

        # Privacy check — scan first 10KB for sensitive patterns
        if self.config.get("sensitive_data_filter", True):
            if self._contains_sensitive_data(file_path):
                raise ValueError(
                    "Document contains potentially sensitive data "
                    "(SSN, credit card, passwords). Upload blocked by privacy filter."
                )

        # Check usage budget
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        est_pages = max(1, int(file_size_mb * 5))  # rough estimate
        if not self._check_usage_budget(pages=est_pages):
            raise ValueError(
                f"Monthly page budget exceeded. "
                f"Limit: {self.config.get('max_monthly_pages', 2000)}"
            )

        # Check if already uploaded (by file hash)
        file_hash = self._hash_file(file_path)
        if file_hash in self.doc_mapping:
            existing = self.doc_mapping[file_hash]
            print(f"    ℹ Document already indexed: {existing['doc_id']}")
            return {
                "doc_id": existing["doc_id"],
                "status": existing.get("status", "ready"),
                "already_indexed": True,
                "filename": existing.get("filename", ""),
            }

        # Upload to PageIndex
        display_name = filename or os.path.basename(file_path)
        print(f"    📤 Uploading to PageIndex: {display_name}...")

        try:
            result = self.client.submit_document(file_path)
            doc_id = result.get("doc_id", "")
            if not doc_id:
                raise ValueError(f"No doc_id in response: {result}")
        except Exception as e:
            raise ConnectionError(f"PageIndex upload failed: {e}")

        # Store mapping
        self.doc_mapping[file_hash] = {
            "doc_id": doc_id,
            "filename": display_name,
            "file_path": os.path.abspath(file_path),
            "file_hash": file_hash,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "estimated_pages": est_pages,
            "status": "processing",
        }
        self._save_mapping()
        self._track_usage(pages=est_pages)

        print(f"    ✓ Uploaded: {doc_id} ({display_name})")
        return {
            "doc_id": doc_id,
            "status": "processing",
            "already_indexed": False,
            "filename": display_name,
        }

    # ─── Document Status ─────────────────────────────────────────────

    def check_status(self, doc_id: str) -> Dict[str, Any]:
        """Check processing status of a document."""
        try:
            ready = self.client.is_retrieval_ready(doc_id)
            status = "ready" if ready else "processing"

            # Update local mapping
            for fh, info in self.doc_mapping.items():
                if info["doc_id"] == doc_id:
                    info["status"] = status
                    self._save_mapping()
                    break

            return {"doc_id": doc_id, "status": status, "ready": ready}
        except Exception as e:
            return {"doc_id": doc_id, "status": "error", "error": str(e)}

    def wait_until_ready(self, doc_id: str, timeout: int = 120,
                         poll_interval: int = 5) -> bool:
        """Poll until document processing is complete."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.check_status(doc_id)
            if status.get("ready"):
                return True
            if status.get("status") == "error":
                return False
            time.sleep(poll_interval)
        return False

    # ─── Retrieval ───────────────────────────────────────────────────

    def chat_retrieve(self, query: str,
                      doc_ids: Optional[List[str]] = None,
                      stream: bool = False) -> str:
        """
        Use PageIndex Chat API for managed agentic retrieval.
        The PageIndex LLM navigates the tree, retrieves relevant sections,
        and generates an answer.

        Args:
            query: The user's question
            doc_ids: Specific doc_ids to query (None = all)
            stream: Whether to stream the response

        Returns:
            The generated answer text
        """
        if not self._connected:
            return ""

        target_ids = doc_ids or self.get_all_doc_ids()
        if not target_ids:
            return ""

        if not self._check_usage_budget(queries=1):
            return "[PageIndex query budget exceeded]"

        messages = [{"role": "user", "content": query}]
        doc_id_param = target_ids[0] if len(target_ids) == 1 else target_ids

        try:
            if stream:
                result_parts = []
                for chunk in self.client.chat_completions(
                    messages=messages,
                    doc_id=doc_id_param,
                    stream=True,
                    enable_citations=True,
                ):
                    if isinstance(chunk, str):
                        result_parts.append(chunk)
                    elif isinstance(chunk, dict):
                        content = chunk.get("choices", [{}])[0].get(
                            "delta", {}
                        ).get("content", "")
                        if content:
                            result_parts.append(content)
                answer = "".join(result_parts)
            else:
                response = self.client.chat_completions(
                    messages=messages,
                    doc_id=doc_id_param,
                    stream=False,
                    enable_citations=True,
                )
                # Handle both dict and string responses
                if isinstance(response, dict):
                    choices = response.get("choices", [])
                    if choices:
                        answer = choices[0].get("message", {}).get("content", "")
                    else:
                        answer = str(response)
                else:
                    answer = str(response)

            self._track_usage(queries=1)
            return answer.strip()

        except Exception as e:
            print(f"    ⚠ PageIndex chat retrieval failed: {e}")
            return ""

    def retrieve_sections(self, query: str,
                          doc_ids: Optional[List[str]] = None,
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document sections as structured results.
        Uses the Chat API with a retrieval-focused prompt to get
        page numbers and raw content.

        Returns:
            List of {page, content, doc_id, score} dicts
        """
        if not self._connected or not self.get_all_doc_ids():
            return []

        target_ids = doc_ids or self.get_all_doc_ids()
        if not target_ids:
            return []

        retrieval_prompt = (
            f"Your job is to retrieve the raw relevant content from the "
            f"document based on the user's query. Return ONLY a JSON array "
            f"of the most relevant sections (up to {top_k}).\n\n"
            f"Query: {query}\n\n"
            f'Return in this exact JSON format:\n'
            f'[{{"page": <number>, "content": "<raw text from document>"}}]'
        )

        messages = [{"role": "user", "content": retrieval_prompt}]
        doc_id_param = target_ids[0] if len(target_ids) == 1 else target_ids

        try:
            response_parts = []
            for chunk in self.client.chat_completions(
                messages=messages,
                doc_id=doc_id_param,
                stream=True,
            ):
                if isinstance(chunk, str):
                    response_parts.append(chunk)
                elif isinstance(chunk, dict):
                    content = chunk.get("choices", [{}])[0].get(
                        "delta", {}
                    ).get("content", "")
                    if content:
                        response_parts.append(content)

            full_response = "".join(response_parts)
            sections = self._extract_json_array(full_response)

            # Add doc_id and score to each result
            for i, section in enumerate(sections[:top_k]):
                section["doc_id"] = target_ids[0] if target_ids else ""
                section["score"] = round(1.0 - (i * 0.1), 2)

            self._track_usage(queries=1)
            return sections[:top_k]

        except Exception as e:
            print(f"    ⚠ PageIndex section retrieval failed: {e}")
            return []

    def get_tree(self, doc_id: str,
                 include_summaries: bool = True) -> Optional[Dict]:
        """Get the hierarchical tree structure for a document."""
        try:
            return self.client.get_tree(
                doc_id, node_summary=include_summaries
            )
        except Exception as e:
            print(f"    ⚠ PageIndex get_tree failed: {e}")
            return None

    # ─── Document Management ─────────────────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all tracked documents with local + remote status."""
        docs = []
        for file_hash, info in self.doc_mapping.items():
            docs.append({
                "doc_id": info["doc_id"],
                "filename": info.get("filename", "Unknown"),
                "uploaded_at": info.get("uploaded_at", ""),
                "status": info.get("status", "unknown"),
                "estimated_pages": info.get("estimated_pages", 0),
                "file_hash": file_hash,
            })
        return docs

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from PageIndex and local mapping."""
        try:
            self.client.delete_document(doc_id)
        except Exception as e:
            print(f"    ⚠ PageIndex delete failed: {e}")

        # Remove from local mapping regardless
        removed = False
        for fh in list(self.doc_mapping.keys()):
            if self.doc_mapping[fh]["doc_id"] == doc_id:
                del self.doc_mapping[fh]
                removed = True
                break
        if removed:
            self._save_mapping()
        return removed

    def get_all_doc_ids(self) -> List[str]:
        """Get all PageIndex doc_ids that are ready for retrieval."""
        return [
            info["doc_id"]
            for info in self.doc_mapping.values()
            if info.get("status") in ("ready", "processing")
        ]

    def get_doc_ids_for_query(self, query: str) -> List[str]:
        """
        Determine which documents are relevant to a query.
        Currently returns all ready doc_ids. Future: semantic matching.
        """
        return [
            info["doc_id"]
            for info in self.doc_mapping.values()
            if info.get("status") == "ready"
        ]

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get local info for a specific document."""
        for info in self.doc_mapping.values():
            if info["doc_id"] == doc_id:
                return info
        return None

    def sync_statuses(self):
        """Refresh status for all 'processing' documents."""
        for fh, info in self.doc_mapping.items():
            if info.get("status") == "processing":
                self.check_status(info["doc_id"])

    # ─── Usage Tracking ──────────────────────────────────────────────

    def get_usage(self) -> Dict[str, Any]:
        """Get current month's usage stats."""
        self._ensure_current_month()
        return {
            "queries_used": self._usage.get("queries", 0),
            "pages_used": self._usage.get("pages", 0),
            "queries_limit": self.config.get("max_monthly_queries", 500),
            "pages_limit": self.config.get("max_monthly_pages", 2000),
            "month": self._usage.get("month", ""),
        }

    def _check_usage_budget(self, queries: int = 0,
                             pages: int = 0) -> bool:
        """Check if operation is within monthly budget."""
        if not self.config.get("track_usage", True):
            return True
        self._ensure_current_month()
        q_limit = self.config.get("max_monthly_queries", 500)
        p_limit = self.config.get("max_monthly_pages", 2000)
        if queries and self._usage.get("queries", 0) + queries > q_limit:
            return False
        if pages and self._usage.get("pages", 0) + pages > p_limit:
            return False
        return True

    def _track_usage(self, queries: int = 0, pages: int = 0):
        """Increment usage counters."""
        self._ensure_current_month()
        self._usage["queries"] = self._usage.get("queries", 0) + queries
        self._usage["pages"] = self._usage.get("pages", 0) + pages
        self._save_usage()

    def _ensure_current_month(self):
        """Reset counters if month has changed."""
        current_month = datetime.now().strftime("%Y-%m")
        if self._usage.get("month") != current_month:
            self._usage = {
                "queries": 0, "pages": 0, "month": current_month
            }
            self._save_usage()

    # ─── Privacy Guard ───────────────────────────────────────────────

    def _contains_sensitive_data(self, file_path: str) -> bool:
        """
        Scan first 10KB of file for sensitive data patterns.
        This is a basic heuristic — not a full PII detector.
        """
        try:
            with open(file_path, "rb") as f:
                sample = f.read(10240).decode("utf-8", errors="ignore")
        except Exception:
            return False  # Can't read → let it through (PDF is binary anyway)

        sensitive_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"password\s*[:=]",                   # Passwords
            r"\b(confidential|top\s+secret)\b",   # Classification markers
        ]
        for pattern in sensitive_patterns:
            if re.search(pattern, sample, re.IGNORECASE):
                return True
        return False

    # ─── Persistence ─────────────────────────────────────────────────

    def _load_mapping(self):
        path = os.path.join(self.data_dir, "doc_mapping.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.doc_mapping = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.doc_mapping = {}

    def _save_mapping(self):
        path = os.path.join(self.data_dir, "doc_mapping.json")
        with open(path, "w") as f:
            json.dump(self.doc_mapping, f, indent=2)

    def _load_usage(self):
        path = os.path.join(self.data_dir, "usage.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self._usage = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        self._ensure_current_month()

    def _save_usage(self):
        path = os.path.join(self.data_dir, "usage.json")
        with open(path, "w") as f:
            json.dump(self._usage, f, indent=2)

    # ─── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(file_path: str) -> str:
        """SHA-256 hash of file contents."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _extract_json_array(text: str) -> List[Dict]:
        """Extract a JSON array from LLM response text."""
        # Try fenced code block first
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try raw JSON array
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get PageIndex integration stats for health endpoint."""
        return {
            "connected": self._connected,
            "enabled": True,
            "documents": len(self.doc_mapping),
            "ready_documents": sum(
                1 for info in self.doc_mapping.values()
                if info.get("status") == "ready"
            ),
            "usage": self.get_usage(),
        }
