"""Background worker for auto-executing approved permission requests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from .retry import RetryMatrix
from .safety import PermissionRequest, PermissionStatus, SafeToolRuntime


ApprovalHandler = Callable[[PermissionRequest], Awaitable[Dict[str, Any]]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(raw: str) -> Optional[datetime]:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class ApprovalExecutionWorker:
    """Polls approved requests and executes them with idempotency safeguards."""

    def __init__(
        self,
        safe_tool_runtime: SafeToolRuntime,
        handlers: Dict[str, ApprovalHandler],
        poll_interval_seconds: float = 2.0,
        execution_timeout_seconds: float = 60.0,
        max_attempts: int = 2,
        retry_matrix: Optional[RetryMatrix] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ):
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if execution_timeout_seconds <= 0:
            raise ValueError("execution_timeout_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self.safe_tool_runtime = safe_tool_runtime
        self.handlers = dict(handlers)
        self.poll_interval_seconds = poll_interval_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self.max_attempts = max_attempts
        self.retry_matrix = retry_matrix or RetryMatrix()
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="approval-execution-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                # Keep worker alive so one failing request does not stop the queue.
                print(f"  ⚠ Approval execution worker cycle failed: {exc}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> None:
        approved = self.safe_tool_runtime.list_permissions_by_status(PermissionStatus.APPROVED)
        for request in approved:
            await self._execute_request_if_needed(request)

    def get_status(self) -> Dict[str, Any]:
        approved = self.safe_tool_runtime.list_permissions_by_status(PermissionStatus.APPROVED)
        summary = {
            "approved_total": len(approved),
            "pending_total": len(self.safe_tool_runtime.list_pending_permissions()),
            "running": 0,
            "waiting_retry": 0,
            "completed": 0,
            "failed": 0,
            "unsupported": 0,
            "idle": 0,
        }

        for request in approved:
            state = self._get_execution_state(request)
            status = str(state.get("status", "")).strip().lower()
            if status in summary:
                summary[status] += 1
            else:
                summary["idle"] += 1

        return {
            "running": self.is_running(),
            "poll_interval_seconds": self.poll_interval_seconds,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "max_attempts": self.max_attempts,
            "summary": summary,
        }

    def _get_execution_state(self, request: PermissionRequest) -> Dict[str, Any]:
        execution = request.metadata.get("_execution")
        if not isinstance(execution, dict):
            execution = {}
            request.metadata["_execution"] = execution
        return execution

    def _is_terminal_state(self, state: Dict[str, Any]) -> bool:
        status = str(state.get("status", "")).strip().lower()
        attempts = int(state.get("attempts", 0) or 0)
        if status in {"completed", "unsupported"}:
            return True
        if status == "failed" and attempts >= self.max_attempts:
            return True
        return False

    def _should_wait_for_retry(self, state: Dict[str, Any], now: datetime) -> bool:
        if str(state.get("status", "")).strip().lower() != "waiting_retry":
            return False
        next_retry_at = _parse_iso_datetime(str(state.get("next_retry_at", "") or ""))
        if next_retry_at is None:
            return False
        return now < next_retry_at

    async def _execute_request_if_needed(self, request: PermissionRequest) -> None:
        now = self._now_fn()
        state = self._get_execution_state(request)

        if self._is_terminal_state(state):
            return
        if str(state.get("status", "")).strip().lower() == "running":
            return
        if self._should_wait_for_retry(state, now):
            return

        attempts = int(state.get("attempts", 0) or 0)
        if attempts >= self.max_attempts:
            return

        handler = self.handlers.get(request.tool_name)
        if handler is None:
            state.update(
                {
                    "status": "unsupported",
                    "attempts": attempts,
                    "finished_at": _utc_now_iso(),
                    "last_error": f"No handler registered for tool '{request.tool_name}'",
                }
            )
            self.safe_tool_runtime.record_permission_execution(
                permission_id=request.permission_id,
                execution_status="unsupported",
                metadata={
                    "tool_name": request.tool_name,
                    "attempts": attempts,
                },
            )
            return

        attempts += 1
        state.update(
            {
                "status": "running",
                "attempts": attempts,
                    "started_at": now.isoformat(),
                "finished_at": None,
                "last_error": "",
                "result": None,
                    "retry_source": "",
                    "next_retry_at": None,
                    "next_backoff_ms": 0,
                    "backoff_ms_total": int(state.get("backoff_ms_total", 0) or 0),
            }
        )

        self.safe_tool_runtime.record_permission_execution(
            permission_id=request.permission_id,
            execution_status="running",
            metadata={"attempts": attempts},
        )

        try:
            result = await asyncio.wait_for(
                handler(request),
                timeout=self.execution_timeout_seconds,
            )
        except Exception as exc:
            retry_source = self.retry_matrix.classify_exception(exc)
            should_retry = self.retry_matrix.should_retry(retry_source, attempts)
            backoff_ms = self.retry_matrix.next_backoff_ms(retry_source, attempts) if should_retry else 0
            next_retry_at = now + timedelta(milliseconds=backoff_ms) if backoff_ms > 0 else None
            backoff_total = int(state.get("backoff_ms_total", 0) or 0) + backoff_ms

            state.update(
                {
                    "status": "waiting_retry" if should_retry else "failed",
                    "finished_at": now.isoformat(),
                    "last_error": str(exc),
                    "result": None,
                    "retry_source": retry_source.value,
                    "next_backoff_ms": backoff_ms,
                    "next_retry_at": next_retry_at.isoformat() if next_retry_at else None,
                    "backoff_ms_total": backoff_total,
                }
            )
            self.safe_tool_runtime.record_permission_execution(
                permission_id=request.permission_id,
                execution_status="failed",
                metadata={
                    "attempts": attempts,
                    "error": str(exc),
                    "retry_source": retry_source.value,
                    "retry_scheduled": should_retry,
                    "retry_in_ms": backoff_ms,
                    "backoff_ms_total": backoff_total,
                },
            )
            return

        result_payload: Dict[str, Any]
        if isinstance(result, dict):
            result_payload = result
        else:
            result_payload = {"value": result}

        state.update(
            {
                "status": "completed",
                "finished_at": now.isoformat(),
                "last_error": "",
                "result": result_payload,
                "next_backoff_ms": 0,
                "next_retry_at": None,
                "backoff_ms_total": int(state.get("backoff_ms_total", 0) or 0),
            }
        )
        self.safe_tool_runtime.record_permission_execution(
            permission_id=request.permission_id,
            execution_status="completed",
            metadata={
                "attempts": attempts,
                "result": result_payload,
                "backoff_ms_total": int(state.get("backoff_ms_total", 0) or 0),
            },
        )
