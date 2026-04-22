"""
VRAM Coordination — Mutual Exclusion between LLM and Whisper GPU usage.
Uses asyncio.Lock to ensure only one GPU-intensive task runs at a time.

The fine-tuned 7B LLM (~14.3 GB) and faster-whisper (~500 MB) both fit in 20 GB VRAM,
but must not run GPU-intensive kernels simultaneously to avoid CUDA OOM spikes.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional


class VRAMGuard:
    """
    Ensures LLM inference and Whisper transcription don't overlap on GPU.

    Usage:
        async with vram_guard.acquire("whisper"):
            result = await transcriber.transcribe(audio)

        async with vram_guard.acquire("llm"):
            result = await llm.generate(prompt)
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_holder: Optional[str] = None
        self._acquire_count = 0
        self._total_wait_ms = 0.0

    @asynccontextmanager
    async def acquire(self, holder: str = "unknown"):
        """Context manager for GPU-exclusive access."""
        t0 = time.time()
        await self._lock.acquire()
        wait_ms = (time.time() - t0) * 1000
        self._total_wait_ms += wait_ms
        self._acquire_count += 1
        self._current_holder = holder

        if wait_ms > 50:
            print(f"  ⏳ VRAMGuard: '{holder}' waited {wait_ms:.0f}ms for GPU lock")

        try:
            yield
        finally:
            self._current_holder = None
            self._lock.release()

    @property
    def is_locked(self) -> bool:
        return self._lock.locked()

    @property
    def current_holder(self) -> Optional[str]:
        return self._current_holder

    def get_stats(self) -> dict:
        return {
            "locked": self.is_locked,
            "current_holder": self._current_holder,
            "total_acquires": self._acquire_count,
            "total_wait_ms": round(self._total_wait_ms, 1),
        }


# Singleton — shared across ambient service and LLM
vram_guard = VRAMGuard()
