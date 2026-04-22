#!/usr/bin/env python3
"""Deep end-to-end validator for the Gemini Live ambient voice pipeline.

This script verifies:
1. Gemini live session lifecycle and always-listening state transitions
2. Live websocket event stream (`session_state`, `live_partial`, `live_final_turn`,
   `assistant_audio_chunk`, `transcript`, `memory_tagging`)
3. STT/TTS behavior via `/api/voice/query` and `/api/tts/synthesize`
4. Conversation persistence with `live_turn_id` + `retention_trace`
5. Continuous orchestration across multiple injected turns within one session

Usage:
  python scripts/deep_ambient_live_test.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import numpy as np
import websockets


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    required: bool = True


class EventCollector:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.events: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.connected = asyncio.Event()
        self._stop = asyncio.Event()

    async def run(self) -> None:
        try:
            async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, max_size=4 * 1024 * 1024) as ws:
                self.connected.set()
                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if isinstance(raw, bytes):
                        continue

                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    payload["_received_at"] = round(time.time(), 3)
                    self.events.append(payload)

                    if payload.get("type") == "ping":
                        await ws.send(json.dumps({"command": "ping"}))
        except Exception as exc:
            self.error = str(exc)
            self.connected.set()

    def stop(self) -> None:
        self._stop.set()

    def count(self, event_type: str) -> int:
        return sum(1 for e in self.events if e.get("type") == event_type)

    def count_final_turns(self, speaker_label: Optional[str] = None) -> int:
        if speaker_label is None:
            return sum(1 for e in self.events if e.get("type") == "live_final_turn")
        return sum(
            1
            for e in self.events
            if e.get("type") == "live_final_turn" and str(e.get("speaker_label", "")).upper() == speaker_label.upper()
        )

    def any_state(self, state_name: str) -> bool:
        return any(
            e.get("type") == "session_state" and str(e.get("state", "")) == state_name
            for e in self.events
        )

    def event_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self.events:
            key = str(e.get("type", "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts


class DeepAmbientLiveTester:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.checks: List[CheckResult] = []
        self.voice_query_results: List[Dict[str, Any]] = []
        self.inject_summaries: List[Dict[str, Any]] = []
        self.live_status_history: List[Dict[str, Any]] = []
        self.start_payload: Dict[str, Any] = {}
        self.stop_payload: Dict[str, Any] = {}
        self.latest_conversation: Optional[Dict[str, Any]] = None

        api_base, host_base = normalize_base_url(args.base_url)
        self.api_base = api_base
        self.host_base = host_base
        self.ws_url = to_ws_url(host_base, "/ws/ambient")

    def _record(self, name: str, passed: bool, detail: str, required: bool = True) -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail, required=required))

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        expected: tuple[int, ...] = (200,),
    ) -> Dict[str, Any]:
        resp = await client.request(method, path, json=payload)
        if resp.status_code not in expected:
            text = resp.text[:800]
            raise RuntimeError(f"{method} {path} -> {resp.status_code}: {text}")
        if not resp.text:
            return {}
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype or resp.text.startswith("{"):
            return resp.json()
        return {"raw": resp.text}

    async def _request_bytes(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        expected: tuple[int, ...] = (200,),
    ) -> bytes:
        resp = await client.request(method, path, json=payload)
        if resp.status_code not in expected:
            text = resp.text[:800]
            raise RuntimeError(f"{method} {path} -> {resp.status_code}: {text}")
        return resp.content

    async def _wait_for(
        self,
        predicate,
        *,
        timeout_s: float,
        interval_s: float = 0.3,
    ) -> Any:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            value = await predicate()
            if value:
                return value
            await asyncio.sleep(interval_s)
        return None

    async def run(self) -> Dict[str, Any]:
        collector = EventCollector(self.ws_url)
        ws_task: Optional[asyncio.Task] = None

        async with httpx.AsyncClient(base_url=self.api_base, timeout=self.args.request_timeout) as client:
            # Cleanup pre-existing sessions so this run is isolated.
            await self._best_effort_stop(client)

            # Configure Gemini live mode for deterministic behavior.
            cfg_payload = {
                "stt_provider": "gemini",
                "tts_provider": "gemini",
                "live_mode": "gemini_live",
                "energy_gate_threshold": self.args.energy_gate_threshold,
                "energy_min_speech_ms": self.args.energy_min_speech_ms,
                "energy_silence_ms": self.args.energy_silence_ms,
            }
            config_response = await self._request_json(client, "POST", "/ambient/config", payload=cfg_payload)
            self._record(
                "config_applied",
                config_response.get("stt_provider") == "gemini"
                and config_response.get("tts_provider") == "gemini"
                and config_response.get("live_mode") == "gemini_live",
                f"stt={config_response.get('stt_provider')} tts={config_response.get('tts_provider')} live_mode={config_response.get('live_mode')}",
            )

            # Start websocket collector before starting live session to capture session_started.
            ws_task = asyncio.create_task(collector.run(), name="ambient-ws-collector")
            await asyncio.wait_for(collector.connected.wait(), timeout=self.args.connect_timeout)
            self._record("ws_connected", collector.error is None, collector.error or f"connected to {self.ws_url}")

            if collector.error is not None:
                raise RuntimeError(f"websocket connection failed: {collector.error}")

            # Start live session.
            self.start_payload = await self._request_json(client, "POST", "/ambient/live/start")
            self._record(
                "live_start_success",
                bool(self.start_payload.get("success")),
                json.dumps(self.start_payload, ensure_ascii=True),
            )

            started = await self._wait_for(
                lambda: self._live_status_if_running(client),
                timeout_s=self.args.state_timeout,
            )
            self._record(
                "live_running",
                started is not None,
                json.dumps(started or {}, ensure_ascii=True),
            )
            if started:
                self.live_status_history.append(started)

            session_id = (started or {}).get("session_id")
            initial_segments = int((started or {}).get("segments_detected", 0))

            # Validate websocket states during startup.
            await asyncio.sleep(1.2)
            self._record(
                "ws_session_state_seen",
                collector.count("session_state") > 0,
                f"session_state_events={collector.count('session_state')}",
            )

            # Prepare and run two turns to verify continuous listening within one session.
            phrases = [
                self.args.phrase1,
                self.args.phrase2,
            ]

            current_expected_segments = initial_segments
            for idx, phrase in enumerate(phrases, start=1):
                turn_result = await self._execute_turn(client, collector, phrase, idx)
                self.inject_summaries.append(turn_result)

                current_expected_segments += 1
                reached = await self._wait_for(
                    lambda: self._live_status_with_min_segments(client, current_expected_segments),
                    timeout_s=self.args.turn_timeout,
                )
                passed = reached is not None
                detail = f"expected_segments>={current_expected_segments}"
                if reached is not None:
                    detail += f" actual={reached.get('segments_detected')} state={reached.get('state')}"
                    self.live_status_history.append(reached)
                self._record(f"segments_increment_turn_{idx}", passed, detail)

            # Session continuity: same session_id after both turns.
            final_running = await self._request_json(client, "GET", "/ambient/live/status")
            self.live_status_history.append(final_running)
            same_session = bool(session_id) and final_running.get("session_id") == session_id
            self._record(
                "single_continuous_session",
                same_session,
                f"started={session_id} final={final_running.get('session_id')}",
            )
            idle_status = await self._wait_for(
                lambda: self._live_status_if_idle(client),
                timeout_s=self.args.state_timeout,
            )
            if idle_status:
                self.live_status_history.append(idle_status)

            # Check expected live websocket signals.
            self._record(
                "live_partial_events",
                collector.count("live_partial") > 0,
                f"count={collector.count('live_partial')}",
            )
            self._record(
                "live_final_turn_events",
                collector.count("live_final_turn") > 0,
                f"count={collector.count('live_final_turn')}",
            )
            self._record(
                "transcript_events",
                collector.count("transcript") > 0,
                f"count={collector.count('transcript')}",
            )
            self._record(
                "memory_tagging_events",
                collector.count("memory_tagging") > 0,
                f"count={collector.count('memory_tagging')}",
                required=False,
            )
            self._record(
                "assistant_audio_chunk_events",
                collector.count("assistant_audio_chunk") > 0,
                f"count={collector.count('assistant_audio_chunk')}",
                required=self.args.strict_assistant_audio,
            )
            self._record(
                "state_returned_idle",
                idle_status is not None,
                f"state={(idle_status or final_running).get('state')}",
            )

            # Stop live session so in-progress conversation is finalized.
            self.stop_payload = await self._request_json(client, "POST", "/ambient/live/stop")
            self._record(
                "live_stop_success",
                bool(self.stop_payload.get("success")),
                json.dumps(self.stop_payload, ensure_ascii=True),
            )

            stopped = await self._wait_for(
                lambda: self._live_status_if_stopped(client),
                timeout_s=self.args.state_timeout,
            )
            self._record(
                "live_stopped",
                stopped is not None,
                json.dumps(stopped or {}, ensure_ascii=True),
            )
            if stopped:
                self.live_status_history.append(stopped)

            # Validate persisted conversations include live metadata.
            await asyncio.sleep(0.8)
            convs = await self._request_json(client, "GET", f"/ambient/conversations?limit={self.args.conversation_limit}&offset=0")
            conv_list = convs.get("conversations", []) if isinstance(convs, dict) else []
            self._record(
                "conversation_created",
                len(conv_list) > 0,
                f"conversation_count={len(conv_list)}",
            )

            if conv_list:
                latest_id = conv_list[0].get("id")
                if latest_id:
                    self.latest_conversation = await self._request_json(client, "GET", f"/ambient/conversations/{latest_id}")
                    turns = self.latest_conversation.get("turns", []) if isinstance(self.latest_conversation, dict) else []

                    has_live_metadata = any(
                        t.get("live_turn_id") and isinstance(t.get("retention_trace"), dict)
                        for t in turns
                    )
                    has_user_turn = any(str(t.get("speaker_label", "")).upper() != "ASSISTANT" for t in turns)
                    has_assistant_turn = any(str(t.get("speaker_label", "")).upper() == "ASSISTANT" for t in turns)

                    self._record(
                        "conversation_turns_present",
                        len(turns) > 0,
                        f"turn_count={len(turns)}",
                    )
                    self._record(
                        "conversation_live_metadata",
                        has_live_metadata,
                        "live_turn_id+retention_trace found" if has_live_metadata else "missing live metadata",
                    )
                    self._record(
                        "conversation_user_turn_present",
                        has_user_turn,
                        "non-assistant turn found" if has_user_turn else "no user-like turns",
                    )
                    self._record(
                        "conversation_assistant_turn_present",
                        has_assistant_turn,
                        "assistant turn found" if has_assistant_turn else "no assistant turns",
                        required=False,
                    )

        if ws_task is not None:
            collector.stop()
            try:
                await asyncio.wait_for(ws_task, timeout=3.0)
            except Exception:
                ws_task.cancel()

        return self._build_report(collector)

    async def _best_effort_stop(self, client: httpx.AsyncClient) -> None:
        for path in ("/ambient/live/stop", "/ambient/stop"):
            try:
                await self._request_json(client, "POST", path)
            except Exception:
                pass

    async def _live_status_if_running(self, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        status = await self._request_json(client, "GET", "/ambient/live/status")
        if status.get("running"):
            return status
        return None

    async def _live_status_if_stopped(self, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        status = await self._request_json(client, "GET", "/ambient/live/status")
        if not status.get("running"):
            return status
        return None

    async def _live_status_with_min_segments(
        self,
        client: httpx.AsyncClient,
        min_segments: int,
    ) -> Optional[Dict[str, Any]]:
        status = await self._request_json(client, "GET", "/ambient/live/status")
        segs = int(status.get("segments_detected", 0) or 0)
        if status.get("running") and segs >= min_segments:
            return status
        return None

    async def _live_status_if_idle(self, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        status = await self._request_json(client, "GET", "/ambient/live/status")
        if status.get("running") and status.get("state") == "idle_listening":
            return status
        return None

    async def _execute_turn(
        self,
        client: httpx.AsyncClient,
        collector: EventCollector,
        phrase: str,
        turn_index: int,
    ) -> Dict[str, Any]:
        # Generate audio via the active Gemini TTS provider to avoid external dependencies.
        wav_bytes = await self._request_bytes(client, "POST", "/tts/synthesize", payload={"text": phrase})
        pcm16, sample_rate = pcm16_mono_from_wav(wav_bytes)

        # Ensure energy level comfortably crosses gate threshold during synthetic injection.
        pcm16 = normalize_rms(pcm16, target_rms=self.args.injection_target_rms)

        payload_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")

        # Control-path STT/TTS check via /voice/query.
        voice_result = await self._request_json(
            client,
            "POST",
            "/voice/query",
            payload={"audio_base64": payload_b64},
        )
        self.voice_query_results.append(
            {
                "turn": turn_index,
                "phrase": phrase,
                "transcript": voice_result.get("transcript", ""),
                "answer_present": bool(voice_result.get("answer")),
                "audio_base64_present": bool(voice_result.get("audio_base64")),
                "stt_provider": voice_result.get("stt_provider"),
                "tts_provider": voice_result.get("tts_provider"),
            }
        )

        self._record(
            f"voice_query_turn_{turn_index}",
            bool(voice_result.get("transcript")) and bool(voice_result.get("answer")),
            f"transcript_len={len(str(voice_result.get('transcript', '')))} answer_len={len(str(voice_result.get('answer', '')))}",
        )
        self._record(
            f"voice_query_audio_turn_{turn_index}",
            bool(voice_result.get("audio_base64")),
            "audio response present" if voice_result.get("audio_base64") else "no TTS audio in voice query response",
            required=False,
        )

        user_final_before = collector.count_final_turns(speaker_label="USER")
        all_final_before = collector.count("live_final_turn")

        inject_result = await self._request_json(
            client,
            "POST",
            "/ambient/live/inject-audio",
            payload={
                "audio_base64": payload_b64,
                "sample_rate": sample_rate,
            },
        )

        # Wait for a new final turn event after injection.
        new_final = await self._wait_for(
            lambda: self._collector_delta(collector, all_final_before),
            timeout_s=self.args.turn_timeout,
            interval_s=0.25,
        )
        self._record(
            f"live_final_after_inject_turn_{turn_index}",
            bool(new_final),
            f"final_turn_count_before={all_final_before} after={collector.count('live_final_turn')}",
            required=False,
        )

        user_final_after = collector.count_final_turns(speaker_label="USER")
        self._record(
            f"user_final_turn_detected_{turn_index}",
            user_final_after > user_final_before,
            f"user_final_before={user_final_before} user_final_after={user_final_after}",
            required=False,
        )

        return {
            "turn": turn_index,
            "phrase": phrase,
            "sample_rate": sample_rate,
            "samples": int(pcm16.size),
            "inject_result": inject_result,
        }

    async def _collector_delta(self, collector: EventCollector, before: int) -> Optional[int]:
        count = collector.count("live_final_turn")
        if count > before:
            return count
        return None

    def _build_report(self, collector: EventCollector) -> Dict[str, Any]:
        required_failed = [c for c in self.checks if c.required and not c.passed]
        optional_failed = [c for c in self.checks if (not c.required) and (not c.passed)]

        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": self.args.base_url,
            "api_base": self.api_base,
            "ws_url": self.ws_url,
            "summary": {
                "required_passed": sum(1 for c in self.checks if c.required and c.passed),
                "required_failed": len(required_failed),
                "optional_failed": len(optional_failed),
                "overall_pass": len(required_failed) == 0,
            },
            "checks": [asdict(c) for c in self.checks],
            "start_payload": self.start_payload,
            "stop_payload": self.stop_payload,
            "live_status_history": self.live_status_history,
            "voice_query_results": self.voice_query_results,
            "inject_summaries": self.inject_summaries,
            "ws_event_counts": collector.event_counts(),
            "ws_recent_events": collector.events[-80:],
            "latest_conversation": self.latest_conversation,
            "ws_error": collector.error,
        }
        return report


def normalize_base_url(base_url: str) -> tuple[str, str]:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/api"):
        return trimmed, trimmed[:-4]
    return f"{trimmed}/api", trimmed


def to_ws_url(host_base: str, path: str) -> str:
    parsed = urlparse(host_base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL: {host_base}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{path}"


def pcm16_mono_from_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif sample_width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16)
    elif sample_width == 4:
        pcm = (np.frombuffer(frames, dtype=np.int32) >> 16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if pcm.size == 0:
        raise ValueError("Decoded WAV contains no audio samples")

    return pcm, int(sample_rate)


def normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    if audio.size == 0:
        return audio

    float_audio = audio.astype(np.float32)
    current_rms = float(np.sqrt(np.mean(float_audio * float_audio)))
    if current_rms <= 1e-6:
        return audio

    gain = float(target_rms) / current_rms
    gain = max(0.1, min(gain, 8.0))

    scaled = np.clip(float_audio * gain, -32768.0, 32767.0).astype(np.int16)
    return scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Gemini Live ambient pipeline test runner")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Backend host URL (with or without /api)")
    parser.add_argument("--output", default="ambient-live-deep-test-results.json", help="Path to write JSON report")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="HTTP request timeout in seconds")
    parser.add_argument("--connect-timeout", type=float, default=12.0, help="WebSocket connect timeout in seconds")
    parser.add_argument("--state-timeout", type=float, default=20.0, help="Timeout for waiting on state transitions")
    parser.add_argument("--turn-timeout", type=float, default=35.0, help="Timeout for waiting on per-turn outcomes")
    parser.add_argument("--conversation-limit", type=int, default=8, help="How many recent conversations to fetch")
    parser.add_argument("--energy-gate-threshold", type=float, default=450.0, help="Energy gate threshold for test run")
    parser.add_argument("--energy-min-speech-ms", type=int, default=220, help="Minimum speech duration in ms")
    parser.add_argument("--energy-silence-ms", type=int, default=280, help="Silence duration to finalize segments in ms")
    parser.add_argument("--injection-target-rms", type=float, default=2000.0, help="RMS normalization target for injected audio")
    parser.add_argument(
        "--phrase1",
        default="Remember this architecture test item 42. Keep this in memory as a technical todo.",
        help="First synthesized phrase used for test turn 1",
    )
    parser.add_argument(
        "--phrase2",
        default="Please summarize the last point and continue listening for the next instruction.",
        help="Second synthesized phrase used for test turn 2",
    )
    parser.add_argument(
        "--strict-assistant-audio",
        action="store_true",
        help="Fail required checks if assistant audio chunks are not observed",
    )
    return parser.parse_args()


def print_summary(report: Dict[str, Any]) -> None:
    summary = report.get("summary", {})
    checks = report.get("checks", [])

    print("DEEP_AMBIENT_LIVE_TEST_SUMMARY")
    print(f"  overall_pass={summary.get('overall_pass')}")
    print(f"  required_passed={summary.get('required_passed')} required_failed={summary.get('required_failed')}")
    print(f"  optional_failed={summary.get('optional_failed')}")

    failed_required = [c for c in checks if c.get("required") and not c.get("passed")]
    failed_optional = [c for c in checks if (not c.get("required")) and (not c.get("passed"))]

    if failed_required:
        print("  required_failures:")
        for item in failed_required:
            print(f"    - {item.get('name')}: {item.get('detail')}")

    if failed_optional:
        print("  optional_failures:")
        for item in failed_optional:
            print(f"    - {item.get('name')}: {item.get('detail')}")


async def async_main(args: argparse.Namespace) -> int:
    tester = DeepAmbientLiveTester(args)
    report = await tester.run()

    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print_summary(report)
    print(f"  report_file={output_path}")

    return 0 if report.get("summary", {}).get("overall_pass") else 1


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(async_main(args))
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"DEEP_AMBIENT_LIVE_TEST_ERROR {exc}")
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
