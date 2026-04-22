"""Phase 2 retry matrix with source-aware classification and bounded backoff."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetrySource(str, Enum):
    TIMEOUT = "timeout"
    NETWORK = "network"
    CAPACITY = "capacity"
    MODEL_TRANSIENT = "model_transient"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_delay_ms: int
    max_delay_ms: int


class RetryMatrix:
    """Deterministic retry policy matrix keyed by failure source."""

    def __init__(self, policies: dict[RetrySource, RetryPolicy] | None = None):
        self._policies = policies or self.default_policies()

    @staticmethod
    def default_policies() -> dict[RetrySource, RetryPolicy]:
        return {
            RetrySource.TIMEOUT: RetryPolicy(max_attempts=3, base_delay_ms=300, max_delay_ms=3000),
            RetrySource.NETWORK: RetryPolicy(max_attempts=3, base_delay_ms=250, max_delay_ms=2500),
            RetrySource.CAPACITY: RetryPolicy(max_attempts=3, base_delay_ms=500, max_delay_ms=4000),
            RetrySource.MODEL_TRANSIENT: RetryPolicy(max_attempts=2, base_delay_ms=350, max_delay_ms=1500),
            RetrySource.DEPENDENCY: RetryPolicy(max_attempts=2, base_delay_ms=400, max_delay_ms=2000),
            RetrySource.UNKNOWN: RetryPolicy(max_attempts=1, base_delay_ms=0, max_delay_ms=0),
        }

    def classify_exception(self, exc: Exception) -> RetrySource:
        if isinstance(exc, TimeoutError):
            return RetrySource.TIMEOUT

        message = str(exc).lower()
        if any(token in message for token in ["timeout", "timed out", "deadline exceeded"]):
            return RetrySource.TIMEOUT
        if any(token in message for token in ["connection reset", "connection refused", "network", "dns", "socket", "ssl"]):
            return RetrySource.NETWORK
        if any(token in message for token in ["rate limit", "429", "capacity", "overloaded", "quota", "too many requests"]):
            return RetrySource.CAPACITY
        if any(token in message for token in ["temporarily unavailable", "try again", "transient", "service unavailable"]):
            return RetrySource.MODEL_TRANSIENT
        if any(token in message for token in ["dependency", "upstream", "gateway", "503"]):
            return RetrySource.DEPENDENCY
        return RetrySource.UNKNOWN

    def policy_for(self, source: RetrySource) -> RetryPolicy:
        return self._policies.get(source, self._policies[RetrySource.UNKNOWN])

    def should_retry(self, source: RetrySource, attempts_so_far: int) -> bool:
        policy = self.policy_for(source)
        return attempts_so_far < policy.max_attempts

    def next_backoff_ms(self, source: RetrySource, attempts_so_far: int) -> int:
        policy = self.policy_for(source)
        if policy.base_delay_ms <= 0:
            return 0

        exponent = max(attempts_so_far - 1, 0)
        delay = policy.base_delay_ms * (2 ** exponent)
        return min(delay, policy.max_delay_ms)
