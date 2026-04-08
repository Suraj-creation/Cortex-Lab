"""Runtime health governor deriving resource tiers from device/system signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .contracts import ResourceTier


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DeviceHealthSnapshot:
    battery_pct: Optional[float] = None
    thermal_state: str = "normal"
    network_state: str = "good"
    charging: bool = False
    cpu_load_pct: Optional[float] = None
    memory_pressure: str = "normal"
    sampled_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "thermal_state": self.thermal_state,
            "network_state": self.network_state,
            "charging": self.charging,
            "cpu_load_pct": self.cpu_load_pct,
            "memory_pressure": self.memory_pressure,
            "sampled_at": self.sampled_at,
        }


class RuntimeHealthGovernor:
    """Resource-tier policy implementation aligned to orchestrator contracts."""

    def __init__(self) -> None:
        self.last_snapshot = DeviceHealthSnapshot()
        self.last_tier = ResourceTier.TIER_1_FULL

    @staticmethod
    def _normalize_thermal(value: str) -> str:
        return (value or "normal").strip().lower()

    @staticmethod
    def _derive_tier(snapshot: DeviceHealthSnapshot) -> ResourceTier:
        thermal = RuntimeHealthGovernor._normalize_thermal(snapshot.thermal_state)
        battery = snapshot.battery_pct

        if thermal == "critical" or (battery is not None and battery < 10):
            return ResourceTier.TIER_4_EMERGENCY
        if thermal == "hot" or (battery is not None and battery < 20):
            return ResourceTier.TIER_3_MINIMUM
        if thermal == "warm" or (battery is not None and battery < 35):
            return ResourceTier.TIER_2_CONSERVATIVE
        return ResourceTier.TIER_1_FULL

    def evaluate(self, snapshot: DeviceHealthSnapshot) -> Dict[str, Any]:
        tier = self._derive_tier(snapshot)
        reasons = []

        if snapshot.battery_pct is not None and snapshot.battery_pct < 35:
            reasons.append("battery_constrained")
        if self._normalize_thermal(snapshot.thermal_state) in {"warm", "hot", "critical"}:
            reasons.append("thermal_pressure")
        if (snapshot.network_state or "").strip().lower() in {"offline", "unstable"}:
            reasons.append("network_degraded")

        self.last_snapshot = snapshot
        self.last_tier = tier

        return {
            "tier": tier.value,
            "reasons": reasons,
            "snapshot": snapshot.to_dict(),
        }

    def sample(
        self,
        *,
        battery_pct: Optional[float] = None,
        thermal_state: str = "normal",
        network_state: str = "good",
        charging: bool = False,
        cpu_load_pct: Optional[float] = None,
        memory_pressure: str = "normal",
    ) -> Dict[str, Any]:
        snapshot = DeviceHealthSnapshot(
            battery_pct=battery_pct,
            thermal_state=thermal_state,
            network_state=network_state,
            charging=charging,
            cpu_load_pct=cpu_load_pct,
            memory_pressure=memory_pressure,
        )
        return self.evaluate(snapshot)


runtime_health_governor = RuntimeHealthGovernor()
