"""
Ambient Configuration — Load/Save ambient listening settings.
JSON-based configuration stored locally in data/ambient_config.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AmbientConfig:
    """All ambient listening configuration."""

    # Master toggle
    enabled: bool = False

    # VAD settings
    vad_threshold: float = 0.5

    # Conversation segmentation
    auto_ingest: bool = True
    silence_timeout_s: int = 120       # 2 min = conversation end
    min_speech_ms: int = 250           # Minimum speech to process

    # Speaker ID
    similarity_threshold: float = 0.70

    # Whisper
    whisper_model_size: str = "small"
    whisper_device: str = "auto"       # auto, cuda, cpu
    whisper_language: Optional[str] = None  # None = auto-detect

    # TTS
    tts_enabled: bool = True
    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0

    # Audio capture
    audio_device: Optional[int] = None  # None = system default

    # Privacy
    record_raw_audio: bool = False     # Archive WAV files (disk-heavy)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmbientConfig":
        """Create config from dict, ignoring unknown fields."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def load_config(data_dir: str = "data") -> AmbientConfig:
    """Load ambient config from disk, or return defaults."""
    path = Path(data_dir) / "ambient_config.json"
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return AmbientConfig.from_dict(data)
        except Exception:
            pass
    return AmbientConfig()


def save_config(config: AmbientConfig, data_dir: str = "data"):
    """Save ambient config to disk."""
    path = Path(data_dir) / "ambient_config.json"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
