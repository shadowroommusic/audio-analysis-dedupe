from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioInfo:
    path: str
    file_name: str
    size_bytes: int
    sha256: str
    container: str | None = None
    codec: str | None = None
    duration_ms: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bits_per_sample: int | None = None
    bitrate_kbps: float | None = None
    audio_sha256: str | None = None
    probe: str = "unavailable"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def analyzed(self) -> bool:
        return self.duration_ms is not None and self.sample_rate is not None


def to_json(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_json(item) for key, item in asdict(value).items()}
    if isinstance(value, (tuple, list)):
        return [to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, float):
        return round(value, 3)
    return value
