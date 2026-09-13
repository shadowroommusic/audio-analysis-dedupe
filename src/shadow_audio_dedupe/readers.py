from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import wave

from .model import AudioInfo

HASH_CHUNK = 1 << 20

WAV_CODECS = {1: "pcm_u8", 2: "pcm_s16le", 3: "pcm_s24le", 4: "pcm_s32le"}

AIFF_COMPRESSION = {
    "NONE": "pcm_s16be",
    "sowt": "pcm_s16le",
    "twos": "pcm_s16be",
    "in24": "pcm_s24be",
    "in32": "pcm_s32be",
    "fl32": "pcm_f32be",
    "fl64": "pcm_f64be",
    "ulaw": "pcm_mulaw",
    "alaw": "pcm_alaw",
}

AU_ENCODINGS = {
    1: ("pcm_mulaw", 8),
    2: ("pcm_alaw", 8),
    3: ("pcm_s16be", 16),
    4: ("pcm_s24be", 24),
    5: ("pcm_s32be", 32),
    6: ("pcm_f32be", 32),
    7: ("pcm_f64be", 64),
    27: ("adpcm_g726", 4),
}

METADATA_FIELDS = (
    "container",
    "codec",
    "duration_ms",
    "sample_rate",
    "channels",
    "bits_per_sample",
    "bitrate_kbps",
    "audio_sha256",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_digest(reader) -> "tuple[str, int]":
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = reader(HASH_CHUNK)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _apply_bitrate(data: dict, byte_count: int | None, fallback_bytes: int | None = None) -> None:
    duration_ms = data.get("duration_ms")
    if not duration_ms:
        return
    payload = byte_count if byte_count else fallback_bytes
    if not payload:
        return
    data["bitrate_kbps"] = round(payload * 8 / (duration_ms / 1000) / 1000, 1)


def read_wav(path: Path, data: dict) -> None:
    """Read a RIFF/WAVE container with the standard library `wave` module."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        compression = handle.getcomptype()
        data["container"] = "wav"
        data["channels"] = channels
        data["sample_rate"] = rate
        data["bits_per_sample"] = width * 8
        if rate:
            data["duration_ms"] = round(frames * 1000 / rate)
        if compression != "NONE":
            data["codec"] = compression.lower()
        else:
            data["codec"] = WAV_CODECS.get(width, f"pcm_s{width * 8}le")
        digest, total = _stream_digest(handle.readframes)
        data["audio_sha256"] = digest
        _apply_bitrate(data, total, path.stat().st_size)


def _read_extended(raw: bytes) -> float:
    """Decode an 80-bit IEEE 754 extended float as used by AIFF sample rates."""
    if len(raw) < 10:
        raise ValueError("truncated 80-bit float")
    exponent = int.from_bytes(raw[:2], "big") & 0x7FFF
    mantissa = int.from_bytes(raw[2:10], "big")
    if exponent == 0 and mantissa == 0:
        return 0.0
    value = math.ldexp(mantissa, exponent - 16383 - 63)
    return -value if raw[0] & 0x80 else value


def _iter_chunks(handle, size: int):
    position = 12
    while position + 8 <= size:
        handle.seek(position)
        chunk_id = handle.read(4)
        raw_size = handle.read(4)
        if len(raw_size) < 4:
            return
        chunk_size = int.from_bytes(raw_size, "big")
        yield chunk_id, position + 8, chunk_size
        position = position + 8 + chunk_size + (chunk_size & 1)


def read_aiff_manual(path: Path, data: dict) -> None:
    """Minimal AIFF/AIFF-C reader used when the deprecated `aifc` module is gone."""
    data["container"] = "aiff"
    size = path.stat().st_size
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[:4] != b"FORM":
            raise ValueError("not an IFF FORM container")
        compression = None
        for chunk_id, body_pos, chunk_size in _iter_chunks(handle, size):
            if chunk_id == b"COMM":
                handle.seek(body_pos)
                body = handle.read(min(chunk_size, 64))
                if len(body) < 18:
                    raise ValueError("truncated COMM chunk")
                channels = int.from_bytes(body[0:2], "big")
                frames = int.from_bytes(body[2:6], "big")
                bits = int.from_bytes(body[6:8], "big")
                rate = _read_extended(body[8:18])
                if len(body) >= 22:
                    compression = body[18:22].decode("latin-1").strip("\x00 ")
                data["channels"] = channels
                data["bits_per_sample"] = bits
                data["sample_rate"] = round(rate) if rate else None
                if rate:
                    data["duration_ms"] = round(frames * 1000 / rate)
                data["codec"] = AIFF_COMPRESSION.get(compression, "pcm_s16be" if compression in (None, "", "NONE") else compression.lower())
            elif chunk_id == b"SSND":
                handle.seek(body_pos + 8)
                remaining = max(chunk_size - 8, 0)
                digest = hashlib.sha256()
                total = 0
                while remaining > 0:
                    chunk = handle.read(min(HASH_CHUNK, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                    remaining -= len(chunk)
                data["audio_sha256"] = digest.hexdigest()
                _apply_bitrate(data, total, size)
                return
    _apply_bitrate(data, None, size)


def read_aiff(path: Path, data: dict) -> None:
    """Read AIFF through the standard library, falling back to a manual parser."""
    try:
        import aifc
    except ImportError:
        read_aiff_manual(path, data)
        return
    data["container"] = "aiff"
    with aifc.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        compression = handle.getcomptype()
        data["channels"] = channels
        data["sample_rate"] = rate
        data["bits_per_sample"] = width * 8
        if rate:
            data["duration_ms"] = round(frames * 1000 / rate)
        data["codec"] = AIFF_COMPRESSION.get(compression, "pcm_s16be" if compression == "NONE" else compression.lower())
        digest, total = _stream_digest(handle.readframes)
        data["audio_sha256"] = digest
        _apply_bitrate(data, total, path.stat().st_size)


def read_au_manual(path: Path, data: dict) -> None:
    """Read a Sun/NeXT `.snd` (AU) header without the deprecated `sunau` module."""
    data["container"] = "au"
    with open(path, "rb") as handle:
        header = handle.read(24)
        if len(header) < 24 or header[:4] != b".snd":
            raise ValueError("not an AU/SND container")
        data_offset = int.from_bytes(header[4:8], "big")
        data_size = int.from_bytes(header[8:12], "big")
        encoding = int.from_bytes(header[12:16], "big")
        rate = int.from_bytes(header[16:20], "big")
        channels = int.from_bytes(header[20:24], "big")
        codec, bits = AU_ENCODINGS.get(encoding, (f"au_encoding_{encoding}", None))
        data["codec"] = codec
        data["sample_rate"] = rate or None
        data["channels"] = channels or None
        data["bits_per_sample"] = bits
        handle.seek(data_offset)
        remaining = data_size or None
        digest = hashlib.sha256()
        total = 0
        while remaining is None or remaining > 0:
            chunk = handle.read(HASH_CHUNK if remaining is None else min(HASH_CHUNK, remaining))
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        data["audio_sha256"] = digest.hexdigest()
        if rate and channels and bits:
            frames = total / (channels * bits / 8)
            data["duration_ms"] = round(frames * 1000 / rate)
        _apply_bitrate(data, total, path.stat().st_size)


def read_au(path: Path, data: dict) -> None:
    """Read AU through the standard library, falling back to a manual parser."""
    try:
        import sunau
    except ImportError:
        read_au_manual(path, data)
        return
    data["container"] = "au"
    with sunau.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        data["channels"] = channels
        data["sample_rate"] = rate
        data["bits_per_sample"] = width * 8
        if rate:
            data["duration_ms"] = round(frames * 1000 / rate)
        data["codec"] = {
            "ULAW": "pcm_mulaw",
            "ALAW": "pcm_alaw",
            "PCM": f"pcm_s{width * 8}be",
        }.get(handle.getcomptype(), handle.getcomptype().lower())
        digest, total = _stream_digest(handle.readframes)
        data["audio_sha256"] = digest
        _apply_bitrate(data, total, path.stat().st_size)


READERS = {
    ".wav": ("wave", read_wav),
    ".wave": ("wave", read_wav),
    ".aif": ("aifc", read_aiff),
    ".aiff": ("aifc", read_aiff),
    ".aifc": ("aifc", read_aiff),
    ".au": ("sunau", read_au),
    ".snd": ("sunau", read_au),
}


def resolve_ffprobe(ffprobe: "str | bool | None" = None) -> str | None:
    if ffprobe is False:
        return None
    if isinstance(ffprobe, str):
        return ffprobe or None
    override = os.environ.get("SHADOW_FFPROBE")
    if override is not None:
        return override or None
    return shutil.which("ffprobe")


def read_ffprobe(executable: str, path: Path, data: dict) -> None:
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate,format_name:stream=codec_name,codec_type,sample_rate,channels,bits_per_sample",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, check=True, timeout=300)
    payload = json.loads(completed.stdout.decode("utf-8", "replace") or "{}")
    container = payload.get("format") or {}
    streams = [item for item in (payload.get("streams") or []) if item.get("codec_type") == "audio"]
    stream = streams[0] if streams else {}
    if container.get("format_name"):
        data["container"] = str(container["format_name"]).split(",")[0]
    if stream.get("codec_name"):
        data["codec"] = stream["codec_name"]
    if container.get("duration"):
        data["duration_ms"] = round(float(container["duration"]) * 1000)
    if stream.get("sample_rate"):
        data["sample_rate"] = int(stream["sample_rate"])
    if stream.get("channels"):
        data["channels"] = int(stream["channels"])
    if stream.get("bits_per_sample"):
        data["bits_per_sample"] = int(stream["bits_per_sample"])
    if container.get("bit_rate"):
        data["bitrate_kbps"] = round(int(container["bit_rate"]) / 1000, 1)


def analyze_file(path: str | Path, ffprobe: "str | bool | None" = None) -> AudioInfo:
    """Read one file without ever modifying it. Failures are reported, not raised."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {target}")
    stat = target.stat()
    data = {field: None for field in METADATA_FIELDS}
    warnings: "list[str]" = []
    probe = "unavailable"
    reader = READERS.get(target.suffix.lower())
    if reader is not None:
        probe = reader[0]
        try:
            reader[1](target, data)
        except Exception as exc:
            warnings.append(f"{probe} reader could not parse the file: {exc}")
    if data["duration_ms"] is None or data["sample_rate"] is None:
        executable = resolve_ffprobe(ffprobe)
        if executable:
            try:
                read_ffprobe(executable, target, data)
                probe = "ffprobe"
            except Exception as exc:
                warnings.append(f"ffprobe failed: {exc}")
        elif reader is None:
            warnings.append("no standard-library reader for this container and ffprobe is not installed; metadata unavailable")
    if data["bitrate_kbps"] is None and data["duration_ms"]:
        data["bitrate_kbps"] = round(stat.st_size * 8 / (data["duration_ms"] / 1000) / 1000, 1)
    return AudioInfo(
        path=str(target),
        file_name=target.name,
        size_bytes=stat.st_size,
        sha256=file_sha256(target),
        probe=probe,
        warnings=tuple(warnings),
        **data,
    )
