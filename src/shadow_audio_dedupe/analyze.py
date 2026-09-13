from __future__ import annotations

from pathlib import Path

from .model import AudioInfo, to_json
from .readers import analyze_file

DEFAULT_AUDIO_SUFFIXES = frozenset(
    {
        ".wav",
        ".wave",
        ".aif",
        ".aiff",
        ".aifc",
        ".au",
        ".snd",
        ".mp3",
        ".m4a",
        ".mp4",
        ".aac",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".wma",
    }
)


def iter_audio_files(
    folder: str | Path,
    recursive: bool = True,
    suffixes: "frozenset[str]" = DEFAULT_AUDIO_SUFFIXES,
) -> "list[Path]":
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    walker = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (path for path in walker if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: str(path).casefold(),
    )


def analyze_folder(
    folder: str | Path,
    recursive: bool = True,
    suffixes: "frozenset[str]" = DEFAULT_AUDIO_SUFFIXES,
    ffprobe: "str | bool | None" = None,
) -> dict:
    """Analyze every matching file in a folder and report what was found."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    files = iter_audio_files(root, recursive=recursive, suffixes=suffixes)
    infos: "list[AudioInfo]" = []
    skipped: "list[dict]" = []
    for path in files:
        try:
            infos.append(analyze_file(path, ffprobe=ffprobe))
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    formats: "dict[str, int]" = {}
    rates: "dict[str, int]" = {}
    for info in infos:
        key = info.codec or info.container or "unknown"
        formats[key] = formats.get(key, 0) + 1
        rate_key = str(info.sample_rate) if info.sample_rate else "unknown"
        rates[rate_key] = rates.get(rate_key, 0) + 1
    return {
        "schema_version": 1,
        "mode": "read-only-analysis",
        "folder": str(root),
        "recursive": recursive,
        "file_count": len(infos),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "files": [to_json(info) for info in infos],
        "summary": {
            "total_size_bytes": sum(info.size_bytes for info in infos),
            "total_duration_ms": sum(info.duration_ms or 0 for info in infos),
            "analyzed_files": sum(info.analyzed for info in infos),
            "unanalyzed_files": sum(not info.analyzed for info in infos),
            "codecs": formats,
            "sample_rates": rates,
        },
        "warnings": [
            "No audio file or vendor database was modified.",
            "Durations come from the container header; a streaming decode can differ by a few milliseconds.",
        ],
    }
