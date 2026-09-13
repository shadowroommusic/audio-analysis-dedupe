from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
import unicodedata

from .analyze import analyze_folder
from .model import AudioInfo

COPY_MARKER_PATTERNS = (
    re.compile(r"\s*[\(\[\{]\s*(?:copy|kopie|コピー|dup(?:licate)?|\d{1,3})\s*[\)\]\}]\s*$", re.IGNORECASE),
    re.compile(r"\s*[-_ ]\s*copy(?:\s*\d{1,3})?\s*$", re.IGNORECASE),
    re.compile(r"\s*[-_]\s*\d{1,2}\s*$"),
)


def normalize_name(value: str) -> str:
    """Normalize a file name so 'Song (1).wav' and 'Song.wav' share a stem."""
    stem = Path(unicodedata.normalize("NFKC", value)).stem.casefold()
    previous = None
    while previous != stem:
        previous = stem
        for pattern in COPY_MARKER_PATTERNS:
            stem = pattern.sub("", stem)
        stem = stem.strip()
    return re.sub(r"[\s_\-.]+", " ", stem).strip()


def has_copy_marker(value: str) -> bool:
    stem = Path(unicodedata.normalize("NFKC", value)).stem.casefold().strip()
    return any(pattern.search(stem) for pattern in COPY_MARKER_PATTERNS)


def _size_similarity(left: AudioInfo, right: AudioInfo) -> float:
    if not left.size_bytes or not right.size_bytes:
        return 0.5
    ratio = min(left.size_bytes, right.size_bytes) / max(left.size_bytes, right.size_bytes)
    return max(0.0, ratio)


def _only_digits_differ(left: str, right: str) -> bool:
    """True when two normalized names differ only in their numbers (e.g. 203 vs 205)."""
    if not left or not right or left == right:
        return False
    return re.sub(r"\d+", "#", left) == re.sub(r"\d+", "#", right)


def _duration_similarity(left: AudioInfo, right: AudioInfo) -> float:
    if not left.duration_ms or not right.duration_ms:
        return 0.5
    delta = abs(left.duration_ms - right.duration_ms)
    return max(0.0, 1 - delta / 10_000)


def _technical_differences(left: AudioInfo, right: AudioInfo) -> "list[str]":
    differences: "list[str]" = []
    if left.size_bytes != right.size_bytes:
        differences.append(f"size differs by {abs(left.size_bytes - right.size_bytes)} bytes ({left.size_bytes} vs {right.size_bytes})")
    if (left.duration_ms or 0) and (right.duration_ms or 0) and left.duration_ms != right.duration_ms:
        differences.append(f"duration differs by {abs((left.duration_ms or 0) - (right.duration_ms or 0))} ms ({left.duration_ms} vs {right.duration_ms})")
    if left.sample_rate != right.sample_rate:
        differences.append(f"sample rate differs ({left.sample_rate} vs {right.sample_rate})")
    if left.channels != right.channels:
        differences.append(f"channels differ ({left.channels} vs {right.channels})")
    if left.bitrate_kbps != right.bitrate_kbps:
        differences.append(f"bitrate differs ({left.bitrate_kbps} vs {right.bitrate_kbps} kbps)")
    if left.bits_per_sample != right.bits_per_sample:
        differences.append(f"bit depth differs ({left.bits_per_sample} vs {right.bits_per_sample} bit)")
    if (left.container, left.codec) != (right.container, right.codec):
        differences.append(f"container/codec differs ({left.container}/{left.codec} vs {right.container}/{right.codec})")
    return differences


def similarity(left: AudioInfo, right: AudioInfo) -> tuple[float, "list[str]", "list[str]"]:
    """Score how likely two files are two versions of the same track."""
    name_left, name_right = normalize_name(left.file_name), normalize_name(right.file_name)
    name_score = SequenceMatcher(None, name_left, name_right).ratio() if name_left or name_right else 0.0
    duration_score = _duration_similarity(left, right)
    size_score = _size_similarity(left, right)
    score = 0.6 * name_score + 0.25 * duration_score + 0.15 * size_score
    reasons: "list[str]" = []
    if name_left == name_right and name_left:
        reasons.append(f"normalized names match: '{name_left}'")
        score = max(score, 0.7 + 0.3 * duration_score)
    elif name_score >= 0.7:
        reasons.append(f"names are {round(name_score * 100)}% similar: '{name_left}' vs '{name_right}'")
    if duration_score >= 0.95 and left.duration_ms and right.duration_ms:
        reasons.append(f"durations match within {abs(left.duration_ms - right.duration_ms)} ms")
    if _only_digits_differ(name_left, name_right):
        # Same series, different number: often a different track, so never rank it as a strong match.
        reasons.append("names differ only by a number, which usually means a different track in a series; confirm by ear")
        score = min(score, 0.8)
    differences = _technical_differences(left, right)
    if differences:
        reasons.append("differences: " + "; ".join(differences))
    return round(min(score, 0.99), 3), reasons, differences


def keep_suggestion(left: AudioInfo, right: AudioInfo) -> "tuple[str, str]":
    """Suggest which file to keep when two files are duplicates. Never deletes anything."""
    left_marker, right_marker = has_copy_marker(left.file_name), has_copy_marker(right.file_name)
    if left_marker != right_marker:
        keep = "b" if left_marker else "a"
        return keep, "prefer the file without a trailing copy/index marker in its name"
    if left.size_bytes != right.size_bytes:
        keep = "a" if left.size_bytes > right.size_bytes else "b"
        return keep, "prefer the larger file (usually the higher-quality rip)"
    if (left.bitrate_kbps or 0) != (right.bitrate_kbps or 0):
        keep = "a" if (left.bitrate_kbps or 0) > (right.bitrate_kbps or 0) else "b"
        return keep, "prefer the higher bitrate"
    return "a", "files look equivalent; keep the first by path order"


def dedupe(
    infos: "list[AudioInfo]",
    threshold: float = 0.72,
    max_candidates: int = 200,
) -> dict:
    """Group exact duplicates and list near-duplicate candidates for manual review."""
    by_sha: "dict[str, list[AudioInfo]]" = {}
    by_audio: "dict[str, list[AudioInfo]]" = {}
    for info in infos:
        by_sha.setdefault(info.sha256, []).append(info)
        if info.audio_sha256:
            by_audio.setdefault(info.audio_sha256, []).append(info)

    exact_groups: "list[dict]" = []
    redundant_bytes = 0
    exact_paths: "set[str]" = set()
    for digest, group in sorted(by_sha.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda info: str(info.path).casefold())
        keep, reason = keep_suggestion(ordered[0], ordered[1])
        keep_path = ordered[0].path if keep == "a" else ordered[1].path
        redundant_bytes += ordered[0].size_bytes * (len(ordered) - 1)
        exact_paths.update(info.path for info in ordered)
        exact_groups.append(
            {
                "sha256": digest,
                "kind": "identical-bytes",
                "file_count": len(ordered),
                "size_bytes": ordered[0].size_bytes,
                "duration_ms": ordered[0].duration_ms,
                "codec": ordered[0].codec,
                "files": [info.path for info in ordered],
                "suggested_keep": keep_path,
                "suggested_keep_reason": reason,
            }
        )

    audio_only_groups: "list[dict]" = []
    for digest, group in by_audio.items():
        distinct_sha = {info.sha256 for info in group}
        if len(group) < 2 or len(distinct_sha) < 2:
            continue
        ordered = sorted(group, key=lambda info: str(info.path).casefold())
        audio_only_groups.append(
            {
                "audio_sha256": digest,
                "kind": "identical-audio-different-encoding",
                "file_count": len(ordered),
                "files": [info.path for info in ordered],
                "containers": sorted({info.container or "unknown" for info in ordered}),
                "note": "The decoded sample stream is identical; containers or headers differ.",
            }
        )

    candidates: "list[dict]" = []
    for index, left in enumerate(infos):
        for right in infos[index + 1 :]:
            if left.sha256 == right.sha256:
                continue
            score, reasons, differences = similarity(left, right)
            if score < threshold:
                continue
            keep, keep_reason = keep_suggestion(left, right)
            candidates.append(
                {
                    "a": left.path,
                    "b": right.path,
                    "score": score,
                    "reasons": reasons,
                    "differences": differences,
                    "same_duration": bool(left.duration_ms and left.duration_ms == right.duration_ms),
                    "same_audio_stream": bool(left.audio_sha256 and left.audio_sha256 == right.audio_sha256),
                    "suggested_keep": left.path if keep == "a" else right.path,
                    "suggested_keep_reason": keep_reason,
                }
            )
    candidates.sort(key=lambda item: (-item["score"], item["a"], item["b"]))
    truncated = len(candidates) > max_candidates
    candidates = candidates[:max_candidates]

    return {
        "schema_version": 1,
        "mode": "read-only-dedupe",
        "file_count": len(infos),
        "exact_groups": exact_groups,
        "identical_audio_groups": audio_only_groups,
        "candidates": candidates,
        "candidates_truncated": truncated,
        "summary": {
            "exact_group_count": len(exact_groups),
            "redundant_file_count": sum(group["file_count"] - 1 for group in exact_groups),
            "redundant_bytes": redundant_bytes,
            "identical_audio_group_count": len(audio_only_groups),
            "candidate_count": len(candidates),
            "duplicate_free_files": len([info for info in infos if info.path not in exact_paths]),
            "threshold": threshold,
        },
        "warnings": [
            "No file was deleted, moved, or modified; this plugin only reports.",
            "Exact groups share the same SHA-256; verify your own backup before removing anything.",
            "Similar candidates are hints from names and technical metadata; listen before acting.",
        ],
    }


def dedupe_folder(
    folder: str,
    recursive: bool = True,
    threshold: float = 0.72,
    max_candidates: int = 200,
    ffprobe: "str | bool | None" = None,
) -> dict:
    """Analyze a folder and immediately group duplicates inside it."""
    report = analyze_folder(folder, recursive=recursive, ffprobe=ffprobe)
    infos = [
        AudioInfo(
            path=item["path"],
            file_name=item["file_name"],
            size_bytes=item["size_bytes"],
            sha256=item["sha256"],
            container=item["container"],
            codec=item["codec"],
            duration_ms=item["duration_ms"],
            sample_rate=item["sample_rate"],
            channels=item["channels"],
            bits_per_sample=item["bits_per_sample"],
            bitrate_kbps=item["bitrate_kbps"],
            audio_sha256=item["audio_sha256"],
            probe=item["probe"],
            warnings=tuple(item["warnings"]),
        )
        for item in report["files"]
    ]
    result = dedupe(infos, threshold=threshold, max_candidates=max_candidates)
    result["folder"] = report["folder"]
    result["recursive"] = report["recursive"]
    result["analysis_summary"] = report["summary"]
    result["skipped"] = report["skipped"]
    return result
