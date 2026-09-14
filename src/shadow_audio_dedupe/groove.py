"""Hummed or tapped groove → drum pattern.

Onset detection on a mono WAV, onsets quantised onto a 16-step bar grid at a given (or estimated)
tempo, each hit classified by spectral centroid into kick / snare / hat. The result is the same
`pattern` shape the synth renders (`shadow-music-generator`'s `render_song`), so a tapped rhythm can
go straight back to audio.

Monophonic and loud-on-transient by design: this is for a hummed or tapped groove, not for pulling
drums out of a finished mix (that is Spotify's basic-pitch / a separation model).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .melody import read_wav_mono

__all__ = ["detect_onsets", "estimate_tempo", "groove_to_pattern", "quantise_to_pattern"]

FRAME = 1024
HOP = 256


def _frames(samples: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    count = max(0, (samples.size - FRAME) // HOP + 1)
    if count == 0:
        return np.zeros((0, FRAME), dtype=np.float32), np.zeros(0, dtype=np.float32)
    window = np.hanning(FRAME).astype(np.float32)
    frames = np.stack([samples[index * HOP : index * HOP + FRAME] for index in range(count)])
    return frames * window, np.arange(count) * HOP


def detect_onsets(samples: np.ndarray, rate: int, *, threshold: float = 0.3, min_gap_ms: int = 60) -> list[int]:
    """Onset positions in milliseconds, from positive energy flux peaks.

    The gate is deliberately low (`mean + 0.3σ`, floor `0.12 × max flux`): measured on a synthetic
    groove, a `mean + 1.6σ` gate found 12 of 16 hits because quiet hats sit close to the noise floor.
    Peaks must still be local maxima, and hits closer than `min_gap_ms` are merged.
    """
    frames, positions = _frames(samples)
    if frames.shape[0] < 3:
        return []
    energy = np.sqrt(np.mean(frames**2, axis=1))
    flux = np.diff(energy, prepend=energy[:1])
    flux = np.maximum(flux, 0.0)
    if flux.max() <= 0:
        return []
    mean = float(flux.mean())
    spread = float(flux.std())
    gate = max(mean + threshold * spread, 0.12 * float(flux.max()))
    gap_frames = max(1, int(min_gap_ms / 1000 * rate / HOP))
    onsets: "list[int]" = []
    last = -gap_frames
    last_peak = 0.0
    lowest_since = float("inf")
    for index in range(1, flux.size - 1):
        # A hit only counts after the energy has fallen away from the previous one: a kick's tail (or
        # its pitch sweep) can make the flux rise twice, which used to register as a second onset.
        if last_peak > 0 and lowest_since > 0.5 * last_peak:
            lowest_since = min(lowest_since, float(energy[index]))
            continue
        if flux[index] < gate:
            lowest_since = min(lowest_since, float(energy[index]))
            continue
        if flux[index] < flux[index - 1] or flux[index] < flux[index + 1]:
            lowest_since = min(lowest_since, float(energy[index]))
            continue
        if index - last < gap_frames:
            lowest_since = min(lowest_since, float(energy[index]))
            continue
        # Report the transient, not the frame's start: a hit lights up the frame that contains it, so
        # the frame centre is the unbiased estimate of when it happened.
        onsets.append(int(round((positions[index] + FRAME / 2) / rate * 1000)))
        last = index
        last_peak = float(energy[index])
        lowest_since = float(energy[index])
    # A hit on the very first sample has no flux to rise from: treat a loud opening as an onset at 0.
    opening = float(energy[:3].max()) if energy.size else 0.0
    floor = float(np.median(energy)) if energy.size else 0.0
    if opening > 2 * floor and (not onsets or onsets[0] > min_gap_ms):
        onsets.insert(0, 0)
    return onsets


def estimate_tempo(onsets: "list[int]", *, fallback: float = 120.0) -> float:
    """Tempo from the median inter-onset interval, folded into a musical range."""
    if len(onsets) < 3:
        return fallback
    intervals = np.diff(sorted(onsets))
    intervals = intervals[intervals > 60]
    if intervals.size == 0:
        return fallback
    bpm = 60_000.0 / float(np.median(intervals))
    while bpm < 70:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return float(round(bpm, 2))


def _centroid(frame: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(frame))
    if spectrum.sum() <= 0:
        return 0.0
    frequencies = np.fft.rfftfreq(frame.size, 1 / rate)
    return float((spectrum * frequencies).sum() / spectrum.sum())


def quantise_to_pattern(
    onsets: "list[int]",
    centroids: "list[float]",
    bpm: float,
    *,
    bars: int,
    steps: int = 16,
) -> dict:
    """Place every onset on the 16-step grid and split it into kick / snare / hat rows."""
    step_ms = 60_000.0 / max(1.0, bpm) / (steps / 4)
    rows = {name: ["."] * (steps * bars) for name in ("kick", "snare", "hat")}
    hits: "list[dict]" = []
    for onset, centroid in zip(onsets, centroids):
        index = int(round(onset / step_ms))
        if index >= steps * bars:
            continue
        if centroid < 500:
            voice = "kick"
        elif centroid > 2500:
            voice = "hat"
        else:
            voice = "snare"
        rows[voice][index] = "x"
        hits.append({"ms": onset, "step": index, "voice": voice, "centroid": int(centroid)})
    return {
        "pattern": {name: "".join(cells) for name, cells in rows.items()},
        "hits": hits,
        "step_ms": round(step_ms, 2),
    }


def groove_to_pattern(path: "str | Path", *, bpm: float | None = None, bars: int = 2, steps: int = 16) -> dict:
    """Read a hummed / tapped WAV and return the drum pattern it contains."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    samples, rate = read_wav_mono(source)
    onsets = detect_onsets(samples, rate)
    tempo = float(bpm) if bpm else estimate_tempo(onsets)
    frames, positions = _frames(samples)
    centroids: "list[float]" = []
    for onset in onsets:
        index = int(round(onset / 1000 * rate / HOP))
        if 0 <= index < frames.shape[0]:
            centroids.append(_centroid(frames[index], rate))
        else:
            centroids.append(0.0)
    quantised = quantise_to_pattern(onsets, centroids, tempo, bars=bars, steps=steps)
    return {
        "schema_version": 1,
        "mode": "groove-to-pattern",
        "source": str(source),
        "bpm": tempo,
        "bars": bars,
        "steps": steps,
        "onset_count": len(onsets),
        "onsets_ms": onsets,
        "pattern": quantised["pattern"],
        "hits": quantised["hits"],
        "warnings": [] if onsets else ["no onsets found — tap or hum louder, closer to the mic"],
    }
