"""Tempo and key of a finished piece of music.

Both answers come from the same two ideas: onsets say *when* something happened (tempo), and the
distribution of energy over pitch classes says *what* it was (key). Neither is a transcription —
this is what a DJ needs on the screen before deciding whether two tracks mix.

Reads WAV directly; anything else goes through the machine's own decoder (ffmpeg, or CoreAudio's
`afconvert` on macOS), because a library is full of mp3 and m4a, not WAV.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .groove import FRAME, HOP, detect_onsets, estimate_tempo
from .melody import read_wav_mono

__all__ = ["decode_audio", "detect_tempo", "detect_key", "estimate_key", "camelot_for"]

#: Pitch-class names, C first, as the key detector names them.
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: Krumhansl–Kessler tonal hierarchies: how strongly each scale degree belongs to the key.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

#: Camelot wheel: DJs read keys as `8A`, not as `A minor`.
CAMELOT_MAJOR = ("B", "F#", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E")
CAMELOT_MINOR = ("Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "Db")


def _decoder() -> "tuple[str, ...] | None":
    """ffmpeg if it is around, else macOS's own converter; `None` when neither is."""
    if (ffmpeg := shutil.which("ffmpeg")) is not None:
        return (ffmpeg, "-v", "error", "-y", "-i")
    if (afconvert := shutil.which("afconvert")) is not None:
        return (afconvert, "-f", "WAVE", "-d", "LEI16", "-o")
    return None


def _read_riff_mono(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Minimal RIFF reader: PCM 8/16/24/32-bit and float, extensible headers included.

    Python's `wave` refuses `WAVE_FORMAT_EXTENSIBLE`, which is exactly what ffmpeg and macOS's
    afconvert write — so the decoder's own output has to be readable here.
    """
    raw = Path(path).expanduser().read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    offset = 12
    fmt: "tuple[int, int, int, int] | None" = None
    payload = b""
    while offset + 8 <= len(raw):
        chunk = raw[offset : offset + 4]
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        body = raw[offset + 8 : offset + 8 + size]
        if chunk == b"fmt ":
            tag = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if tag == 0xFFFE and len(body) >= 26:
                tag = int.from_bytes(body[24:26], "little")
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            payload = body
        offset += 8 + size + (size % 2)
    if fmt is None or payload == b"":
        raise ValueError(f"no audio data in {path}")
    tag, channels, rate, bits = fmt
    if tag == 3:
        data = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 8:
        data = (np.frombuffer(payload, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        data = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        packed = np.frombuffer(payload, dtype=np.uint8)
        packed = packed[: (packed.size // 3) * 3].reshape(-1, 3)
        values = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        data = np.where(values >= 1 << 23, values - (1 << 24), values).astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 32:
        data = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV format tag {tag} ({bits}-bit) in {path}")
    if channels > 1:
        data = data[: (data.size // channels) * channels].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), rate


def decode_audio(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Mono float samples plus their rate, whatever container the file is in."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(str(target))
    if target.suffix.lower() in (".wav", ".wave"):
        try:
            return read_wav_mono(target)
        except Exception:
            pass  # an extensible header: let the decoder below deal with it
    decoder = _decoder()
    if decoder is None:
        raise ValueError("reading this format needs ffmpeg (or macOS afconvert); WAV works as it is")
    with tempfile.TemporaryDirectory(prefix="shadow-listen-") as tmp:
        decoded = Path(tmp) / "decoded.wav"
        argv = [*decoder, str(decoded), str(target)] if decoder[0].endswith("afconvert") else [*decoder, str(target), str(decoded)]
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0 or not decoded.exists():
            detail = (result.stderr or result.stdout or "decoder failed").strip().splitlines()
            raise ValueError(f"could not decode {target.name}: {detail[-1] if detail else 'unknown error'}")
        try:
            return read_wav_mono(decoded)
        except Exception:
            return _read_riff_mono(decoded)


def _fit(onsets: "list[int]", bpm: float, steps: int = 16) -> "tuple[float, float]":
    """How well the onsets fit `bpm`'s 16th grid, and with which phase: `(score, phase)`."""
    if len(onsets) < 4 or bpm <= 0:
        return 0.0, 0.0
    step_ms = 60_000.0 / bpm / (steps / 4)
    tolerance = step_ms * 0.28
    grid = np.asarray(onsets, dtype=np.float64) / step_ms
    best, best_phase = 0.0, 0.0
    for phase in np.linspace(0.0, 1.0, 21)[:-1]:
        within = np.abs(((grid - phase + 0.5) % 1.0) - 0.5) * step_ms
        score = float(np.mean(within <= tolerance))
        if score > best:
            best, best_phase = score, float(phase)
    return best, best_phase


def _refine_tempo(onsets: "list[int]", estimate: float, steps: int = 16) -> "tuple[float, float]":
    """Search around the interval estimate for the tempo the onsets fit best.

    `estimate_tempo` takes the median inter-onset interval; a 0.5 % error there drifts past the
    tolerance by the end of a long file (measured: a 174 BPM click scored 0.41 confidence at 174.93
    and 1.0 after refining to 174.02).
    """
    if len(onsets) < 4 or estimate <= 0:
        return estimate, 0.0
    best_bpm, best_score = float(estimate), _fit(onsets, estimate, steps)[0]
    for candidate in np.arange(estimate * 0.97, estimate * 1.03, 0.05):
        score, _ = _fit(onsets, float(candidate), steps)
        # a tie goes to the tempo closest to the estimate, so the search cannot wander
        if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and abs(candidate - estimate) < abs(best_bpm - estimate)):
            best_bpm, best_score = float(candidate), score
    return round(best_bpm, 2), best_score


def _grid_confidence(onsets: "list[int]", bpm: float, steps: int = 16) -> float:
    """How well the onsets sit on the 16th grid of `bpm`, 0–1.

    A tempo estimate is only worth acting on when the hits agree with it: a rock-solid click track
    matches nearly every onset, a pad or a spoken word lands wherever it lands.
    """
    if len(onsets) < 4 or bpm <= 0:
        return 0.0
    step_ms = 60_000.0 / bpm / (steps / 4)
    tolerance = step_ms * 0.28
    grid = np.asarray(onsets, dtype=np.float64) / step_ms
    # distance to the *nearest 16th*, not to the downbeat: a hit is on the grid whether or not it is
    # the first step of the bar
    within = np.minimum(grid % 1.0, 1.0 - (grid % 1.0))
    return round(float(np.mean(within * step_ms <= tolerance)), 3)


def detect_tempo(path: "str | Path", *, steps: int = 16) -> dict:
    """Estimate the tempo of one file, with a confidence that says whether to trust it."""
    source = Path(path).expanduser()
    samples, rate = decode_audio(source)
    onsets = detect_onsets(samples, rate)
    coarse = estimate_tempo(onsets)
    bpm, confidence = _refine_tempo(onsets, coarse, steps=steps)
    return {
        "schema_version": 1,
        "mode": "detect-tempo",
        "source": str(source),
        "bpm": bpm,
        "coarse_bpm": coarse,
        "confidence": round(float(confidence), 3),
        "onsets": len(onsets),
        "duration_ms": int(round(samples.size / rate * 1000)),
        "sample_rate": rate,
    }


def _chroma(samples: np.ndarray, rate: int) -> np.ndarray:
    """Energy per pitch class (C…B), from the spectral peaks of every frame.

    Measured on this machine while tuning (2026-09-15): plain squared magnitude named *F major* for an
    Am–F–C–G progression and for an A-minor render, because the summed spectrum is dominated by the
    harmonics that happen to be loud. Taking local spectral peaks, compressing them with `log1p` and
    boosting the bass (where the root is) named the relative pair C / A minor for both — the honest
    answer for material that only uses F, G, A, C and E. Every step here is measured, not taste: see
    `tests/test_listen_unittest.py`.
    """
    count = max(0, (samples.size - FRAME) // HOP + 1)
    if count == 0:
        return np.zeros(12, dtype=np.float64)
    window = np.hanning(FRAME).astype(np.float32)
    frequencies = np.fft.rfftfreq(FRAME, 1.0 / rate)
    audible = (frequencies >= 55.0) & (frequencies <= 2000.0)
    indices = np.nonzero(audible)[0]
    if indices.size == 0:
        return np.zeros(12, dtype=np.float64)
    # 440 Hz is A (pitch class 9), so the octave number has to be offset by 9 before folding —
    # without it every chroma bin lands three semitones sharp (measured: A minor read as Cm).
    pitch_class = (np.round(12 * np.log2(frequencies[indices] / 440.0)) + 9).astype(int) % 12
    weight = (frequencies[indices] / 110.0) ** -0.5
    chroma = np.zeros(12, dtype=np.float64)
    for index in range(count):
        frame = samples[index * HOP : index * HOP + FRAME] * window
        magnitude = np.abs(np.fft.rfft(frame))[indices]
        peaks = np.zeros_like(magnitude, dtype=bool)
        peaks[1:-1] = (magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])
        np.add.at(chroma, pitch_class, np.log1p(magnitude * 50.0) * peaks * weight)
    total = float(np.sum(chroma))
    return chroma / total if total > 0 else chroma


def estimate_key(chroma: np.ndarray) -> "tuple[str, str, float, list[dict]]":
    """Correlate a chroma vector with the 24 keys: `(key, camelot, confidence, alternates)`.

    A four-chord loop is genuinely ambiguous — C major and A minor share every note — so the answer
    comes back with its runners-up and their correlations instead of pretending to be certain.
    """
    if float(np.sum(chroma)) <= 0:
        return "C", "8B", 0.0, []
    scores: "list[tuple[float, str, str]]" = []
    for root in range(12):
        major = float(np.corrcoef(np.roll(MAJOR_PROFILE, root), chroma)[0, 1])
        minor = float(np.corrcoef(np.roll(MINOR_PROFILE, root), chroma)[0, 1])
        scores.append((major, NOTE_NAMES[root], "B"))
        scores.append((minor, f"{NOTE_NAMES[root]}m", "A"))
    scores.sort(reverse=True)
    best = scores[0][0] if np.isfinite(scores[0][0]) else 0.0
    alternates = [
        {"key": name, "camelot": camelot_for(name, mode), "correlation": round(float(score), 3)}
        for score, name, mode in scores[1:4]
    ]
    confidence = round(float(max(0.0, min(1.0, best))), 3)
    return scores[0][1], camelot_for(scores[0][1], scores[0][2]), confidence, alternates


#: Enharmonic spellings: the detector names sharps, the Camelot wheel happens to print flats.
ENHARMONIC = {"C#": "Db", "D#": "Eb", "G#": "Ab", "A#": "Bb", "Db": "C#", "Eb": "D#", "Ab": "G#", "Bb": "A#"}


def camelot_for(name: str, mode: str) -> str:
    """Camelot code for a key name: `A minor` → `8A`, `C major` → `8B`, `C# major` → `3B`."""
    root = name[:-1] if name.endswith("m") else name
    table = CAMELOT_MINOR if mode == "A" else CAMELOT_MAJOR
    for spelling in (root, ENHARMONIC.get(root, root)):
        if spelling in table:
            return f"{table.index(spelling) + 1}{mode}"
    return ""


def detect_key(path: "str | Path") -> dict:
    """Estimate the key of one file (Krumhansl–Schmuckler on its chroma)."""
    source = Path(path).expanduser()
    samples, rate = decode_audio(source)
    chroma = _chroma(samples, rate)
    name, camelot, confidence, alternates = estimate_key(chroma)
    return {
        "schema_version": 1,
        "mode": "detect-key",
        "source": str(source),
        "key": name,
        "camelot": camelot,
        "confidence": confidence,
        "alternates": alternates,
        "chroma": [round(float(value), 4) for value in chroma],
    }
