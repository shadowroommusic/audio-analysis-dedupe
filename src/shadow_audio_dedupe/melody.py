"""Hummed melody → MIDI.

Monophonic pitch tracking with numpy only (autocorrelation + median smoothing) and a hand-rolled
Type-0 MIDI writer, so the tool works on a machine with no audio libraries installed. For
polyphonic or percussive material swap `detect_notes` for Spotify's basic-pitch (Apache-2.0) — the
rest of the pipeline (notes → MIDI → clip) stays the same.
"""
from __future__ import annotations

import json
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Note", "detect_notes", "hum_to_midi", "read_wav_mono", "write_midi_type0"]

FRAME = 2048
HOP = 256
FMIN = 55.0  # A1
FMAX = 1200.0  # ~D6, enough for humming


@dataclass(frozen=True)
class Note:
    """One detected note."""

    start_ms: int
    end_ms: int
    midi: int
    confidence: float

    @property
    def name(self) -> str:
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        return f"{names[self.midi % 12]}{self.midi // 12 - 1}"

    def to_json(self) -> dict:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "midi": self.midi,
            "name": self.name,
            "confidence": round(self.confidence, 3),
        }


def read_wav_mono(path: "str | Path") -> "tuple[np.ndarray, int]":
    """Read a PCM WAV as mono float32 in [-1, 1] (the browser records exactly this shape)."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width * 8} bit")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _frame_pitch(frame: "np.ndarray", rate: int, fmin: float, fmax: float) -> "tuple[float, float]":
    """Autocorrelation pitch for one frame.

    Returns ``(hz, confidence)``; ``hz`` is 0 when the frame is too quiet to judge.
    """
    windowed = (frame - frame.mean()) * np.hanning(frame.size)
    rms = float(np.sqrt(np.mean(windowed**2)))
    if rms < 1e-4:
        return 0.0, 0.0
    size = frame.size
    spectrum = np.fft.rfft(windowed, 2 * size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    if acf[0] <= 0:
        return 0.0, 0.0
    acf = acf / acf[0]
    low = max(1, int(rate / fmax))
    high = min(size - 1, int(rate / fmin))
    if high <= low:
        return 0.0, 0.0
    window = acf[low:high]
    if window.size < 3:
        return 0.0, 0.0
    # Only a local maximum counts as a period. White noise decays monotonically, and its highest
    # value sits at the first lag of the window — reading that as a pitch was a real bug (measured on
    # a noisy recording, reported as 1.2 kHz).
    interior = np.where((window[1:-1] > window[:-2]) & (window[1:-1] >= window[2:]))[0] + 1
    if interior.size == 0:
        return 0.0, 0.0
    best = int(interior[int(np.argmax(window[interior]))])
    confidence = float(window[best])
    if confidence < 0.3:
        return 0.0, 0.0
    # Prefer the lowest strong peak: it is the fundamental, not an octave-down multiple.
    strong = interior[window[interior] >= 0.85 * confidence]
    index = int(strong[0]) + low
    confidence = float(acf[index])
    if 0 < index < size - 1:
        a, b, c = float(acf[index - 1]), float(acf[index]), float(acf[index + 1])
        denominator = a - 2 * b + c
        if denominator != 0:
            index += max(-0.5, min(0.5, 0.5 * (a - c) / denominator))
    return rate / index, confidence


def detect_notes(
    samples: "np.ndarray",
    rate: int,
    *,
    min_note_ms: int = 120,
    min_confidence: float = 0.5,
    pitch_tolerance: float = 0.7,
    gap_ms: int = 60,
) -> "list[Note]":
    """Frame-wise pitch tracking, then note segmentation.

    Args:
        samples: mono float samples in [-1, 1].
        rate: sample rate.
        min_note_ms: notes shorter than this are dropped (breath / noise).
        min_confidence: autocorrelation peak below this counts as unvoiced.
        pitch_tolerance: semitones a note may drift before it is split.
        gap_ms: unvoiced gaps shorter than this are bridged.
    """
    hop_ms = HOP / rate * 1000.0
    count = max(0, (samples.size - FRAME) // HOP + 1)
    if count == 0:
        return []
    pitches = np.zeros(count, dtype=np.float64)
    voiced = np.zeros(count, dtype=bool)
    for index in range(count):
        frame = samples[index * HOP : index * HOP + FRAME]
        hz, confidence = _frame_pitch(frame, rate, FMIN, FMAX)
        if confidence >= min_confidence and FMIN <= hz <= FMAX:
            pitches[index] = hz
            voiced[index] = True
    if not voiced.any():
        return []

    # Smooth the pitch track so vibrato does not split notes.
    smoothed = pitches.copy()
    for index in range(count):
        if not voiced[index]:
            continue
        window = pitches[max(0, index - 2) : index + 3]
        window = window[window > 0]
        if window.size:
            smoothed[index] = float(np.median(window))

    midi = np.zeros(count, dtype=np.float64)
    with np.errstate(divide="ignore"):
        midi[voiced] = 69 + 12 * np.log2(smoothed[voiced] / 440.0)

    groups: "list[tuple[int, int]]" = []
    start = None
    previous = 0.0
    for index in range(count):
        if not voiced[index]:
            if start is not None:
                groups.append((start, index - 1))
                start = None
            continue
        if start is None:
            start = index
            previous = midi[index]
            continue
        if abs(midi[index] - previous) > pitch_tolerance:
            groups.append((start, index - 1))
            start = index
        previous = midi[index]
    if start is not None:
        groups.append((start, count - 1))

    notes: "list[Note]" = []
    for first, last in groups:
        duration_ms = (last - first + 1) * hop_ms
        if duration_ms < min_note_ms:
            continue
        values = midi[first : last + 1]
        values = values[values > 0]
        if values.size == 0:
            continue
        pitch = int(round(float(np.median(values))))
        notes.append(
            Note(
                start_ms=int(round(first * hop_ms)),
                end_ms=int(round((last + 1) * hop_ms)),
                midi=max(0, min(127, pitch)),
                confidence=1.0,
            )
        )

    # Bridge tiny gaps and merge repeats of the same pitch.
    merged: "list[Note]" = []
    for note in notes:
        if merged and note.midi == merged[-1].midi and note.start_ms - merged[-1].end_ms <= gap_ms:
            previous_note = merged[-1]
            merged[-1] = Note(previous_note.start_ms, note.end_ms, note.midi, previous_note.confidence)
        else:
            merged.append(note)
    return merged


def write_midi_type0(notes: "list[Note]", path: "str | Path", *, tempo_bpm: float = 120.0) -> Path:
    """Write the notes as a Type-0 MIDI file (one track), using only the standard library."""
    ticks_per_beat = 480
    target = Path(path)
    events: "list[tuple[int, int, bytes]]" = []

    def to_ticks(ms: float) -> int:
        return int(round(ms / 60000.0 * tempo_bpm * ticks_per_beat))

    micros_per_beat = int(round(60_000_000 / tempo_bpm))
    events.append((0, 0, b"\xff\x51\x03" + struct.pack(">I", micros_per_beat)[1:]))
    for order, note in enumerate(notes):
        channel = 0
        events.append((to_ticks(note.start_ms), 1, bytes([0x90 | channel, note.midi, 100])))
        events.append((to_ticks(note.end_ms), 0, bytes([0x80 | channel, note.midi, 64])))
    events.sort(key=lambda item: (item[0], item[1]))

    track = bytearray()
    previous_tick = 0
    for tick, _order, payload in events:
        delta = tick - previous_tick
        previous_tick = tick
        track += _varint(delta)
        track += payload
    track += _varint(0) + b"\xff\x2f\x00"  # end of track

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    target.write_bytes(header + chunk)
    return target


def _varint(value: int) -> bytes:
    """MIDI variable-length quantity."""
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def hum_to_midi(
    path: "str | Path",
    out_path: "str | Path | None" = None,
    *,
    min_note_ms: int = 120,
    tempo_bpm: float = 120.0,
    min_confidence: float = 0.5,
) -> dict:
    """Detect the melody in a WAV recording and write it as MIDI next to it."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    samples, rate = read_wav_mono(source)
    notes = detect_notes(samples, rate, min_note_ms=min_note_ms, min_confidence=min_confidence)
    target = Path(out_path).expanduser() if out_path else source.with_suffix(".mid")
    write_midi_type0(notes, target, tempo_bpm=tempo_bpm)
    return {
        "schema_version": 1,
        "mode": "hum-to-midi",
        "source": str(source),
        "midi_path": str(target),
        "sample_rate": rate,
        "duration_ms": int(round(samples.size / rate * 1000)) if rate else 0,
        "note_count": len(notes),
        "notes": [note.to_json() for note in notes],
        "tempo_bpm": tempo_bpm,
        "warnings": [] if notes else ["no pitched notes found — hum louder or longer"],
    }


def _main() -> int:  # pragma: no cover - convenience for manual runs
    import argparse

    parser = argparse.ArgumentParser(prog="hum-to-midi", description="Detect a hummed melody and write MIDI.")
    parser.add_argument("wav")
    parser.add_argument("--out")
    parser.add_argument("--min-note-ms", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(hum_to_midi(args.wav, args.out, min_note_ms=args.min_note_ms), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
