import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_audio_dedupe.groove import detect_onsets, estimate_tempo, groove_to_pattern

RATE = 16000
KICK = "x...x...x...x...x...x...x...x..."
HAT = "..x...x...x...x...x...x...x...x."


def kick(ms: int = 120) -> "np.ndarray":
    count = int(RATE * ms / 1000)
    time = np.arange(count) / RATE
    return (np.sin(2 * np.pi * (48 + 80 * np.exp(-time * 30)) * time) * np.exp(-time * 16)).astype(np.float32)


def hat(ms: int = 40) -> "np.ndarray":
    count = int(RATE * ms / 1000)
    time = np.arange(count) / RATE
    noise = np.random.default_rng(5).standard_normal(count)
    return (noise * np.exp(-time * 90) * 0.5).astype(np.float32)


def write_groove(path: Path, bpm: float, pattern: dict, bars: int = 2) -> Path:
    step_ms = 60_000.0 / bpm / 4
    total = np.zeros(int(RATE * step_ms * 16 * bars / 1000), dtype=np.float32)
    for row, sound in (("kick", kick), ("hat", hat)):
        for index, symbol in enumerate((pattern.get(row) or "." * 16) * bars):
            if symbol != "x":
                continue
            start = int(RATE * index * step_ms / 1000)
            hit = sound()
            end = min(total.size, start + hit.size)
            total[start:end] += hit[: end - start]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(total, -1, 1) * 32767).astype("<i2").tobytes())
    return path


class GrooveTests(unittest.TestCase):
    def test_round_trip_of_a_house_groove(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_groove(Path(tmp) / "groove.wav", 124, {"kick": "x...x...x...x...", "hat": "..x...x...x...x."})
            result = groove_to_pattern(source, bpm=124, bars=2)
            self.assertEqual(result["onset_count"], 16)
            self.assertEqual(result["pattern"]["kick"], KICK)
            self.assertEqual(result["pattern"]["hat"], HAT)
            self.assertEqual(result["pattern"]["snare"], "." * 32)
            self.assertEqual(result["bpm"], 124)

    def test_tempo_is_estimated_when_not_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_groove(Path(tmp) / "groove.wav", 128, {"kick": "x...x...x...x..."})
            result = groove_to_pattern(source, bars=2)
            self.assertAlmostEqual(result["bpm"], 128, delta=4)

    def test_silence_reports_no_onsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "quiet.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(RATE)
                handle.writeframes(np.zeros(RATE, dtype="<i2").tobytes())
            result = groove_to_pattern(source, bars=2)
            self.assertEqual(result["onset_count"], 0)
            self.assertTrue(result["warnings"])

    def test_estimate_tempo_folds_into_a_musical_range(self):
        slow = estimate_tempo([0, 1000, 2000, 3000])   # 60 BPM → doubled
        fast = estimate_tempo([0, 200, 400, 600])      # 300 BPM → halved
        self.assertGreaterEqual(slow, 70)
        self.assertLessEqual(fast, 190)

    def test_detects_onsets_at_the_right_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_groove(Path(tmp) / "four.wav", 120, {"kick": "x...x...x...x..."}, bars=1)
            from shadow_audio_dedupe.melody import read_wav_mono

            samples, rate = read_wav_mono(source)
            onsets = detect_onsets(samples, rate)
            # 120 BPM → a kick every 500 ms
            self.assertEqual(len(onsets), 4)
            for expected, actual in zip([0, 500, 1000, 1500], onsets):
                self.assertLess(abs(expected - actual), 30)


if __name__ == "__main__":
    unittest.main()
