"""Tempo and key detection, measured on synthetic material with a known answer.

The synthetic cases are deliberately simple — a click track has one obvious tempo, a triad one
obvious key — so that a failure here means the analysis, not the music.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_audio_dedupe.listen import camelot_for, decode_audio, detect_key, detect_tempo, estimate_key

RATE = 44100


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return path


def click_track(bpm: float, seconds: float = 12.0) -> np.ndarray:
    time = np.arange(int(RATE * seconds)) / RATE
    signal = np.zeros_like(time)
    beat = 60.0 / bpm
    for index in range(int(seconds / beat)):
        start = int(index * beat * RATE)
        envelope = np.exp(-np.linspace(0, 14, int(0.09 * RATE)))
        signal[start : start + envelope.size] += envelope * np.sin(
            2 * np.pi * 90 * np.linspace(0, 0.09, envelope.size)
        )
    return (signal / max(1e-9, float(np.abs(signal).max())) * 0.8).astype(np.float32)


def chord(root: int, minor: bool = False, seconds: float = 1.0) -> np.ndarray:
    semitones = [root, root + (3 if minor else 4), root + 7]
    time = np.arange(int(RATE * seconds)) / RATE
    out = np.zeros_like(time)
    for semitone in semitones:
        base = 440.0 * 2 ** ((semitone - 69) / 12)
        for harmonic in range(1, 6):
            out += np.sin(2 * np.pi * base * harmonic * time) * (0.6**harmonic)
    return (out / float(np.abs(out).max())).astype(np.float32)


class TempoTests(unittest.TestCase):
    def test_a_click_track_keeps_its_tempo(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bpm in (120.0, 128.0, 174.0):
                path = write_wav(Path(tmp) / f"click-{int(bpm)}.wav", click_track(bpm))
                result = detect_tempo(path)
                self.assertLess(abs(result["bpm"] - bpm), 1.0, f"{bpm} → {result}")
                self.assertGreater(result["confidence"], 0.8, f"{bpm} → {result}")
                self.assertGreaterEqual(result["onsets"], 20)

    def test_a_pad_without_hits_is_not_confident(self):
        time = np.arange(RATE * 4) / RATE
        pad = (np.sin(2 * np.pi * 110 * time) * 0.4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            result = detect_tempo(write_wav(Path(tmp) / "pad.wav", pad))
            # onsets are rare in a sustained tone, so whatever tempo comes back must not be trusted
            self.assertLess(result["confidence"], 0.8, result)


class KeyTests(unittest.TestCase):
    def test_tonic_triads_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            minor = detect_key(write_wav(Path(tmp) / "a-minor.wav", chord(57, True) * 0.8))
            self.assertEqual((minor["key"], minor["camelot"]), ("Am", "8A"))
            major = detect_key(write_wav(Path(tmp) / "c-major.wav", chord(60) * 0.8))
            self.assertEqual((major["key"], major["camelot"]), ("C", "8B"))

    def test_a_loop_reports_the_relative_pair_as_its_runner_up(self):
        """Am–F–C–G uses only F, G, A, C and E: C major and A minor are both true, so both are said."""
        progression = np.concatenate([chord(57, True), chord(53), chord(60), chord(55)]) * 0.8
        with tempfile.TemporaryDirectory() as tmp:
            result = detect_key(write_wav(Path(tmp) / "loop.wav", progression))
            self.assertIn(result["camelot"], {"8A", "8B"}, result)
            candidates = {entry["camelot"] for entry in result["alternates"]}
            self.assertTrue({"8A", "8B"} <= candidates | {result["camelot"]}, result)
            self.assertGreater(result["confidence"], 0.8, result)

    def test_camelot_codes_follow_the_wheel(self):
        self.assertEqual(camelot_for("Am", "A"), "8A")
        self.assertEqual(camelot_for("C", "B"), "8B")
        self.assertEqual(camelot_for("Em", "A"), "9A")
        self.assertEqual(camelot_for("F#m", "A"), "11A")
        self.assertEqual(camelot_for("Db", "B"), "3B")


class DecodeTests(unittest.TestCase):
    def test_an_extensible_wav_header_is_readable(self):
        """macOS's own converter writes `WAVE_FORMAT_EXTENSIBLE`; the stdlib reader refuses it."""
        time = np.arange(RATE) / RATE
        tone = (np.sin(2 * np.pi * 220 * time) * 0.5).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_wav(Path(tmp) / "plain.wav", tone)
            if shutil.which("afconvert") is None:
                self.skipTest("afconvert is not available")
            extended = Path(tmp) / "extended.wav"
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@44100", str(path), str(extended)], check=True)
            samples, rate = decode_audio(extended)
            self.assertEqual(rate, RATE)
            self.assertGreater(float(np.abs(samples).max()), 0.3)

    @unittest.skipUnless(shutil.which("ffmpeg") or shutil.which("afconvert"), "needs a decoder")
    def test_a_compressed_file_is_decoded_before_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_wav(Path(tmp) / "click.wav", click_track(124.0))
            compressed = Path(tmp) / "click.m4a"
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                command = [ffmpeg, "-v", "error", "-y", "-i", str(source), "-c:a", "aac", "-b:a", "192000", str(compressed)]
            else:
                command = ["afconvert", "-f", "m4af", "-d", "aac", "-b", "192000", str(source), str(compressed)]
            subprocess.run(command, check=True, capture_output=True)
            result = detect_tempo(compressed)
            self.assertLess(abs(result["bpm"] - 124.0), 1.5, result)
            self.assertEqual(os.path.basename(result["source"]), "click.m4a")


if __name__ == "__main__":
    unittest.main()
