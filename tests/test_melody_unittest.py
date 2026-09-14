import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_audio_dedupe.melody import detect_notes, hum_to_midi, read_wav_mono, write_midi_type0

RATE = 16000


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def tone(midi: int, ms: int, amplitude: float = 0.35) -> "np.ndarray":
    """A sung-ish note: sine with a soft attack and a light vibrato.

    The vibrato is phase modulation (`sin(2πft + m·sin(2πrt))`, m = 0.15) — modulating the frequency
    term itself would sweep the pitch far more than a singer does.
    """
    count = int(RATE * ms / 1000)
    time = np.arange(count) / RATE
    phase = 2 * math.pi * midi_to_hz(midi) * time + 0.15 * np.sin(2 * math.pi * 5.5 * time)
    wave_ = amplitude * np.sin(phase)
    return (wave_ * (1 - np.exp(-time * 40))).astype(np.float32)


def write_wav(path: Path, samples: "np.ndarray") -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


class MelodyTests(unittest.TestCase):
    def test_detects_a_clean_four_note_melody(self):
        melody = [60, 62, 64, 67]
        samples = np.concatenate([tone(midi, 400) for midi in melody] + [np.zeros(RATE // 3, dtype=np.float32)])
        notes = detect_notes(samples, RATE, min_note_ms=150, min_confidence=0.4)
        self.assertEqual([note.midi for note in notes], melody)
        # the notes should keep their order and roughly their length
        for note in notes:
            self.assertGreater(note.end_ms - note.start_ms, 250)

    def test_ignores_silence_and_noise(self):
        noise = (np.random.default_rng(7).standard_normal(RATE) * 0.02).astype(np.float32)
        self.assertEqual(detect_notes(noise, RATE, min_note_ms=120, min_confidence=0.4), [])
        self.assertEqual(detect_notes(np.zeros(RATE, dtype=np.float32), RATE), [])

    def test_noise_does_not_report_an_octave_high_pitch(self):
        """A decaying autocorrelation used to be read as its first lag (a bogus ~1.2 kHz note)."""
        rng = np.random.default_rng(11)
        for scale in (0.005, 0.02, 0.08):
            noise = (rng.standard_normal(RATE) * scale).astype(np.float32)
            for note in detect_notes(noise, RATE, min_note_ms=120, min_confidence=0.4):
                self.assertLess(note.midi, 84)  # never invent a very high note from noise

    def test_quiet_recording_with_a_burst_still_finds_the_note(self):
        rng = np.random.default_rng(3)
        silence = rng.standard_normal(RATE // 2).astype(np.float32) * 0.002
        burst = tone(64, 500, amplitude=0.2)
        tail = rng.standard_normal(RATE // 3).astype(np.float32) * 0.002
        notes = detect_notes(np.concatenate([silence, burst, tail]), RATE, min_note_ms=150, min_confidence=0.4)
        self.assertEqual([note.midi for note in notes], [64])

    def test_octave_does_not_change_the_note_names(self):
        low = detect_notes(np.concatenate([tone(48, 450)]), RATE, min_note_ms=150, min_confidence=0.4)
        high = detect_notes(np.concatenate([tone(72, 450)]), RATE, min_note_ms=150, min_confidence=0.4)
        self.assertEqual([note.midi for note in low], [48])
        self.assertEqual([note.midi for note in high], [72])
        self.assertEqual(low[0].name, "C3")
        self.assertEqual(high[0].name, "C5")

    def test_hum_to_midi_writes_a_parseable_type0_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_wav(Path(tmp) / "hum.wav", np.concatenate([tone(60, 350), tone(67, 350)]))
            result = hum_to_midi(source, min_note_ms=150, min_confidence=0.4)
            target = Path(result["midi_path"])
            self.assertTrue(target.is_file())
            self.assertEqual(result["note_count"], 2)
            data = target.read_bytes()
            self.assertEqual(data[:4], b"MThd")
            self.assertEqual(struct.unpack(">HHH", data[8:14]), (0, 1, 480))  # format 0, one track, 480 ppq
            self.assertEqual(data[14:18], b"MTrk")
            self.assertEqual(data.count(bytes([0x90])), 2)  # two note-ons
            self.assertEqual(data.count(bytes([0x80])), 2)  # two note-offs

    def test_read_wav_mono_mixes_channels_and_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            samples = np.zeros(200, dtype="<i2")
            samples[0::2] = 16384  # left channel
            samples[1::2] = 16384  # right channel
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(RATE)
                handle.writeframes(samples.tobytes())
            data, rate = read_wav_mono(path)
            self.assertEqual(rate, RATE)
            self.assertEqual(data.size, 100)
            self.assertAlmostEqual(float(data.max()), 0.5, places=2)

    def test_write_midi_orders_events_by_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            from shadow_audio_dedupe.melody import Note

            notes = [Note(500, 900, 64, 1.0), Note(0, 400, 60, 1.0)]
            target = write_midi_type0(notes, Path(tmp) / "two.mid")
            data = target.read_bytes()
            first_note_on = data.index(bytes([0x90, 60, 100]))
            second_note_on = data.index(bytes([0x90, 64, 100]))
            self.assertLess(first_note_on, second_note_on)


if __name__ == "__main__":
    unittest.main()
