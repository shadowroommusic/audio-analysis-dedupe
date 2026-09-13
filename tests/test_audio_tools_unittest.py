from __future__ import annotations

import hashlib
import math
import os
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from shadow_audio_dedupe.analyze import analyze_file, analyze_folder
from shadow_audio_dedupe.dedupe import dedupe, dedupe_folder, normalize_name
from shadow_audio_dedupe.readers import read_aiff_manual, read_au_manual

RATE = 8000


def sine_frames(sample_count: int, rate: int = RATE, frequency: float = 440.0) -> bytes:
    samples = [int(20_000 * math.sin(2 * math.pi * frequency * index / rate)) for index in range(sample_count)]
    return struct.pack(f"<{sample_count}h", *samples)


def write_wav(path: Path, frames: bytes, channels: int = 1, rate: int = RATE, width: int = 2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(frames)


def extended80(value: float) -> bytes:
    if value == 0:
        return b"\x00" * 10
    exponent = math.floor(math.log2(value))
    mantissa = int(value / (2 ** exponent) * (1 << 63))
    return struct.pack(">H", 16383 + exponent) + mantissa.to_bytes(8, "big")


def chunk(chunk_id: bytes, body: bytes) -> bytes:
    padded = body + (b"\x00" if len(body) % 2 else b"")
    return chunk_id + struct.pack(">I", len(body)) + padded


def write_aiff(path: Path, frames: bytes, compression: str = "NONE", channels: int = 1, rate: int = RATE, bits: int = 16) -> None:
    frame_count = len(frames) // max(channels * bits // 8, 1)
    comm = struct.pack(">hIh", channels, frame_count, bits) + extended80(rate)
    form_type = b"AIFF"
    if compression != "NONE":
        form_type = b"AIFC"
        name = b"shadow"
        comm += compression.encode("latin-1") + bytes([len(name)]) + name
    ssnd = struct.pack(">II", 0, 0) + frames
    body = form_type + chunk(b"COMM", comm) + chunk(b"SSND", ssnd)
    path.write_bytes(b"FORM" + struct.pack(">I", len(body)) + body)


def write_au(path: Path, frames: bytes, encoding: int = 3, channels: int = 1, rate: int = RATE) -> None:
    header = b".snd" + struct.pack(">IIIII", 24, len(frames), encoding, rate, channels)
    path.write_bytes(header + frames)


class TempFilesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="shadow-audio-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)


class AudioAnalysisTests(TempFilesTestCase):
    def test_wave_metadata_is_reported(self) -> None:
        path = self.root / "kick.wav"
        write_wav(path, sine_frames(RATE))
        info = analyze_file(path)
        self.assertEqual(info.container, "wav")
        self.assertEqual(info.codec, "pcm_s16le")
        self.assertEqual(info.sample_rate, RATE)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.bits_per_sample, 16)
        self.assertEqual(info.duration_ms, 1000)
        self.assertEqual(info.size_bytes, path.stat().st_size)
        self.assertEqual(info.probe, "wave")
        self.assertTrue(info.analyzed)
        self.assertAlmostEqual(info.bitrate_kbps or 0, 128.0, places=1)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            analyze_file(self.root / "missing.wav")

    def test_manual_aiff_parser_reads_a_handmade_container(self) -> None:
        frames = sine_frames(RATE)
        path = self.root / "handmade.aiff"
        write_aiff(path, frames)
        data: dict = {}
        read_aiff_manual(path, data)
        self.assertEqual(data["container"], "aiff")
        self.assertEqual(data["sample_rate"], RATE)
        self.assertEqual(data["channels"], 1)
        self.assertEqual(data["duration_ms"], 1000)
        self.assertEqual(data["codec"], "pcm_s16be")
        self.assertEqual(data["audio_sha256"], hashlib.sha256(frames).hexdigest())

    def test_manual_aiff_parser_reads_aifc_compression_type(self) -> None:
        frames = sine_frames(RATE // 2)
        path = self.root / "handmade-sowt.aifc"
        write_aiff(path, frames, compression="sowt")
        data: dict = {}
        read_aiff_manual(path, data)
        self.assertEqual(data["codec"], "pcm_s16le")
        self.assertEqual(data["duration_ms"], 500)

    def test_manual_au_parser_reads_a_handmade_container(self) -> None:
        frames = sine_frames(RATE)
        path = self.root / "handmade.au"
        write_au(path, frames)
        data: dict = {}
        read_au_manual(path, data)
        self.assertEqual(data["container"], "au")
        self.assertEqual(data["codec"], "pcm_s16be")
        self.assertEqual(data["sample_rate"], RATE)
        self.assertEqual(data["channels"], 1)
        self.assertEqual(data["duration_ms"], 1000)
        self.assertEqual(data["audio_sha256"], hashlib.sha256(frames).hexdigest())

    def test_unknown_container_without_ffprobe_reports_a_warning(self) -> None:
        path = self.root / "mystery.mp3"
        path.write_bytes(b"\x00" * 64)
        with mock.patch.dict(os.environ, {"SHADOW_FFPROBE": ""}):
            info = analyze_file(path)
        self.assertEqual(info.probe, "unavailable")
        self.assertIsNone(info.duration_ms)
        self.assertFalse(info.analyzed)
        self.assertTrue(any("ffprobe is not installed" in warning for warning in info.warnings))
        self.assertEqual(len(info.sha256), 64)

    def test_folder_analysis_summarizes_formats_and_ignores_other_files(self) -> None:
        write_wav(self.root / "one.wav", sine_frames(RATE))
        write_wav(self.root / "two.wav", sine_frames(RATE, frequency=220.0))
        (self.root / "notes.txt").write_text("not audio", encoding="utf-8")
        nested = self.root / "crate"
        nested.mkdir()
        write_wav(nested / "three.wav", sine_frames(RATE, frequency=110.0))
        report = analyze_folder(self.root)
        self.assertEqual(report["file_count"], 3)
        self.assertEqual(report["summary"]["codecs"], {"pcm_s16le": 3})
        self.assertEqual(report["summary"]["total_duration_ms"], 3000)
        self.assertEqual(report["mode"], "read-only-analysis")
        self.assertEqual(report["skipped_count"], 0)
        top_level_only = analyze_folder(self.root, recursive=False)
        self.assertEqual(top_level_only["file_count"], 2)


class DedupeTests(TempFilesTestCase):
    def test_cli_level_folder_report_is_read_only_and_stable(self) -> None:
        write_wav(self.root / "Track A.wav", sine_frames(RATE))
        report = dedupe_folder(str(self.root))
        self.assertEqual(report["mode"], "read-only-dedupe")
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["summary"]["exact_group_count"], 0)
        self.assertTrue(any("No file was deleted" in warning for warning in report["warnings"]))

    def test_exact_duplicates_are_grouped_and_nothing_is_deleted(self) -> None:
        original = self.root / "Track A.wav"
        copy = self.root / "Track A (1).wav"
        write_wav(original, sine_frames(RATE))
        shutil.copyfile(original, copy)
        report = dedupe_folder(str(self.root))
        self.assertEqual(report["summary"]["exact_group_count"], 1)
        group = report["exact_groups"][0]
        self.assertEqual(group["file_count"], 2)
        self.assertEqual(sorted(group["files"]), sorted([str(original), str(copy)]))
        self.assertEqual(group["suggested_keep"], str(original))
        self.assertEqual(report["summary"]["redundant_bytes"], original.stat().st_size)
        self.assertTrue(original.exists() and copy.exists())

    def test_near_duplicate_candidate_lists_explicit_differences(self) -> None:
        short = self.root / "Track A.wav"
        longer = self.root / "Track A (1).wav"
        write_wav(short, sine_frames(RATE))
        write_wav(longer, sine_frames(RATE * 3 // 2))
        report = dedupe_folder(str(self.root))
        self.assertEqual(report["summary"]["exact_group_count"], 0)
        self.assertGreaterEqual(report["summary"]["candidate_count"], 1)
        candidate = report["candidates"][0]
        self.assertEqual({candidate["a"], candidate["b"]}, {str(short), str(longer)})
        self.assertGreater(candidate["score"], 0.72)
        self.assertIn("normalized names match", candidate["reasons"][0])
        self.assertTrue(any("duration differs by 500 ms" in difference for difference in candidate["differences"]))
        self.assertTrue(any("size differs" in difference for difference in candidate["differences"]))
        self.assertEqual(candidate["suggested_keep"], str(short))

    def test_same_audio_stream_in_two_containers_is_reported(self) -> None:
        frames = sine_frames(RATE)
        wave_path = self.root / "Track B.wav"
        aiff_path = self.root / "Track B.aiff"
        au_path = self.root / "Track B.au"
        write_wav(wave_path, frames)
        write_aiff(aiff_path, frames)
        write_au(au_path, frames)
        infos = [analyze_file(path) for path in (wave_path, aiff_path, au_path)]
        self.assertEqual(len({info.audio_sha256 for info in infos}), 1)
        self.assertEqual(len({info.sha256 for info in infos}), 3)
        report = dedupe(infos)
        self.assertEqual(report["summary"]["identical_audio_group_count"], 1)
        self.assertEqual(report["identical_audio_groups"][0]["file_count"], 3)
        self.assertEqual(report["identical_audio_groups"][0]["containers"], ["aiff", "au", "wav"])

    def test_unrelated_files_stay_below_the_threshold(self) -> None:
        first = self.root / "Ambient Pad.wav"
        second = self.root / "Hard Kick.wav"
        write_wav(first, sine_frames(RATE, frequency=220.0))
        write_wav(second, sine_frames(RATE * 4, frequency=1000.0))
        report = dedupe_folder(str(self.root))
        self.assertEqual(report["summary"]["candidate_count"], 0)
        self.assertEqual(report["summary"]["duplicate_free_files"], 2)

    def test_numbered_series_names_are_flagged_as_weak_candidates(self) -> None:
        first = self.root / "Sample 101.wav"
        second = self.root / "Sample 102.wav"
        write_wav(first, sine_frames(RATE, frequency=220.0))
        write_wav(second, sine_frames(RATE, frequency=1000.0))
        report = dedupe_folder(str(self.root))
        self.assertEqual(report["summary"]["candidate_count"], 1)
        candidate = report["candidates"][0]
        self.assertLessEqual(candidate["score"], 0.8)
        self.assertTrue(any("different track in a series" in reason for reason in candidate["reasons"]))

    def test_copy_markers_are_normalized(self) -> None:
        self.assertEqual(normalize_name("Track A (1).wav"), normalize_name("Track A.wav"))
        self.assertEqual(normalize_name("Track A（1）.wav"), normalize_name("Track A.wav"))
        self.assertEqual(normalize_name("Track A - Copy 2.aiff"), normalize_name("Track A.aiff"))
        self.assertEqual(normalize_name("Track A-1.au"), normalize_name("Track A.au"))
        self.assertEqual(normalize_name("Track A_2.wav"), normalize_name("Track A.wav"))
        self.assertNotEqual(normalize_name("Track A.wav"), normalize_name("Track B.wav"))
        # Real titles keep their number; only copy markers are stripped.
        self.assertNotEqual(normalize_name("Demo Track 1.wav"), normalize_name("Demo Track 2.wav"))

    def test_numbered_titles_are_not_treated_as_exact_copies(self) -> None:
        first = self.root / "Demo Track 1.wav"
        second = self.root / "Demo Track 2.wav"
        write_wav(first, sine_frames(RATE, frequency=220.0))
        write_wav(second, sine_frames(RATE, frequency=1000.0))
        report = dedupe_folder(str(self.root))
        self.assertEqual(report["summary"]["exact_group_count"], 0)
        self.assertEqual(len(report["exact_groups"]), 0)
        candidate = report["candidates"][0]
        self.assertTrue(any("different track in a series" in reason for reason in candidate["reasons"]))
        self.assertEqual(candidate["suggested_keep"], str(first))


if __name__ == "__main__":
    unittest.main()
