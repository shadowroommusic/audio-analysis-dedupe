# Internals

Maintainer notes for `audio-analysis-dedupe`. The user-facing docs live in
[../README.md](../README.md).

## How analysis works

- WAV (`.wav`, `.wave`) uses the standard library `wave` module.
- AIFF/AIFF-C (`.aif`, `.aiff`, `.aifc`) uses `aifc` when present, with a built-in IFF chunk parser
  as a fallback for Python versions where `aifc` was removed.
- AU/SND (`.au`, `.snd`) uses `sunau`, with a built-in header parser as a fallback.
- Every other container (MP3, M4A, FLAC, OGG, …) is read through `ffprobe` when it is on `PATH` or
  pointed at via `SHADOW_FFPROBE`. Without `ffprobe` the files are still hashed and listed, with a
  warning that metadata is unavailable.

For PCM containers the plugin additionally records `audio_sha256` over the decoded sample stream,
so the same recording stored as WAV, AIFF and AU is detected even though the files differ byte for
byte.

## Dedupe pipeline

1. Hash the raw file (SHA-256) → `exact_groups`.
2. Hash the decoded PCM stream → `identical_audio_groups`.
3. Normalise file names (strip trailing `(1)`, `[2]`, `- copy`, `_copy2`, `- 1`, full-width `（1）`,
   …) and compare name + duration + size within the threshold → `candidates`, each with `reasons`
   and an explicit `differences` list (duration, size, sample rate, channels, bitrate, bit depth,
   container/codec).
4. Summarise: redundant file count, redundant bytes, duplicate-free files, active threshold.

## Tests

`tests/` builds its own WAV, AIFF/AIFF-C and AU fixtures with the standard library, so no audio
sample files are required:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
