# Audio Analysis & Dedupe

This independent ShadowRoom Music plugin (a Shadow Producers tool) inspects an audio folder and answers two questions:

1. What is inside these files (size, duration, sample rate, channels, codec, bitrate)?
2. Which files are duplicates, and how do two versions of the same track differ?

It is read-only by design. It never deletes, moves, renames, or rewrites a file, and it never touches a vendor database.

## Dependency-free analysis

- WAV (`.wav`, `.wave`) is parsed with the standard library `wave` module.
- AIFF/AIFF-C (`.aif`, `.aiff`, `.aifc`) is parsed with the standard library `aifc` module, with a built-in IFF chunk parser as the fallback for Python versions where `aifc` was removed.
- AU/SND (`.au`, `.snd`) is parsed with `sunau`, with a built-in header parser as the fallback.
- Everything else (MP3, M4A, FLAC, OGG, …) is read through `ffprobe` when it is available on `PATH`, or through `SHADOW_FFPROBE=/path/to/ffprobe`. Without `ffprobe`, unsupported containers are still hashed and listed, with a warning that metadata is unavailable.

For PCM containers the plugin also records an `audio_sha256` of the decoded sample stream, so the same recording stored as WAV, AIFF and AU is detected even though the files themselves differ byte for byte.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .

# Analyze files or whole folders
.venv/bin/shadow-audio-dedupe analyze --path ~/Music/Crate --output analysis.json

# Group exact duplicates and near-duplicate candidates
.venv/bin/shadow-audio-dedupe dedupe --folder ~/Music/Crate --threshold 0.72 --output duplicates.json
```

## What the dedupe report contains

- `exact_groups`: files with an identical SHA-256, plus a `suggested_keep` and the reason for that suggestion.
- `identical_audio_groups`: files whose decoded sample stream is identical but whose container or header differs.
- `candidates`: near-duplicates found through the normalized file name, duration and size, each with `reasons` and an explicit `differences` list (duration, size, sample rate, channels, bitrate, bit depth, container/codec).
- `summary`: redundant file count, redundant bytes, duplicate-free files and the active threshold.

A trailing `(1)`, `[2]`, `- copy`, `_copy2`, `- 1` or full-width `（1）` marker is stripped before names are compared, which is exactly the "song a（1） vs song a" case.

## MCP

`.mcp.json` exposes two read-only tools:

- `analyze_file` — analyze one file.
- `dedupe_folder` — analyze a folder (`recursive`, `threshold`, `max_candidates`).

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests build their own WAV fixtures, AIFF/AIFF-C and AU containers with the standard library, so no audio sample files are required.

## License

MIT for this plugin. It has no runtime dependency; optional `ffprobe` usage is subject to the FFmpeg license of your local installation.
