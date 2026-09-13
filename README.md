# Audio Analysis & Dedupe

An MCP server that inspects an audio folder and answers two questions: *what is inside these
files?*, and *which of them are duplicates?*

Works with any MCP-compatible agent or client.

[中文说明](README.zh-CN.md) · License: [AGPL-3.0](LICENSE)

## Features

- **Library analysis.** Size, duration, sample rate, channels, codec and bitrate for every file in
  a folder (recursively if you want).
- **Exact duplicates.** Byte-identical files, grouped by SHA-256 with a suggested file to keep.
- **Same audio, different container.** Files whose decoded PCM stream is identical (for example the
  same master exported as WAV, AIFF and AU) are reported separately.
- **Near-duplicate candidates.** Name (with `(1)`, `- copy`, `（1）` style suffixes normalised),
  duration and size are compared, with an explicit list of differences for each candidate pair.
- **Read-only by design.** It never deletes, moves, renames or rewrites a file, and never touches a
  vendor database.

## Requirements

| | |
| --- | --- |
| OS | macOS, Linux or Windows |
| Python | 3.9 or newer |
| Optional | `ffprobe` on `PATH` (or `SHADOW_FFPROBE`) for formats outside WAV/AIFF/AU |

## Install

### As a Codex plugin

```sh
codex plugin marketplace add shadowroommusic/audio-analysis-dedupe
codex plugin add audio-analysis-dedupe@shadowroom
```

### In any other MCP client

```json
{
  "mcpServers": {
    "audio-analysis-dedupe": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/audio-analysis-dedupe"
    }
  }
}
```

### CLI only

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/shadow-audio-dedupe --help
```

## Configuration

| Option | Default | Used for |
| --- | --- | --- |
| `SHADOW_FFPROBE` | auto-detected | path to `ffprobe` for non-PCM containers |
| `--threshold` | `0.72` | similarity threshold for near-duplicate candidates |
| `--output` | stdout | write the JSON report to a file |

## Tools

| Tool | What it does |
| --- | --- |
| `analyze_file` | Analyse a single file (format, duration, sample rate, channels, bitrate) |
| `dedupe_folder` | Analyse a folder and group duplicates (`recursive`, `threshold`, `max_candidates`) |

CLI equivalents: `shadow-audio-dedupe analyze --path <file-or-folder>` and
`shadow-audio-dedupe dedupe --folder <folder>`.

## Usage

```sh
# what is in this crate?
.venv/bin/shadow-audio-dedupe analyze --path ~/Music/Crate --output analysis.json

# what is duplicated, and how do the versions differ?
.venv/bin/shadow-audio-dedupe dedupe --folder ~/Music/Crate --threshold 0.72 --output duplicates.json
```

The dedupe report contains `exact_groups`, `identical_audio_groups`, `candidates` (each with
`reasons` and `differences`) and a `summary` with the redundant file count and bytes.

## Safety

- Read-only: nothing is modified, and no vendor database is opened.
- Reports are written only to the path you pass with `--output` (or to stdout).

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Metadata unavailable for MP3/FLAC | Install `ffmpeg`/`ffprobe`, or set `SHADOW_FFPROBE` to its path. |
| Everything looks like a near-duplicate | Raise `--threshold` (for example `0.85`). |
| Very large folders are slow | Narrow the folder, or run `analyze` first to see what is inside. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Implementation notes live in
[docs/internals.md](docs/internals.md).

## License

AGPL-3.0 — see [LICENSE](LICENSE). Optional `ffprobe` usage is subject to the license of your local
FFmpeg installation.
