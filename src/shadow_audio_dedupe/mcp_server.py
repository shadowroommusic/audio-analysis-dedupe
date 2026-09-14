from __future__ import annotations

import json
import sys

from .analyze import analyze_file, analyze_folder
from .dedupe import dedupe_folder
from .melody import hum_to_midi
from .model import to_json

TOOLS = {
    "analyze_file": "Report size, duration, sample rate, channels, codec and bitrate for one audio file.",
    "dedupe_folder": "Group exact SHA-256 duplicates and list near-duplicate candidates in a folder. Never deletes files.",
    "hum_to_midi": "Detect the melody in a monophonic WAV recording (a hum) and write it as a Type-0 MIDI file.",
}

SERVER_NAME = "audio-analysis-dedupe"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSION = "2024-11-05"

# Methods other MCP clients probe during capability negotiation. Answering with a
# valid empty result keeps the server usable from any agent, not just Codex.
EMPTY_RESULTS = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
    "logging/setLevel": {},
}


def _schema(name: str) -> dict:
    if name == "analyze_file":
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute path to an audio file."}},
            "required": ["path"],
            "additionalProperties": False,
        }
    if name == "hum_to_midi":
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "WAV recording of the hummed melody."},
                "out_path": {"type": "string", "description": "Where to write the .mid (default: next to the WAV)."},
                "min_note_ms": {"type": "integer", "default": 120, "description": "Shorter blips are treated as noise."},
                "tempo_bpm": {"type": "number", "default": 120},
                "min_confidence": {"type": "number", "default": 0.5},
            },
            "required": ["path"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Absolute path to the folder to scan."},
            "recursive": {"type": "boolean", "default": True},
            "threshold": {"type": "number", "default": 0.72},
            "max_candidates": {"type": "integer", "default": 200},
        },
        "required": ["folder"],
        "additionalProperties": False,
    }


def response(request_id: object, result: object = None, error: object = None) -> dict:
    value = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        value["error"] = {"code": -32000, "message": str(error)}
    else:
        value["result"] = result
    return value


def handle(message: dict) -> "dict | None":
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        requested = params.get("protocolVersion")
        return response(
            request_id,
            {
                "protocolVersion": requested if isinstance(requested, str) and requested else PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "ping":
        return response(request_id, {})
    if method in EMPTY_RESULTS:
        return response(request_id, EMPTY_RESULTS[method])
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if method == "tools/list":
        return response(
            request_id,
            {"tools": [{"name": name, "description": description, "inputSchema": _schema(name)} for name, description in TOOLS.items()]},
        )
    if method != "tools/call":
        return response(request_id, error=f"Unsupported method: {method}")
    name = params.get("name")
    arguments = params.get("arguments") or {}
    try:
        if name == "analyze_file":
            value = to_json(analyze_file(arguments["path"]))
        elif name == "dedupe_folder":
            value = dedupe_folder(
                arguments["folder"],
                recursive=bool(arguments.get("recursive", True)),
                threshold=float(arguments.get("threshold", 0.72)),
                max_candidates=int(arguments.get("max_candidates", 200)),
            )
        elif name == "hum_to_midi":
            value = hum_to_midi(
                arguments["path"],
                arguments.get("out_path"),
                min_note_ms=int(arguments.get("min_note_ms", 120)),
                tempo_bpm=float(arguments.get("tempo_bpm", 120.0)),
                min_confidence=float(arguments.get("min_confidence", 0.5)),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
        return response(request_id, {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]})
    except Exception as exc:
        # Tool failures are reported inside the result (MCP `isError`), not as protocol errors.
        return response(request_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            result = handle(json.loads(line))
            if result is not None:
                print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps(response(None, error=exc), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
