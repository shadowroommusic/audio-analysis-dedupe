from __future__ import annotations

import json
import sys

from .analyze import analyze_file, analyze_folder
from .dedupe import dedupe_folder
from .model import to_json

TOOLS = {
    "analyze_file": "Report size, duration, sample rate, channels, codec and bitrate for one audio file.",
    "dedupe_folder": "Group exact SHA-256 duplicates and list near-duplicate candidates in a folder. Never deletes files.",
}


def _schema(name: str) -> dict:
    if name == "analyze_file":
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute path to an audio file."}},
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
        return response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "audio-analysis-dedupe", "version": "0.1.0"},
            },
        )
    if method == "notifications/initialized":
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
        else:
            raise ValueError(f"Unknown tool: {name}")
        return response(request_id, {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]})
    except Exception as exc:
        return response(request_id, error=exc)


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
