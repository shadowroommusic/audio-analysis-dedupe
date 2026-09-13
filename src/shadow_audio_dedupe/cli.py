from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_file, analyze_folder
from .dedupe import dedupe_folder
from .model import to_json


def dump(value: object, output: str | None) -> None:
    text = json.dumps(to_json(value), ensure_ascii=False, indent=2) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="shadow-audio-dedupe",
        description="Offline, read-only audio analysis and duplicate detection. Nothing is ever deleted.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyze audio files or folders.")
    analyze.add_argument("--path", action="append", required=True, metavar="PATH", help="File or folder; repeat for more.")
    analyze.add_argument("--no-recursive", action="store_true", help="Only look at the top level of each folder.")
    analyze.add_argument("--output", help="Write the JSON report to this file instead of stdout.")
    dedupe_command = commands.add_parser("dedupe", help="Group exact duplicates and list near-duplicate candidates.")
    dedupe_command.add_argument("--folder", required=True, help="Folder to scan.")
    dedupe_command.add_argument("--no-recursive", action="store_true", help="Only look at the top level.")
    dedupe_command.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold between 0 and 1 (default 0.72).")
    dedupe_command.add_argument("--max-candidates", type=int, default=200, help="Cap the candidate list (default 200).")
    dedupe_command.add_argument("--output", help="Write the JSON report to this file instead of stdout.")
    args = parser.parse_args()

    if args.command == "analyze":
        results = []
        for raw in args.path:
            target = Path(raw).expanduser()
            if target.is_dir():
                results.append(analyze_folder(target, recursive=not args.no_recursive))
            else:
                results.append(to_json(analyze_file(target)))
        dump(
            {
                "schema_version": 1,
                "mode": "read-only-analysis",
                "targets": results,
                "warnings": ["No audio file or vendor database was modified."],
            },
            args.output,
        )
        return 0

    dump(
        dedupe_folder(
            args.folder,
            recursive=not args.no_recursive,
            threshold=args.threshold,
            max_candidates=args.max_candidates,
        ),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
