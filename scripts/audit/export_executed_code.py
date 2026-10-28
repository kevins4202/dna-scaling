"""Save tool code from this project's local session history, excluding messages and outputs.

The archive is evidence of issued code, including failed attempts. It is not a script
that should be replayed. Only files belonging to the requested thread are inspected.
"""

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path


SECRET_PATTERNS = [
    (
        re.compile(
            r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----.*?-----END (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
            re.S,
        ),
        "[REDACTED PRIVATE KEY]",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[REDACTED API KEY]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[REDACTED TOKEN]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED ACCESS KEY]"),
    (
        re.compile(
            r'(?i)(authorization["\s:]+bearer\s+)(?!\$|\{|<)[A-Za-z0-9._-]{16,}'
        ),
        r"\1[REDACTED]",
    ),
]


def redact(text):
    total = 0
    for pattern, replacement in SECRET_PATTERNS:
        text, count = pattern.subn(replacement, text)
        total += count
    return text, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument(
        "--sessions-root", type=Path, default=Path.home() / ".codex/sessions"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f-]{36}", args.thread_id):
        raise SystemExit("Expected a thread UUID")
    paths = sorted(args.sessions_root.rglob(f"*{args.thread_id}*.jsonl"))
    if not paths:
        raise SystemExit("No matching session files")
    records, sources, seen = [], [], set()
    redactions = duplicates = incomplete_lines = 0
    for path in paths:
        content = path.read_bytes()
        sources.append(
            {"file": path.name, "sha256": hashlib.sha256(content).hexdigest()}
        )
        for line in content.splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                incomplete_lines += 1
                continue
            payload = item.get("payload", {})
            if not isinstance(payload, dict) or payload.get("type") not in (
                "function_call",
                "custom_tool_call",
            ):
                continue
            name = payload.get("name", "").split(".")[-1]
            if name not in (
                "exec",
                "exec_command",
                "apply_patch",
                "js",
                "python",
                "bash",
            ):
                continue
            code = payload.get("input", payload.get("arguments", ""))
            if not isinstance(code, str):
                code = json.dumps(code)
            key = (
                payload.get("call_id", payload.get("id")),
                hashlib.sha256(code.encode()).hexdigest(),
            )
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            code, count = redact(code)
            redactions += count
            records.append(
                {
                    "timestamp": item.get("timestamp"),
                    "session_file": path.name,
                    "call_id": key[0],
                    "tool": name,
                    "code": code,
                }
            )
    records.sort(key=lambda row: (row["timestamp"] or "", row["call_id"] or ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in records))
    receipt = {
        "exported_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "thread_id": args.thread_id,
        "issued_code_records": len(records),
        "source_snapshots": sources,
        "redactions": redactions,
        "duplicate_calls_removed": duplicates,
        "incomplete_lines_skipped": incomplete_lines,
        "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "scope": "Tool code only from available matching project sessions. Includes failed attempts and edits. No user/assistant messages or tool outputs. Not for automatic replay.",
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k != "source_snapshots"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
