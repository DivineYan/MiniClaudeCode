from __future__ import annotations

import time
from pathlib import Path

from reconstructed_minicode.tools.base import ToolDefinition, ToolResult
from reconstructed_minicode.utils.workspace import resolve_tool_path

DEFAULT_READ_LINES = 200
MAX_READ_LINES = 2000

_file_cache: dict[tuple[str, float], tuple[list[str], float]] = {}
_FILE_CACHE_TTL = 2.0


def _get_cached_lines(target: Path) -> list[str]:
    try:
        mtime = target.stat().st_mtime
        cache_key = (str(target), mtime)
        if cache_key in _file_cache:
            lines, t = _file_cache[cache_key]
            if time.monotonic() - t <= _FILE_CACHE_TTL:
                return lines
        now = time.monotonic()
        expired = [k for k, (_, t) in _file_cache.items() if now - t > _FILE_CACHE_TTL]
        for k in expired:
            del _file_cache[k]
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        _file_cache[cache_key] = (lines, time.monotonic())
        return lines
    except OSError:
        return target.read_text(encoding="utf-8").splitlines(keepends=True)


def _validate(input_data: dict) -> dict:
    path = input_data.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    offset = int(input_data.get("offset", 1))
    limit = int(input_data.get("limit", DEFAULT_READ_LINES))
    if offset < 1:
        raise ValueError("offset (line number) must be >= 1")
    if limit < 1 or limit > MAX_READ_LINES:
        raise ValueError(f"limit must be between 1 and {MAX_READ_LINES}")
    return {"path": path, "offset": offset, "limit": limit}


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"], "read")
    try:
        lines = _get_cached_lines(target)
    except UnicodeDecodeError:
        return ToolResult(ok=False, output=f"File {input_data['path']} is binary.")

    offset = input_data["offset"]   # 1-based line number
    limit = input_data["limit"]
    total = len(lines)

    start = max(0, offset - 1)      # convert to 0-based index
    end = min(total, start + limit)
    chunk = "".join(lines[start:end])
    truncated = end < total

    header = "\n".join([
        f"FILE: {input_data['path']}",
        f"LINES: {start + 1}-{end} of {total}",
        f"TRUNCATED: {'yes (more content at offset ' + str(end + 1) + ', read only if needed)' if truncated else 'no'}",
        "",
    ])
    return ToolResult(ok=True, output=header + chunk)


read_file_tool = ToolDefinition(
    name="read_file",
    description="Read a text file by line number. Use grep_files first to find the relevant line, then read from that line.",
    input_schema={"type": "object", "properties": {
        "path": {"type": "string", "description": "File path relative to workspace root"},
        "offset": {"type": "number", "description": "Start line number, 1-based (default 1). Use the line number from grep results."},
        "limit": {"type": "number", "description": f"Number of lines to read (default {DEFAULT_READ_LINES}, max {MAX_READ_LINES})."},
    }, "required": ["path"]},
    validator=_validate,
    run=_run,
)
