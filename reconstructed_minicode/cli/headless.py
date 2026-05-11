"""MiniCode Headless Runner — non-interactive, one-shot execution.

Usage:
  python -m minicode.headless "帮我分析这个项目的结构"
  echo "解释这段代码" | python -m minicode.headless
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HeadlessResult:
    response: str
    tokens_used: int = 0
    compression_count: int = 0
    messages_dropped: int = 0
    turns: int = 0
    error: str | None = None


def run_headless(prompt: str | None = None, verbose: bool = False) -> HeadlessResult:
    """Run agent in headless mode. Returns HeadlessResult with response + metrics."""
    from reconstructed_minicode.agent.context import ContextManager, estimate_messages_tokens
    from reconstructed_minicode.agent.loop import run_agent_turn
    from reconstructed_minicode.config import load_runtime_config
    from reconstructed_minicode.extensions.memory import MemoryManager
    from reconstructed_minicode.model.registry import create_model_adapter
    from reconstructed_minicode.security.permissions import PermissionManager
    from reconstructed_minicode.security.risk import PermissionMode
    from reconstructed_minicode.session.prompt import build_system_prompt
    from reconstructed_minicode.tools import create_default_tool_registry
    from reconstructed_minicode.utils.logging import setup_logging, get_logger

    setup_logging(level=os.environ.get("MINI_CODE_LOG_LEVEL", "WARNING"))
    logger = get_logger("headless")

    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("Usage: python -m minicode.headless <prompt>", file=sys.stderr)
            sys.exit(1)

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    cwd = str(Path.cwd())

    try:
        runtime = load_runtime_config(cwd)
    except Exception as exc:
        return HeadlessResult(response=f"Config error: {exc}", error=str(exc))

    tools = create_default_tool_registry(cwd, runtime=runtime)
    bypass = os.environ.get("MINI_CODE_BYPASS_PERMISSIONS") == "1"
    permissions = PermissionManager(cwd, prompt=None, auto_mode=PermissionMode.BYPASS if bypass else None)
    memory_mgr = MemoryManager(project_root=Path(cwd))
    context_manager = ContextManager(model=runtime.get("model", "default"))
    model = create_model_adapter(model=runtime.get("model", ""), tools=tools, runtime=runtime)

    messages = [
        {"role": "system", "content": build_system_prompt(
            cwd, permissions.get_summary(),
            {"skills": tools.get_skills(), "mcpServers": tools.get_mcp_servers(),
             "memory_context": memory_mgr.get_relevant_context()},
        )},
        {"role": "user", "content": prompt},
    ]

    turns = 0

    def on_tool_start(name: str, inp: dict) -> None:
        if verbose:
            print(f"  [tool] {name}({str(inp).replace(chr(10), ' ')})")

    def on_tool_result(name: str, output: str, is_err: bool) -> None:
        if verbose:
            tag = "ERR" if is_err else "ok"
            print(f"  [tool] {name} → {tag}: {output.replace(chr(10), ' ')}")

    def on_msg(content: str) -> None:
        nonlocal turns
        turns += 1
        if verbose:
            print(f"  [msg] {content}")

    logger.info("Headless run: %s", prompt[:80])

    try:
        result_messages = run_agent_turn(
            model=model, tools=tools, messages=messages, cwd=cwd,
            permissions=permissions, context_manager=context_manager, runtime=runtime,
            on_assistant_message=on_msg, on_tool_start=on_tool_start,
            on_tool_result=on_tool_result,
        )
        last = next((m for m in reversed(result_messages) if m["role"] == "assistant"), None)
        response = last["content"] if last else "(no response)"
        tokens = estimate_messages_tokens(result_messages)
        compressions = len(context_manager.compaction_history)
        dropped = sum(h.get("messages_removed", 0) for h in context_manager.compaction_history)
        return HeadlessResult(response=response, tokens_used=tokens,
                              compression_count=compressions, messages_dropped=dropped, turns=turns)
    except Exception as exc:
        logger.error("Headless error: %s", exc)
        return HeadlessResult(response=f"Error: {exc}", error=str(exc))
    finally:
        try:
            tools.dispose()
        except Exception:
            pass


def main() -> None:
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    result = run_headless(prompt, verbose=True)
    print(result.response)


if __name__ == "__main__":
    main()
