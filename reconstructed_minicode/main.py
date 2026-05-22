from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from reconstructed_minicode.agent.context import ContextManager
from reconstructed_minicode.agent.loop import run_agent_turn
from reconstructed_minicode.agent.state import create_app_store
from reconstructed_minicode.cli.commands import try_handle_local_command
from reconstructed_minicode.cli.manage import maybe_handle_management_command
from reconstructed_minicode.cli.shortcuts import parse_local_tool_shortcut
from reconstructed_minicode.config import load_runtime_config
from reconstructed_minicode.extensions.agent_loader import AgentRegistry, discover_agents
from reconstructed_minicode.extensions.hooks import load_hooks_from_config
from reconstructed_minicode.extensions.memory import MemoryManager
from reconstructed_minicode.security.permissions import PermissionManager
from reconstructed_minicode.session.history import load_history_entries, save_history_entries
from reconstructed_minicode.session.prompt import build_system_prompt
from reconstructed_minicode.tools import create_default_tool_registry
from reconstructed_minicode.tools.base import ToolContext
from reconstructed_minicode.tui.app import run_tty_app
from reconstructed_minicode.tui.transcript import format_transcript_text
from reconstructed_minicode.tui.types import TranscriptEntry
from reconstructed_minicode.utils.logging import get_logger, setup_logging
from reconstructed_minicode.utils.workspace import resolve_tool_path
from reconstructed_minicode.model.registry import create_model_adapter

logger = get_logger("main")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniCode Python - A lightweight terminal coding assistant",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="SESSION_ID",
        help="Resume a previous session ('latest' or session ID)",
    )
    parser.add_argument("--list-sessions", action="store_true", help="List saved sessions and exit")
    parser.add_argument("--session", default=None, metavar="SESSION_ID", help="Start with a specific session ID")
    parser.add_argument("--install", action="store_true", help="Run the interactive installer")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration and exit")
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    parser.add_argument(
        "--cwd", default=None, metavar="DIR",
        help="Working directory to open (defaults to current terminal directory)",
    )
    parser.add_argument(
        "path", nargs="?", default=None, metavar="PATH",
        help="Working directory to open (positional shorthand for --cwd)",
    )
    return parser.parse_args()


def _make_cli_permission_prompt():
    def _prompt(request: dict) -> dict:
        print(f"\n{request.get('summary', 'Permission Request')}")
        choices = request.get("choices", [])
        if choices:
            for choice in choices:
                print(f"  [{choice.get('key', '')}] {choice.get('label', '')}")
            answer = input("Choose: ").strip()
            for choice in choices:
                if answer == choice.get("key"):
                    return {"decision": choice.get("decision", "allow_once")}
        answer = input("Allow? (y/n): ").strip().lower()
        return {"decision": "allow_once" if answer in ("y", "yes") else "deny_once"}
    return _prompt


def _render_banner(runtime: dict | None, cwd: str, permission_summary: list[str], counts: dict[str, int]) -> str:
    model = runtime["model"] if runtime else "unconfigured"
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║  MiniCode Python - Your Terminal Coding Assistant       ║",
        "╠══════════════════════════════════════════════════════════╣",
        f"║  Model: {model:<46} ║",
        f"║  CWD: {cwd:<50} ║",
    ]
    for perm in permission_summary[:2]:
        lines.append(f"║  {perm:<60} ║")
    lines.append("╠══════════════════════════════════════════════════════════╣")
    lines.append(
        f"║  Skills: {counts['skillCount']:>2} | MCP: {counts['mcpCount']:>2} | "
        f"Transcript: {counts['transcriptCount']:>3}                    ║"
    )
    lines.append("╚══════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def _append_transcript(transcript: list[TranscriptEntry], **kwargs) -> None:
    transcript.append(TranscriptEntry(id=len(transcript) + 1, **kwargs))


def _save_transcript_file(cwd: str, permissions, transcript: list[TranscriptEntry], output_path: str) -> str:
    target = resolve_tool_path(ToolContext(cwd=cwd, permissions=permissions), output_path, "write")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_transcript_text(transcript), encoding="utf-8")
    return str(target)


def _build_system_message(
    cwd: str,
    permissions,
    tools,
    memory_mgr: MemoryManager | None = None,
    agent_registry: AgentRegistry | None = None,
) -> dict:
    extras = {
        "skills": tools.get_skills(),
        "mcpServers": tools.get_mcp_servers(),
    }
    if memory_mgr:
        extras["memory_context"] = memory_mgr.get_relevant_context()
    return {"role": "system", "content": build_system_prompt(cwd, permissions.get_summary(), extras, agent_registry=agent_registry)}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _load_runtime(cwd: str) -> dict | None:
    try:
        return load_runtime_config(cwd)
    except Exception as e:
        print(
            f"Warning: Failed to load runtime config: {e}\n"
            "How to fix:\n"
            "  1. export ANTHROPIC_MODEL=claude-sonnet-4-20250514\n"
            "  2. export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Or edit ~/.mini-code/settings.json\n"
            "Falling back to mock model.\n",
            file=sys.stderr,
        )
        return None


def _init_components(args: argparse.Namespace, cwd: str):
    """Initialise all runtime components and return them as a bundle."""
    runtime = _load_runtime(cwd)

    prompt_handler = _make_cli_permission_prompt() if sys.stdin.isatty() else None
    tools = create_default_tool_registry(cwd, runtime=runtime)
    permissions = PermissionManager(cwd, prompt=prompt_handler)
    model = create_model_adapter(
        model=runtime.get("model", "") if runtime else "",
        tools=tools,
        runtime=runtime,
        force_mock=runtime is None,
    )

    load_hooks_from_config(runtime or {}, cwd)

    context_mgr = ContextManager(model=runtime.get("model", "default")) if runtime else None
    memory_mgr = MemoryManager(project_root=Path(cwd))
    agent_registry = discover_agents(cwd)

    app_store = create_app_store(initial={
        "session_id": args.session or "new",
        "workspace": cwd,
        "model": runtime.get("model", "mock") if runtime else "mock",
    })

    logger.info(
        "Components initialised (model=%s, agents=%d)",
        app_store.get_state().model,
        len(agent_registry.list()),
    )
    return runtime, tools, permissions, model, context_mgr, memory_mgr, agent_registry, app_store


# ---------------------------------------------------------------------------
# CLI mode (stdin pipe)
# ---------------------------------------------------------------------------

def _run_cli_mode(
    *,
    cwd: str,
    runtime: dict | None,
    tools,
    permissions,
    model,
    context_mgr,
    app_store,
    messages: list,
    history: list,
    transcript: list[TranscriptEntry],
    agent_registry: AgentRegistry | None = None,
) -> None:
    for raw_input in sys.stdin:
        user_input = raw_input.strip()
        if not user_input:
            continue
        if user_input == "/exit":
            break

        if user_input.startswith("/transcript-save "):
            output_path = user_input[len("/transcript-save "):].strip()
            if not output_path:
                print("Usage: /transcript-save <path>")
                continue
            print(f"Saved transcript to {_save_transcript_file(cwd, permissions, transcript, output_path)}")
            continue

        # Slash commands
        local_result = try_handle_local_command(user_input, tools=tools, agent_registry=agent_registry)
        if user_input == "/tools":
            local_result = "\n".join(f"{t.name}: {t.description}" for t in tools.list())
        if local_result is not None:
            _append_transcript(transcript, kind="user", body=user_input)
            _append_transcript(transcript, kind="assistant", body=local_result)
            print(local_result)
            continue

        # Tool shortcuts (e.g. /read path)
        shortcut = parse_local_tool_shortcut(user_input)
        if shortcut is not None:
            _append_transcript(transcript, kind="user", body=user_input)
            result = tools.execute(shortcut["toolName"], shortcut["input"], context=ToolContext(cwd=cwd, permissions=permissions))
            _append_transcript(transcript, kind="tool", body=result.output, toolName=shortcut["toolName"], status="success" if result.ok else "error")
            print(result.output)
            continue

        # Agent turn
        _append_transcript(transcript, kind="user", body=user_input)
        messages.append({"role": "user", "content": user_input})
        history.append(user_input)
        save_history_entries(history)

        messages[0] = _build_system_message(cwd, permissions, tools, agent_registry=agent_registry)
        permissions.begin_turn()
        messages = run_agent_turn(
            model=model,
            tools=tools,
            messages=messages,
            cwd=cwd,
            permissions=permissions,
            store=app_store,
            context_manager=context_mgr,
            runtime=runtime,
            agent_registry=agent_registry,
        )
        permissions.end_turn()

        last = next((m for m in reversed(messages) if m["role"] == "assistant"), None)
        if last:
            _append_transcript(transcript, kind="assistant", body=last["content"])
            print(last["content"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    setup_logging(level=args.log_level)

    if args.validate_config:
        from reconstructed_minicode.config import format_config_diagnostic
        print(format_config_diagnostic())
        return

    if args.install:
        from reconstructed_minicode.cli.install import main as install_main
        install_main()
        return

    raw_cwd = args.cwd or args.path
    if raw_cwd:
        target = Path(raw_cwd).expanduser().resolve()
        if not target.is_dir():
            print(f"Error: '{raw_cwd}' is not a directory.", file=sys.stderr)
            sys.exit(1)
        cwd = str(target)
    else:
        cwd = str(Path.cwd())
    argv = sys.argv[1:]
    management_argv = [argv[0]] if argv and not argv[0].startswith("--") else []
    if maybe_handle_management_command(cwd, management_argv):
        return

    runtime, tools, permissions, model, context_mgr, memory_mgr, agent_registry, app_store = _init_components(args, cwd)

    messages = [_build_system_message(cwd, permissions, tools, memory_mgr, agent_registry)]
    history = load_history_entries()
    transcript: list[TranscriptEntry] = []

    print(_render_banner(
        runtime, cwd, permissions.get_summary(),
        {"transcriptCount": 0, "skillCount": len(tools.get_skills()), "mcpCount": len(tools.get_mcp_servers())},
    ))

    try:
        if not sys.stdin.isatty():
            _run_cli_mode(
                cwd=cwd, runtime=runtime, tools=tools, permissions=permissions,
                model=model, context_mgr=context_mgr, app_store=app_store,
                messages=messages, history=history, transcript=transcript,
                agent_registry=agent_registry,
            )
        else:
            run_tty_app(
                runtime=runtime, tools=tools, model=model, messages=messages,
                cwd=cwd, permissions=permissions,
                resume_session=args.resume,
                list_sessions_only=args.list_sessions,
                agent_registry=agent_registry,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            tools.dispose()
        except Exception as e:
            logger.warning("Error disposing tools: %s", e)


if __name__ == "__main__":
    main()
