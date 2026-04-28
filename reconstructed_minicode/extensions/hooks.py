"""Hooks event system for MiniCode Python.

Inspired by Claude Code's hooks system (PreToolUse, PostToolUse, Stop, etc.)
and plugin event listeners.

Provides lifecycle hooks for:
- Tool execution (pre/post)
- Agent lifecycle (start/stop)
- Session events (save/resume)
- User interactions (input/output)

Hooks can trigger external scripts, logging, or custom behaviors.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Hook events
# ---------------------------------------------------------------------------

class HookEvent(str, Enum):
    """Lifecycle hook events."""
    # Tool lifecycle
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    STOP = "stop"                        # Agent completed final response
    SUBAGENT_START = "subagent_start"
    SUBAGENT_STOP = "subagent_stop"

    # Session events
    SESSION_SAVE = "session_save"
    SESSION_RESUME = "session_resume"

    # User interactions
    USER_INPUT = "user_input"
    ASSISTANT_OUTPUT = "assistant_output"

    # System
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


# ---------------------------------------------------------------------------
# Hook context
# ---------------------------------------------------------------------------

@dataclass
class HookContext:
    """Context passed to hook handlers."""
    event: HookEvent
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def tool_name(self) -> str | None:
        return self.data.get("tool_name")
    
    @property
    def tool_input(self) -> Any:
        return self.data.get("tool_input")
    
    @property
    def tool_output(self) -> str | None:
        return self.data.get("tool_output")
    
    @property
    def is_error(self) -> bool:
        return self.data.get("is_error", False)
    
    @property
    def session_id(self) -> str | None:
        return self.data.get("session_id")
    
    @property
    def user_input(self) -> str | None:
        return self.data.get("user_input")
    
    @property
    def assistant_output(self) -> str | None:
        return self.data.get("assistant_output")


# ---------------------------------------------------------------------------
# Hook handler
# ---------------------------------------------------------------------------

HookHandler = Callable[[HookContext], None]
AsyncHookHandler = Callable[[HookContext], Any]


@dataclass
class HookRegistration:
    """Registered hook with metadata."""
    event: HookEvent
    handler: HookHandler | AsyncHookHandler
    is_async: bool = False
    enabled: bool = True
    description: str = ""
    created_at: float = field(default_factory=time.time)
    call_count: int = 0
    last_called: float | None = None
    total_duration_ms: int = 0


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class HookManager:
    """Manages hook registrations and executions.
    
    Inspired by Claude Code's hooks system and plugin event listeners.
    """
    
    def __init__(self):
        self._hooks: dict[HookEvent, list[HookRegistration]] = {
            event: [] for event in HookEvent
        }
        self._enabled = True
    
    def register(
        self,
        event: HookEvent,
        handler: HookHandler | AsyncHookHandler,
        description: str = "",
    ) -> Callable[[], None]:
        """Register a hook for an event.
        
        Args:
            event: Event to hook into
            handler: Handler function (sync or async)
            description: Human-readable description
        
        Returns:
            Unregister function
        """
        import asyncio
        
        registration = HookRegistration(
            event=event,
            handler=handler,
            is_async=asyncio.iscoroutinefunction(handler),
            description=description,
        )
        
        self._hooks[event].append(registration)
        
        def unregister():
            if registration in self._hooks[event]:
                self._hooks[event].remove(registration)
        
        return unregister
    
    async def fire(self, event: HookEvent, **kwargs: Any) -> list[Any]:
        """Fire an event, calling all registered hooks.
        
        Args:
            event: Event to fire
            **kwargs: Data to pass to hooks
        
        Returns:
            List of hook results
        """
        if not self._enabled:
            return []
        
        context = HookContext(event=event, data=kwargs)
        results = []
        
        for registration in self._hooks[event]:
            if not registration.enabled:
                continue
            
            start_time = time.time()
            try:
                if registration.is_async:
                    result = await registration.handler(context)
                else:
                    result = registration.handler(context)
                
                registration.call_count += 1
                registration.last_called = time.time()
                
                duration_ms = int((time.time() - start_time) * 1000)
                registration.total_duration_ms += duration_ms
                
                results.append(result)
            
            except Exception as e:
                # Don't let hook errors break main flow
                results.append(f"Hook error: {e}")
        
        return results
    
    def fire_sync(self, event: HookEvent, **kwargs: Any) -> list[Any]:
        """Fire event synchronously (for sync hooks only)."""
        if not self._enabled:
            return []
        
        context = HookContext(event=event, data=kwargs)
        results = []
        
        for registration in self._hooks[event]:
            if not registration.enabled or registration.is_async:
                continue
            
            start_time = time.time()
            try:
                result = registration.handler(context)
                registration.call_count += 1
                registration.last_called = time.time()
                
                duration_ms = int((time.time() - start_time) * 1000)
                registration.total_duration_ms += duration_ms
                
                results.append(result)
            
            except Exception as e:
                results.append(f"Hook error: {e}")
        
        return results
    
    def enable(self) -> None:
        """Enable all hooks."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable all hooks."""
        self._enabled = False
    
    def get_hook_stats(self, event: HookEvent | None = None) -> dict[str, Any]:
        """Get hook execution statistics."""
        if event:
            hooks = self._hooks.get(event, [])
        else:
            hooks = [h for hooks_list in self._hooks.values() for h in hooks_list]
        
        return {
            "total_hooks": len(hooks),
            "enabled_hooks": sum(1 for h in hooks if h.enabled),
            "total_calls": sum(h.call_count for h in hooks),
            "total_duration_ms": sum(h.total_duration_ms for h in hooks),
        }
    
    def format_hook_status(self) -> str:
        """Format hook status for display."""
        lines = ["Hooks Status", "=" * 50, ""]
        
        for event in HookEvent:
            hooks = self._hooks[event]
            if not hooks:
                continue
            
            lines.append(f"{event.value}:")
            for hook in hooks:
                status = "✓" if hook.enabled else "✗"
                lines.append(
                    f"  {status} {hook.description or hook.handler.__name__} "
                    f"({hook.call_count} calls, {hook.total_duration_ms}ms)"
                )
            lines.append("")
        
        stats = self.get_hook_stats()
        lines.extend([
            "-" * 50,
            f"Total hooks: {stats['total_hooks']}",
            f"Enabled: {stats['enabled_hooks']}",
            f"Total calls: {stats['total_calls']}",
            f"Total duration: {stats['total_duration_ms']}ms",
        ])
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in hooks
# ---------------------------------------------------------------------------

def create_logging_hook(log_file: Path | None = None) -> HookHandler:
    """Create a logging hook that records all events.
    
    Args:
        log_file: Optional file to log to
    
    Returns:
        Hook handler function
    """
    def handler(ctx: HookContext) -> None:
        timestamp = time.strftime("%H:%M:%S", time.localtime(ctx.timestamp))
        message = f"[{timestamp}] {ctx.event.value}"
        
        if ctx.tool_name:
            message += f" tool={ctx.tool_name}"
        if ctx.session_id:
            message += f" session={ctx.session_id[:8]}"
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
    
    return handler


def create_script_hook(script_path: Path) -> AsyncHookHandler:
    """Create a hook that executes an external script.
    
    Args:
        script_path: Path to script to execute
    
    Returns:
        Async hook handler function
    """
    async def handler(ctx: HookContext) -> str:
        try:
            # On Windows, CreateProcess can't directly execute script files
            # (.py, .sh, etc.).  Detect the script type and invoke through
            # the appropriate interpreter / shell.
            script_str = str(script_path)
            suffix = script_path.suffix.lower()
            if sys.platform == "win32" and suffix in (".py", ".sh", ".bat", ".cmd", ".ps1"):
                if suffix == ".py":
                    cmd_prefix = [sys.executable, script_str]
                elif suffix in (".bat", ".cmd"):
                    cmd_prefix = ["cmd", "/c", script_str]
                elif suffix == ".ps1":
                    cmd_prefix = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_str]
                else:
                    # .sh on Windows — try bash if available, fall back to sh
                    cmd_prefix = ["bash", script_str]
            else:
                cmd_prefix = [script_str]

            process = await asyncio.create_subprocess_exec(
                *cmd_prefix,
                ctx.event.value,
                *([str(v) for v in ctx.data.values()]),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode("utf-8", errors="replace")
            else:
                return f"Script failed: {stderr.decode('utf-8', errors='replace')}"
        
        except Exception as e:
            return f"Script execution failed: {e}"
    
    return handler


# ---------------------------------------------------------------------------
# Config-driven hooks (Claude Code-style settings.json format)
# ---------------------------------------------------------------------------

HOOK_BLOCKED = "__hook_blocked__"
HOOK_TIMEOUT_SECONDS = 10

_CONFIG_EVENT_MAP: dict[str, HookEvent] = {
    "PreToolUse": HookEvent.PRE_TOOL_USE,
    "PostToolUse": HookEvent.POST_TOOL_USE,
    "PostToolUseFailure": HookEvent.POST_TOOL_USE_FAILURE,
    "Stop": HookEvent.STOP,
    "AgentStart": HookEvent.AGENT_START,
    "AgentStop": HookEvent.AGENT_STOP,
    "SubagentStart": HookEvent.SUBAGENT_START,
    "SubagentStop": HookEvent.SUBAGENT_STOP,
    "UserInput": HookEvent.USER_INPUT,
    "AssistantOutput": HookEvent.ASSISTANT_OUTPUT,
    "SessionSave": HookEvent.SESSION_SAVE,
    "SessionResume": HookEvent.SESSION_RESUME,
    "Startup": HookEvent.STARTUP,
    "Shutdown": HookEvent.SHUTDOWN,
}


_SIMPLE_MATCHER_RE = re.compile(r'^[A-Za-z0-9_|]+$')


def _matches_tool(tool_name: str | None, matcher: str) -> bool:
    """Match tool name against a matcher pattern.

    Rules (mirrors Claude Code):
    - empty / "*"          → always match
    - only [A-Za-z0-9_|]  → exact string or |-separated list
    - anything else        → treat as regex
    """
    if not matcher or matcher == "*":
        return True
    if not tool_name:
        return True
    if _SIMPLE_MATCHER_RE.match(matcher):
        return tool_name in {s.strip() for s in matcher.split("|")}
    try:
        return bool(re.search(matcher, tool_name))
    except re.error:
        return fnmatch.fnmatch(tool_name.lower(), matcher.lower())


def _parse_if_condition(condition: str) -> tuple[str | None, str | None]:
    """Parse 'ToolName(pattern)' into (tool_name_pattern, input_pattern)."""
    m = re.match(r'^(\w+)\((.+)\)$', condition.strip())
    if m:
        return m.group(1), m.group(2)
    return condition.strip() or None, None


def _check_if_condition(condition: str | None, ctx: "HookContext") -> bool:
    if not condition:
        return True
    tool_pat, input_pat = _parse_if_condition(condition)
    if tool_pat and ctx.tool_name:
        if not fnmatch.fnmatch(ctx.tool_name.lower(), tool_pat.lower()):
            return False
    if input_pat and ctx.tool_input is not None:
        input_str = json.dumps(ctx.tool_input) if not isinstance(ctx.tool_input, str) else ctx.tool_input
        if not fnmatch.fnmatch(input_str.lower(), f"*{input_pat.lower()}*"):
            return False
    return True


def _run_hook_command(command: str, ctx: "HookContext", cwd: str) -> tuple[int, str]:
    """Run a hook command. Returns (exit_code, stdout). Exit code 2 means block."""
    env = os.environ.copy()
    env["MINI_CODE_PROJECT_DIR"] = cwd
    env["CLAUDE_PROJECT_DIR"] = cwd
    env["MINI_CODE_EVENT"] = ctx.event.value
    if ctx.tool_name:
        env["MINI_CODE_TOOL_NAME"] = ctx.tool_name
    if ctx.tool_input is not None:
        env["MINI_CODE_TOOL_INPUT"] = json.dumps(ctx.tool_input)
    if ctx.tool_output:
        env["MINI_CODE_TOOL_OUTPUT"] = ctx.tool_output[:4000]
    # Expose file_path directly for format-on-save hooks (Edit/Write tools)
    if isinstance(ctx.tool_input, dict) and "file_path" in ctx.tool_input:
        env["MINI_CODE_FILE_PATH"] = str(ctx.tool_input["file_path"])

    expanded = os.path.expandvars(command)
    # On Windows, .sh files need an explicit interpreter since cmd.exe can't run them.
    if sys.platform == "win32" and expanded.lower().endswith(".sh"):
        expanded = f"bash {expanded}"
    try:
        result = subprocess.run(
            expanded, shell=True, capture_output=True,
            timeout=HOOK_TIMEOUT_SECONDS, env=env, cwd=cwd,
        )
        return result.returncode, result.stdout.decode("utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired:
        return 1, f"Hook timed out after {HOOK_TIMEOUT_SECONDS}s"
    except Exception as exc:
        return 1, f"Hook failed: {exc}"


def _make_command_handler(
    command: str,
    matcher: str,
    condition: str | None,
    cwd: str,
) -> "HookHandler":
    def handler(ctx: "HookContext") -> str | None:
        if not _matches_tool(ctx.tool_name, matcher):
            return None
        if not _check_if_condition(condition, ctx):
            return None
        exit_code, output = _run_hook_command(command, ctx, cwd)
        if exit_code == 2:
            return HOOK_BLOCKED + (f": {output}" if output else "")
        return output or None
    return handler


def load_hooks_from_config(settings: dict[str, Any], cwd: str) -> None:
    """Register hooks declared in the 'hooks' section of settings.json.

    Supported format (mirrors Claude Code):
    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "hooks": [{"type": "command", "if": "Bash(rm *)", "command": "..."}]
          }
        ]
      }
    }

    Exit code 2 from a PreToolUse command blocks the tool call.
    """
    hooks_config = settings.get("hooks", {})
    if not isinstance(hooks_config, dict):
        return

    for event_name, matcher_blocks in hooks_config.items():
        event = _CONFIG_EVENT_MAP.get(event_name)
        if event is None or not isinstance(matcher_blocks, list):
            continue
        for block in matcher_blocks:
            if not isinstance(block, dict):
                continue
            matcher = str(block.get("matcher", "*"))
            for entry in block.get("hooks", []):
                if not isinstance(entry, dict) or entry.get("type") != "command":
                    continue
                command = str(entry.get("command", "")).strip()
                if not command:
                    continue
                condition = entry.get("if") or None
                handler = _make_command_handler(command, matcher, condition, cwd)
                register_hook(event, handler, description=f"{event_name}[{matcher}]: {command[:60]}")


def fire_pre_tool_hook(tool_name: str, tool_input: Any, step: int) -> str | None:
    """Fire PRE_TOOL_USE hooks. Returns block reason if any hook exits with code 2, else None."""
    results = fire_hook_sync(HookEvent.PRE_TOOL_USE, tool_name=tool_name, tool_input=tool_input, step=step)
    for r in results:
        if isinstance(r, str) and r.startswith(HOOK_BLOCKED):
            return r[len(HOOK_BLOCKED):].lstrip(": ") or "Blocked by hook"
    return None


def fire_post_tool_hook(tool_name: str, tool_input: Any, tool_output: str, ok: bool, step: int) -> None:
    """Fire POST_TOOL_USE or POST_TOOL_USE_FAILURE depending on result."""
    event = HookEvent.POST_TOOL_USE if ok else HookEvent.POST_TOOL_USE_FAILURE
    fire_hook_sync(event, tool_name=tool_name, tool_input=tool_input, tool_output=tool_output, step=step)


def fire_stop_hook(step: int, tool_errors: int) -> None:
    """Fire STOP hook after agent completes its final response."""
    fire_hook_sync(HookEvent.STOP, step=step, tool_errors=tool_errors)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_hook_manager = HookManager()


def get_hook_manager() -> HookManager:
    """Get global hook manager."""
    return _hook_manager


def register_hook(
    event: HookEvent,
    handler: HookHandler | AsyncHookHandler,
    description: str = "",
) -> Callable[[], None]:
    """Register a hook (convenience function)."""
    return _hook_manager.register(event, handler, description)


async def fire_hook(event: HookEvent, **kwargs: Any) -> list[Any]:
    """Fire a hook event (convenience function)."""
    return await _hook_manager.fire(event, **kwargs)


def fire_hook_sync(event: HookEvent, **kwargs: Any) -> list[Any]:
    """Fire a hook event synchronously (convenience function)."""
    return _hook_manager.fire_sync(event, **kwargs)
