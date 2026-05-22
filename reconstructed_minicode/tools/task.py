"""Task tool — spawn a sub-agent to handle complex multi-step tasks.

Inspired by Claude Code's Task tool which launches an independent agent loop
with its own context window, isolated from the main conversation.

The sub-agent runs a full agent loop (model + tools) with:
- Its own system prompt tailored to the task type
- A filtered tool set based on the agent type
- A turn limit to prevent runaway execution
- Result summarized back into the parent context
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from reconstructed_minicode.tools.base import ToolDefinition, ToolResult
from reconstructed_minicode.extensions.agent_loader import discover_agents
from reconstructed_minicode.model.registry import create_model_adapter
from reconstructed_minicode.config import load_runtime_config
from reconstructed_minicode.tools.base import ToolRegistry
# ---------------------------------------------------------------------------
# Agent type definitions
# ---------------------------------------------------------------------------

AGENT_TYPES = {
    "explore": {
        "name": "Explore",
        "description": "Fast, read-only agent for codebase exploration and search",
        "system_prompt": (
            "You are an exploration agent. Your job is to quickly search and "
            "understand codebases. You should be fast and focused on finding "
            "relevant files and understanding structure. "
            "You can only use read-only tools. "
            "When done, provide a concise summary of your findings."
        ),
        "allowed_tools": {"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "get_ast_info"},
        "max_turns": 5,
    },
    "plan": {
        "name": "Plan",
        "description": "Thorough agent for gathering context and understanding code",
        "system_prompt": (
            "You are a planning agent. Your job is to thoroughly understand "
            "the codebase and task before acting. Read multiple files, trace "
            "code paths, and build a complete mental model. "
            "You can only use read-only tools. "
            "When done, provide a detailed analysis with actionable recommendations."
        ),
        "allowed_tools": {"read_file", "list_files", "grep_files", "file_tree", "find_symbols", "find_references", "get_ast_info", "code_review"},
        "max_turns": 8,
    },
    "general": {
        "name": "General",
        "description": "Full-featured agent for complex multi-step tasks",
        "system_prompt": (
            "You are a general-purpose coding agent. You can read, write, "
            "and modify code. Follow best practices and explain your changes. "
            "Break complex tasks into smaller steps. "
            "When done, provide a summary of what you did and any important findings."
        ),
        "allowed_tools": None,  # None = all tools allowed
        "max_turns": 15,
    },
}


def _validate(input_data: dict) -> dict:
    description = input_data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")

    agent_type = input_data.get("agent_type", "general")
    # Built-in types are validated here; custom names are validated at runtime
    # against the AgentRegistry (loaded from .mini-code/agents/)
    if agent_type not in AGENT_TYPES:
        # Treat as a custom agent name — validated in _run against registry
        pass

    return {
        "description": description.strip(),
        "agent_type": agent_type,
        "prompt": input_data.get("prompt", description.strip()),
    }


def _run(input_data: dict, context) -> ToolResult:
    """Execute a sub-agent task.

    Supports both built-in agent types (explore/plan/general) and custom
    agents loaded from .mini-code/agents/*.md via AgentRegistry.
    """


    agent_type = input_data["agent_type"]
    task_prompt = input_data["prompt"]

    # --- Resolve agent definition (built-in or custom) ---
    if agent_type in AGENT_TYPES:
        raw = AGENT_TYPES[agent_type]
        agent_def = {
            "name": raw["name"],
            "description": raw["description"],
            "system_prompt": raw["system_prompt"],
            "allowed_tools": raw["allowed_tools"],
            "model_override": None,
            "max_turns": raw["max_turns"],
        }
    else:
        # Look up custom agent from registry (injected via _runtime)
        registry = None
        if hasattr(context, "_runtime") and context._runtime:
            registry = context._runtime.get("_agent_registry")
        if registry is None:
            registry = discover_agents(context.cwd)

        custom = registry.find(agent_type)
        if custom is None:
            valid = ", ".join(AGENT_TYPES.keys()) + (
                (", " + ", ".join(registry.names())) if registry.names() else ""
            )
            return ToolResult(
                ok=False,
                output=f"Unknown agent type '{agent_type}'. Available: {valid}",
            )
        agent_def = {
            "name": custom.name,
            "description": custom.description,
            "system_prompt": custom.system_prompt,
            "allowed_tools": custom.allowed_tools,
            "model_override": custom.model,
            "max_turns": custom.max_turns,
        }
    
    # --- Resolve runtime config ---
    runtime = (getattr(context, "_runtime", None) or {})
    if not runtime:
        try:

            runtime = load_runtime_config(context.cwd)
        except Exception:
            pass
    if not runtime:
        return ToolResult(
            ok=False,
            output="Cannot run sub-agent: no model configuration available. Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL.",
        )

    # --- Build tool registry ---
    from reconstructed_minicode.tools import create_default_tool_registry
    full_tools = create_default_tool_registry(context.cwd, runtime=runtime)
    allowed = agent_def["allowed_tools"]
    if allowed is not None:
        tools = ToolRegistry([t for t in full_tools.list() if t.name in set(allowed)])
    else:
        tools = full_tools

    # --- Create model adapter (support per-agent model override) ---
    model_name = agent_def.get("model_override") or runtime.get("model", "")
    model = create_model_adapter(model=model_name, tools=tools, runtime=runtime)

    # --- Permissions ---
    from reconstructed_minicode.security.permissions import PermissionManager
    if allowed is not None:
        sub_permissions = PermissionManager(context.cwd, prompt=None)
    else:
        sub_permissions = PermissionManager(
            context.cwd,
            prompt=getattr(context.permissions, "prompt", None),
        )

    # --- Build isolated message list ---
    sub_messages = [
        {
            "role": "system",
            "content": (
                agent_def["system_prompt"]
                + f"\n\nCurrent cwd: {context.cwd}"
                + "\n\nIMPORTANT: When you have completed your task, end with <final> and provide your findings."
                + " Do not ask the user questions — work autonomously with the tools available."
                + " Be concise and focused."
            ),
        },
        {"role": "user", "content": task_prompt},
    ]

    # --- Run ---
    start_time = time.time()
    max_turns = agent_def["max_turns"]
    try:
        from reconstructed_minicode.agent.loop import run_agent_turn
        result_messages = run_agent_turn(
            model=model,
            tools=tools,
            messages=sub_messages,
            cwd=context.cwd,
            permissions=sub_permissions,
            max_steps=max_turns,
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            output=f"Sub-agent ({agent_def['name']}) failed: {type(e).__name__}: {e}",
        )

    elapsed = time.time() - start_time

    # --- Extract result ---
    final_message = next(
        (m["content"] for m in reversed(result_messages)
         if m.get("role") == "assistant" and (m.get("content") or "").strip()),
        "(sub-agent completed without a final message)",
    )

    tool_calls_count = sum(1 for m in result_messages if m.get("tool_calls"))
    user_messages_count = sum(1 for m in result_messages if m.get("role") == "user")

    header = (
        f"[Sub-agent '{agent_def['name']}' completed]\n"
        f"  Turns: {user_messages_count}  Tool calls: {tool_calls_count}  Duration: {elapsed:.1f}s\n"
    )

    MAX_RESULT_LEN = 8000
    result_text = final_message
    if len(result_text) > MAX_RESULT_LEN:
        result_text = result_text[:MAX_RESULT_LEN] + f"\n\n... (truncated, {len(final_message)} chars total)"

    return ToolResult(ok=True, output=header + "\n" + result_text)


task_tool = ToolDefinition(
    name="task",
    description=(
        "Launch a sub-agent to handle a complex task autonomously. "
        "The sub-agent runs in its own isolated context with a turn limit. "
        "Built-in types: 'explore' (fast read-only search), 'plan' (thorough analysis), 'general' (full tools). "
        "Custom agents defined in .mini-code/agents/ can be referenced by name. "
        "The sub-agent's final result is returned to you."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short 3-5 word description of the task",
            },
            "prompt": {
                "type": "string",
                "description": "Full task description for the sub-agent. If not provided, uses 'description'.",
            },
            "agent_type": {
                "type": "string",
                "description": (
                    "Agent type to use. Built-in: 'explore' (fast, read-only), "
                    "'plan' (thorough, read-only), 'general' (full tools, default). "
                    "Or use the name of a custom agent from .mini-code/agents/."
                ),
            },
        },
        "required": ["description"],
    },
    validator=_validate,
    run=_run,
)
