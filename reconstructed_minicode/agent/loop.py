from __future__ import annotations

import concurrent.futures
import json
from typing import Any, Callable

from reconstructed_minicode.agent.context import ContextManager, estimate_message_tokens
from reconstructed_minicode.agent.state import Store, AppState, increment_tool_calls, set_busy, set_idle
from reconstructed_minicode.extensions.hooks import HookEvent, fire_hook_sync, fire_post_tool_hook, fire_pre_tool_hook, fire_stop_hook
from reconstructed_minicode.security.permissions import PermissionManager
from reconstructed_minicode.tools.base import ToolContext, ToolRegistry, ToolResult
from reconstructed_minicode.types import AgentStep, ChatMessage, ModelAdapter
from reconstructed_minicode.utils.logging import get_logger

from reconstructed_minicode.reliability.agent_metrics import AgentMetricsCollector
from reconstructed_minicode.reliability.agent_intelligence import ErrorClassifier, NudgeGenerator, ToolScheduler

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reconstructed_minicode.extensions.memory_injector import MemoryInjector

logger = get_logger("agent_loop")

# Nudge constants: injected as user messages to recover from stalled turns
NUDGE_CONTINUE = (
    "Continue immediately from your <progress> update with concrete tool calls, "
    "code changes, or an explicit <final> answer only if the task is complete."
)
NUDGE_AFTER_TOOL_RESULT = (
    "Continue from your progress update. You have already used tools in this turn, "
    "so treat plain status text as progress, not a final answer. Respond with the "
    "next concrete tool call, code change, or an explicit <final> answer only if "
    "the task is truly complete."
)
NUDGE_AFTER_EMPTY_RESPONSE = (
    "Your last response was empty after recent tool results. Continue immediately "
    "by trying the next concrete step, adapting to any tool errors, or giving an "
    "explicit <final> answer only if the task is complete."
)
NUDGE_AFTER_EMPTY_NO_TOOLS = (
    "Your last response was empty. Continue immediately with concrete tool calls, "
    "code changes, or an explicit <final> answer only if the task is complete."
)
RESUME_AFTER_PAUSE = (
    "Resume from the previous pause and continue immediately with the next concrete "
    "tool call, code change, or an explicit <final> answer only if the task is complete."
)
RESUME_AFTER_MAX_TOKENS = (
    "Your previous response hit max_tokens during thinking before producing the next "
    "actionable step. Resume immediately and continue with the next concrete tool call, "
    "code change, or an explicit <final> answer only if the task is complete."
)
NUDGE_REPEATED_FAILURE = (
    "You have tried the exact same tool call {tool_name} {count} times and it keeps "
    "failing with the same error. Stop retrying. Read the file at the relevant lines "
    "to see the actual content, then construct a corrected call based on what you see."
)
NUDGE_REPEATED_SAME_SEARCH = (
    "You have run the exact same {tool_name} call {count} times and gotten the same "
    "result every time. This search is not yielding new information — stop repeating it. "
    "Try a different search term, use get_outline or list_files to explore the structure, "
    "or re-read the problem statement to reconsider your approach."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_empty(content: str) -> bool:
    return not content.strip()


def _format_diagnostics(
    stop_reason: str | None,
    block_types: list[str] | None,
    ignored_block_types: list[str] | None,
) -> str:
    parts: list[str] = []
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    if block_types:
        parts.append(f"blocks={','.join(block_types)}")
    if ignored_block_types:
        parts.append(f"ignored={','.join(ignored_block_types)}")
    return f" Diagnostics: {'; '.join(parts)}." if parts else ""


def _is_recoverable_thinking_stop(
    *,
    is_empty: bool,
    stop_reason: str | None,
    ignored_block_types: list[str] | None,
) -> bool:
    return (
        is_empty
        and stop_reason in {"pause_turn", "max_tokens"}
        and "thinking" in (ignored_block_types or [])
    )


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

def _maybe_compact(
    context_manager: ContextManager,
    messages: list[ChatMessage],
    on_assistant_message: Callable[[str], None] | None,
) -> list[ChatMessage]:
    context_manager.messages = messages
    stats = context_manager.get_stats()
    logger.info(
        "Context: %d tokens (%.0f%%), %d messages",
        stats.total_tokens,
        stats.usage_percentage,
        stats.messages_count,
    )
    if context_manager.should_auto_compact():
        logger.warning("Context near limit, auto-compacting...")
        messages = context_manager.compact_messages()
        if on_assistant_message:
            on_assistant_message(context_manager.get_context_summary())
    return messages


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def _call_model(
    model: ModelAdapter,
    messages: list[ChatMessage],
    on_stream_chunk: Callable[[str], None] | None,
    on_assistant_message: Callable[[str], None] | None,
) -> tuple[AgentStep | None, str | None]:
    """Call the model and return (step, None) on success or (None, fallback) on error."""
    try:
        step = model.next(messages, on_stream_chunk=on_stream_chunk)
        return step, None
    except KeyboardInterrupt:
        raise
    except ConnectionError as e:
        fallback = f"Network error (connection failed or dropped): {e}"
        logger.error("Model API connection error: %s", e)
    except TimeoutError as e:
        fallback = f"Model API timeout: {e}"
        logger.error("Model API timeout: %s", e)
    except Exception as e:
        fallback = f"Model API error ({type(e).__name__}): {e}"
        logger.error("Model API error (%s): %s", type(e).__name__, e)

    if on_assistant_message:
        on_assistant_message(fallback)
    return None, fallback


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_single_tool(
    call: dict,
    tools: ToolRegistry,
    cwd: str,
    permissions: Any | None,
    runtime: dict | None,
    store: Any | None,
    step: int,
    on_tool_start: Callable[[str, dict], None] | None,
    on_tool_result: Callable[[str, str, bool], None] | None,
) -> ToolResult:
    """Execute one tool call with crash protection.

    Serial callers pass store and UI callbacks; concurrent callers pass None
    for both so that UI updates are deferred to the result-processing phase.
    """
    tool_name = call["toolName"]
    tool_input = call["input"]

    try:
        if on_tool_start:
            on_tool_start(tool_name, tool_input)
        if store:
            store.set_state(set_busy(tool_name))

        block_reason = fire_pre_tool_hook(tool_name, tool_input, step)
        if block_reason:
            if store:
                store.set_state(set_idle())
            return ToolResult(ok=False, output=f"Blocked by hook: {block_reason}")

        result = tools.execute(
            tool_name,
            tool_input,
            ToolContext(cwd=cwd, permissions=permissions, _runtime=runtime),
        )

        if store:
            store.set_state(increment_tool_calls())
            store.set_state(set_idle())
        if on_tool_result:
            on_tool_result(tool_name, result.output, not result.ok)

        return result

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]).strip()
        logger.error("Tool pipeline crashed (%s): %s", type(exc).__name__, exc)
        if store:
            try:
                store.set_state(set_idle())
            except Exception:
                pass
        return ToolResult(
            ok=False,
            output=f"[{type(exc).__name__}] Tool pipeline crashed: {exc}\nTraceback:\n{tb}",
        )


def _run_tool_calls(
    calls: list[dict],
    tools: ToolRegistry,
    cwd: str,
    permissions: Any | None,
    runtime: dict | None,
    store: Any | None,
    step: int,
    on_tool_start: Callable[[str, dict], None] | None,
    on_tool_result: Callable[[str, str, bool], None] | None,
    tool_scheduler: ToolScheduler | None = None,
    metrics_collector: AgentMetricsCollector | None = None,
) -> list[tuple[dict, ToolResult]]:
    """Partition calls into concurrent-safe and serial, execute both, return ordered results."""
    if len(calls) == 1:
        if metrics_collector:
            metrics_collector.start_tool(calls[0]["toolName"])
        result = _execute_single_tool(
            calls[0], tools, cwd, permissions, runtime, store, step,
            on_tool_start, on_tool_result,
        )
        if metrics_collector:
            metrics_collector.end_tool(success=result.ok, error=result.output if not result.ok else "")
        return [(calls[0], result)]

    # Use ToolScheduler for intelligent partitioning if available, else simple flag check
    if tool_scheduler:
        concurrent_calls, serial_calls = tool_scheduler.schedule_calls(calls, tools)
        max_workers = tool_scheduler.get_recommended_max_workers(concurrent_calls)
    else:
        concurrent_calls = [c for c in calls if (t := tools.find(c["toolName"])) and t.is_concurrency_safe]
        serial_calls = [c for c in calls if not ((t := tools.find(c["toolName"])) and t.is_concurrency_safe)]
        max_workers = min(len(concurrent_calls), 8)

    results: list[tuple[dict, ToolResult]] = []

    if concurrent_calls:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mc-tool",
        ) as pool:
            future_map = {
                pool.submit(
                    _execute_single_tool,
                    call, tools, cwd, permissions, runtime,
                    None, step, None, None,
                ): call
                for call in concurrent_calls
            }
            for future in concurrent.futures.as_completed(future_map):
                call = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ToolResult(ok=False, output=f"Concurrent execution error: {exc}")
                results.append((call, result))

    for call in serial_calls:
        if metrics_collector:
            metrics_collector.start_tool(call["toolName"])
        result = _execute_single_tool(
            call, tools, cwd, permissions, runtime, store, step,
            on_tool_start, on_tool_result,
        )
        if metrics_collector:
            metrics_collector.end_tool(success=result.ok, error=result.output if not result.ok else "")
        results.append((call, result))
        if result.awaitUser:
            break

    # Record conflicts between concurrent tools that both failed
    if tool_scheduler:
        failed_concurrent = [call for call, res in results if not res.ok and call in concurrent_calls]
        for i, call_a in enumerate(failed_concurrent):
            for call_b in failed_concurrent[i + 1:]:
                tool_scheduler.record_conflict(call_a["toolName"], call_b["toolName"])

    # Restore original call order
    order = {call["id"]: i for i, call in enumerate(calls)}
    results.sort(key=lambda pair: order.get(pair[0]["id"], 999))
    return results


def _append_tool_messages(
    call: dict,
    result: ToolResult,
    messages: list[ChatMessage],
    is_concurrent: bool,
    step: int,
    store: Any | None,
    on_tool_start: Callable[[str, dict], None] | None,
    on_tool_result: Callable[[str, str, bool], None] | None,
) -> None:
    """Fire deferred hooks/UI for concurrent tools, then append call+result messages."""
    if is_concurrent:
        if on_tool_start:
            on_tool_start(call["toolName"], call["input"])
        if store:
            store.set_state(set_busy(call["toolName"]))
            store.set_state(increment_tool_calls())
            store.set_state(set_idle())
        fire_hook_sync(
            HookEvent.PRE_TOOL_USE,
            tool_name=call["toolName"],
            tool_input=call["input"],
            step=step,
        )

    fire_post_tool_hook(call["toolName"], call["input"], result.output, result.ok, step)

    if is_concurrent and on_tool_result:
        on_tool_result(call["toolName"], result.output, not result.ok)

    messages.append({
        "role": "assistant_tool_call",
        "toolUseId": call["id"],
        "toolName": call["toolName"],
        "input": call["input"],
    })
    messages.append({
        "role": "tool_result",
        "toolUseId": call["id"],
        "toolName": call["toolName"],
        "content": result.output,
        "isError": not result.ok,
    })


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent_turn(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    messages: list[ChatMessage],
    cwd: str,
    permissions: PermissionManager | None = None,
    store: Store[AppState] | None = None,
    max_steps: int = 60,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_result: Callable[[str, str, bool], None] | None = None,
    on_assistant_message: Callable[[str], None] | None = None,
    on_progress_message: Callable[[str], None] | None = None,
    on_assistant_stream_chunk: Callable[[str], None] | None = None,
    context_manager: ContextManager | None = None,
    runtime: dict | None = None,
    metrics_collector: AgentMetricsCollector | None = None,
    memory_injector: MemoryInjector | None = None,
) -> list[ChatMessage]:
    current_messages = list(messages)
    saw_tool_result = False
    empty_retry = 0
    thinking_retry = 0
    tool_error_count = 0
    step = 0
    # Loop detection: track total frequency of each call key within this turn
    _call_freq: dict[str, int] = {}
    _nudged_keys: set[str] = set()

    tool_scheduler = ToolScheduler(metrics_collector=metrics_collector)

    if context_manager:
        current_messages = _maybe_compact(context_manager, current_messages, on_assistant_message)

    try:
        while step < max_steps:
            step += 1
            fire_hook_sync(HookEvent.AGENT_START, step=step, cwd=cwd)

            next_step, fallback = _call_model(
                model, current_messages, on_assistant_stream_chunk, on_assistant_message,
            )
            if next_step is None:
                current_messages.append({"role": "assistant", "content": fallback})
                return current_messages

            # --- Pure assistant response (no tool calls) ---
            if next_step.type == "assistant":
                content = next_step.content
                kind = getattr(next_step, "kind", None)
                is_empty = _is_empty(content)
                diagnostics = next_step.diagnostics

                # Progress message: keep looping
                if not is_empty and kind == "progress":
                    if on_progress_message:
                        on_progress_message(content)
                    current_messages.append({"role": "assistant_progress", "content": content})
                    current_messages.append({"role": "user", "content": NUDGE_CONTINUE})
                    continue

                # Thinking was truncated: recover up to 3 times
                stop_reason = diagnostics.stopReason if diagnostics else None
                ignored = diagnostics.ignoredBlockTypes if diagnostics else None
                if _is_recoverable_thinking_stop(is_empty=is_empty, stop_reason=stop_reason, ignored_block_types=ignored) and thinking_retry < 3:
                    thinking_retry += 1
                    msg = (
                        "Model hit max_tokens during thinking; requesting the next step."
                        if stop_reason == "max_tokens"
                        else "Model returned pause_turn; requesting the next step."
                    )
                    if on_progress_message:
                        on_progress_message(msg)
                    current_messages.append({"role": "assistant_progress", "content": msg})
                    current_messages.append({
                        "role": "user",
                        "content": RESUME_AFTER_MAX_TOKENS if stop_reason == "max_tokens" else RESUME_AFTER_PAUSE,
                    })
                    continue

                # Empty response: nudge up to 2 times
                if is_empty and empty_retry < 2:
                    empty_retry += 1
                    current_messages.append({
                        "role": "user",
                        "content": NUDGE_AFTER_EMPTY_RESPONSE if saw_tool_result else NUDGE_AFTER_EMPTY_NO_TOOLS,
                    })
                    continue

                # Empty and out of retries: report and stop
                if is_empty:
                    diag = _format_diagnostics(stop_reason, diagnostics.blockTypes if diagnostics else None, ignored)
                    if saw_tool_result:
                        fallback = (
                            f"Model returned an empty response after tool execution and the turn was stopped. "
                            f"There were {tool_error_count} tool error(s); retry or choose a different approach.{diag}"
                            if tool_error_count > 0
                            else f"Model returned an empty response after tool execution and the turn was stopped. "
                            f"Retry or ask the model to continue the remaining steps.{diag}"
                        )
                    else:
                        fallback = f"Model returned an empty response and the turn was stopped.{diag}"
                    if on_assistant_message:
                        on_assistant_message(fallback)
                    current_messages.append({"role": "assistant", "content": fallback})
                    return current_messages

                # Normal final response
                if on_assistant_message:
                    on_assistant_message(content)
                current_messages.append({"role": "assistant", "content": content})
                return current_messages

            # --- Tool calls ---
            if next_step.content:
                role = "assistant_progress" if next_step.contentKind == "progress" else "assistant"
                if role == "assistant_progress":
                    if on_progress_message:
                        on_progress_message(next_step.content)
                    current_messages.append({"role": role, "content": next_step.content})
                    current_messages.append({"role": "user", "content": NUDGE_CONTINUE})
                else:
                    if on_assistant_message:
                        on_assistant_message(next_step.content)
                    current_messages.append({"role": role, "content": next_step.content})

            if not next_step.calls and next_step.content and next_step.contentKind != "progress":
                return current_messages

            calls = next_step.calls
            if metrics_collector:
                metrics_collector.start_turn(step)
            results = _run_tool_calls(
                calls, tools, cwd, permissions, runtime, store, step,
                on_tool_start, on_tool_result,
                tool_scheduler=tool_scheduler,
                metrics_collector=metrics_collector,
            )
            if metrics_collector:
                total_tokens = sum(
                    estimate_message_tokens(m) for m in current_messages
                ) if context_manager else 0
                metrics_collector.end_turn(total_tokens=total_tokens)

            for call, result in results:
                tool_def = tools.find(call["toolName"])
                is_concurrent = bool(tool_def and tool_def.is_concurrency_safe and len(calls) > 1)
                if not result.ok:
                    classified = ErrorClassifier.classify(result.output, tool_name=call["toolName"])
                    nudge_msg = NudgeGenerator.generate(classified, retry_count=tool_error_count)
                    extra = "\n\n[System note: " + nudge_msg + "]"
                    if memory_injector is not None:
                        failure_mems = memory_injector.inject_on_failure(result.output, call["toolName"])
                        if failure_mems:
                            extra += "\n\n" + memory_injector.format_for_prompt(failure_mems)
                    result = ToolResult(
                        ok=False,
                        output=result.output + extra,
                        awaitUser=result.awaitUser,
                    )
                _append_tool_messages(
                    call, result, current_messages, is_concurrent, step,
                    store, on_tool_start, on_tool_result,
                )
                saw_tool_result = True
                if not result.ok:
                    tool_error_count += 1
                call_key = f"{call['toolName']}:{json.dumps(call['input'], sort_keys=True)}"
                _call_freq[call_key] = _call_freq.get(call_key, 0) + 1
                freq = _call_freq[call_key]
                if freq >= 3 and call_key not in _nudged_keys:
                    _nudged_keys.add(call_key)
                    if not result.ok:
                        nudge = NUDGE_REPEATED_FAILURE.format(
                            tool_name=call["toolName"], count=freq
                        )
                    else:
                        nudge = NUDGE_REPEATED_SAME_SEARCH.format(
                            tool_name=call["toolName"], count=freq
                        )
                    current_messages.append({"role": "user", "content": nudge})
                if result.awaitUser:
                    if on_assistant_message:
                        on_assistant_message(result.output)
                    current_messages.append({"role": "assistant", "content": result.output})
                    return current_messages

        fallback = "Reached the maximum tool step limit for this turn."
        if on_assistant_message:
            on_assistant_message(fallback)
        current_messages.append({"role": "assistant", "content": fallback})
        return current_messages

    finally:
        fire_hook_sync(HookEvent.AGENT_STOP, step=step, tool_errors=tool_error_count)
        fire_stop_hook(step=step, tool_errors=tool_error_count)
