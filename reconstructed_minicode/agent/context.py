from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from reconstructed_minicode.config import MINI_CODE_DIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_WINDOWS = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-haiku-3-20240307": 100_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    "openrouter/auto": 200_000,
    "anthropic/claude-sonnet-4": 200_000,
    "anthropic/claude-opus-4": 200_000,
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "google/gemini-2.5-pro": 1_000_000,
    "google/gemini-2.5-flash": 1_000_000,
    "meta-llama/llama-4-maverick": 1_000_000,
    "deepseek/deepseek-r1": 128_000,
    "deepseek/deepseek-chat": 128_000,
    "qwen/qwen3-235b-a22b": 128_000,
    "minimax/minimax-m1": 1_000_000,
    "default": 128_000,
}

AUTOCOMPACT_THRESHOLD = 0.95
MIN_MESSAGES_TO_KEEP = 10

# Tool categories used by compaction phases
_EDIT_TOOLS = frozenset({"edit_file", "write_file", "modify_file", "patch_file", "multi_edit"})
_READ_TOOLS = frozenset({"read_file", "list_files", "grep_files", "file_tree"})
_SEARCH_TOOLS = frozenset({"grep_files", "find_symbols", "find_references", "web_search", "web_fetch"})
_COMMAND_TOOLS = frozenset({"run_command", "execute_command", "bash"})


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CJK_PATTERN = re.compile(r'[一-鿿぀-ゟ゠-ヿ가-힯]')

# Cache token estimates: same text always produces the same count.
# Capped at 1024 entries to bound memory use.
_token_cache: dict[int, int] = {}
_TOKEN_CACHE_MAX = 1024


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    CJK characters count as ~1.5 chars/token; ASCII as ~4 chars/token.
    Uses a hash-keyed LRU cache to avoid recomputing identical strings.
    """
    if not text:
        return 0
    key = hash(text)
    if key in _token_cache:
        return _token_cache[key]
    cjk_count = len(_CJK_PATTERN.findall(text))
    ascii_chars = len(text) - cjk_count
    result = max(1, int(cjk_count / 1.5 + ascii_chars / 4.0))
    if len(_token_cache) < _TOKEN_CACHE_MAX:
        _token_cache[key] = result
    return result


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens for a single message dict."""
    ROLE_OVERHEAD = {
        "system": 3,
        "user": 4,
        "assistant": 3,
        "assistant_tool_call": 7,
        "tool_result": 6,
        "assistant_progress": 3,
    }
    tokens = ROLE_OVERHEAD.get(message.get("role", ""), 3)
    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    if "input" in message:
        inp = message["input"]
        tokens += estimate_tokens(json.dumps(inp) if isinstance(inp, dict) else str(inp))
    return tokens


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r'```[\w]*\n(.{20,300}?)```', re.DOTALL)
_DECISION_KEYWORDS = re.compile(
    r'(?:decided|decision|chose|chosen|will use|using|switching to|'
    r'implemented|fixed|resolved|refactored|migrated|upgraded|'
    r'recommend|should|must|need to|going to|plan to|'
    r'approach:|strategy:|solution:|conclusion:)',
    re.IGNORECASE,
)


@dataclass
class _ExtractedInfo:
    user_intents: list[str] = field(default_factory=list)
    file_paths: set[str] = field(default_factory=set)
    key_tool_results: list[str] = field(default_factory=list)
    assistant_conclusions: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


def _extract_from_messages(messages: list[dict[str, Any]]) -> _ExtractedInfo:
    info = _ExtractedInfo()
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and content.strip():
            preview = content.strip().replace("\n", " ")
            info.user_intents.append(preview[:200] + ("..." if len(preview) > 200 else ""))

        elif role == "assistant" and content and content.strip():
            text = content.strip()
            for sentence in text.replace("\n", " ").split(". "):
                if _DECISION_KEYWORDS.search(sentence):
                    dec = sentence.strip()[:180]
                    if dec and dec not in info.decisions:
                        info.decisions.append(dec)
            for match in _CODE_FENCE_RE.finditer(text):
                snippet = match.group(1).strip()
                if len(snippet) >= 20 and len(info.code_snippets) < 5:
                    info.code_snippets.append(snippet[:300])
            info.assistant_conclusions.append(text[:200].replace("\n", " "))

        elif role == "assistant_tool_call":
            tool_name = msg.get("toolName", "unknown")
            info.tool_names.append(tool_name)
            inp = msg.get("input", {})
            if tool_name in _EDIT_TOOLS:
                path = inp.get("path") or inp.get("filePath", "")
                if path:
                    info.file_paths.add(path)
            if tool_name in _SEARCH_TOOLS:
                pattern = inp.get("pattern") or inp.get("query", "")
                if pattern:
                    info.file_paths.add(f"search:{pattern[:80]}")
            if tool_name in _COMMAND_TOOLS:
                cmd = inp.get("command", "")
                cmd_name = cmd.split()[0] if cmd.split() else ""
                if cmd_name:
                    info.key_tool_results.append(f"ran: {cmd_name}")

        elif role == "tool_result":
            tool_name = msg.get("toolName", "")
            is_error = msg.get("isError", False)
            if is_error:
                preview = content.strip()[:150].replace("\n", " ")
                info.key_tool_results.append(f"ERROR({tool_name}): {preview}")
            elif tool_name in _EDIT_TOOLS and content.strip():
                info.key_tool_results.append(f"{tool_name} ok: {content.strip()[:100].replace(chr(10), ' ')}")
            elif tool_name in _READ_TOOLS and content.strip():
                first_line = content.strip().split("\n")[0][:100]
                if "/" in first_line or "\\" in first_line:
                    info.file_paths.add(first_line.strip())

    return info


def _build_layered_summary(info: _ExtractedInfo, max_summary_tokens: int = 2000) -> str:
    """Assemble a budget-aware summary from extracted info, most important layers first."""
    lines: list[str] = []
    # Budget fractions per layer: intents, decisions+paths, tool results, conclusions, code, tool log
    budgets = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]

    def _used() -> int:
        return estimate_tokens("\n".join(lines))

    if info.user_intents:
        budget = int(max_summary_tokens * budgets[0])
        lines.append("## User requests:")
        for i, intent in enumerate(info.user_intents[:12]):
            if _used() > budget:
                lines.append(f"  ... and {len(info.user_intents) - i} more")
                break
            lines.append(f"- {intent}")

    if info.decisions or info.file_paths:
        budget = int(max_summary_tokens * (budgets[0] + budgets[1]))
        if info.decisions:
            lines.append("## Key decisions:")
            for dec in info.decisions[:8]:
                if _used() > budget:
                    break
                lines.append(f"- {dec}")
        if info.file_paths:
            real = sorted(p for p in info.file_paths if not p.startswith("search:"))
            searched = sorted(p[7:] for p in info.file_paths if p.startswith("search:"))
            path_line = f"## Files: {', '.join(real[:20])}"
            if len(real) > 20:
                path_line += f" (+{len(real)-20} more)"
            if searched:
                path_line += f"\n## Searched: {', '.join(searched[:5])}"
            if _used() + estimate_tokens(path_line) <= budget:
                lines.append(path_line)

    if info.key_tool_results:
        budget = int(max_summary_tokens * sum(budgets[:3]))
        lines.append("## Key results:")
        for r in info.key_tool_results[:15]:
            if _used() > budget:
                break
            lines.append(f"- {r}")

    if info.assistant_conclusions:
        budget = int(max_summary_tokens * sum(budgets[:4]))
        lines.append("## Conclusions:")
        for c in info.assistant_conclusions[:8]:
            if _used() > budget:
                break
            lines.append(f"- {c}")

    if info.code_snippets:
        budget = int(max_summary_tokens * sum(budgets[:5]))
        lines.append("## Code patterns:")
        for snippet in info.code_snippets[:3]:
            block = f"```\n{snippet}\n```"
            if _used() + estimate_tokens(block) > budget:
                break
            lines.append(block)

    if info.tool_names:
        counts = Counter(info.tool_names)
        summary = ", ".join(
            f"{n}×{c}" if c > 1 else n for n, c in counts.most_common()
        )
        lines.append(f"## Tools: {summary}")

    return "\n".join(lines)


def _summarize_removed_messages(messages: list[dict[str, Any]], max_tokens: int = 2000) -> str:
    if not messages:
        return ""
    return _build_layered_summary(_extract_from_messages(messages), max_tokens)


# ---------------------------------------------------------------------------
# Compaction phase helpers
# ---------------------------------------------------------------------------

def _head_tail_truncate(content: str, keep_chars: int) -> str:
    """Truncate content to keep_chars, preserving head (70%) and tail (30%)."""
    lines = content.split("\n")
    head: list[str] = []
    tail: list[str] = []
    head_chars = tail_chars = 0

    for line in lines:
        if head_chars + len(line) + 1 > keep_chars * 0.7:
            break
        head.append(line)
        head_chars += len(line) + 1

    for line in reversed(lines):
        if tail_chars + len(line) + 1 > keep_chars * 0.3:
            break
        tail.insert(0, line)
        tail_chars += len(line) + 1

    omitted = len(lines) - len(head) - len(tail)
    result = "\n".join(head)
    if omitted > 0:
        result += f"\n... [{omitted} lines truncated for compaction] ...\n"
    result += "\n".join(tail)
    return result


def _truncate_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 2: truncate oversized tool results in-place, keeping head+tail."""
    THRESHOLDS = {
        "error": 4000,
        "edit": 3000,
        "read": 1500,
        "default": 2000,
    }
    result = list(messages)
    for i, m in enumerate(result):
        if m.get("role") != "tool_result":
            continue
        content = m.get("content", "")
        if not content:
            continue
        tool_name = m.get("toolName", "")
        is_error = m.get("isError", False)
        if is_error:
            threshold = THRESHOLDS["error"]
        elif tool_name in _EDIT_TOOLS:
            threshold = THRESHOLDS["edit"]
        elif tool_name in _READ_TOOLS:
            threshold = THRESHOLDS["read"]
        else:
            threshold = THRESHOLDS["default"]
        if len(content) > threshold:
            result[i] = {**m, "content": _head_tail_truncate(content, threshold)}
    return result


def _compress_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase 3: replace adjacent tool_call+tool_result pairs with inline summaries."""
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if (
            msg.get("role") == "assistant_tool_call"
            and i + 1 < len(messages)
            and messages[i + 1].get("role") == "tool_result"
        ):
            summary = _compress_tool_pair(msg, messages[i + 1])
            result.append({"role": "assistant", "content": summary})
            i += 2
        else:
            result.append(msg)
            i += 1
    return result


def _compress_tool_pair(call_msg: dict[str, Any], result_msg: dict[str, Any]) -> str:
    """Compress one tool_call+result pair into a single-line summary."""
    tool_name = call_msg.get("toolName", "unknown")
    inp = call_msg.get("input", {})
    result_content = result_msg.get("content", "")
    is_error = result_msg.get("isError", False)

    if is_error:
        error_text = result_content.strip()[:200].replace("\n", " ")
        return f"[Tool {tool_name} ERROR: {error_text}]"

    if tool_name in _EDIT_TOOLS:
        path = inp.get("path") or inp.get("filePath", "unknown")
        if tool_name == "multi_edit":
            return f"[Edited {path}: {len(inp.get('edits', []))} changes applied]"
        return f"[Edited {path}: ok]"

    if tool_name in _READ_TOOLS:
        path = inp.get("path") or inp.get("filePath", "")
        if path:
            return f"[Read {path}: {result_content.count(chr(10)) + 1} lines]"
        return f"[{tool_name}: completed]"

    if tool_name in _SEARCH_TOOLS:
        pattern = inp.get("pattern") or inp.get("query", "")
        matches = [l for l in result_content.split("\n") if l.strip() and not l.startswith("#")]
        return f"[Searched '{pattern[:50]}': {len(matches)} results]"

    if tool_name in _COMMAND_TOOLS:
        cmd = inp.get("command", "")
        cmd_name = cmd.split()[0] if cmd.split() else "command"
        exit_info = ""
        for line in result_content.split("\n"):
            if "exit code" in line.lower():
                exit_info = f" ({line.strip()[:50]})"
                break
        return f"[Ran {cmd_name}{exit_info}]"

    brief = result_content.strip()[:100].replace("\n", " ")
    return f"[{tool_name}: {brief}]" if brief else f"[{tool_name}: completed]"


def _remove_by_priority(messages: list[dict[str, Any]], target_tokens: int) -> list[dict[str, Any]]:
    """Phase 4: remove lowest-priority messages (oldest first) until under target."""
    PRIORITY = {
        "user": 0,           # highest — encodes intent
        "assistant": 1,      # high — encodes conclusions
        "assistant_tool_call": 2,
        "tool_result": 3,    # lowest — should already be compressed
    }
    PROTECTED_RECENT = 6
    result = list(messages)
    while estimate_messages_tokens(result) > target_tokens and len(result) > MIN_MESSAGES_TO_KEEP:
        removable_end = max(MIN_MESSAGES_TO_KEEP, len(result) - PROTECTED_RECENT)
        best_idx = None
        best_priority = -1
        for idx in range(removable_end):
            p = PRIORITY.get(result[idx].get("role", ""), 1)
            if p > best_priority:
                best_priority = p
                best_idx = idx
        if best_idx is None:
            break
        del result[best_idx]
    return result


# ---------------------------------------------------------------------------
# Context tracking
# ---------------------------------------------------------------------------

@dataclass
class ContextStats:
    total_tokens: int = 0
    context_window: int = 0
    usage_percentage: float = 0.0
    messages_count: int = 0
    system_tokens: int = 0
    conversation_tokens: int = 0
    tool_calls_count: int = 0
    is_near_limit: bool = False
    should_compact: bool = False


@dataclass
class ContextManager:
    """Tracks token usage and compacts the conversation when near the context limit."""
    model: str = "default"
    context_window: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    compaction_history: list[dict[str, Any]] = field(default_factory=list)
    _token_cache: dict[int, int] = field(default_factory=dict, repr=False)
    _compaction_level: int = field(default=0)  # increases after each compaction

    # Target context usage after each compaction level (0=first, 1=second, 2+=deep)
    _COMPACTION_LEVELS = [0.70, 0.50, 0.30]

    def __post_init__(self) -> None:
        if self.context_window == 0:
            self.context_window = DEFAULT_CONTEXT_WINDOWS.get(
                self.model, DEFAULT_CONTEXT_WINDOWS["default"]
            )

    def update_model(self, model: str) -> None:
        self.model = model
        self.context_window = DEFAULT_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOWS["default"])

    def add_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._token_cache[id(message)] = estimate_message_tokens(message)

    def get_stats(self) -> ContextStats:
        if not self.messages:
            return ContextStats(context_window=self.context_window)

        system_tokens = conversation_tokens = tool_calls = 0
        for msg in self.messages:
            tokens = self._token_cache.get(id(msg))
            if tokens is None:
                tokens = estimate_message_tokens(msg)
                self._token_cache[id(msg)] = tokens
            if msg.get("role") == "system":
                system_tokens += tokens
            else:
                conversation_tokens += tokens
            if msg.get("role") == "assistant_tool_call":
                tool_calls += 1

        total = system_tokens + conversation_tokens
        usage_pct = (total / self.context_window * 100) if self.context_window > 0 else 0
        return ContextStats(
            total_tokens=total,
            context_window=self.context_window,
            usage_percentage=usage_pct,
            messages_count=len(self.messages),
            system_tokens=system_tokens,
            conversation_tokens=conversation_tokens,
            tool_calls_count=tool_calls,
            is_near_limit=usage_pct >= 80,
            should_compact=usage_pct >= AUTOCOMPACT_THRESHOLD * 100,
        )

    def should_auto_compact(self) -> bool:
        # Each compaction level lowers the trigger threshold by 10%, floor 60%
        threshold = max(0.60, AUTOCOMPACT_THRESHOLD - self._compaction_level * 0.10)
        return self.get_stats().usage_percentage >= threshold * 100

    def compact_messages(self) -> list[dict[str, Any]]:
        """Compact messages through four progressive phases until under the target token count.

        Phase 1 — drop progress messages (zero semantic value).
        Phase 2 — truncate oversized tool results in-place (keep head + tail).
        Phase 3 — compress tool_call+result pairs into single-line summaries.
        Phase 4 — remove lowest-priority messages oldest-first.

        The target shrinks with each successive compaction (70% → 50% → 30%).
        """
        stats = self.get_stats()
        if not stats.should_compact:
            return self.messages

        target_pct = self._COMPACTION_LEVELS[min(self._compaction_level, 2)]
        target_tokens = int(self.context_window * target_pct)

        system = [m for m in self.messages if m.get("role") == "system"]
        other = [m for m in self.messages if m.get("role") != "system"]

        # Phase 1: drop progress messages
        filtered = [m for m in other if m.get("role") != "assistant_progress"]
        if estimate_messages_tokens(filtered) <= target_tokens:
            return self._finalize_compaction(system, other, filtered, stats, target_tokens)

        # Phase 2: truncate oversized tool results
        filtered = _truncate_tool_results(filtered)
        if estimate_messages_tokens(filtered) <= target_tokens:
            return self._finalize_compaction(system, other, filtered, stats, target_tokens)

        # Phase 3: compress tool_call+result pairs into summaries
        compressed = _compress_tool_pairs(filtered)
        if estimate_messages_tokens(compressed) <= target_tokens:
            return self._finalize_compaction(system, other, compressed, stats, target_tokens)

        # Phase 4: priority-based removal, oldest first
        compressed = _remove_by_priority(compressed, target_tokens)
        return self._finalize_compaction(system, other, compressed, stats, target_tokens)

    def _finalize_compaction(
        self,
        system_messages: list[dict[str, Any]],
        original_other: list[dict[str, Any]],
        filtered: list[dict[str, Any]],
        stats: ContextStats,
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        retained_ids = {id(m) for m in filtered}
        removed = [m for m in original_other if id(m) not in retained_ids]
        summary = _summarize_removed_messages(removed)

        removed_count = len(original_other) - len(filtered)
        after_pct = estimate_messages_tokens(filtered) / self.context_window * 100 if self.context_window > 0 else 0

        marker = {
            "role": "system",
            "content": (
                f"[Context compacted at {time.strftime('%H:%M:%S')}. "
                f"{removed_count} messages removed. "
                f"Token usage: {stats.usage_percentage:.0f}% → {after_pct:.0f}%]"
                + (f"\n\nSummary of removed conversation:\n{summary}" if summary else "")
            ),
        }

        compacted = system_messages + [marker] + filtered
        self.compaction_history.append({
            "timestamp": time.time(),
            "before_tokens": stats.total_tokens,
            "after_tokens": estimate_messages_tokens(compacted),
            "messages_removed": len(self.messages) - len(compacted),
            "compaction_level": self._compaction_level,
        })
        self._compaction_level = min(self._compaction_level + 1, 3)
        self.messages = compacted
        self._token_cache = {
            id(m): self._token_cache.get(id(m), estimate_message_tokens(m))
            for m in compacted
        }
        return compacted

    def get_context_summary(self) -> str:
        stats = self.get_stats()
        if stats.messages_count == 0:
            return "Context: empty"
        status = "✓" if not stats.is_near_limit else ("🔴" if stats.should_compact else "⚠")
        return (
            f"Context: {status} {stats.usage_percentage:.0f}% "
            f"({stats.total_tokens:,}/{stats.context_window:,} tokens, "
            f"{stats.messages_count} msgs, {stats.tool_calls_count} tools)"
        )

    def format_context_details(self) -> str:
        stats = self.get_stats()
        lines = [
            "Context Window Usage",
            "=" * 50,
            f"Model: {self.model}",
            f"Context window: {stats.context_window:,} tokens",
            "",
            f"Total tokens: {stats.total_tokens:,}",
            f"Usage: {stats.usage_percentage:.1f}%",
            f"Messages: {stats.messages_count}",
            f"Tool calls: {stats.tool_calls_count}",
            "",
        ]
        if stats.should_compact:
            lines += ["⚠️  WARNING: Context is near capacity!", "Auto-compaction will trigger soon.", ""]
        if self.compaction_history:
            lines.append("Compaction History:")
            for comp in self.compaction_history[-3:]:
                ts = time.strftime("%H:%M:%S", time.localtime(comp["timestamp"]))
                lines.append(
                    f"  {ts}: {comp['messages_removed']} messages removed, "
                    f"{comp['before_tokens']:,} → {comp['after_tokens']:,} tokens"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_context_state(manager: ContextManager) -> None:
    state_path = MINI_CODE_DIR / "context_state.json"
    MINI_CODE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "model": manager.model,
        "context_window": manager.context_window,
        "messages": manager.messages,
        "compaction_history": manager.compaction_history[-10:],
        "_compaction_level": manager._compaction_level,
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_context_state() -> ContextManager | None:
    state_path = MINI_CODE_DIR / "context_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manager = ContextManager(
            model=state.get("model", "default"),
            context_window=state.get("context_window", 0),
            messages=state.get("messages", []),
            compaction_history=state.get("compaction_history", []),
        )
        if "_compaction_level" in state:
            manager._compaction_level = state["_compaction_level"]
        return manager
    except (json.JSONDecodeError, KeyError):
        return None


def clear_context_state() -> None:
    state_path = MINI_CODE_DIR / "context_state.json"
    if state_path.exists():
        state_path.unlink()
