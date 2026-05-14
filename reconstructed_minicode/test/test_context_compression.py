"""Engineering tests for context compression: timing, token reduction, phase coverage."""
import time
import pytest
from reconstructed_minicode.agent.context import ContextManager, estimate_message_tokens, estimate_messages_tokens


SMALL_WINDOW = 10_000   # small context window to trigger compression easily
TOOL_RESULT_BODY = "x" * 800   # ~200 tokens per tool result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conversation(num_pairs: int, include_progress: bool = False) -> list[dict]:
    """Build a realistic conversation: system + user/assistant pairs + tool calls."""
    messages = [{"role": "system", "content": "You are a coding assistant." * 20}]
    for i in range(num_pairs):
        messages.append({"role": "user", "content": f"Task {i}: please implement feature {i}."})
        messages.append({"role": "assistant", "content": f"I'll implement feature {i} now."})
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": "write_file", "arguments": f'{{"path": "f{i}.py"}}'}}],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": TOOL_RESULT_BODY,
        })
        if include_progress:
            messages.append({"role": "assistant_progress", "content": f"Working on feature {i}..."})
    return messages


def _fill_to_pct(manager: ContextManager, target_pct: float) -> list[dict]:
    """Build messages and shrink context_window so usage_pct == target_pct."""
    messages = [{"role": "system", "content": "You are a coding assistant." * 20}]
    for i in range(40):
        messages += [
            {"role": "user", "content": f"Do task {i} in detail. " * 5},
            {"role": "assistant", "content": f"Completing task {i}. " * 5},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": "read_file", "arguments": f'{{"path": "f{i}.py"}}'}}],
            },
            {"role": "tool", "tool_call_id": f"c{i}", "content": TOOL_RESULT_BODY},
        ]
    total = estimate_messages_tokens(messages)
    # Set context_window so that total / context_window == target_pct
    manager.context_window = int(total / target_pct)
    manager.messages = messages
    return messages


# ---------------------------------------------------------------------------
# Trigger threshold tests
# ---------------------------------------------------------------------------

class TestCompressionTrigger:
    def test_no_trigger_below_threshold(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        _fill_to_pct(mgr, 0.80)
        assert not mgr.should_auto_compact(), "Should NOT trigger at 80%"

    def test_triggers_at_95_pct(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        _fill_to_pct(mgr, 0.96)
        assert mgr.should_auto_compact(), "Should trigger at 96%"

    def test_threshold_lowers_after_compaction(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        _fill_to_pct(mgr, 0.96)
        mgr.compact_messages()
        # After first compaction, threshold drops to 85%
        assert mgr._compaction_level == 1
        _fill_to_pct(mgr, 0.87)
        assert mgr.should_auto_compact(), "Second compaction should trigger at 85%"


# ---------------------------------------------------------------------------
# Phase coverage tests
# ---------------------------------------------------------------------------

class TestCompressionPhases:
    def test_phase1_drops_progress_messages(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        msgs = _make_conversation(5, include_progress=True)
        progress_count = sum(1 for m in msgs if m.get("role") == "assistant_progress")
        assert progress_count > 0

        # Shrink context_window so compaction triggers
        total = estimate_messages_tokens(msgs)
        mgr.context_window = int(total / 0.96)
        mgr.messages = msgs

        compacted = mgr.compact_messages()
        after_progress = sum(1 for m in compacted if m.get("role") == "assistant_progress")
        assert after_progress == 0, "Phase 1 must remove all progress messages"

    def test_phase2_truncates_large_tool_results(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        # Fill with large tool results
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"task {i}"})
            msgs.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"}}],
            })
            msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                         "content": "A" * 2000})  # very large result ~500 tokens each
        mgr.messages = msgs
        _fill_to_pct(mgr, 0.96)

        compacted = mgr.compact_messages()
        tool_results = [m for m in compacted if m.get("role") == "tool"]
        if tool_results:
            max_len = max(len(m["content"]) for m in tool_results if isinstance(m.get("content"), str))
            assert max_len < 2000, "Phase 2 should have truncated large tool results"


# ---------------------------------------------------------------------------
# Token reduction tests
# ---------------------------------------------------------------------------

class TestTokenReduction:
    def test_tokens_reduced_after_compaction(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        _fill_to_pct(mgr, 0.96)

        before = mgr.get_stats().total_tokens
        mgr.compact_messages()
        after = mgr.get_stats().total_tokens

        assert after < before, "Compaction must reduce token count"
        reduction_pct = (before - after) / before * 100
        print(f"\nToken reduction: {before} → {after} ({reduction_pct:.1f}% saved)")

    def test_first_compaction_targets_70_pct(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        _fill_to_pct(mgr, 0.96)
        mgr.compact_messages()

        after_pct = mgr.get_stats().usage_percentage
        assert after_pct <= 75.0, f"After first compaction should be ≤70-75%, got {after_pct:.1f}%"

    def test_compaction_history_recorded(self):
        mgr = ContextManager(context_window=SMALL_WINDOW)
        _fill_to_pct(mgr, 0.96)
        mgr.compact_messages()

        assert len(mgr.compaction_history) == 1
        record = mgr.compaction_history[0]
        assert record["before_tokens"] > record["after_tokens"]
        assert record["compaction_level"] == 0


# ---------------------------------------------------------------------------
# Performance / timing tests
# ---------------------------------------------------------------------------

class TestCompressionPerformance:
    @pytest.mark.parametrize("num_pairs", [20, 50, 100])
    def test_compaction_latency(self, num_pairs):
        mgr = ContextManager(context_window=SMALL_WINDOW * 5)
        msgs = _make_conversation(num_pairs)
        mgr.messages = msgs
        # Force compaction by patching usage check
        mgr._compaction_level = 0
        # Artificially set context_window small to trigger
        mgr.context_window = estimate_messages_tokens(msgs) + 10

        start = time.perf_counter()
        mgr.compact_messages()
        elapsed = time.perf_counter() - start

        print(f"\n  {num_pairs} pairs ({len(msgs)} msgs): {elapsed*1000:.1f}ms")
        assert elapsed < 1.0, f"Compaction of {num_pairs} pairs took {elapsed:.3f}s — too slow"

    def test_repeated_compaction_does_not_degrade(self):
        """Three successive compactions should each complete in under 1s."""
        mgr = ContextManager(context_window=SMALL_WINDOW)
        timings = []
        for round_ in range(3):
            _fill_to_pct(mgr, 0.96)
            start = time.perf_counter()
            mgr.compact_messages()
            timings.append(time.perf_counter() - start)

        print(f"\n  Compaction timings: {[f'{t*1000:.1f}ms' for t in timings]}")
        for i, t in enumerate(timings):
            assert t < 1.0, f"Round {i+1} compaction took {t:.3f}s"


# ---------------------------------------------------------------------------
# Token estimation accuracy
# ---------------------------------------------------------------------------

class TestTokenEstimation:
    def test_estimation_consistent_with_sum(self):
        """estimate_messages_tokens should equal sum of per-message estimates."""
        msgs = _make_conversation(10)
        total_sum = sum(estimate_message_tokens(m) for m in msgs)
        total_bulk = estimate_messages_tokens(msgs)
        assert total_sum == total_bulk

    def test_cjk_content_estimated_higher_than_ascii(self):
        """CJK tokens should estimate more tokens per character than ASCII."""
        ascii_msg = {"role": "user", "content": "a" * 100}
        cjk_msg = {"role": "user", "content": "中" * 100}
        assert estimate_message_tokens(cjk_msg) > estimate_message_tokens(ascii_msg), \
            "CJK 100 chars should yield more tokens than ASCII 100 chars"

    def test_empty_message_has_minimal_tokens(self):
        msg = {"role": "user", "content": ""}
        assert estimate_message_tokens(msg) < 10
