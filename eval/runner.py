"""SWE-bench evaluation runner for MiniCode.

Usage:
    python eval/runner.py --instances eval/instances.jsonl --label baseline --n 5
    python eval/runner.py --instances eval/instances.jsonl --label focus --n 5
    python eval/runner.py --compare eval/results/baseline.json eval/results/focus.json

Instances file: one JSON object per line (SWE-bench Lite format).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.env import clone_repo, apply_patch, check_tests_pass
from eval.metrics import EvalRun, TaskResult, print_comparison
from reconstructed_minicode.agent.context import estimate_messages_tokens


# ---------------------------------------------------------------------------
# Instrumented agent runner
# ---------------------------------------------------------------------------

def _run_agent_on_instance(
    instance: dict[str, Any],
    repo_path: Path,
    runtime: dict,
) -> tuple[str, int, int, int, int]:
    """Run MiniCode agent on a SWE-bench instance.

    Returns: (patch, tokens_used, compressions, messages_dropped, turns)
    """
    from reconstructed_minicode.agent.loop import run_agent_turn
    from reconstructed_minicode.agent.context import ContextManager
    from reconstructed_minicode.model.registry import create_model_adapter
    from reconstructed_minicode.security.permissions import PermissionManager
    from reconstructed_minicode.tools import create_default_tool_registry

    cwd = str(repo_path)
    tools = create_default_tool_registry(cwd, runtime=runtime)
    model = create_model_adapter(
        model=runtime.get("model", ""),
        tools=tools,
        runtime=runtime,
    )
    context_manager = ContextManager(model=runtime.get("model", "default"))
    permissions = PermissionManager(cwd, prompt=None)  # auto-approve for eval

    system_prompt = (
        "You are an expert software engineer. "
        "Solve the GitHub issue described below by editing the repository files. "
        "When done, summarize what you changed."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance["problem_statement"]},
    ]

    tokens_before = estimate_messages_tokens(messages)
    turns_taken = 0
    on_turn = lambda: None

    def count_turn(*_):
        nonlocal turns_taken
        turns_taken += 1

    result_messages = run_agent_turn(
        model=model,
        tools=tools,
        messages=messages,
        cwd=cwd,
        permissions=permissions,
        context_manager=context_manager,
        runtime=runtime,
        on_assistant_message=count_turn,
        max_steps=30,
    )

    tokens_used = estimate_messages_tokens(result_messages)

    # Collect compaction stats
    compressions = len(context_manager.compaction_history)
    messages_dropped = sum(
        h.get("messages_removed", 0) for h in context_manager.compaction_history
    )

    # Get the diff of what was changed
    import subprocess
    diff_result = subprocess.run(
        ["git", "diff"], cwd=cwd, capture_output=True
    )
    patch = diff_result.stdout.decode("utf-8", errors="replace")

    return patch, tokens_used, compressions, messages_dropped, turns_taken


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(
    instances: list[dict[str, Any]],
    label: str,
    runtime: dict,
    output_path: Path,
    workdir: Path,
) -> EvalRun:
    run = EvalRun(label=label, model=runtime.get("model", "unknown"))

    for i, instance in enumerate(instances):
        iid = instance["instance_id"]
        print(f"\n[{i+1}/{len(instances)}] {iid}")

        start = time.time()
        error = None
        success = False
        patch = ""
        tokens = 0
        compressions = 0
        dropped = 0
        turns = 0

        repo_path = None
        try:
            repo_path = clone_repo(instance["repo"], instance["base_commit"], workdir)

            patch, tokens, compressions, dropped, turns = _run_agent_on_instance(
                instance, repo_path, runtime
            )
            print(f"  patch: {len(patch)} chars | tokens: {tokens:,} | "
                  f"compressions: {compressions} | dropped: {dropped}")

            if patch.strip():
                apply_patch(repo_path, patch)

            import json as _json
            fail_to_pass = _json.loads(instance.get("FAIL_TO_PASS", "[]"))
            success = check_tests_pass(repo_path, fail_to_pass)
            print(f"  success: {success}")

        except KeyboardInterrupt:
            print("  Interrupted.")
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}")

        finally:
            # Reset repo for next run
            if repo_path and repo_path.exists():
                import subprocess
                subprocess.run(
                    ["git", "checkout", "."],
                    cwd=str(repo_path), capture_output=True,
                )

        run.results.append(TaskResult(
            instance_id=iid,
            success=success,
            tokens_used=tokens,
            compression_count=compressions,
            messages_dropped=dropped,
            turns=turns,
            duration_seconds=time.time() - start,
            error=error,
            patch_generated=patch[:2000],  # truncate for storage
        ))

        run.save(output_path)
        print(f"  saved → {output_path}")

    return run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench eval runner for MiniCode")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run evaluation")
    run_p.add_argument("--instances", required=True, help="Path to .jsonl file")
    run_p.add_argument("--label", required=True, help="Run label, e.g. 'baseline'")
    run_p.add_argument("--n", type=int, default=5, help="Number of instances to evaluate")
    run_p.add_argument("--output", default="eval/results", help="Output directory")
    run_p.add_argument("--workdir", default="eval/repos", help="Temp dir for cloned repos")
    run_p.add_argument("--model", default=None, help="Override model")

    cmp_p = sub.add_parser("compare", help="Compare two eval runs")
    cmp_p.add_argument("baseline", help="Path to baseline results JSON")
    cmp_p.add_argument("focus", help="Path to focus results JSON")

    args = parser.parse_args()

    if args.cmd == "compare":
        from eval.metrics import EvalRun, print_comparison
        baseline = EvalRun.load(Path(args.baseline))
        focus = EvalRun.load(Path(args.focus))
        print_comparison(baseline, focus)
        return

    if args.cmd == "run":
        instances_path = Path(args.instances)
        instances = []
        with open(instances_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))
        instances = instances[: args.n]

        # Load runtime config
        config_path = Path.home() / ".mini-code" / "settings.json"
        runtime = {}
        if config_path.exists():
            try:
                settings = json.loads(config_path.read_text(encoding="utf-8"))
                runtime = {
                    "model": args.model or settings.get("model", ""),
                    **{k: v for k, v in settings.get("env", {}).items()},
                }
            except Exception:
                pass
        if args.model:
            runtime["model"] = args.model

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.label}.json"

        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        run = run_eval(instances, args.label, runtime, output_path, workdir)

        print(f"\n{'='*55}")
        print(f"  {args.label}: {run.n} tasks")
        print(f"  Success: {sum(r.success for r in run.results)}/{run.n} ({run.task_success_rate*100:.0f}%)")
        print(f"  Total tokens: {run.total_tokens:,}")
        print(f"  Avg tokens/task: {run.avg_tokens_per_task:,.0f}")
        print(f"  Avg compressions: {run.avg_compressions:.1f}")
        print(f"  Avg msgs dropped: {run.avg_messages_dropped:.1f}")
        print(f"{'='*55}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
