"""SWE-bench evaluation runner for MiniCode.

Usage:
    python eval/runner.py run --instances eval/instances.jsonl --label baseline --n 5
    python eval/runner.py merge --label baseline --swebench eval/results/swebench.json
    python eval/runner.py compare eval/results/baseline.json eval/results/focus.json

Patches are saved to eval/predictions/<label>.jsonl for upload to swebench.com.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.env import clone_repo
from eval.metrics import EvalRun, TaskResult, print_comparison


def _run_on_instance(instance: dict[str, Any], repo_path: Path) -> tuple[str, int, int, int, int]:
    from reconstructed_minicode.cli.headless import run_headless
    prompt = (
        "Fix the following GitHub issue by editing the source code directly. Hints are important.\n"
        "\n"
        "Rules:\n"
        "- Before editing, read the relevant test file to understand the exact expected behavior.\n"
        "- A bug often requires edits in MORE THAN ONE place. After each fix, re-read the entire\n"
        "  function/file to find other instances of the same bug pattern.\n"
        "- Check related methods: if method A is broken, check sibling methods B and C too.\n"
        "- Do NOT modify test files — only fix source code.\n"
        "- Do not use web_search — all information is in the local codebase.\n"
        "- Do not run any shell commands or test scripts.\n"
        "\n"
        + instance["problem_statement"]
    )
    hints = instance.get("hints_text", "").strip()
    if hints:
        prompt += f"\n\n--- Hints ---\n{hints}"

    orig_cwd = os.getcwd()
    os.chdir(str(repo_path))
    os.environ["MINI_CODE_BYPASS_PERMISSIONS"] = "1"
    try:
        result = run_headless(prompt, verbose=True)
    finally:
        os.chdir(orig_cwd)
        os.environ.pop("MINI_CODE_BYPASS_PERMISSIONS", None)

    diff = subprocess.run(["git", "diff"], cwd=str(repo_path), capture_output=True)
    patch = diff.stdout.decode("utf-8", errors="replace")

    return patch, result.tokens_used, result.compression_count, result.messages_dropped, result.turns


def run_eval(
    instances: list[dict[str, Any]],
    label: str,
    output_path: Path,
    predictions_path: Path,
    workdir: Path,
) -> EvalRun:
    from reconstructed_minicode.config import load_runtime_config
    runtime = load_runtime_config()
    model = runtime.get("model", "unknown")

    run = EvalRun(label=label, model=model)
    predictions: list[dict] = []

    for i, instance in enumerate(instances):
        iid = instance["instance_id"]
        print(f"\n[{i+1}/{len(instances)}] {iid}")

        start = time.time()
        error = None
        patch = ""
        tokens = compressions = dropped = turns = 0

        repo_path = None
        try:
            repo_path = clone_repo(instance["repo"], instance["base_commit"], workdir)
            patch, tokens, compressions, dropped, turns = _run_on_instance(instance, repo_path)
            print(f"  patch: {len(patch)} chars")

        except KeyboardInterrupt:
            print("  Interrupted.")
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}")

        finally:
            if repo_path and repo_path.exists():
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(repo_path), capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=str(repo_path), capture_output=True)

        run.results.append(TaskResult(
            instance_id=iid,
            tokens_used=tokens,
            compression_count=compressions,
            messages_dropped=dropped,
            turns=turns,
            duration_seconds=time.time() - start,
            error=error,
            patch_generated=patch[:2000],
        ))
        predictions.append({
            "instance_id": iid,
            "model_patch": patch,
            "model_name_or_path": model,
        })

        run.save(output_path)
        _save_jsonl(predictions, predictions_path)
        print(f"  saved → {output_path}")

    return run


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_results(label: str, results_dir: Path, swebench_path: Path) -> None:
    metrics_path = results_dir / f"{label}.json"
    run = EvalRun.load(metrics_path)
    raw = json.loads(swebench_path.read_text(encoding="utf-8"))
    resolved = set(raw) if isinstance(raw, list) else {k for k, v in raw.items() if v}
    updated = 0
    for r in run.results:
        if r.instance_id in resolved or r.instance_id in raw:
            r.success = r.instance_id in resolved if isinstance(raw, list) else bool(raw.get(r.instance_id))
            updated += 1
    run.save(metrics_path)
    print(f"Updated {updated}/{run.n} | success: {sum(r.success for r in run.results if r.success)}/{run.n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run")
    run_p.add_argument("--instances", required=True)
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--n", type=int, default=5)
    run_p.add_argument("--offset", type=int, default=0)
    run_p.add_argument("--output", default="eval/results")
    run_p.add_argument("--workdir", default="eval/repos")

    merge_p = sub.add_parser("merge")
    merge_p.add_argument("--label", required=True)
    merge_p.add_argument("--swebench", required=True)
    merge_p.add_argument("--output", default="eval/results")

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("baseline")
    cmp_p.add_argument("focus")

    args = parser.parse_args()

    if args.cmd == "compare":
        print_comparison(EvalRun.load(Path(args.baseline)), EvalRun.load(Path(args.focus)))

    elif args.cmd == "merge":
        merge_results(args.label, Path(args.output), Path(args.swebench))

    elif args.cmd == "run":
        with open(args.instances, encoding="utf-8") as f:
            instances = [json.loads(l) for l in f if l.strip()]
        instances = instances[args.offset: args.offset + args.n]

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(args.workdir).mkdir(parents=True, exist_ok=True)

        run = run_eval(
            instances=instances,
            label=args.label,
            output_path=output_dir / f"{args.label}.json",
            predictions_path=Path("eval/predictions") / f"{args.label}.jsonl",
            workdir=Path(args.workdir),
        )
        print(f"\n{'='*50}")
        print(f"  {args.label}: {run.n} tasks | tokens: {run.total_tokens:,}")
        print(f"  Upload eval/predictions/{args.label}.jsonl → swebench.com")
        print(f"{'='*50}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
