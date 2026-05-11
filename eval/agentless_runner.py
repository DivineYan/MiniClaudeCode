"""Agentless SWE-bench evaluation runner.

Two-stage fixed pipeline (no autonomous tool use):
  Stage 1 – Localization
      1a. file tree + issue → relevant files (JSON list)
      1b. relevant files contents + issue → edit locations (free text)
  Stage 2 – Repair
      sample n patches at temperature 0.8, pick first that git-applies cleanly

Usage:
    python eval/agentless_runner.py --instances eval/instances.jsonl --label agentless_v1 --n 5
    python eval/agentless_runner.py --instances eval/instances.jsonl --label agentless_v1 --n 5 --samples 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.env import clone_repo
from eval.runner import _save_jsonl


# ── LLM call ────────────────────────────────────────────────────────

def _call_llm(messages: list[dict], runtime: dict, temperature: float = 0.0, timeout: int = 300) -> str:
    """Single non-streaming LLM call. Returns text content."""
    import urllib.request
    import urllib.error

    base_url = (
        os.environ.get("OPENAI_BASE_URL", "")
        or os.environ.get("OPENAI_API_BASE", "")
        or runtime.get("openaiBaseUrl", "")
        or runtime.get("customBaseUrl", "")
        or "https://api.openai.com"
    ).rstrip("/")
    api_key = (
        os.environ.get("OPENAI_API_KEY", "")
        or runtime.get("openaiApiKey", "")
        or runtime.get("customApiKey", "")
        or ""
    )

    body: dict[str, Any] = {
        "model": runtime["model"],
        "messages": messages,
        "temperature": temperature,
    }
    if runtime.get("maxOutputTokens"):
        body["max_tokens"] = runtime["maxOutputTokens"]
    # Agentless uses single-turn calls — skip thinking to avoid timeouts

    url = (
        f"{base_url}/chat/completions"
        if base_url.endswith(("/v1", "/v4"))
        else f"{base_url}/v1/chat/completions"
    )
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **runtime.get("_custom_headers", {}),
    }
    req = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(4):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"].get("content", "") or ""
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt >= 3:
                raise
            time.sleep(2 ** attempt)
    return ""


# ── Stage 1a: file-level localization ───────────────────────────────

def _file_tree(repo_path: Path, max_lines: int = 400) -> str:
    """Python source files tracked by git, excluding tests/migrations/docs."""
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=str(repo_path), capture_output=True)
    lines = r.stdout.decode("utf-8", errors="replace").splitlines()
    src = [
        l for l in lines
        if not any(x in l for x in ("/tests/", "/test_", "_test.py", "/migrations/", "/docs/"))
    ]
    return "\n".join(src[:max_lines])


def localize_files(problem: str, hints: str, repo_path: Path, runtime: dict) -> list[str]:
    """Ask model which files to edit. Returns validated list of existing paths."""
    issue = problem + (f"\n\nHints:\n{hints}" if hints else "")
    tree = _file_tree(repo_path)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a software engineer triaging a bug report. "
                "Given an issue description and a repository file listing, "
                "identify which source files most likely need to be changed to fix the issue. "
                'Respond ONLY with a valid JSON array of file paths, e.g. ["a/b.py"]. '
                "Maximum 5 files. No explanation."
            ),
        },
        {"role": "user", "content": f"Issue:\n{issue}\n\nRepository files:\n{tree}"},
    ]
    raw = _call_llm(messages, runtime, temperature=0.0)
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        files = json.loads(m.group())
        return [f for f in files if isinstance(f, str) and (repo_path / f).exists()][:5]
    except (json.JSONDecodeError, TypeError):
        return []


# ── Stage 1b: edit-location localization ────────────────────────────

def localize_edits(
    problem: str, hints: str, files: list[str], repo_path: Path, runtime: dict
) -> str:
    """Given file list, ask model for specific functions/lines to change."""
    issue = problem + (f"\n\nHints:\n{hints}" if hints else "")

    snippets = []
    for path in files:
        full = repo_path / path
        if not full.exists():
            continue
        content = full.read_text(encoding="utf-8", errors="replace")
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"
        snippets.append(f"=== {path} ===\n{content}")

    if not snippets:
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a software engineer fixing a bug. "
                "Given the issue and relevant source files, identify exactly which "
                "functions, classes, or line ranges need to be modified. "
                "Be specific: list file path + function/class name + what needs to change."
            ),
        },
        {"role": "user", "content": f"Issue:\n{issue}\n\n" + "\n\n".join(snippets)},
    ]
    return _call_llm(messages, runtime, temperature=0.0)


# ── Stage 2: repair ──────────────────────────────────────────────────

_PATCH_RE = re.compile(r"(diff --git |--- a/|--- )", re.MULTILINE)


def _extract_patch(text: str) -> str:
    m = _PATCH_RE.search(text)
    return text[m.start():].strip() if m else text.strip()


def _applies_cleanly(patch: str, repo_path: Path) -> bool:
    tmp = repo_path / "_agentless_tmp.patch"
    try:
        tmp.write_bytes(patch.replace("\r\n", "\n").encode("utf-8"))
        r = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", tmp.as_posix()],
            cwd=str(repo_path),
            capture_output=True,
        )
        return r.returncode == 0
    finally:
        tmp.unlink(missing_ok=True)


def repair(
    problem: str,
    hints: str,
    edit_context: str,
    files: list[str],
    repo_path: Path,
    runtime: dict,
    n_samples: int,
) -> str:
    """Sample n patches; return first that applies, else first sample."""
    issue = problem + (f"\n\nHints:\n{hints}" if hints else "")

    snippets = []
    for path in files:
        full = repo_path / path
        if not full.exists():
            continue
        content = full.read_text(encoding="utf-8", errors="replace")
        if len(content) > 4000:
            content = content[:4000] + "\n... (truncated)"
        snippets.append(f"=== {path} ===\n{content}")

    context = "\n\n".join(snippets)
    if edit_context:
        context = f"Edit locations:\n{edit_context}\n\n{context}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a software engineer fixing a GitHub issue. "
                "Generate a unified diff patch.\n"
                "Rules:\n"
                "- Use unified diff format with 'diff --git a/... b/...' headers\n"
                "- Include 3 lines of context before and after each change\n"
                "- Only modify source files, not test files\n"
                "- Output ONLY the patch, no explanation"
            ),
        },
        {"role": "user", "content": f"Issue:\n{issue}\n\nRelevant code:\n{context}"},
    ]

    samples: list[str] = []
    for i in range(n_samples):
        temp = 0.0 if i == 0 else 0.8
        raw = _call_llm(messages, runtime, temperature=temp)
        patch = _extract_patch(raw)
        if patch:
            samples.append(patch)

    if not samples:
        return ""

    for patch in samples:
        if _applies_cleanly(patch, repo_path):
            return patch
    return samples[0]


# ── Full pipeline per instance ───────────────────────────────────────

def run_on_instance(
    instance: dict[str, Any], repo_path: Path, runtime: dict, n_samples: int
) -> str:
    problem = instance["problem_statement"]
    hints = instance.get("hints_text", "").strip()

    print("  [1/3] localize files ...")
    files = localize_files(problem, hints, repo_path, runtime)
    print(f"        → {files}")
    if not files:
        print("  [!] no files identified, skipping")
        return ""

    print("  [2/3] localize edits ...")
    edit_ctx = localize_edits(problem, hints, files, repo_path, runtime)

    print(f"  [3/3] repair (samples={n_samples}) ...")
    return repair(problem, hints, edit_ctx, files, repo_path, runtime, n_samples)


# ── Eval loop ────────────────────────────────────────────────────────

def run_eval(
    instances: list[dict[str, Any]],
    label: str,
    predictions_path: Path,
    workdir: Path,
    n_samples: int,
) -> None:
    from reconstructed_minicode.config import load_runtime_config
    runtime = load_runtime_config()
    model = runtime.get("model", "unknown")
    print(f"Model: {model} | instances: {len(instances)} | samples/instance: {n_samples}\n")

    predictions: list[dict] = []

    for i, instance in enumerate(instances):
        iid = instance["instance_id"]
        print(f"[{i+1}/{len(instances)}] {iid}")

        patch = ""
        repo_path = None
        try:
            repo_path = clone_repo(instance["repo"], instance["base_commit"], workdir)
            patch = run_on_instance(instance, repo_path, runtime, n_samples)
            print(f"  patch: {len(patch)} chars")
        except KeyboardInterrupt:
            print("  Interrupted.")
            break
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        finally:
            if repo_path and repo_path.exists():
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(repo_path), capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=str(repo_path), capture_output=True)

        predictions.append({"instance_id": iid, "model_patch": patch, "model_name_or_path": model})
        _save_jsonl(predictions, predictions_path)

    no_patch = sum(1 for p in predictions if not p["model_patch"].strip())
    print(f"\n{'='*50}")
    print(f"  {label}: {len(predictions)} tasks | no patch: {no_patch}")
    print(f"  predictions → {predictions_path}")
    print(f"{'='*50}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=5, help="number of instances to run")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--samples", type=int, default=3, help="repair samples per instance")
    p.add_argument("--workdir", default="eval/repos")
    p.add_argument("--output", default="eval/predictions")
    args = p.parse_args()

    with open(args.instances, encoding="utf-8") as f:
        instances = [json.loads(l) for l in f if l.strip()]
    instances = instances[args.offset: args.offset + args.n]

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.output) / f"{args.label}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    run_eval(
        instances=instances,
        label=args.label,
        predictions_path=pred_path,
        workdir=Path(args.workdir),
        n_samples=args.samples,
    )


if __name__ == "__main__":
    main()
