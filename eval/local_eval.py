"""Local SWE-bench approximation: apply check + GT diff + optional test run.

Usage:
    python eval/local_eval.py eval/predictions/astropy_glm_lf.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _normalize_patch(patch: str) -> str:
    """Keep only +/- lines for comparison, strip index/diff headers."""
    lines = []
    for line in patch.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            lines.append(line)
    return "\n".join(lines)


def _patch_similarity(ours: str, gt: str) -> float:
    a = _normalize_patch(ours)
    b = _normalize_patch(gt)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _git(args: list[str], cwd: Path, input: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        input=input,
    )


_TMP_PATCH = Path(__file__).parent / "_tmp_apply.patch"


def _write_patch(patch: str) -> None:
    _TMP_PATCH.write_bytes(patch.replace("\r\n", "\n").encode("utf-8"))


def _apply_patch(repo_path: Path, patch: str) -> tuple[bool, str]:
    _write_patch(patch)
    r = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", _TMP_PATCH.absolute().as_posix()],
        cwd=str(repo_path),
        capture_output=True,
    )
    err = (r.stdout + r.stderr).decode("utf-8", errors="replace").strip()[:200]
    return r.returncode == 0, err


def _check_patch(repo_path: Path, patch: str) -> tuple[bool, str]:
    _write_patch(patch)
    r = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", _TMP_PATCH.absolute().as_posix()],
        cwd=str(repo_path),
        capture_output=True,
    )
    err = (r.stdout + r.stderr).decode("utf-8", errors="replace").strip()[:200]
    return r.returncode == 0, err


def _run_tests(repo_path: Path, test_ids: list[str], timeout: int = 60) -> tuple[int, int, str]:
    """Run specific tests. Returns (passed, total, output)."""
    if not test_ids:
        return 0, 0, "no tests"

    # Convert swebench test IDs to pytest node IDs
    # swebench uses format: path::test_name
    pytest_ids = test_ids[:5]  # limit to 5 to keep it fast

    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest"] + pytest_ids + ["-x", "--tb=no", "-q", "--no-header"],
            cwd=str(repo_path),
            capture_output=True,
            timeout=timeout,
        )
        out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
        # Count passed/failed from pytest summary
        passed = out.count(" passed")
        failed = out.count(" failed") + out.count(" error")
        if "passed" in out:
            # Try to parse "X passed"
            import re
            m = re.search(r"(\d+) passed", out)
            p = int(m.group(1)) if m else 0
            m2 = re.search(r"(\d+) failed", out)
            f = int(m2.group(1)) if m2 else 0
            return p, p + f, out[-500:]
        return 0, len(pytest_ids), out[-300:]
    except subprocess.TimeoutExpired:
        return 0, len(pytest_ids), "TIMEOUT"
    except Exception as e:
        return 0, len(pytest_ids), f"ERROR: {e}"


def evaluate(predictions_path: Path, instances_path: Path, repos_dir: Path, run_tests: bool) -> None:
    with open(predictions_path, encoding="utf-8") as f:
        predictions = {json.loads(l)["instance_id"]: json.loads(l) for l in f if l.strip()}

    with open(instances_path, encoding="utf-8") as f:
        instances = {json.loads(l)["instance_id"]: json.loads(l) for l in f if l.strip()}

    rows = []
    n = len(predictions)

    for i, (iid, pred) in enumerate(predictions.items()):
        patch = pred.get("model_patch", "")
        inst = instances.get(iid, {})
        repo_name = inst.get("repo", "").split("/")[-1]
        repo_path = repos_dir / repo_name
        base_commit = inst.get("base_commit", "")
        gt_patch = inst.get("patch", "")
        fail_tests = inst.get("FAIL_TO_PASS", [])

        print(f"\n[{i+1}/{n}] {iid}")

        if not patch.strip():
            rows.append({"id": iid, "status": "no_patch", "sim": 0.0})
            print("  → no patch")
            continue

        if not repo_path.exists():
            rows.append({"id": iid, "status": "no_repo", "sim": 0.0})
            print(f"  → repo not found: {repo_path}")
            continue

        # Reset to base commit
        _git(["reset", "--hard", base_commit], repo_path)
        _git(["clean", "-fd"], repo_path)

        # Check patch applies
        ok, err = _check_patch(repo_path, patch)
        sim = _patch_similarity(patch, gt_patch)
        if not ok:
            rows.append({"id": iid, "status": "apply_fail", "sim": sim, "err": err})
            print(f"  → apply FAILED (sim={sim:.0%}): {err[:80]}")
            continue

        if not run_tests:
            rows.append({"id": iid, "status": "apply_ok", "sim": sim})
            print(f"  → apply OK  sim={sim:.0%}")
            continue

        # Apply and run tests
        _apply_patch(repo_path, patch)
        passed, total, out = _run_tests(repo_path, fail_tests)
        status = "pass" if total > 0 and passed == total else ("fail" if total > 0 else "no_test")
        rows.append({"id": iid, "status": status, "sim": sim, "passed": passed, "total": total})
        print(f"  → tests {passed}/{total}  sim={sim:.0%}  [{status}]")
        if "TIMEOUT" in out or "ERROR" in out:
            print(f"     {out[:100]}")

        # Cleanup
        _git(["reset", "--hard", base_commit], repo_path)
        _git(["clean", "-fd"], repo_path)

    # Summary
    print(f"\n{'='*60}")
    apply_ok  = [r for r in rows if r["status"] not in ("no_patch", "no_repo", "apply_fail")]
    apply_fail = [r for r in rows if r["status"] == "apply_fail"]
    no_patch  = [r for r in rows if r["status"] == "no_patch"]
    passed    = [r for r in rows if r["status"] == "pass"]
    high_sim  = [r for r in apply_ok if r["sim"] >= 0.9]

    print(f"Total:        {n}")
    print(f"No patch:     {len(no_patch)}")
    print(f"Apply failed: {len(apply_fail)}")
    print(f"Apply OK:     {len(apply_ok)}")
    if run_tests:
        print(f"Tests pass:   {len(passed)}  ← estimated resolved")
    print(f"Sim ≥ 90%:    {len(high_sim)}  ← likely correct (GT match)")
    print(f"{'='*60}")

    if high_sim:
        print("High similarity instances:")
        for r in sorted(high_sim, key=lambda x: -x["sim"]):
            print(f"  {r['id']:45s}  sim={r['sim']:.0%}")

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", help="path to predictions .jsonl")
    p.add_argument("--instances", default="eval/instances.jsonl")
    p.add_argument("--repos", default="eval/repos")
    p.add_argument("--test", action="store_true", help="actually run FAIL_TO_PASS tests (slow)")
    p.add_argument("--save", help="save results JSON to this path")
    p.add_argument("--dump-failed", help="write apply-failed instance IDs to this file")
    args = p.parse_args()

    rows = evaluate(
        predictions_path=Path(args.predictions),
        instances_path=Path(args.instances),
        repos_dir=Path(args.repos),
        run_tests=args.test,
    )

    if args.save:
        Path(args.save).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Results saved → {args.save}")

    if args.dump_failed:
        failed_ids = [r["id"] for r in rows if r["status"] == "apply_fail"]
        Path(args.dump_failed).write_text(
            "\n".join(failed_ids) + "\n", encoding="utf-8"
        )
        print(f"Failed IDs ({len(failed_ids)}) → {args.dump_failed}")


if __name__ == "__main__":
    main()
