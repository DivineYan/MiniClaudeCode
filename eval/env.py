"""Repository environment setup for SWE-bench evaluation.

Each task needs:
1. A clean repo at base_commit
2. A way to apply a patch
3. A way to run the test suite and collect pass/fail
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def clone_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    """Clone repo at base_commit into workdir/<repo_name>. Returns the repo path."""
    repo_name = repo.split("/")[-1]
    dest = workdir / repo_name
    if dest.exists():
        return dest

    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(dest)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=str(dest), check=True, capture_output=True,
    )
    return dest


def apply_patch(repo_path: Path, patch: str) -> bool:
    """Apply a unified diff patch. Returns True if successful."""
    if not patch.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_file = f.name
    try:
        result = subprocess.run(
            ["git", "apply", "--whitespace=fix", patch_file],
            cwd=str(repo_path), capture_output=True,
        )
        return result.returncode == 0
    finally:
        Path(patch_file).unlink(missing_ok=True)


def run_tests(repo_path: Path, test_ids: list[str], timeout: int = 120) -> dict[str, bool]:
    """Run specific test IDs, return {test_id: passed} mapping."""
    if not test_ids:
        return {}

    result = subprocess.run(
        ["python", "-m", "pytest", "--tb=no", "-q", "--no-header"] + test_ids,
        cwd=str(repo_path),
        capture_output=True, timeout=timeout,
    )
    output = result.stdout.decode("utf-8", errors="replace")

    # Parse pytest output: "PASSED" / "FAILED" / "ERROR" per test
    passed: dict[str, bool] = {}
    for line in output.splitlines():
        for tid in test_ids:
            test_name = tid.split("::")[-1]
            if test_name in line:
                passed[tid] = "PASSED" in line or "passed" in line
                break

    # If we couldn't parse individually, use overall return code
    if not passed:
        all_pass = result.returncode == 0
        passed = {tid: all_pass for tid in test_ids}

    return passed


def check_tests_pass(repo_path: Path, test_ids: list[str]) -> bool:
    """Return True if ALL given test IDs pass."""
    if not test_ids:
        return True
    results = run_tests(repo_path, test_ids)
    return all(results.get(tid, False) for tid in test_ids)
