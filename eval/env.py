"""Repository setup for SWE-bench evaluation."""
from __future__ import annotations

import subprocess
from pathlib import Path


def clone_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    """Clone repo at base_commit into workdir/<repo_name>. Returns the repo path."""
    repo_name = repo.split("/")[-1]
    dest = workdir / repo_name
    if (dest / ".git").exists() and any(p.name != ".git" for p in dest.iterdir()):
        # Repo already cloned — hard reset + clean to remove any agent-created files,
        # then checkout the correct base commit for this instance
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(dest), capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=str(dest), capture_output=True)
        subprocess.run(["git", "checkout", base_commit], cwd=str(dest), check=True, capture_output=True)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(dest)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=str(dest), check=True, capture_output=True,
    )
    return dest
