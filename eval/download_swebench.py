"""Download SWE-bench Lite dataset and save as instances.jsonl.

Tries three methods in order:
  1. huggingface_hub  (pip install huggingface_hub)
  2. datasets library (pip install datasets)
  3. Direct HTTP download (no extra deps)

Usage:
    python eval/download_swebench.py
    python eval/download_swebench.py --split test --n 20 --out eval/instances.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DATASET_ID = "princeton-nlp/SWE-bench_Lite"

# Direct parquet URLs (HuggingFace CDN, no auth needed for public datasets)
PARQUET_URLS = {
    "test": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/data/test-00000-of-00001.parquet",
    "dev":  "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/data/dev-00000-of-00001.parquet",
}

REQUIRED_FIELDS = [
    "instance_id", "repo", "base_commit", "problem_statement",
    "hints_text", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    "version", "environment_setup_commit", "created_at",
]


# ---------------------------------------------------------------------------
# Method 1: huggingface_hub
# ---------------------------------------------------------------------------

def _download_via_hub(split: str, n: int | None) -> list[dict] | None:
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd  # noqa: F401 — needed to read parquet
    except ImportError:
        return None

    print("Downloading via huggingface_hub...")
    try:
        local = hf_hub_download(
            repo_id=DATASET_ID,
            filename=f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        import pandas as pd
        df = pd.read_parquet(local)
        rows = df.to_dict(orient="records")
        return rows[:n] if n else rows
    except Exception as e:
        print(f"  huggingface_hub failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Method 2: datasets library
# ---------------------------------------------------------------------------

def _download_via_datasets(split: str, n: int | None) -> list[dict] | None:
    try:
        import datasets as ds
    except ImportError:
        return None

    print("Downloading via 'datasets' library...")
    try:
        dataset = ds.load_dataset(DATASET_ID, split=split, trust_remote_code=True)
        rows = [dict(row) for row in dataset]
        return rows[:n] if n else rows
    except Exception as e:
        print(f"  datasets library failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Method 3: direct HTTP (parquet → need pandas/pyarrow)
# ---------------------------------------------------------------------------

def _download_via_http(split: str, n: int | None) -> list[dict] | None:
    url = PARQUET_URLS.get(split)
    if not url:
        print(f"No direct URL known for split '{split}'")
        return None

    try:
        import pandas as pd
        import io
    except ImportError:
        print("pandas not installed; cannot read parquet directly.")
        print("Install with: pip install pandas pyarrow")
        return None

    print(f"Downloading parquet directly from HuggingFace CDN...")
    print(f"  URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "minicode-eval/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        print(f"  Downloaded {len(data)/1024/1024:.1f} MB")
        df = pd.read_parquet(io.BytesIO(data))
        rows = df.to_dict(orient="records")
        return rows[:n] if n else rows
    except Exception as e:
        print(f"  Direct HTTP failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Normalise row fields
# ---------------------------------------------------------------------------

def _normalise(row: dict) -> dict:
    """Ensure list fields stored as JSON strings are kept as strings (our runner uses json.loads)."""
    out = {}
    for k in REQUIRED_FIELDS:
        v = row.get(k, "")
        if isinstance(v, list):
            out[k] = json.dumps(v)
        elif v is None:
            out[k] = ""
        else:
            out[k] = v
    # pass through any extra fields
    for k, v in row.items():
        if k not in out:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_local_parquet(split: str, n: int | None) -> list[dict] | None:
    """Read a manually downloaded parquet file from the eval/ directory."""
    candidates = [
        Path(f"eval/{split}-00000-of-00001.parquet"),
        Path(f"eval/swebench_lite_{split}.parquet"),
        Path(f"eval/{split}.parquet"),
    ]
    for path in candidates:
        if path.exists():
            try:
                import pandas as pd
                print(f"Reading local file: {path}")
                df = pd.read_parquet(path)
                rows = df.to_dict(orient="records")
                return rows[:n] if n else rows
            except Exception as e:
                print(f"  Failed to read {path}: {e}")
    return None


def download(split: str = "test", n: int | None = None, out: Path = Path("eval/instances.jsonl")) -> None:
    rows = (
        _load_local_parquet(split, n)
        or _download_via_hub(split, n)
        or _download_via_datasets(split, n)
        or _download_via_http(split, n)
    )

    if rows is None:
        print("\nAll download methods failed.")
        print("Install one of: pip install huggingface_hub pandas pyarrow")
        print("            or: pip install datasets")
        sys.exit(1)

    rows = [_normalise(r) for r in rows]

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} instances → {out}")
    print(f"Fields: {', '.join(REQUIRED_FIELDS)}")

    # Quick sanity check
    sample = rows[0]
    print(f"\nSample instance:")
    print(f"  instance_id : {sample.get('instance_id')}")
    print(f"  repo        : {sample.get('repo')}")
    print(f"  FAIL_TO_PASS: {str(sample.get('FAIL_TO_PASS', ''))[:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SWE-bench Lite")
    parser.add_argument("--split", default="test", choices=["test", "dev"],
                        help="Dataset split (default: test = 300 instances)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N instances (default: all)")
    parser.add_argument("--out", default="eval/instances.jsonl",
                        help="Output .jsonl path")
    args = parser.parse_args()

    download(split=args.split, n=args.n, out=Path(args.out))
