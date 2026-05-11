"""Re-run instances with empty patches and merge results back.

Usage:
    python eval/rerun_empty.py --predictions eval/predictions/astropy_glm.jsonl \\
                               --instances eval/instances.jsonl \\
                               --label astropy_glm

For each empty-patch instance this script will:
  1. Run the eval runner in a subprocess
  2. Save stdout+stderr to eval/logs/<instance_id>.log
  3. Merge the new patch into the predictions file
  4. Merge the new TaskResult into eval/results/<label>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _find_empty(predictions_path: Path) -> list[str]:
    empty = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("model_patch", "").strip():
                empty.append(rec["instance_id"])
    return empty


def _load_instances(instances_path: Path, instance_ids: set[str]) -> list[dict]:
    found = []
    with open(instances_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("instance_id") in instance_ids:
                found.append(rec)
    return found


def _run_one(instance: dict, label_tmp: str, workdir: Path, log_path: Path) -> None:
    """Run a single instance via the eval runner, capturing output to log_path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(json.dumps(instance) + "\n")
        tmp_instances = tf.name

    try:
        cmd = [
            sys.executable, "eval/runner.py", "run",
            "--instances", tmp_instances,
            "--label", label_tmp,
            "--n", "1",
            "--output", "eval/results",
            "--workdir", str(workdir),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_f:
            subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        os.unlink(tmp_instances)


def _load_predictions(path: Path) -> dict[str, dict]:
    preds = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    preds[rec["instance_id"]] = rec
    return preds


def _save_predictions(preds: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in preds.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_results_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"label": "", "model": "", "results": []}


def _save_results_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="eval/predictions/astropy_glm.jsonl")
    parser.add_argument("--instances", default="eval/instances.jsonl")
    parser.add_argument("--label", default="astropy_glm")
    parser.add_argument("--workdir", default="eval/repos")
    parser.add_argument("--log-dir", default="eval/logs")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    instances_path = Path(args.instances)
    results_path = Path("eval/results") / f"{args.label}.json"
    log_dir = Path(args.log_dir)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Find empty patches
    empty_ids = _find_empty(predictions_path)
    if not empty_ids:
        print("No empty patches found.")
        return
    print(f"Found {len(empty_ids)} empty-patch instances: {empty_ids}")

    # 2. Load matching instances
    instances = _load_instances(instances_path, set(empty_ids))
    found_ids = {i["instance_id"] for i in instances}
    missing = set(empty_ids) - found_ids
    if missing:
        print(f"WARNING: {len(missing)} instance(s) not found in {instances_path}: {missing}")

    # 3. Run each instance
    for inst in instances:
        iid = inst["instance_id"]
        label_tmp = f"_tmp_{iid.replace('/', '_').replace('-', '_')}"
        log_path = log_dir / f"{iid}.log"
        print(f"\n[running] {iid}  →  {log_path}")
        _run_one(inst, label_tmp, workdir, log_path)
        print(f"  done, log saved")

        # 4a. Merge predictions
        tmp_pred_path = Path("eval/predictions") / f"{label_tmp}.jsonl"
        if tmp_pred_path.exists():
            tmp_preds = _load_predictions(tmp_pred_path)
            if iid in tmp_preds and tmp_preds[iid].get("model_patch", "").strip():
                main_preds = _load_predictions(predictions_path)
                main_preds[iid] = tmp_preds[iid]
                # Restore original model name
                main_preds[iid]["model_name_or_path"] = main_preds[iid].get(
                    "model_name_or_path", args.label
                )
                _save_predictions(main_preds, predictions_path)
                print(f"  patch merged ({len(tmp_preds[iid]['model_patch'])} chars)")
            else:
                print(f"  still empty patch after re-run")
            tmp_pred_path.unlink(missing_ok=True)
        else:
            print(f"  WARNING: no predictions file at {tmp_pred_path}")

        # 4b. Merge results JSON
        tmp_results_path = Path("eval/results") / f"{label_tmp}.json"
        if tmp_results_path.exists():
            tmp_data = _load_results_json(tmp_results_path)
            main_data = _load_results_json(results_path)
            tmp_by_id = {r["instance_id"]: r for r in tmp_data.get("results", [])}
            main_results = main_data.get("results", [])
            for j, r in enumerate(main_results):
                if r["instance_id"] == iid and iid in tmp_by_id:
                    main_results[j] = tmp_by_id[iid]
                    break
            else:
                if iid in tmp_by_id:
                    main_results.append(tmp_by_id[iid])
            main_data["results"] = main_results
            _save_results_json(main_data, results_path)
            print(f"  results JSON updated")
            tmp_results_path.unlink(missing_ok=True)

    print(f"\nDone. Logs in {log_dir}/")
    print(f"Updated predictions: {predictions_path}")
    print(f"Updated results: {results_path}")


if __name__ == "__main__":
    main()
