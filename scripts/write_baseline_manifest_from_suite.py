#!/usr/bin/env python3
"""Write baseline_manifest.json files for checkpoints referenced by a suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="Suite JSON with baseline runs")
    return parser.parse_args()


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite)
    if not suite_path.exists():
        raise SystemExit(f"Missing suite: {suite_path}")

    cfg = json.loads(suite_path.read_text())
    defaults = cfg.get("defaults", {})
    runs = cfg.get("runs", [])
    if not runs:
        raise SystemExit("Suite contains no runs.")

    for spec in runs:
        ckpt_path = spec.get("save_model_path") or spec.get("load_model_path")
        if not ckpt_path:
            continue
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            print(f"Skipping missing checkpoint: {ckpt}")
            continue

        manifest_path = ckpt.parent / "baseline_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {}

        task = spec.get("task", "unknown")
        seed = int(spec.get("seed", 0))
        run_id = spec.get("run_id", "baseline")
        train_steps = int(defaults.get("train_steps", 0))
        ft_steps = int(defaults.get("ft_steps", 0))
        digest = _hash_file(ckpt)

        manifest[str(ckpt)] = {
            "task": task,
            "seed": seed,
            "run_id": run_id,
            "train_steps": train_steps,
            "ft_steps": ft_steps,
            "hash": digest,
            "status": "reused",
            "updated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "path": str(ckpt),
            "abs_path": str(ckpt.resolve()),
        }

        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
