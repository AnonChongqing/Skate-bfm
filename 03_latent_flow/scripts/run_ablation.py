from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from skate_bfm_flow.evaluation.ablation import config_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--matrix", required=True, help="YAML mapping from dotted config key to values")
    parser.add_argument("--script", default="03_latent_flow/scripts/train_online_sac.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    rows = []
    for run_id, overrides in enumerate(config_matrix(matrix)):
        command = [sys.executable, args.script, "--config", args.config]
        for override in overrides:
            command.extend(("--set", override))
        command.extend(("--set", f"experiment.name=ablation_{run_id:04d}"))
        return_code = 0 if args.dry_run else subprocess.run(command, check=False).returncode
        rows.append({"run_id": run_id, "overrides": overrides, "return_code": return_code})
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(f"prepared {len(rows)} ablation runs")


if __name__ == "__main__":
    main()
