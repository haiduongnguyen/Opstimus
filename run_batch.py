from __future__ import annotations

import argparse
import json

from pipelines.batch import run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple anomaly detection configs and aggregate results")
    parser.add_argument(
        "--config-root",
        default="config",
        help="Config file or directory containing JSON configs",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/batch_runs",
        help="Directory to store leaderboard and batch summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_batch(args.config_root, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
