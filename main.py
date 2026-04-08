from __future__ import annotations

import argparse
import json

from pipelines import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly detection and RCA pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
