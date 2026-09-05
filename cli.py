"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .experiment import run_experiment
from .validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magfield")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validate", help="run analytical physics checks")
    validation.add_argument("--segments", type=int, default=256)
    reproduction = subparsers.add_parser("reproduce", help="run the configured experiment")
    reproduction.add_argument("--config", default="configs/quick_reproduction.yaml")
    reproduction.add_argument("--output-dir", default="results")
    reproduction.add_argument("--backend", choices=["scipy", "cvxpy"], default="scipy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        summary = run_validation(args.segments)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0 if summary.passed else 1
    summary = run_experiment(args.config, output_dir=args.output_dir, backend=args.backend)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

