"""CLI: python -m g2metric --help."""

import argparse
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description="MM845 - metric induced by a G2 structure")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate data, train all models and produce report")
    run.add_argument("--config", default="configs/full.json")
    run.add_argument("--output", default="outputs/full")
    generate = commands.add_parser("generate", help="Generate and validate synthetic data only")
    generate.add_argument("--config", default="configs/full.json")
    generate.add_argument("--output", default="data/generated/g2.npz")
    predict = commands.add_parser("predict", help="Predict metrics for a .npy array (...,35)")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .experiment import read_config, run_experiment
            run_experiment(read_config(args.config), args.output)
        elif args.command == "generate":
            from .data import generate_and_save
            from .experiment import read_config
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            generate_and_save(args.output, **read_config(args.config)["data"])
            print(f"Saved {args.output}")
        else:
            from .geometry import metric_exact
            from .training import load_predictor
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            phi = np.load(args.input, allow_pickle=False)
            if phi.ndim not in (1, 2) or phi.shape[-1] != 35 or phi.size == 0 or not np.isfinite(phi).all():
                raise ValueError("Expected a finite nonempty (35,) or (N,35) input")
            metric_exact(phi)  # Reject forms outside the positive orbit/orientation.
            prediction = load_predictor(args.checkpoint)(phi)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.output).open("wb") as stream:
                np.save(stream, prediction, allow_pickle=False)
            print(f"Saved metrics {prediction.shape} to {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
