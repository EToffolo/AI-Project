"""Build the English report's vector figures and numbers from a completed run.

Run from the repository root. The generated assets are kept beside the LaTeX
source so the report can be compiled without the unversioned experiment outputs.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("ridge", "mlp", "mlp_augmented")
LABELS = {"ridge": "Ridge", "mlp": "MLP", "mlp_augmented": "MLP + rotations"}
COLORS = {"ridge": "#697386", "mlp": "#176A96", "mlp_augmented": "#C65A25"}


def build(results, output):
    config = json.loads((results / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("A completed experiment is required")
    with (results / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(model, seed, size, domain) for model in MODELS
                for seed in config["seeds"] for size in config["train_sizes"]
                for domain in ("id", "ood")}
    actual = {(row["model"], int(row["seed"]), int(row["train_size"]), row["domain"])
              for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Incomplete or duplicated metric records")

    def values(model, domain, field, size=None):
        size = max(config["train_sizes"]) if size is None else size
        selected = sorted((row for row in rows if row["model"] == model
                           and row["domain"] == domain and int(row["train_size"]) == size),
                          key=lambda row: int(row["seed"]))
        return np.array([float(row[field]) for row in selected])

    def stat(model, domain, field, size=None, factor=1):
        array = factor * values(model, domain, field, size)
        return float(array.mean()), float(array.std(ddof=1))

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.30), layout="constrained")
    sizes = sorted(config["train_sizes"])
    for ax, domain, title in zip(axes, ("id", "ood"), ("In-distribution", "Extrapolation")):
        for model in MODELS:
            stats = np.array([stat(model, domain, "error_median", size, 100) for size in sizes])
            ax.plot(sizes, stats[:, 0], "o-", color=COLORS[model], label=LABELS[model],
                    lw=1.5, markersize=4)
            ax.fill_between(sizes, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1],
                            color=COLORS[model], alpha=0.16, linewidth=0)
        ax.set(xscale="log", xticks=sizes, xticklabels=[f"{size:,}" for size in sizes],
               xlabel="Training source examples", ylabel="Median relative error (%)", title=title)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.20)
    axes[1].legend(loc="upper right", frameon=False)
    fig.savefig(output / "learning-curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), layout="constrained")
    factors = config["scale_factors"]
    for ax, domain, title in zip(axes, ("id", "ood"), ("ID sources", "OOD sources")):
        for model in MODELS:
            stats = np.array([stat(model, domain, f"scale_{factor}_defect_median")
                              for factor in factors])
            ax.plot(factors, stats[:, 0], "o-", color=COLORS[model], label=LABELS[model],
                    lw=1.5, markersize=4)
            ax.fill_between(factors, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1],
                            color=COLORS[model], alpha=0.16, linewidth=0)
        ax.set(xscale="log", xticks=factors, xticklabels=[f"{factor:g}" for factor in factors],
               xlabel="Form multiplier t", ylabel="Median homogeneity defect", title=title)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.20)
    axes[1].legend(loc="upper left", frameon=False)
    fig.savefig(output / "homogeneity.pdf")
    plt.close(fig)

    size, seed = max(config["train_sizes"]), config["seeds"][0]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.75), layout="constrained", sharey=True)
    for ax, model in zip(axes, ("mlp", "mlp_augmented")):
        history = json.loads((results / "histories" / f"{model}_n{size}_seed{seed}.json")
                             .read_text(encoding="utf-8"))
        epochs = [row["epoch"] for row in history["history"]]
        ax.plot(epochs, [row["train_loss"] for row in history["history"]],
                label="Training", color=COLORS[model], lw=1.4)
        ax.plot(epochs, [row["val_loss"] for row in history["history"]],
                label="Validation", color="#5C6470", lw=1.2)
        ax.axvline(history["best_epoch"], color="#9298A0", linestyle=":", lw=1)
        ax.set(title=LABELS[model], xlabel="Epoch")
        ax.grid(axis="y", alpha=0.20)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean squared relative error")
    fig.savefig(output / "training-curves.pdf")
    plt.close(fig)

    accuracy = []
    for model in MODELS:
        cells = []
        for domain in ("id", "ood"):
            mean, sd = stat(model, domain, "error_median", factor=100)
            cells.append(rf"${mean:.2f}\pm{sd:.2f}$")
            mean, _ = stat(model, domain, "error_p95", factor=100)
            cells.append(f"{mean:.2f}")
        accuracy.append(LABELS[model] + " & " + " & ".join(cells) + r" \\")
    geometry = []
    for model in MODELS:
        cells = []
        for domain in ("id", "ood"):
            for field in ("eq_so7_median", "eq_gl7_median"):
                mean, sd = stat(model, domain, field)
                cells.append(rf"${mean:.4f}\pm{sd:.4f}$")
        geometry.append(LABELS[model] + " & " + " & ".join(cells) + r" \\")
    macros = [f"% Generated from {(results / 'metrics.csv').as_posix()}; do not edit numeric entries by hand.",
              "\\newcommand{\\accuracyrows}{\n" + "\n".join(accuracy) + "\n}",
              "\\newcommand{\\geometryrows}{\n" + "\n".join(geometry) + "\n}"]
    for domain, prefix in (("id", "ID"), ("ood", "OOD")):
        base = stat("mlp", domain, "error_median")[0]
        aug = stat("mlp_augmented", domain, "error_median")[0]
        macros.append(rf"\newcommand{{\{prefix}AccuracyGain}}{{{100 * (1 - aug / base):.2f}}}")
        base = stat("mlp", domain, "eq_so7_median")[0]
        aug = stat("mlp_augmented", domain, "eq_so7_median")[0]
        macros.append(rf"\newcommand{{\{prefix}EquivarianceGain}}{{{100 * (1 - aug / base):.2f}}}")
        for model, key in (("mlp", "Plain"), ("mlp_augmented", "Augmented")):
            mean, _ = stat(model, domain, "scale_4.0_error_median", factor=100)
            macros.append(rf"\newcommand{{\{prefix}{key}ScaleFourError}}{{{mean:.1f}}}")
    (output / "results-summary.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"Built three vector figures and numerical tables in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/full"))
    parser.add_argument("--output", type=Path, default=Path("docs/relatorio/relatorio-assets"))
    args = parser.parse_args()
    build(args.results, args.output)
