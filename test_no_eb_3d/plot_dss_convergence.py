"""Plot measured one-cycle errors and an optional independent Richardson holdout."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory, out, holdout=None):
    def read(name):
        return json.loads((directory / name).read_text())

    levels = [read(f"{level}-score.json") for level in ("coarse", "middle", "fine")]
    guide = read("richardson-score.json")
    records = [*levels, guide] + ([holdout] if holdout is not None else [])
    if len({(row["trial"], row["control_sha256"]) for row in records}) != 1:
        raise ValueError(
            "The convergence plot requires the same trial and control at every resolution"
        )
    grids = [v["grid"] for v in levels]
    ticks = list(grids)
    if holdout is not None:
        if (
            holdout["grid"] <= grids[-1]
            or holdout["steps"] * grids[-1] != levels[-1]["steps"] * holdout["grid"]
        ):
            raise ValueError(
                "Holdout resolution must extend the same refinement sequence"
            )
        ticks.append(holdout["grid"])
    specifications = (
        ("velocity_return", "Velocity"),
        ("vorticity_return", "Vorticity"),
        ("m19_return", "Complex spiral mode 19"),
    )
    fig, panels = plt.subplots(1, 3, figsize=(10.5, 3.9), layout="constrained")
    fig.suptitle(
        f"Trial {guide['trial']}  ·  One-cycle return error under refinement",
        fontsize=15,
    )
    for ax, (metric, title) in zip(panels, specifications):
        y = np.array([100 * row[metric] for row in levels])
        ax.plot(grids, y, marker="o", color="#225e78", lw=2, label="Optimization grids")
        all_y = list(y)
        if holdout is None:
            for x, v in zip(grids, y):
                ax.annotate(
                    f"{v:.2f}%",
                    (x, v),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        else:
            check = holdout["metrics"][metric]
            predicted, measured = 100 * check["prediction"], 100 * check["measured"]
            ax.plot(
                [grids[-1], ticks[-1]],
                [y[-1], predicted],
                color="#99501f",
                ls="--",
                lw=1.2,
            )
            ax.plot(
                ticks[-1],
                predicted,
                "D",
                color="#99501f",
                mfc="white",
                mew=1.6,
                ms=8,
                label="Holdout prediction",
            )
            ax.plot(
                ticks[-1],
                measured,
                "x",
                color="#222222",
                ms=9,
                mew=2,
                label="Holdout measurement",
            )
            all_y.extend((predicted, measured))
        span = max(max(all_y) - min(all_y), 0.2)
        ax.set(
            xticks=ticks,
            xticklabels=[f"{n}³" for n in ticks],
            ylim=(min(all_y) - 0.18 * span, max(all_y) + 0.36 * span),
            xlim=(ticks[0] - 10, ticks[-1] + 10),
            xlabel="Grid",
            title=title,
        )
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.24)
    panels[0].set_ylabel("Return error (%)")
    if holdout is not None:
        fig.legend(
            *panels[0].get_legend_handles_labels(),
            loc="outside lower center",
            ncols=3,
            frameon=False,
        )
    for ext in ("png", "pdf"):
        fig.savefig(out / f"convergence.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--holdout", type=Path)
    args = parser.parse_args()
    plot(
        args.trial,
        args.out,
        json.loads(args.holdout.read_text()) if args.holdout else None,
    )
