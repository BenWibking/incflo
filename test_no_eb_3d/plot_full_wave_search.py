"""Plot the phase-zero axial wave vorticity and the independently measured shape change."""

import argparse
import json
from pathlib import Path

import numpy as np
from plot_rotating_wave import meridian_modes, plt
from solve_full_rotating_wave import read


def draw(runs, path):
    extent, height = 72.0, 6.5
    axis = np.linspace(-extent, extent, 384)
    xx, yy = np.meshgrid(axis, axis)
    rr, phi = np.hypot(xx, yy), np.arctan2(yy, xx)
    radius = np.linspace(0.0001, rr.max() + 0.01, 800)
    R = np.hypot(radius, height)
    data = []
    for checkpoint, log in runs:
        cfg, b, c = read(checkpoint)
        history = json.loads(log.read_text())
        if any(
            history["setup"][k] != cfg[k]
            for k in ("m", "omega", "damping", "amplitude", "basis", "base_sha256")
        ):
            raise ValueError(
                "The evolution and velocity checkpoint describe different waves"
            )
        mode = meridian_modes({**cfg, "kappa": cfg["m"]}, b, c, radius, height, "curl")
        vort = (height * mode[:, :, 0] - radius * mode[:, :, 1]) / R
        field = sum(
            2 * (np.interp(rr, radius, v) * np.exp(1j * j * cfg["m"] * phi)).real
            for j, v in enumerate(vort[1:], 1)
        )
        complete = [
            r for r in history["runs"] if r.get("status", "completed") == "completed"
        ]
        best = max(complete or history["runs"], key=lambda r: r["steps"])
        data.append((cfg, field, best["history"]))
    fig, axes = plt.subplots(
        1, len(data) + 1, figsize=(3.65 * (len(data) + 1), 3.4), layout="constrained"
    )
    lim = np.quantile(np.abs(np.stack([f for _, f, _ in data])), 0.995)
    change = axes[-1]
    for ax, (cfg, field, rows) in zip(axes[:-1], data, strict=True):
        im = ax.imshow(
            field,
            extent=(-extent, extent, -extent, extent),
            origin="lower",
            vmin=-lim,
            vmax=lim,
            cmap="RdBu_r",
        )
        ax.set(
            title=f"m = {cfg['m']}",
            xlabel="x",
            ylabel="y",
            xticks=[-60, 0, 60],
            yticks=[-60, 0, 60],
        )
        t = np.array([r["s"] for r in rows]) * abs(cfg["omega"]) / (2 * np.pi)
        change.plot(
            t,
            [r["layer_phase_aligned_change"] for r in rows],
            ".-",
            label=f"m = {cfg['m']}",
        )
    fig.colorbar(
        im,
        ax=axes[:-1].tolist(),
        location="bottom",
        shrink=0.69,
        pad=0.04,
        extend="both",
        label="Axial wave vorticity at z = 6.5, initial phase",
    )
    change.set(
        title="Independent time evolution",
        xlabel="Fraction of a predicted period",
        ylabel="Relative shape change on the layer",
        xlim=(0, 1),
        ylim=(0, None),
    )
    change.grid(axis="y", alpha=0.2)
    change.spines[["top", "right"]].set_visible(False)
    change.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        nargs=2,
        metavar=("CHECKPOINT", "EVOLUTION_JSON"),
        action="append",
        required=True,
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    draw(a.run, a.out)


if __name__ == "__main__":
    main()
