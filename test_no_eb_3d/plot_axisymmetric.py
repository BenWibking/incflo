"""Compare a fixed-torque axisymmetric equilibrium to the digitized final-profile mean."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
from plot_rotating_wave import centers, meridian_modes, plt
from scipy.interpolate import RegularGridInterpolator
from solve_axisymmetric import read


def draw(args):
    cfg, basis, coeff = read(args.checkpoint)
    with np.load(args.profile, allow_pickle=False) as data:
        r, z = centers(float(data["r"][-1]), 200), centers(float(data["z"][-1]), 144)
        rr, zz = np.meshgrid(r, z)
        positions = np.stack([zz, rr], axis=-1)
        ref = [
            RegularGridInterpolator((data["z"], data["r"]), data[k])(positions)
            for k in ("mean_wr", "mean_wphi", "mean_wz")
        ]
    state = meridian_modes(
        {**cfg, "kappa": 0, "omega": 0}, basis, [coeff], rr, zz, "velocity"
    )[0].real
    R = np.hypot(rr, zz)
    computed = [
        (rr * state[..., 0] + zz * state[..., 1]) / R,
        state[..., 2],
        (zz * state[..., 0] - rr * state[..., 1]) / R,
    ]
    histories = [json.loads(f.read_text()) for f in args.history]
    if "verification" in cfg:
        source = cfg["verification"]
        measurements = source["measurements"]
    else:
        source = json.loads((args.checkpoint.parent / "axisymmetric.json").read_text())
        measurements = source["stages"][cfg["stage"]]["audits"]
    selected = measurements[-1]
    for history in histories:
        for key in ("drive_r", "drive_z", "reference_force", "g", "nu"):
            if not np.isclose(cfg[key], history[key], rtol=1e-10, atol=0):
                raise ValueError(
                    f"Plot histories at the same physical controls; {key} differs"
                )
    balance_tol = 100 * source.get("balance_tolerance", 0.03)
    field_title = (
        "Equilibrated axisymmetric field"
        if cfg["balance_passed"]
        else "Axisymmetric trial field"
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stixsans",
        }
    )
    fig = plt.figure(figsize=(12.2, 7.8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.45, 0.85],
        width_ratios=[1, 1, 0.035],
        left=0.09,
        right=0.95,
        top=0.84,
        bottom=0.12,
        hspace=0.58,
        wspace=0.23,
    )
    a, b, cb, d = (
        fig.add_subplot(s) for s in (gs[0, 0], gs[0, 1], gs[0, 2], gs[1, :2])
    )
    peak = max(ref[1].max(), computed[1].max())
    norm = Normalize(vmin=min(0, computed[1].min()), vmax=peak)
    fig.text(
        0.09,
        0.959,
        "Axisymmetric flow at fixed torque",
        fontsize=21,
        weight="bold",
        color="#172b38",
        va="top",
    )
    fig.text(
        0.09,
        0.905,
        rf"$g={cfg['g']:g}$,  $\nu=1$    ·    Torque amplitude {cfg['force']:.3g}    ·    "
        rf"Momentum imbalance {selected['residual_over_force']:.2%} of the applied force",
        fontsize=11,
        color="#3d5667",
    )
    for ax, pieces, title in (
        (a, ref, "Mean reconstructed from the final profile"),
        (b, computed, field_title),
    ):
        radial, swirl, vertical = pieces
        im = ax.pcolormesh(
            rr, zz, swirl, cmap="viridis", norm=norm, shading="nearest", rasterized=True
        )
        streams = ax.streamplot(
            r,
            z,
            radial,
            vertical,
            density=(0.9, 0.78),
            color="white",
            linewidth=0.78,
            arrowsize=0.85,
        )
        streams.lines.set_path_effects(
            [pe.Stroke(linewidth=1.4, foreground="#334252", alpha=0.55), pe.Normal()]
        )
        ax.set(
            aspect="equal",
            xlim=(0, 75),
            ylim=(0, 50),
            xlabel="Cylindrical radius",
            ylabel="Height",
        )
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
    fig.colorbar(im, cax=cb, label="Azimuthal velocity")
    colors = plt.colormaps["viridis"](np.linspace(0.12, 0.78, len(histories)))
    for color, history in zip(colors, histories, strict=True):
        stages = history["stages"]
        values = np.array([s["fraction"] for s in stages])
        monotone = np.all(np.diff(values) >= 0) or np.all(np.diff(values) <= 0)
        n = history["unknowns"]
        d.plot(
            [s["fraction"] for s in stages],
            [100 * s["audits"][-1]["residual_over_force"] for s in stages],
            ".-" if monotone else ".",
            color=color,
            lw=1.75,
            ms=5,
            label=f"Degree {history['basis']['degree']}, {n:,} coefficients",
        )
    d.axhline(
        balance_tol,
        ls="--",
        color="#646b70",
        lw=1.05,
        label=f"{balance_tol:g}% balance criterion",
    )
    if histories:
        ticks = sorted(
            {s["fraction"] for history in histories for s in history["stages"]}
        )
        coord, xlabel, xscale = (
            cfg["fraction"],
            "Torque relative to the linear calibration",
            "log",
        )
    else:
        ticks = sorted({m["grid"][0] for m in measurements})
        coord, xlabel, xscale = selected["grid"][0], "Radial quadrature nodes", "linear"
        for degree in sorted({m["pressure_degree"] for m in measurements}):
            rows = [m for m in measurements if m["pressure_degree"] == degree]
            d.plot(
                [m["grid"][0] for m in rows],
                [100 * m["residual_over_force"] for m in rows],
                ".-",
                label=f"Pressure degree {degree}",
            )
    d.scatter(
        coord,
        100 * selected["residual_over_force"],
        facecolor="white",
        edgecolor="#8e3b26",
        s=60,
        lw=1.8,
        zorder=5,
        label="Plotted field",
    )
    d.set(
        xlabel=xlabel,
        ylabel="Residual / force (%)",
        xscale=xscale,
        yscale="log",
    )
    d.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
    d.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    d.grid(axis="y", alpha=0.2, which="both")
    d.set_title(
        "Independent momentum balance under refinement",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    d.legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8)
    # Leave the legend inside the drawing. The two physical-space panels retain equal scale.
    box = d.get_position()
    d.set_position([box.x0, box.y0, box.width * 0.77, box.height])
    d.xaxis.set_tick_params(which="minor", bottom=False)
    cb.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.text(
        0.09,
        0.033,
        "Color shows swirl. Streamlines show flow in the meridional plane. All distances and velocities are in similarity units.",
        fontsize=9,
        color="#48606e",
    )
    fig.savefig(args.out, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("rotating_figure_seed.npz"),
    )
    p.add_argument("--history", action="append", default=[], type=Path)
    p.add_argument("--out", type=Path, required=True)
    torch.set_num_threads(1)
    draw(p.parse_args())
