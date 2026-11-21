"""Plot a saved half-space wave and its measured unforced residual history."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from similarity_rotating import WaveMeridian
from solve_rotating_wave import read_state

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import EngFormatter, LogLocator, MaxNLocator, NullLocator


def meridian_modes(cfg, basis, coeff, radius, height, field):
    """Evaluate paired (cylindrical r, z) points, returning physical spherical Fourier coefficients at s=0."""
    radius, height = np.broadcast_arrays(radius, height)
    R = np.hypot(radius, height).ravel()
    if np.any(R == 0) or np.any(radius <= 0) or np.any(height < 0):
        raise ValueError("Sample positive cylindrical radii in the upper half-space")
    eta = height.ravel() / R
    tail = (1 - cfg["g"]) / cfg["g"]
    phase = cfg["omega"] / (2 * cfg["g"]) * np.log1p((R / basis.phase_radius) ** 2)
    modes = np.zeros((len(coeff), len(R), 3), dtype=complex)
    for start in range(0, len(R), 2048):
        ix = slice(start, start + 2048)
        mer = WaveMeridian(basis, R[ix] / (R[ix] + basis.radius), eta[ix])
        for j, cj in enumerate(coeff):
            op = mer.operator(
                j, cfg["kappa"], cfg["g"], cfg["omega"], tail, field, "cpu"
            )
            offset = 0
            for shape, components in zip(op.shapes, op.parts, strict=True):
                count = math.prod(shape)
                block = cj[offset : offset + count].reshape(shape)
                offset += count
                for i, terms in enumerate(components):
                    for rr, aa in terms:
                        modes[j, ix, i] += ((rr @ block) * aa).sum(axis=1)
            modes[j, ix] *= np.exp(1j * j * phase[ix, None])
    return modes.reshape((len(coeff), *radius.shape, 3))


def centers(limit, n):
    return (np.arange(n) + 0.5) * limit / n


def draw(args):
    cfg, basis, coeff = read_state(args.checkpoint)
    if cfg["forcing"] != 0:
        raise ValueError("The zero-force figure requires a zero-force checkpoint")
    histories = [
        json.loads(p.read_text())
        for p in (args.history or [args.checkpoint.parent / "polish.json"])
    ]
    origin = args.checkpoint.parent.resolve()
    history_paths = [
        p.parent.resolve()
        for p in (args.history or [args.checkpoint.parent / "polish.json"])
    ]
    selected = history_paths.index(origin)
    check = histories[selected][
        "best_h_audit" if args.checkpoint.name == "best-h.npz" else "audit"
    ]
    n = 180
    core, outer = args.core, args.outer
    cg, og = centers(core, n), centers(outer, n)
    mean = meridian_modes(cfg, basis, coeff[:1], cg[None, :], cg[:, None], "velocity")[
        0, :, :, 2
    ].real
    full = meridian_modes(cfg, basis, coeff, og[None, :], og[:, None], "velocity")
    speed = np.sqrt(2 * (abs(full[1:]) ** 2).sum(axis=(0, 3)))
    r = centers(core, 360)
    R = np.hypot(r, args.height)
    curl = meridian_modes(cfg, basis, coeff, r, args.height, "curl")
    axial = (args.height * curl[:, :, 0] - r * curl[:, :, 1]) / R
    psi = centers(2 * math.pi, n)
    vorticity = (
        axial[0].real
        + 2 * (np.exp(1j * np.outer(psi, np.arange(1, len(coeff)))) @ axial[1:]).real
    )
    sector = 360 / cfg["kappa"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#273441",
            "text.color": "#162633",
            "mathtext.fontset": "stixsans",
        }
    )
    fig = plt.figure(figsize=(11.0, 8.05), facecolor="white")
    spec = fig.add_gridspec(
        2,
        2,
        left=0.083,
        right=0.945,
        bottom=0.11,
        top=0.833,
        width_ratios=(1, 1),
        height_ratios=(1.16, 0.9),
        hspace=0.59,
        wspace=0.36,
    )
    a, b, c, d = (
        fig.add_subplot(spec[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    fig.text(
        0.083,
        0.963,
        "Half-space rotating-wave candidate",
        fontsize=21,
        weight="bold",
        va="top",
    )
    status = "Passed numerical checks" if check["accepted"] else "Unconverged candidate"
    measured = check["measurements"][-1]
    fig.text(
        0.083,
        0.907,
        rf"{status}    ·    Zero forcing    ·    $\kappa={cfg['kappa']:g}$    ·    "
        rf"$g={cfg['g']:.4f}$    ·    $\omega={cfg['omega']:.3f}$    ·    {basis.harmonics} harmonics",
        fontsize=11,
        color="#465a68",
    )
    for ax, array, window, title, signed in (
        (a, mean, core, "Mean azimuthal velocity", True),
        (b, speed, outer, "Rotating wave speed (RMS)", False),
    ):
        im = ax.imshow(
            array,
            origin="lower",
            extent=(0, window, 0, window),
            aspect="equal",
            cmap="RdBu_r" if signed else "viridis",
            norm=CenteredNorm() if signed else None,
            vmin=None if signed else 0,
            interpolation="nearest",
        )
        ax.set(
            xlabel="Cylindrical radius",
            ylabel="Height",
            xlim=(0, window),
            ylim=(0, window),
        )
        ax.set_title(title, loc="left", size=12, weight="bold", pad=8)
        fig.colorbar(im, ax=ax, fraction=0.049, pad=0.035, ticks=MaxNLocator(nbins=5))
    if outer > core:
        b.add_patch(
            Rectangle(
                (0, 0),
                core,
                core,
                fill=False,
                linewidth=1.0,
                edgecolor="white",
                ls="--",
            )
        )
    im = c.imshow(
        vorticity,
        origin="lower",
        extent=(0, core, 0, sector),
        aspect="auto",
        cmap="RdBu_r",
        norm=CenteredNorm(),
        interpolation="nearest",
    )
    c.set(xlabel="Cylindrical radius", ylabel="Azimuth (degrees)")
    c.set_yticks(
        [0, sector / 2, sector], labels=["0", f"{sector / 2:.1f}", f"{sector:.1f}"]
    )
    c.set_title(
        rf"Axial vorticity at $z={args.height:g}$, one sector",
        loc="left",
        size=12,
        weight="bold",
        pad=8,
    )
    fig.colorbar(im, ax=c, fraction=0.049, pad=0.035, ticks=MaxNLocator(nbins=5))
    offset = 0
    for i, history in enumerate(histories):
        records = history["iterations"]
        steps = np.array([v["iteration"] for v in records])
        h = np.array([v["residual_h"] for v in records])
        color = ["#295e8a", "#c45c20", "#38794c"][i % 3]
        d.semilogy(
            offset + steps,
            h,
            color=color,
            lw=2,
            label=f"{history['basis']['harmonics']} harmonics",
        )
        if i == selected:
            it = cfg.get("iteration", records[-1]["iteration"])
            idx = list(steps).index(it)
            d.scatter(
                offset + it,
                h[idx],
                s=67,
                facecolor="white",
                edgecolor=color,
                linewidth=2,
                zorder=4,
                label="Plotted field",
            )
        offset += steps[-1]
    d.grid(axis="y", alpha=0.2, which="both")
    d.set(
        xlabel="Cumulative zero-force iterations", ylabel=r"Absolute residual $\|R\|_H$"
    )
    d.set_title(
        "Momentum balance during the search",
        loc="left",
        size=12,
        weight="bold",
        pad=8,
    )
    d.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    d.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    d.yaxis.set_major_formatter(EngFormatter(sep=""))
    d.yaxis.set_minor_locator(NullLocator())
    d.legend(frameon=False, fontsize=9, loc="best")
    magnitude = measured["residual_h"]
    exponent = math.floor(math.log10(magnitude)) if magnitude > 0 else 0
    fig.text(
        0.083,
        0.024,
        rf"Similarity units. Dashed square shows the inner view. "
        rf"On the finer grid $\|R\|_H={magnitude / 10**exponent:.2f}\times10^{{{exponent}}}$, "
        rf"relative defect {measured['relative_h']:.1%}.",
        size=9,
        color="#465a68",
    )
    fig.savefig(args.out, dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--history",
        action="append",
        type=Path,
        help="Consecutive polish histories, in transfer order",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--core", type=float, default=80)
    parser.add_argument("--outer", type=float, default=400)
    parser.add_argument("--height", type=float, default=6.5)
    options = parser.parse_args()
    if not (0 < options.core <= options.outer and options.height > 0):
        parser.error("Require positive windows and slice height")
    torch.set_num_threads(1)
    draw(options)
