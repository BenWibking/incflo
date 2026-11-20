"""Replay an optimized velocity and compare its vorticity at both ends of one cycle."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from compare_dss_cycle import RADIUS, interpolate, read_snapshot, relative_l2
from optimize_dss import Cycle, S
from scipy.fft import irfft, rfft
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

INPUTS = Path(__file__).with_name("inputs.dss_cycle")
HEIGHT = 6.5
PLOT_AXIS = np.linspace(-55, 55, 221)
RADII = np.linspace(0, 65, 131)


def refine_velocity(velocity, n):
    """Sample the same periodic trigonometric interpolant at finer cell centers."""
    source_n = velocity.shape[-1]
    if n < source_n:
        raise ValueError("The replay grid must be at least as fine as the saved grid")
    if n == source_n:
        return velocity
    modes = np.arange(source_n // 2 + 1)
    shift = np.exp(1j * np.pi * modes * (1 / n - 1 / source_n))
    if source_n % 2 == 0:
        # Split the real coarse Nyquist cosine between its two fine-grid modes.
        shift[-1] *= 0.5
    for axis in (1, 2, 3):
        spectrum = rfft(velocity, axis=axis, norm="forward")
        shape = [1] * velocity.ndim
        shape[axis] = len(shift)
        spectrum *= shift.reshape(shape)
        velocity = irfft(spectrum, n=n, axis=axis, norm="forward")
    return velocity


def replay(report, control, exe, inputs, out, n, steps):
    source_n = report["grid"]
    count = 3 * source_n**3
    if control.stat().st_size != count * np.dtype(np.float64).itemsize:
        raise ValueError(f"Expected {count} float64 velocity values in {control}")
    velocity = np.fromfile(control, dtype=np.float64).reshape(
        3, source_n, source_n, source_n
    )
    if not np.isfinite(velocity).all() or steps < 1:
        raise ValueError("Replay requires a finite velocity and a positive step count")
    cycle = Cycle(exe, inputs, n, steps)
    _, error, _ = cycle.run(
        refine_velocity(velocity, n), out, gradient=False, plots=True
    )
    if (n, steps) == (source_n, report["steps"]) and not np.isclose(
        error, report["final_return"], rtol=2e-7, atol=2e-10
    ):
        raise ValueError(
            f"Replay error {error} differs from saved return {report['final_return']}"
        )
    return error


def centered_curl(velocity, axes):
    def derivative(v, axis):
        dx = axes[axis][1] - axes[axis][0]
        return (np.roll(v, -1, axis) - np.roll(v, 1, axis)) / (2 * dx)

    u, v, w = velocity
    return np.stack(
        (
            derivative(w, 1) - derivative(v, 2),
            derivative(u, 2) - derivative(w, 0),
            derivative(v, 0) - derivative(u, 1),
        )
    )


def sample_phase(plotfile, phase):
    t, axes, fields = read_snapshot(plotfile, ("velx", "vely", "velz", "vort"))
    if not math.isclose(t, -math.expm1(-phase), abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"Unexpected time {t} in {plotfile}")
    curl = centered_curl(fields[:3], axes)
    if not np.allclose(np.linalg.norm(curl, axis=0), fields[3], rtol=2e-6, atol=1e-10):
        raise ValueError(
            f"Centered curl disagrees with native incflo vorticity in {plotfile}"
        )
    a = math.exp(-phase / 2)
    targets = tuple(a * axis for axis in axes)
    interpolator = RegularGridInterpolator(axes, np.moveaxis(curl, 0, -1))

    def on_slice(x, y):
        points = np.stack((x, y, np.full_like(x, HEIGHT)), axis=-1)
        return a**2 * interpolator(a * points)

    phi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ring = on_slice(RADII[:, None] * np.cos(phi), RADII[:, None] * np.sin(phi))
    axial = ring[..., 2]
    return {
        "axes": axes,
        "time": t,
        "velocity": np.stack([a * interpolate(v, axes, targets) for v in fields[:3]]),
        "omega": np.stack([a**2 * interpolate(v, axes, targets) for v in curl]),
        "slice": on_slice(*np.meshgrid(PLOT_AXIS, PLOT_AXIS))[..., 2],
        "rms": np.sqrt(np.mean(np.sum(ring**2, axis=-1), axis=1)),
        "axial_rms": np.sqrt(np.mean(axial**2, axis=1)),
        "axial_mean": axial.mean(axis=1),
        "m19": 2 * np.mean(axial * np.exp(-19j * phi), axis=1),
    }


def ring_norm(mode):
    mask = RADII**2 + HEIGHT**2 <= RADIUS**2
    r = RADII[mask]
    return float(np.sqrt(trapezoid(np.abs(mode[mask]) ** 2 * r, r) / trapezoid(r, r)))


def measure(directory, steps, save_profile=False, title=""):
    """Measure the same normalized return at both fidelities, using projected endpoints."""
    import yt

    yt.set_log_level(40)
    initial = sample_phase(directory / "plt00000", 0)
    final = sample_phase(directory / f"plt{steps:05}", S)
    axes = initial["axes"]
    if any(not np.array_equal(x, y) for x, y in zip(axes, final["axes"])):
        raise ValueError("Endpoint grids differ")
    x, y, z = np.ix_(*axes)
    core = x * x + y * y + z * z <= RADIUS**2
    velocity_error = float(relative_l2(final["velocity"], initial["velocity"], core))
    if not np.isclose(
        velocity_error,
        np.loadtxt(directory / "dss_result.txt")[1],
        rtol=2e-7,
        atol=2e-10,
    ):
        raise ValueError("Plotfile velocity return differs from the native objective")
    vorticity_error = float(relative_l2(final["omega"], initial["omega"], core))
    initial_m19 = ring_norm(initial["m19"])
    if initial_m19 == 0:
        raise ValueError("The initial slice has zero azimuthal mode 19")
    mode_error = ring_norm(final["m19"] - initial["m19"]) / initial_m19
    metrics = {
        "velocity_return": velocity_error,
        "vorticity_return": vorticity_error,
        "m19_return": mode_error,
        "m19_retention": ring_norm(final["m19"]) / initial_m19,
        "score": max(velocity_error, vorticity_error, mode_error),
        "grid": len(axes[0]),
        "steps": steps,
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError(f"Nonfinite recurrence metric in {directory}")
    if save_profile:
        save_products(directory, initial, final, title)
    return metrics


def save_products(directory, initial, final, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = (initial, final)
    axes = initial["axes"]
    np.savez_compressed(
        directory / "vorticity-3d.npz",
        vorticity=np.stack([p["omega"] for p in phases]),
        axis_order="phase,component,y1,y2,y3",
        component=["omega1", "omega2", "omega3"],
        y1=axes[0],
        y2=axes[1],
        y3=axes[2],
        similarity_time=[0, S],
        physical_time=[p["time"] for p in phases],
    )
    columns = [RADII]
    names = ["cylindrical_radius"]
    for key in ("rms", "axial_rms", "axial_mean", "m19"):
        for p, label in zip(phases, ("initial", "final")):
            columns.append(np.abs(p[key]) if key == "m19" else p[key])
            names.append(f"{key}_{label}")
    np.savetxt(
        directory / "radial-vorticity.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.12g",
    )

    fig = plt.figure(figsize=(9, 7.2), layout="constrained")
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.7, 0.9], width_ratios=[1, 1, 0.045], hspace=0.16
    )
    fig.suptitle(rf"Vorticity at similarity height $y_3={HEIGHT:g}$  ·  {title}")
    limit = max(np.abs(p["slice"]).max() for p in phases)
    for i, (p, name) in enumerate(zip(phases, ("Initial field", "After one cycle"))):
        ax = fig.add_subplot(gs[0, i])
        im = ax.pcolormesh(
            PLOT_AXIS,
            PLOT_AXIS,
            p["slice"],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
            rasterized=True,
        )
        ax.set(
            title=f"{name}   $s={(0, S)[i]:g}$",
            xlabel=r"$y_1$",
            ylabel=r"$y_2$" if i == 0 else "",
            aspect="equal",
            xticks=[-50, -25, 0, 25, 50],
            yticks=[-50, -25, 0, 25, 50],
        )
    fig.colorbar(
        im,
        cax=fig.add_subplot(gs[0, 2]),
        label=r"Axial vorticity  $(\nabla_y\times W)_3$",
    )
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(RADII, initial["rms"], color="#245b73", lw=2, label="Initial field")
    ax.plot(RADII, final["rms"], "--", color="#ae4923", lw=2, label="After one cycle")
    ax.set(
        xlim=(0, RADII[-1]),
        ylim=(0, None),
        xlabel=r"Cylindrical radius  $\sqrt{y_1^2+y_2^2}$",
        ylabel=r"RMS $|\nabla_y\times W|$",
        title="Vorticity magnitude averaged around each ring",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"vorticity-profile.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="contains selected-search.json and accepted-velocity.bin",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--exe", type=Path)
    source.add_argument("--from-plots", action="store_true")
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    out = args.directory.resolve()
    report = json.loads((out / "selected-search.json").read_text())
    reference = out / "selected-fine.json"
    target = json.loads(reference.read_text()) if reference.exists() else report
    grid = target["grid"] if args.grid is None else args.grid
    steps = target["steps"] if args.steps is None else args.steps
    if args.from_plots:
        if args.grid is not None or args.steps is not None:
            parser.error("Grid or timestep changes require --exe")
    else:
        replay(
            report,
            out / "accepted-velocity.bin",
            args.exe,
            args.inputs,
            out,
            grid,
            steps,
        )
    label = f"Seed {report['random_seed']}"
    if "trial" in target:
        label = f"Trial {target['trial']} · {label}"
    metrics = measure(out, steps, save_profile=True, title=label)
    if metrics["grid"] != grid:
        raise ValueError(f"Expected a {grid}³ plotfile")
    if (
        reference.exists()
        and (grid, steps) == (target["grid"], target["steps"])
        and not np.isclose(metrics["score"], target["score"], rtol=2e-6, atol=2e-9)
    ):
        raise ValueError("Replayed score differs from the selected fine result")
    (out / "vorticity-diagnostics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
