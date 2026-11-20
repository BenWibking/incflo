"""Compare the incflo and Quokka seed experiments at matched similarity phases."""

import argparse
import math
from pathlib import Path

import numpy as np
import yt

PERIOD = 0.36
RADIUS = 50.0


def read_snapshot(path, fields):
    ds = yt.load(str(path), hint="boxlib")
    if ds.index.max_level != 0:
        raise ValueError(f"Expected a single-level dataset in {path}")
    dims = np.asarray(ds.domain_dimensions)
    lo, hi = np.asarray(ds.domain_left_edge), np.asarray(ds.domain_right_edge)
    dx = (hi - lo) / dims
    axes = tuple(lo[d] + (0.5 + np.arange(dims[d])) * dx[d] for d in range(3))
    grid = ds.covering_grid(0, ds.domain_left_edge, dims)
    native = {name: (kind, name) for kind, name in ds.field_list}
    u = np.stack([np.asarray(grid[native[f]]) for f in fields])
    if not np.isfinite(u).all():
        raise ValueError(f"Nonfinite velocity in {path}")
    return float(ds.current_time), axes, u


def interpolate(field, axes, targets):
    """Separable trilinear interpolation at a Cartesian product of coordinates."""
    for d, (source, target) in enumerate(zip(axes, targets)):
        if target[0] < source[0] or target[-1] > source[-1]:
            raise ValueError("Sample coordinates extend beyond the cell centers")
        index = np.interp(target, source, np.arange(source.size))
        lower = np.minimum(index.astype(int), source.size - 2)
        shape = [1, 1, 1]
        shape[d] = target.size
        w = (index - lower).reshape(shape)
        field = (1 - w) * np.take(field, lower, axis=d) + w * np.take(
            field, lower + 1, axis=d
        )
    return field


def relative_l2(field, reference, mask):
    delta = (field - reference)[:, mask]
    return np.sqrt(np.sum(delta**2) / np.sum(reference[:, mask] ** 2))


def comparison(incdir, quokkadir, figure):
    incfiles = sorted(p for p in incdir.glob("plt*") if p.name[3:].isdigit())
    phases = (0.0, PERIOD / 2, PERIOD)
    if len(incfiles) != len(phases):
        raise ValueError(
            "Expected three incflo plotfiles: initial, half-period, full period"
        )
    fields = (("velocity_x", "velocity_y", "velocity_z"), ("velx", "vely", "velz"))
    print("s       t             Quokka return   incflo return   incflo vs Quokka")
    rows = []
    for index, phase in enumerate(phases):
        paths = (quokkadir / f"plt_dss{2 * index:05}", incfiles[index])
        scale = math.exp(-phase / 2)
        target_time = -math.expm1(-phase)
        velocities, slices = [], []
        for path, names in zip(paths, fields):
            time, axes, u = read_snapshot(path, names)
            if not math.isclose(time, target_time, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"{path}: expected physical time {target_time}, got {time}"
                )
            if index == 0 and not velocities:
                start_axes = axes
                x, y, z = np.ix_(*axes)
                mask = x * x + y * y + z * z <= RADIUS**2
            if not all(np.array_equal(c, c0) for c, c0 in zip(axes, start_axes)):
                raise ValueError(f"Physical grids differ at {path}")
            targets = [scale * c for c in axes]
            velocities.append(
                np.stack([scale * interpolate(v, axes, targets) for v in u])
            )
            if figure is not None and index:
                dx, dy = (c[1] - c[0] for c in axes[:2])
                omega = (np.roll(u[1], -1, axis=0) - np.roll(u[1], 1, axis=0)) / (
                    2 * dx
                ) - (np.roll(u[0], -1, axis=1) - np.roll(u[0], 1, axis=1)) / (2 * dy)
                y_plot = np.linspace(-55, 55, 221)
                plane = interpolate(
                    omega,
                    axes,
                    (scale * y_plot, scale * y_plot, np.array([scale * 6.5])),
                )
                slices.append(scale**2 * plane[:, :, 0])
        if index == 0:
            initial = velocities
        dq, di = (relative_l2(v, v0, mask) for v, v0 in zip(velocities, initial))
        between = relative_l2(velocities[1], velocities[0], mask)
        print(
            f"{phase:.2f}    {target_time:.10f}  {dq:13.9f}   {di:13.9f}   {between:15.9f}"
        )
        if slices:
            rows.append((slices[0], slices[1], slices[1] - slices[0]))
    print(
        "Maximum absolute initial velocity difference on the full mesh: "
        f"{np.max(np.abs(initial[1] - initial[0])):.9g}"
    )
    if figure is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, panels = plt.subplots(
            2, 3, figsize=(9.4, 6.3), layout="constrained", sharex=True, sharey=True
        )
        limit = max(np.max(np.abs(f)) for row in rows for f in row)
        for rid, row in enumerate(rows):
            for col, field in enumerate(row):
                ax = panels[rid, col]
                im = ax.pcolormesh(
                    y_plot,
                    y_plot,
                    field.T,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                ax.set_aspect("equal")
                if rid == 0:
                    ax.set_title(("Quokka", "incflo", "incflo − Quokka")[col])
                if rid == 1:
                    ax.set_xlabel(r"$y_1$")
                if col == 0:
                    ax.set_ylabel(rf"$s={phases[rid + 1]:.2f}$" + "\n" + r"$y_2$")
        fig.colorbar(im, ax=panels, label=r"$(1-t)\,\omega_z$", shrink=0.75)
        fig.suptitle(r"Modeled 19-fold seed on the similarity plane $y_3=6.5$")
        fig.savefig(figure, dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incflo_run", type=Path)
    parser.add_argument("quokka_run", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    yt.set_log_level(40)
    comparison(args.incflo_run, args.quokka_run, args.figure)
