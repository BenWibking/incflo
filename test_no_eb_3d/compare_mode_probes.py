"""Compare Fourier velocity fields on a common full-space quadrature and plot the scan."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from probe_rotating_modes import FullSphereBasis
from similarity_elements import gauss_elements
from similarity_rotating import WaveMeridian
from solve_similarity_orbit import write_json


def read_scan(directory):
    scan = {}
    for path in sorted(directory.glob("m*.npz")):
        with np.load(path, allow_pickle=False) as data:
            cfg = json.loads(str(data["config"]))
            scan[cfg["m"]] = (cfg, data["initial"].copy(), data["final"].copy())
    if not scan:
        raise ValueError(f"No mode checkpoints in {directory}")
    return scan


def field_pair(sample, rho, eta):
    cfg, initial, final = sample
    b = FullSphereBasis(**cfg["basis"])
    mer = WaveMeridian(b, rho / (rho + b.radius), eta)
    v = mer.operator(1, cfg["m"], cfg["g"], 0, cfg["tail"], "velocity", "cpu")
    return [v.apply(torch.as_tensor(c)).numpy() for c in (initial, final)]


def compare(a, b, weight):
    aa = np.sum(weight * abs(a) ** 2)
    bb = np.sum(weight * abs(b) ** 2)
    if min(aa, bb) <= 0:
        raise ValueError("Both fields must have a nonzero core norm")
    inner = np.sum(weight * a.conj() * b)
    ab = abs(inner)
    rotation = inner / ab if ab else 1.0
    return {
        "overlap": float(ab / np.sqrt(aa * bb)),
        "relative_change": float(
            np.sqrt(np.sum(weight * abs(rotation * a - b) ** 2) / bb)
        ),
    }


def measure(scans, grid):
    keys = ("base_sha256", "g", "nu", "torque", "time", "tail", "domain")
    cfg = next(iter(scans[-1].values()))[0]
    radius = cfg["basis"]["radius"]
    if any(sc.keys() != scans[-1].keys() for sc in scans):
        raise ValueError("Supply scans of the same modes")
    settings = [sample[0] for scan in scans for sample in scan.values()]
    if any(
        any(row[k] != cfg[k] for k in keys) or row["basis"]["radius"] != radius
        for row in settings
    ):
        raise ValueError("The scans use different physical controls or radial maps")
    breaks = sorted({s for row in settings for s in row["basis"]["sigma_breaks"]})
    eb = sorted({s for row in settings for s in row["basis"]["eta_breaks"]})
    sig, ws = gauss_elements(breaks, grid)
    eta, wa = gauss_elements([-e for e in eb[:0:-1]] + eb, grid)
    rho = radius * sig / (1 - sig)
    wr = ws * radius**3 * sig**2 / (1 - sig) ** 4 / (1 + (rho / 50.0) ** 2) ** 2
    weight = (wr[:, None] * wa)[:, :, None]
    r, z = rho[:, None] * np.sqrt(1 - eta**2), rho[:, None] * eta
    results = {}
    for m in sorted(scans[-1]):
        a, b = [field_pair(scan[m], rho, eta) for scan in scans[-2:]]
        energy = [(weight * abs(u) ** 2).sum(axis=-1) for u in b]
        results[m] = {
            "initial_between_meshes": compare(a[0], b[0], weight),
            "final_between_meshes": compare(a[1], b[1], weight),
            "final_vs_initial_finest": compare(b[0], b[1], weight),
            "initial_and_final_core_energy_centroids_rz": [
                [float(np.sum(e * position) / e.sum()) for position in (r, z)]
                for e in energy
            ],
        }
    return {"common_meridional_quadrature": [len(sig), len(eta)], "modes": results}


def plot(scans, audit, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (gain, shape) = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained")
    ordered = sorted(scans[-1])
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(scans)))
    for scan, color in zip(scans, colors, strict=True):
        rows = [scan[m][0] for m in ordered]
        label = f"{rows[0]['degrees_of_freedom']} coefficients"
        gain.plot(
            ordered,
            [r["end_core_amplitude_gain"] for r in rows],
            ".-",
            color=color,
            label=label,
            linewidth=1.6,
        )
    rows = [scans[-1][m][0] for m in ordered]
    gain.plot(
        ordered,
        [r["end_control_core_gain_same_packet"] for r in rows],
        "--",
        color="#9a563c",
        label="Zero-flow control, finest mesh",
    )
    gain.axhline(1, ls=":", color="#3f454b", lw=1)
    gain.set(
        ylabel="Final / initial core velocity norm",
        title="Velocity amplitude in the core",
    )
    diagnostic = audit["modes"]
    for key, label, color in [
        ("final_vs_initial_finest", "Initial vs. final field, finest mesh", "#a04f39"),
        ("final_between_meshes", "Final fields on two finest meshes", "#155f78"),
    ]:
        shape.plot(
            ordered,
            [diagnostic[m][key]["overlap"] for m in ordered],
            "o-",
            ms=4,
            color=color,
            label=label,
        )
    shape.set(
        ylabel="Normalized overlap after best rotation",
        ylim=(-0.035, 1.045),
        title="Does the velocity pattern repeat?",
    )
    for ax in (gain, shape):
        ax.set(xticks=ordered, xlabel="Azimuthal wave number m")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Linear waves on the fixed-torque base  ·  similarity time {rows[0]['time']:g}"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scans", type=Path, nargs="+", help="Run directories, coarsest to finest"
    )
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--modes", type=int, nargs="+", help="Compare a common subset")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.scans) < 2 or args.grid < 1:
        parser.error("Supply at least two scans and a positive common quadrature size")
    scans = [read_scan(d) for d in args.scans]
    if args.modes:
        if any(set(args.modes) - scan.keys() for scan in scans):
            parser.error("Every run must contain each selected mode")
        scans = [{m: scan[m] for m in args.modes} for scan in scans]
    audit = measure(scans, args.grid)
    write_json(args.out / "field-comparison.json", audit)
    plot(scans, audit, args.out / "mode-probe.png")


if __name__ == "__main__":
    main()
