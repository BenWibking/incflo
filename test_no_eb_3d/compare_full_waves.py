"""Compare nonlinear wave shapes at a common phase on an independent full-space grid."""

import argparse
from pathlib import Path

import numpy as np
import torch
from probe_rotating_modes import FullSphereBasis
from similarity_rotating import WaveGeometry, WaveSpace
from solve_full_rotating_wave import read
from solve_similarity_orbit import write_json


def compare(a, b, grid):
    pairs = [read(p) for p in (a, b)]
    old, new = [p[0] for p in pairs]
    if any(old[k] != new[k] for k in ("m", "base_sha256", "g", "nu")):
        raise ValueError(
            "Use waves with the same base, wave number and similarity equations"
        )
    mesh = FullSphereBasis(
        sigma_breaks=sorted(
            set(pairs[0][1].sigma_breaks) | set(pairs[1][1].sigma_breaks)
        ),
        eta_breaks=sorted(set(pairs[0][1].eta_breaks) | set(pairs[1][1].eta_breaks)),
        harmonics=0,
    )
    waves = []
    for cfg, basis, coeff in pairs:
        sp = WaveSpace(basis, grid, "cpu", cfg["m"], quadrature_basis=mesh)
        geo = WaveGeometry(sp, cfg["m"], cfg["g"], cfg["omega"])
        waves.append(
            [
                v.apply(torch.as_tensor(c))
                * torch.exp(1j * j * geo.twist * sp.log_radius)
                for j, (v, c) in enumerate(zip(geo.velocity, coeff, strict=True))
            ]
        )
    w = sp.core_weight[..., None]
    u, v = waves
    angle = torch.angle((w * u[1].conj() * v[1]).sum())
    count = max(len(u), len(v))
    u.extend([torch.zeros_like(u[0])] * (count - len(u)))
    v.extend([torch.zeros_like(v[0])] * (count - len(v)))
    err = sum(
        (w * (a - b * torch.exp(-1j * j * angle)).abs().square()).sum()
        for j, (a, b) in enumerate(zip(u[1:], v[1:], strict=True), 1)
    )
    norm = sum((w * x.abs().square()).sum() for x in v[1:])
    overlap = (
        abs((w * u[1].conj() * v[1]).sum())
        / ((w * u[1].abs().square()).sum() * (w * v[1].abs().square()).sum()).sqrt()
    )
    rho = sp.rho[:, None]
    eta = sp.meridian.eta
    rr, zz = rho * np.sqrt(1 - eta**2), rho * eta
    energy = sum((w * x.abs().square()).sum(-1).numpy() for x in v[1:])
    energy /= energy.sum()
    return {
        "m": new["m"],
        "wave_relative_change": float((err / norm).sqrt()),
        "fundamental_overlap": float(overlap),
        "mean_correction_difference_rms": float(
            (w * (u[0] - v[0]).abs().square()).sum().sqrt()
        ),
        "new_energy_centroid_rz": [float((energy * r).sum()) for r in (rr, zz)],
        "new_core_energy_fraction_cylindrical_layer": float(
            energy[(rr > 20) & (rr < 65) & (abs(zz) < 20)].sum()
        ),
        "fitted_phase_difference": float(angle),
        "quadrature": list(sp.weight.shape),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("old", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--grid", type=int, default=320)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_json(a.out, compare(a.old, a.new, a.grid))


if __name__ == "__main__":
    main()
