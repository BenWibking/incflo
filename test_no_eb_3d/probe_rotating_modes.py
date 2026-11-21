"""Evolve localized Fourier disturbances on a frozen, forced similarity equilibrium."""

import argparse
import hashlib
import json
from dataclasses import asdict
from functools import cached_property, partial
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from similarity_elements import Splines
from similarity_rotating import WaveBasis, WaveMeridian, WaveSpace, dense_columns
from solve_axisymmetric import read
from solve_similarity_orbit import atomic, backend, write_json


class FullSphereBasis(WaveBasis):
    @property
    def full_eta(self):
        return tuple(-x for x in self.eta_breaks[:0:-1]) + self.eta_breaks

    @cached_property
    def groups(self):
        # Both hemispheres are independent and joined by interior splines at z=0.
        # A and B are the radial and polar potentials used by WaveMeridian.
        def pair(cr, ca):
            return (
                Splines(self.sigma_breaks, self.degree, cr),
                Splines(self.full_eta, self.degree, ca),
            )

        return [super().groups[0], {"A": pair(2, 1), "B": pair(1, 2)}]


MESHES = {
    "coarse": ((0, 0.32, 0.50, 0.64, 0.78, 1), (0, 0.12, 0.28, 0.55, 1)),
    "fine": (
        (0, 0.32, 0.50, 0.57, 0.64, 0.70, 0.78, 1),
        (0, 0.06, 0.12, 0.28, 0.40, 0.55, 1),
    ),
    "finer": (
        (0, 0.20, 0.32, 0.43, 0.50, 0.57, 0.61, 0.64, 0.70, 0.78, 0.88, 1),
        (0, 0.06, 0.12, 0.20, 0.28, 0.40, 0.55, 0.75, 1),
    ),
    "refined": (
        (
            0,
            0.20,
            0.26,
            0.32,
            0.38,
            0.43,
            0.465,
            0.50,
            0.535,
            0.57,
            0.61,
            0.64,
            0.67,
            0.70,
            0.74,
            0.78,
            0.88,
            1,
        ),
        (0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.28, 0.34, 0.40, 0.55, 0.65, 0.75, 1),
    ),
}
TAIL = 3.0  # A localized finite-energy disturbance on the slow-decaying base.


def packets(rho, eta, m):
    """Two solenoidal packets with independent cylindrical radial and vertical velocity."""
    sin_theta = np.sqrt(1 - eta**2)[None, :]
    r, z = rho[:, None] * sin_theta, rho[:, None] * eta
    # A compact C3 radial bump avoids any azimuthal singularity at the axis.
    q = (r - 40.0) / 24.0
    p = np.maximum(1 - q**2, 0)
    b = np.exp(-0.5 * ((z - 6.5) / 6.0) ** 2)
    f = p**4 * b
    dr = -8 * q * p**3 * b / 24.0
    dz = -(z - 6.5) * f / 36.0
    # i*m*v_phi = -[d_r(r*v_r) + r*d_z(v_z)] for exp(i*m*phi).
    radial = np.stack((sin_theta * f, eta * f, 1j * (f + r * dr) / m), axis=-1)
    vertical = np.stack((eta * f, -sin_theta * f, 1j * r * dz / m), axis=-1)
    return np.stack((radial, vertical), axis=-1)


def base_fields(cfg, basis, coeff, sp):
    mer = WaveMeridian(basis, sp.rho / (sp.rho + basis.radius), sp.meridian.eta)
    g = cfg["g"]
    c = torch.as_tensor(coeff, device=sp.device, dtype=torch.complex128)
    return [
        mer.operator(0, 0, g, 0, (1 - g) / g, kind, sp.device).apply(c)
        for kind in ("velocity", "curl")
    ]


def advection(sp, m, g, base, curl_base, n):
    """Weak projection of (curl v) x U + (curl U) x v, integrating in radial slabs."""
    result = torch.zeros((n, n), device=sp.device, dtype=torch.complex128)
    sig = sp.rho / (sp.rho + sp.basis.radius)
    for i in range(0, len(sig), 16):
        rows = slice(i, i + 16)
        mer = WaveMeridian(sp.basis, sig[rows], sp.meridian.eta)
        v, curl = [
            dense_columns(mer.operator(1, m, g, 0, TAIL, kind, sp.device))
            for kind in ("velocity", "curl")
        ]
        adv = torch.linalg.cross(
            curl, base[rows].reshape(-1, 3, 1), dim=1
        ) + torch.linalg.cross(curl_base[rows].reshape(-1, 3, 1), v, dim=1)
        w = sp.weight[rows].reshape(-1, 1, 1)
        result.addmm_(v.reshape(-1, n).mH, (w * adv).reshape(-1, n))
    return result.cpu().numpy()


def hermitian(a):
    return (a + a.conj().T) * 0.5


def project(sp, m, g, base, curl_base):
    def op(kind):
        return sp.meridian.operator(1, m, g, 0, TAIL, kind, sp.device)

    v = op("velocity")
    mass = hermitian(v.gram(v, sp.wr, sp.wa))
    d = op("linear")
    linear = v.gram(d, sp.wr, sp.wa)
    curl = op("curl")
    enstrophy = curl.gram(curl, sp.wr, sp.wa)
    # For U=0 the Hermitian part is known by integration by parts on full space.
    identity = enstrophy + (1 - 2.5 * g) * mass
    energy_error = la.norm(hermitian(linear) - identity) / la.norm(identity)
    if energy_error > 1e-6:
        raise ValueError(f"Unresolved viscous or dilation operator: {energy_error:g}")
    core_window = (1 + (sp.rho / 50.0) ** 2) ** -2
    core = v.gram(v, sp.wr * core_window, sp.wa)
    adv = advection(sp, m, g, base, curl_base, len(mass))
    scale = np.sqrt(mass.diagonal().real)
    scaled = mass / scale[:, None] / scale[None, :]
    chol = la.cholesky(scaled, lower=True)
    E = (
        la.solve_triangular(chol.conj().T, np.eye(len(mass)), lower=False)
        / scale[:, None]
    )
    P = E.conj().T
    target = torch.as_tensor(packets(sp.rho, sp.meridian.eta, m), device=sp.device)
    z = P @ np.column_stack(
        [v.adjoint(sp.weight[..., None] * t).cpu().numpy() for t in target.unbind(-1)]
    )
    wcore = sp.weight * sp.tensor(core_window)[:, None]
    c = torch.as_tensor(E @ z, device=sp.device)
    fits = []
    for cj, t in zip(c.T, target.unbind(-1), strict=True):
        num = (wcore[..., None] * (v.apply(cj) - t).abs().square()).sum()
        den = (wcore[..., None] * t.abs().square()).sum()
        fits.append(float((num / den).sqrt()))
    return (
        -(P @ (linear + adv) @ E),
        -(P @ linear @ E),
        hermitian(P @ core @ E),
        E,
        z,
        {
            "degrees_of_freedom": len(mass),
            "seed_core_relative_errors": fits,
            "unforced_linear_energy_identity_error": float(energy_error),
        },
    )


def propagate(K, K0, core, z, end_time):
    # Only two initial polarizations are optimized. The evolution uses all spatial DOFs.
    dt = end_time / 24
    step = la.expm(dt * K)
    history = [z]
    for _ in range(24):
        history.append(step @ history[-1])
    before, after = [hermitian(x.conj().T @ core @ x) for x in (z, history[-1])]
    values, vectors = la.eigh(after, before)
    mix = vectors[:, -1]
    seq = [x @ mix for x in history]
    wseq = [core @ x for x in seq]
    energy = np.array([np.vdot(x, y).real for x, y in zip(seq, wseq, strict=True)])
    overlaps = np.array(
        [np.vdot(x, w) for x, w in zip(seq[:-1], wseq[1:], strict=True)]
    )
    control = la.expm(end_time * K0) @ seq[0]
    omega = -np.angle(overlaps).sum() / end_time
    report = {
        "end_core_amplitude_gain": float(np.sqrt(values[-1])),
        "pure_radial_and_vertical_gains": np.sqrt(
            after.diagonal().real / before.diagonal().real
        ).tolist(),
        "end_control_core_gain_same_packet": float(
            np.sqrt(np.vdot(control, core @ control).real / energy[0])
        ),
        "end_global_amplitude_gain": float(la.norm(seq[-1]) / la.norm(seq[0])),
        "core_gains": np.sqrt(energy / energy[0]).tolist(),
        "mean_phase_frequency": float(omega),
        "phase_increment_max_rad": float(np.abs(np.angle(overlaps)).max()),
        "endpoint_shape_overlap": float(
            abs(np.vdot(seq[0], wseq[-1])) / np.sqrt(energy[0] * energy[-1])
        ),
    }
    return report, seq[0], seq[-1], mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path(__file__).with_name("axisymmetric_base.npz")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh", choices=MESHES, default="fine")
    parser.add_argument("--grid", type=int, default=160)
    parser.add_argument("--modes", type=int, nargs="+", default=list(range(14, 21)))
    parser.add_argument("--time", type=float, default=0.36)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if (
        not np.isfinite(args.time)
        or args.time <= 0
        or min(args.modes) < 3
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error("Supply a positive finite time and distinct integer modes >= 3")
    cfg, old, c = read(args.base)
    if cfg["nu"] != 1:
        parser.error("The weak linear operator assumes viscosity one")
    sigma, eta = MESHES[args.mesh]
    b = FullSphereBasis(sigma_breaks=sigma, eta_breaks=eta, harmonics=1)
    sp = WaveSpace(b, args.grid, backend(args.device), args.modes[0])
    base, curl_base = base_fields(cfg, old, c, sp)
    meta = {
        "basis": asdict(b),
        "domain": "full_sphere",
        "tail": TAIL,
        "g": cfg["g"],
        "nu": cfg["nu"],
        "torque": cfg["force"],
        "time": args.time,
        "mesh": args.mesh,
        "meridional_quadrature": list(sp.weight.shape),
        "base_sha256": hashlib.sha256(args.base.read_bytes()).hexdigest(),
        "device": str(sp.device),
        "device_name": torch.cuda.get_device_name()
        if sp.device.type == "cuda"
        else "cpu",
    }
    for m in args.modes:
        K, K0, W, E, z, info = project(sp, m, cfg["g"], base, curl_base)
        result, start, final, mix = propagate(K, K0, W, z, args.time)
        row = {
            **meta,
            **info,
            **result,
            "m": m,
            "mean_pattern_speed": result["mean_phase_frequency"] / m,
        }
        stem = f"m{m:02d}"
        atomic(
            args.out / (stem + ".npz"),
            partial(
                np.savez_compressed,
                initial=E @ start,
                final=E @ final,
                polarization=mix,
                config=json.dumps(row),
            ),
        )
        write_json(args.out / (stem + ".json"), row)
        print(
            f"m={m} core gain={result['end_core_amplitude_gain']:.5f} "
            f"control={result['end_control_core_gain_same_packet']:.5f} "
            f"omega={result['mean_phase_frequency']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
