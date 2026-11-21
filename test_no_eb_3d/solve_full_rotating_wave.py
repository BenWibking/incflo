"""Finite-amplitude rotating waves about a fixed-torque, full-space similarity flow."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from probe_rotating_modes import TAIL, FullSphereBasis, packets
from scipy.optimize import least_squares
from similarity_elements import PressureFit
from similarity_rotating import (
    RotatingWave,
    WaveGeometry,
    WaveMeridian,
    WaveSpace,
    dense_columns,
    lift_coefficients,
)
from solve_axisymmetric import read as read_base
from solve_axisymmetric import torque_shape
from solve_rotating_wave import correct
from solve_similarity_orbit import atomic, backend, write_json

# These nested trial knots are contained in the independent base-flow audit mesh.
MESHES = {
    "start": ((0, 0.45, 0.63, 1), (0, 0.12, 0.30, 1)),
    "middle": (
        (0, 0.36, 0.45, 0.54, 0.63, 0.73, 1),
        (0, 0.06, 0.12, 0.20, 0.30, 0.53, 1),
    ),
    "fine": (
        (0, 0.18, 0.36, 0.45, 0.54, 0.63, 0.73, 0.88, 1),
        (0, 0.06, 0.12, 0.20, 0.30, 0.53, 0.76, 0.88, 1),
    ),
}


class Background:
    def __init__(self, sp, source):
        cfg, basis, coeff = source
        self.g, self.force = cfg["g"], cfg["force"]
        if cfg["nu"] != 1:
            raise ValueError("The rotating-wave operator uses unit viscosity")
        mer = WaveMeridian(basis, sp.rho / (sp.rho + basis.radius), sp.meridian.eta)
        c = torch.as_tensor(coeff, device=sp.device, dtype=torch.complex128)
        self.fields = [
            mer.operator(0, 0, self.g, 0, (1 - self.g) / self.g, k, sp.device).apply(c)
            for k in ("velocity", "curl", "linear")
        ]
        eta = sp.meridian.eta
        z, r = sp.rho[:, None] * eta, sp.rho[:, None] * np.sqrt(1 - eta**2)
        self.drive = torch.zeros_like(self.fields[0])
        self.drive[..., 2] = sp.tensor(
            torque_shape(r, z, cfg["drive_r"], cfg["drive_z"])
        )
        # Localizing the auxiliary damping preserves the unforced far-field balance.
        self.window = sp.tensor(1 / (1 + (sp.rho / 60.0) ** 2))[:, None, None]
        self.window = self.window.expand(*sp.weight.shape, 1)
        self.mean_norm = math.sqrt(
            float((sp.core_weight[..., None] * self.fields[0].abs().square()).sum())
        )


def inflow_weight(sp):
    # The radial seed has meridional velocity f e_r. Its squared envelope
    # localizes this measurement on the northern layer used to start the scan.
    meridional = packets(sp.rho, sp.meridian.eta, sp.kappa)[..., :2, 0]
    return sp.core_weight * sp.tensor(np.sum(abs(meridional) ** 2, axis=-1))


class TotalGeometry(WaveGeometry):
    def __init__(self, sp, g, omega, background):
        super().__init__(sp, sp.kappa, g, omega)
        self.background = background

    def fields(self, cs):
        fields = super().fields(cs)
        for parts, base in zip(fields, self.background.fields, strict=True):
            parts[0] = parts[0] + base
        return fields


class FixedTorqueWave(RotatingWave):
    """The mean is a free full-sphere correction to the saved axisymmetric field."""

    def __init__(self, sp, bg, omega, damping, coeff, amplitude, torque_fraction=1.0):
        self.bg, self.torque_fraction = bg, torque_fraction
        self.damp_scale = max(1.0, abs(damping))
        super().__init__(sp, bg.g, omega, coeff, amplitude)
        self.start[-2] = damping / self.damp_scale
        self.load.fill(0.0)  # No manufactured wave load.

    def make_geometry(self, g, omega):
        return TotalGeometry(self.space, g, omega, self.bg)

    def parameters(self, x):
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite wave iterate")
        return self.bg.g, self.omega_scale * x[-1]

    def damping_weight(self, x, j):
        return self.damp_scale * x[-2] * self.bg.window if j else None

    def applied_terms(self, x, U, raw):
        return [raw[0] - self.torque_fraction * self.bg.force * self.bg.drive] + [
            r + self.damping_weight(x, j) * U[j] for j, r in enumerate(raw[1:], 1)
        ]


def packet(sp, path, m, base_hash):
    with np.load(path, allow_pickle=False) as f:
        src = json.loads(str(f["config"]))
        coeff = f["initial"].copy()
    if (
        src["m"] != m
        or src["base_sha256"] != base_hash
        or src["domain"] != "full_sphere"
    ):
        raise ValueError(
            "Supply a full-space packet at this wave number on the same base"
        )
    b = FullSphereBasis(**src["basis"])
    mer = WaveMeridian(b, sp.rho / (sp.rho + b.radius), sp.meridian.eta)
    u = mer.operator(1, m, src["g"], 0, src["tail"], "velocity", sp.device).apply(
        torch.as_tensor(coeff, device=sp.device)
    )
    return src["mean_phase_frequency"], u


def equilibrate_mean(sp, bg):
    """Correct the public field to the mean equations of this Galerkin space."""
    g = bg.g
    T = dense_columns(
        sp.meridian.operator(0, 0, g, 0, TAIL, "velocity", sp.device)
    ).real
    V, C, L = [
        dense_columns(sp.meridian.operator(0, 0, g, 0, (1 - g) / g, k, sp.device)).real
        for k in ("velocity", "curl", "linear")
    ]
    n = V.shape[-1]
    P = (sp.weight.reshape(-1, 1, 1) * T).reshape(-1, n).T.contiguous()
    base, rot, linear = (u.real.reshape(-1, 3) for u in bg.fields)
    drive = bg.force * bg.drive.real.reshape(-1, 3)

    def terms(x):
        v = sp.tensor(x)
        return base + V @ v, rot + C @ v, linear + L @ v - drive

    def fun(x):
        u, c, l = terms(x)
        return (P @ (l + torch.linalg.cross(c, u)).reshape(-1)).cpu().numpy()

    def jac(x):
        u, c, _ = terms(x)
        a = (
            L
            + torch.linalg.cross(C, u[:, :, None], dim=1)
            + torch.linalg.cross(c[:, :, None], V, dim=1)
        )
        return (P @ a.reshape(-1, n)).cpu().numpy()

    start = np.zeros(n)
    A = jac(start)
    row = 1 / np.maximum(la.norm(A, axis=1), 1e-30)
    col = 1 / np.maximum(la.norm(A * row[:, None], axis=0), 1e-30)
    sol = least_squares(
        lambda y: row * fun(col * y),
        start,
        jac=lambda y: row[:, None] * jac(col * y) * col,
        method="lm",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=80,
    )
    coeff = col * sol.x
    u, c, _ = terms(coeff)
    change = (
        math.sqrt(float((sp.core_weight.reshape(-1, 1) * (u - base).square()).sum()))
        / bg.mean_norm
    )
    info = {
        "initial": float(la.norm(row * fun(start))),
        "final": float(la.norm(sol.fun)),
        "relative_core_velocity_change": change,
        "evaluations": sol.nfev,
    }
    return coeff.astype(complex), (u.to(torch.complex128), c.to(torch.complex128)), info


def neutral_seed(sp, bg, omega, target, mean_fields, selection):
    """Track a packet-overlapping damped eigenpair until the damping is real."""
    m, g = sp.kappa, bg.g
    target = target.reshape(-1)
    w = sp.core_weight[..., None].expand(*sp.weight.shape, 3).reshape(-1)
    target = target / torch.sqrt(torch.vdot(target, w * target).real)
    previous = None
    best = None
    history = []
    for _ in range(13):

        def op(kind, tail, frequency=omega):
            return sp.meridian.operator(1, m, g, frequency, tail, kind, sp.device)

        test = op("velocity", TAIL)
        v, c, linear = (op(k, (1 - g) / g) for k in ("velocity", "curl", "linear"))
        T, V, C = (dense_columns(o) for o in (test, v, c))
        n = V.shape[-1]
        project = (sp.weight.reshape(-1, 1, 1) * T).reshape(-1, n).mH
        swirl = torch.linalg.cross(C, mean_fields[0].reshape(-1, 3, 1), dim=1)
        swirl += torch.linalg.cross(mean_fields[1].reshape(-1, 3, 1), V, dim=1)
        A = (
            test.gram(linear, sp.wr, sp.wa)
            + (project @ swirl.reshape(-1, n)).cpu().numpy()
        )
        D = (project @ (bg.window.reshape(-1, 1, 1) * V).reshape(-1, n)).cpu().numpy()
        row = 1 / np.maximum(la.norm(A, axis=1), 1e-30)
        col = 1 / np.maximum(la.norm(row[:, None] * A, axis=0), 1e-30)
        lam, q = la.eig(row[:, None] * A * col, row[:, None] * D * col)
        valid = np.isfinite(lam) & (abs(lam.real) < 60) & (abs(lam.imag) < 60)
        if not valid.any():
            break
        lam, q = lam[valid], col[:, None] * q[:, valid]
        # Compare physical phase-zero fields as the trial winding changes with omega.
        phase = (
            torch.exp(1j * omega / (2 * g) * sp.log_radius)
            .expand(*sp.weight.shape, 3)
            .reshape(-1)
        )
        Z = V.reshape(-1, n)
        gram = (Z.mH @ (w[:, None] * Z)).cpu().numpy()
        norms = np.sqrt(np.maximum((q.conj() * (gram @ q)).sum(axis=0).real, 1e-300))
        q /= norms
        weights = (Z.mH @ (w * phase.conj() * target)).cpu().numpy()
        overlaps = abs(weights.conj() @ q)
        screened = []
        if previous is None:
            score = overlaps / np.sqrt(1 + (lam.real / 4) ** 2 + (lam.imag / 5) ** 2)
            if selection == "residual":
                pressure = PressureFit(
                    sp,
                    1.0,
                    0,
                    1,
                    2,
                    angular_order=m,
                    radial_frequency=-1j * omega / (2 * g),
                )

                def projected(field, pressure=pressure):
                    return field - pressure.fit(field)

                for idx in np.argsort(score)[-24:]:
                    column = torch.as_tensor(q[:, idx].copy(), device=sp.device)
                    u, rot, l = (p.apply(column) for p in (v, c, linear))
                    nonlinear = torch.linalg.cross(
                        rot, mean_fields[0].reshape_as(rot)
                    ) + torch.linalg.cross(mean_fields[1].reshape_as(u), u)
                    pl, pn, pd = map(
                        projected, (l, nonlinear, lam[idx] * bg.window * u)
                    )
                    err = pl + pn - pd

                    def norm(f, weight):
                        return float(
                            (weight[..., None] * f.abs().square()).sum().sqrt()
                        )

                    strong = norm(err, sp.weight) / max(
                        sum(norm(f, sp.weight) for f in (pl, pn, pd)), 1e-300
                    )
                    core = norm(err, sp.core_weight) / max(
                        abs(omega) * norm(u, sp.core_weight), 1e-300
                    )
                    screened.append(
                        {
                            "index": int(idx),
                            "relative_h": strong,
                            "core_defect": core,
                            "packet_overlap": float(overlaps[idx]),
                            "damping": float(-lam[idx].real),
                            "imaginary_defect": float(lam[idx].imag),
                        }
                    )
                # Select a candidate that actually balances momentum and overlaps the
                # seed. Very remote eigenfunctions can have small absolute defects.
                eligible = [
                    r
                    for r in screened
                    if r["packet_overlap"] >= 0.25 * overlaps.max()
                    and abs(r["imaginary_defect"]) < 0.75 * max(1.0, abs(omega))
                ]
                if eligible:
                    winner = min(
                        eligible, key=lambda r: r["relative_h"] + r["core_defect"]
                    )
                    score = np.zeros_like(score)
                    score[winner["index"]] = 1.0
        else:
            score = overlaps
        i = int(np.argmax(score))
        val = lam[i]
        selected = q[:, i].copy()
        physical = phase * (Z @ torch.as_tensor(selected, device=sp.device))
        history.append(
            {
                "omega": float(omega),
                "damping": float(-val.real),
                "imaginary_defect": float(val.imag),
                "overlap": float(overlaps[i]),
            }
        )
        if screened:
            history[-1]["screened"] = screened
        if best is None or abs(val.imag) < best[0]:
            best = (abs(val.imag), omega, -val.real, selected)
        if abs(val.imag) < 1e-7:
            break
        slope = -1.5
        if previous is not None and abs(omega - previous[0]) > 1e-9:
            derivative = (val.imag - previous[1]) / (omega - previous[0])
            if abs(derivative) > 0.1:
                slope = derivative
        previous = (omega, val.imag)
        target = physical
        omega -= float(np.clip(val.imag / slope, -3, 3))
    if best is None:
        raise ValueError("No finite rotating eigenpair in the resolved wave space")
    _, omega, damping, c1 = best
    coeff = [
        np.zeros(sum(a.count * b.count if a else 1 for a, b in group.values()), complex)
        for group in sp.basis.groups
    ]
    coeff[1] = c1
    return float(omega), float(damping), coeff, history


def save(path, cfg, coeff):
    data = {f"c{j}": c for j, c in enumerate(coeff)}
    atomic(path, partial(np.savez_compressed, config=json.dumps(cfg), **data))


def read(path):
    with np.load(path, allow_pickle=False) as f:
        cfg = json.loads(str(f["config"]))
        if cfg.get("kind") != "full_sphere_fixed_torque_wave":
            raise ValueError("Supply a full-space nonlinear checkpoint")
        b = FullSphereBasis(**cfg["basis"])
        cs = [f[f"c{j}"].copy() for j in range(b.harmonics + 1)]
    for c, group in zip(cs, b.groups, strict=True):
        n = sum(a.count * b.count if a else 1 for a, b in group.values())
        if c.shape != (n,) or not np.isfinite(c).all():
            raise ValueError("Invalid full-space wave coefficients")
    return cfg, b, cs


def base_pressure_gradient(sp, bg, base_basis):
    north = WaveSpace(
        base_basis, sp.grid, sp.device, sp.kappa, angular_grid=len(sp.wa) // 2
    )
    half = len(sp.wa) // 2
    high = PressureFit(north, 1.0, 0, 0, 9 - base_basis.degree)
    base_raw = bg.fields[2] + torch.linalg.cross(bg.fields[1], bg.fields[0])
    grad = high.fit(base_raw[:, half:])
    return torch.cat((grad.flip((1,)) * sp.tensor([1.0, -1.0, 1.0]), grad), dim=1)


def audit(source, cfg, b, coeff, grid, pressure_degree, device):
    # Same independent mesh for different trial spaces. The base knots include every
    # trial knot. The pressure minimization sees all quadratic harmonics through 2J.
    _, base_b, _ = source
    pb = FullSphereBasis(
        degree=3,
        sigma_breaks=base_b.sigma_breaks,
        eta_breaks=base_b.eta_breaks,
        harmonics=0,
        radius=b.radius,
        phase_radius=b.phase_radius,
    )
    sp = WaveSpace(pb, grid, device, cfg["m"])
    bg = Background(sp, source)
    trial = WaveSpace(b, grid, device, cfg["m"], quadrature_basis=pb)
    base_grad = base_pressure_gradient(sp, bg, base_b)
    geo = TotalGeometry(trial, bg.g, cfg["omega"], bg)
    c = [torch.as_tensor(x, device=device, dtype=torch.complex128) for x in coeff]
    U, _, L, raw = geo.raw(c, all_modes=True)
    damping = cfg["damping"]
    torque = cfg["torque_fraction"] * bg.force * bg.drive
    rows = []
    core_error = core_rate = 0.0

    def square(v, j, weight):
        return (2 if j else 1) * float((weight[..., None] * v.abs().square()).sum())

    for j, r in enumerate(raw):
        pr = PressureFit(
            sp,
            1.0,
            0,
            j,
            pressure_degree - pb.degree,
            angular_order=cfg["m"] * j,
            radial_frequency=-1j * j * geo.twist,
        )

        def project(v, pressure=pr):
            return v - pressure.fit(v)

        zero = torch.zeros_like(r)
        l = project(L[j]) if j < len(L) else zero
        nonlinear = project(r - (L[j] if j < len(L) else zero))
        aux = project(damping * bg.window * U[j]) if 0 < j < len(U) else zero
        driven = l + nonlinear + aux - (torque if j == 0 else zero)
        if j == 0:
            # Retain the resolved axisymmetric pressure of the base as an affine
            # pressure; fit the unrestricted pressure correction on the full sphere.
            driven = project(r - base_grad) - torque
        weight = 2 * math.pi * sp.weight
        rows.append(
            {
                k: math.sqrt(square(v, j, weight))
                for k, v in (
                    ("residual", driven),
                    ("linear", l),
                    ("nonlinear", nonlinear),
                    ("damping", aux),
                )
            }
        )
        if j:
            core_error += square(driven, j, sp.core_weight)
            if j < len(U):
                core_rate += square(j * cfg["omega"] * U[j], j, sp.core_weight)

    def norm(key, subset):
        return math.sqrt(sum(r[key] ** 2 for r in subset))

    wave = rows[1:]
    force_norm = math.sqrt(square(torque, 0, 2 * math.pi * sp.weight))
    denom = sum(norm(k, wave) for k in ("linear", "nonlinear", "damping"))
    mean_delta = U[0] - bg.fields[0]
    iw = inflow_weight(sp)
    rr = sp.rho[:, None] * np.sqrt(1 - sp.meridian.eta**2)
    zz = sp.rho[:, None] * sp.meridian.eta
    energy = sum(sp.core_weight * u.abs().square().sum(-1) for u in U[1:])
    region = sp.tensor((rr > 20) & (rr < 65) & (abs(zz) < 20))
    return {
        "inflow_fundamental_over_base": math.sqrt(
            square(U[1], 1, iw) / max(square(bg.fields[0], 0, iw), 1e-300)
        ),
        "inflow_region_energy_fraction": float((energy * region).sum())
        / max(float(energy.sum()), 1e-300),
        "energy_centroid_rz": [
            float((energy * sp.tensor(loc)).sum()) / max(float(energy.sum()), 1e-300)
            for loc in (rr, zz)
        ],
        "quadrature": [len(sp.wr), len(sp.wa), trial.phase_nodes],
        "pressure_degree": pressure_degree,
        "residual_h": norm("residual", rows),
        "wave_residual_h": norm("residual", wave),
        "wave_relative_h": norm("residual", wave) / max(denom, 1e-300),
        "mean_residual_over_reference_force": rows[0]["residual"]
        / max(
            bg.force * math.sqrt(square(bg.drive, 0, 2 * math.pi * sp.weight)), 1e-300
        ),
        "force_h": force_norm,
        "modes": rows,
        "outside_h": norm("residual", rows[len(U) :]),
        "corotating_core_defect": math.sqrt(core_error / max(core_rate, 1e-300)),
        "fundamental_rms": sp.wave_rms(U[1]),
        "mean_change_rms": math.sqrt(square(mean_delta, 0, sp.core_weight)),
        "harmonic_rms": [
            math.sqrt(square(u, j, sp.core_weight)) for j, u in enumerate(U)
        ],
    }


def run(args):
    source = read_base(args.base)
    digest = hashlib.sha256(args.base.read_bytes()).hexdigest()
    device = backend(args.device)
    grids = args.audit_grids
    history = []
    args.out.mkdir(parents=True, exist_ok=False)
    if args.audit_only:
        cfg, b, c = read(args.audit_only)
        if cfg["base_sha256"] != digest:
            raise ValueError("Checkpoint and base hashes differ")
        rows = [
            audit(source, cfg, b, c, n, p, device)
            for n, p in zip(grids, args.pressure_degrees, strict=True)
        ]
        write_json(args.out / "audit.json", rows)
        return
    sigma, eta = MESHES[args.mesh]
    b = FullSphereBasis(sigma_breaks=sigma, eta_breaks=eta, harmonics=args.harmonics)
    audit_mesh = FullSphereBasis(
        sigma_breaks=source[1].sigma_breaks,
        eta_breaks=source[1].eta_breaks,
        harmonics=0,
    )
    sp = WaveSpace(b, args.grid, device, args.m, quadrature_basis=audit_mesh)
    bg = Background(sp, source)
    info = {
        "kind": "full_sphere_fixed_torque_wave",
        "basis": asdict(b),
        "m": args.m,
        "g": bg.g,
        "nu": 1.0,
        "base_sha256": digest,
        "reference_torque": bg.force,
        "torque_fraction": 1.0,
        "search_quadrature": [len(sp.wr), len(sp.wa), sp.phase_nodes],
        "device": str(device),
    }
    if args.restart:
        old, oldb, c = read(args.restart)
        if (old["base_sha256"], old["m"], old["torque_fraction"]) != (
            digest,
            args.m,
            1.0,
        ):
            raise ValueError(
                "Restart must match the base, wave number and fixed torque"
            )
        c = lift_coefficients(oldb, b, c) if oldb != b else c
        omega, damping = old["omega"], old["damping"]
        report = {"restart": str(args.restart)}
    else:
        omega, target = packet(sp, args.packet, args.m, digest)
        mean, fields, mean_info = equilibrate_mean(sp, bg)
        omega, damping, c, trace = neutral_seed(
            sp, bg, omega, target, fields, args.seed_selection
        )
        c[0] = mean
        report = {
            "seed_selection": args.seed_selection,
            "packet": str(args.packet),
            "neutral_seed": trace,
            "mean_on_trial_space": mean_info,
        }
        print(
            f"m={args.m} neutral seed omega={omega:.6g} damping={damping:.6g}",
            flush=True,
        )
    report.update(metadata=info, stages=history)
    for k, ratio in enumerate(args.amplitudes):
        amp = ratio * bg.mean_norm
        eq = FixedTorqueWave(sp, bg, omega, damping, c, amp)
        x = eq.retract(eq.start)

        def state(point, problem):
            return [v.detach().cpu().numpy() for v in problem.coefficients(point)]

        cfg = {
            **info,
            "omega": float(omega),
            "damping": float(damping),
            "amplitude": amp,
            "amplitude_ratio": ratio,
        }
        initial = state(x, eq)
        save(args.out / f"initial-{k:02d}.npz", cfg, initial)
        initial_norm = float(la.norm(eq.evaluate(x, 0)))
        border = np.zeros(len(x) + 1)
        border[-1] = 1.0
        z, ok, status = correct(eq, np.r_[x, 0.0], border, args.tol, args.iterations)
        y = z[:-1]
        if not np.isfinite(y).all():
            y = x
            ok = False
        omega = float(eq.parameters(y)[1])
        damping = float(eq.damp_scale * y[-2])
        c = state(y, eq)
        cfg.update(omega=omega, damping=damping, weak_converged=ok)
        save(args.out / f"stage-{k:02d}.npz", cfg, c)
        row = {
            "amplitude_ratio": ratio,
            "amplitude": amp,
            "weak_initial": initial_norm,
            "solver": status,
            "weak_converged": ok,
            "omega": omega,
            "damping": damping,
        }
        history.append(row)
        if ok:
            report["last_converged_checkpoint"] = f"stage-{k:02d}.npz"
        write_json(args.out / "search.json", report)
        print(
            f"ratio={ratio:g} weak={status['residual']} omega={omega:.7g} damping={damping:.7g} {status['reason']}",
            flush=True,
        )
        if not ok:
            break
    # Audit both ends of the final nonlinear correction using the same finer grid.
    del eq
    before, _, ic = read(args.out / f"initial-{k:02d}.npz")
    report["initial_audit"] = audit(
        source, before, b, ic, grids[-1], args.pressure_degrees[-1], device
    )
    report["audits"] = [
        audit(source, cfg, b, c, n, p, device)
        for n, p in zip(grids, args.pressure_degrees, strict=True)
    ]
    last = report["audits"][-1]
    report["status"] = "weak_fixed_torque_root" if ok else "nonlinear_trial"
    print(
        f"audited m={args.m} wave relative H={last['wave_relative_h']:.6g} core defect={last['corotating_core_defect']:.6g}",
        flush=True,
    )
    write_json(args.out / "search.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base", type=Path, default=Path(__file__).with_name("axisymmetric_base.npz")
    )
    group = p.add_mutually_exclusive_group(required=True)
    for name in ("packet", "restart", "audit-only"):
        group.add_argument("--" + name, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--m", type=int, choices=range(3, 101), default=19)
    p.add_argument("--mesh", choices=MESHES, default="start")
    p.add_argument("--harmonics", type=int, default=2)
    p.add_argument("--grid", type=int, default=128)
    p.add_argument("--audit-grids", nargs="+", type=int, default=[160, 224])
    p.add_argument("--pressure-degrees", nargs="+", type=int, default=[4, 5])
    p.add_argument("--amplitudes", nargs="+", type=float, default=[0.03, 0.10])
    p.add_argument("--iterations", type=int, default=45)
    p.add_argument(
        "--seed-selection", choices=("overlap", "residual"), default="overlap"
    )
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = p.parse_args()
    if (
        args.harmonics < 2
        or min(args.amplitudes) <= 0
        or not np.isfinite(args.amplitudes).all()
        or len(args.audit_grids) != len(args.pressure_degrees)
        or min(args.audit_grids) <= args.grid
        or max(args.pressure_degrees) < 4
        or any(p < 3 for p in args.pressure_degrees)
    ):
        p.error(
            "Use at least two harmonics, positive amplitudes and independent finer audit grids with matching pressure degrees"
        )
    run(args)


if __name__ == "__main__":
    main()
