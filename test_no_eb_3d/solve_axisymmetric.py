"""Equilibrate a full-space axisymmetric similarity flow under a prescribed odd swirl torque."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from scipy.interpolate import RegularGridInterpolator
from similarity_elements import PressureFit
from similarity_rotating import (
    WaveBasis,
    WaveMeridian,
    WaveSpace,
    dense_columns,
    lift_coefficients,
)
from solve_similarity_orbit import atomic, backend, write_json


def cutoff(x):
    t = np.clip((1 - x) / 0.2, 0, 1)
    return t**3 * (10 + t * (-15 + 6 * t))


def torque_shape(radius, height, drive_r, drive_z):
    x, y = radius / drive_r, height / drive_z
    return x * y * np.exp(1 - 0.5 * (x**2 + y**2))


class MeanGrid:
    def __init__(
        self, basis, grid, device, g, drive_r, drive_z, profile, pressure_extra
    ):
        self.basis, self.g = basis, g
        self.space = sp = WaveSpace(basis, grid, device, 3)
        self.ops = [
            sp.meridian.operator(0, 0, g, 0, (1 - g) / g, name, device)
            for name in ("velocity", "curl", "linear")
        ]
        self.pressure = PressureFit(sp, 1, 0, 0, pressure_extra)
        self.pressure_degree = basis.degree + pressure_extra
        # Reflection of the northern meridian includes the southern one. The
        # azimuthal integral is exact for m=0, including the nonlinear product.
        self.weights = (4 * math.pi * sp.weight).reshape(-1)
        eta = sp.meridian.eta[None, :]
        sn = np.sqrt(1 - eta**2)
        rr, zz = sp.rho[:, None] * sn, sp.rho[:, None] * eta
        self.r, self.z = rr, zz
        self.torque = torch.zeros((rr.size, 3), device=device, dtype=torch.float64)
        self.torque[:, 2] = sp.tensor(torque_shape(rr, zz, drive_r, drive_z)).ravel()
        with np.load(profile, allow_pickle=False) as im:
            r, z = im["r"], im["z"]
            points = np.stack([np.minimum(zz, z[-1]), np.minimum(rr, r[-1])], axis=-1)
            ur, up, uz = (
                RegularGridInterpolator((z, r), im[k])(points)
                for k in ("mean_wr", "mean_wphi", "mean_wz")
            )
            self.figure = sp.tensor(
                np.stack([sn * ur + eta * uz, eta * ur - sn * uz, up], axis=-1)
            ).reshape(-1, 3)
            self.observed = (
                self.weights
                * sp.tensor(cutoff(rr / r[-1]) * cutoff(zz / z[-1])).ravel()
            )
        self.swirl_weight = self.observed * self.torque[:, 2]
        self.swirl_weight /= self.swirl_weight.sum()
        self.figure_swirl = float(self.swirl_weight @ self.figure[:, 2])
        self.drive_norm = math.sqrt(float(self.weights @ self.torque[:, 2].square()))
        self.drive_exact = math.sqrt(
            math.pi**1.5 * math.e**2 * drive_r**2 * drive_z / 2
        )

    def probe(self, coeff, radius, height):
        R = np.hypot(radius, height)
        eta, sn = np.array(height) / R, np.array(radius) / R
        m = WaveMeridian(self.basis, R / (R + self.basis.radius), eta)
        u = (
            m.operator(0, 0, self.g, 0, (1 - self.g) / self.g, "velocity", "cpu")
            .apply(torch.as_tensor(coeff, dtype=torch.complex128))
            .real.numpy()
        )
        u = u[np.arange(len(R)), np.arange(len(R))]
        return np.column_stack(
            (sn * u[:, 0] + eta * u[:, 1], u[:, 2], eta * u[:, 0] - sn * u[:, 1])
        )

    def audit(self, coeff, force):
        sp = self.space
        c = torch.as_tensor(coeff, dtype=torch.complex128, device=sp.device)
        u, curl, linear = [op.apply(c).real.reshape(-1, 3) for op in self.ops]
        nonlinear = torch.linalg.cross(curl, u)
        shape = (*sp.weight.shape, 3)

        def projected(v):
            return v - self.pressure.fit(
                v.reshape(shape).to(torch.complex128)
            ).real.reshape(-1, 3)

        def norm(v):
            return math.sqrt(float(self.weights @ v.square().sum(dim=1)))

        pl, pn = projected(linear), projected(nonlinear)
        unforced = pl + pn
        forced = unforced - force * self.torque
        f = abs(force) * self.drive_norm
        deficit = norm(forced)
        size = norm(pl) + norm(pn)
        peak = int(u[:, 2].argmax())
        probes = self.probe(coeff, [30, 0], [0, 24])
        return {
            "grid": [len(sp.wr), len(sp.wa)],
            "azimuth": "exact axisymmetric integral",
            "pressure_degree": self.pressure_degree,
            "residual_h": deficit,
            "residual_over_force": deficit / max(f, 1e-300),
            "unforced_h": norm(unforced),
            "force_h": f,
            "force_over_projected_terms": f / max(size, 1e-300),
            "force_over_raw_terms": f / max(norm(linear) + norm(nonlinear), 1e-300),
            "relative_residual": deficit / max(size + f, 1e-300),
            "swirl_mean": float(self.swirl_weight @ u[:, 2]),
            "figure_swirl_mean": self.figure_swirl,
            "max_swirl": float(u[:, 2].max()),
            "max_swirl_rz": [self.r.flat[peak], self.z.flat[peak]],
            "radial_at_r30_z0": probes[0, 0],
            "vertical_at_r0_z24": probes[1, 2],
            "figure_relative_velocity_error": math.sqrt(
                float(self.observed @ (u - self.figure).square().sum(1))
                / float(self.observed @ self.figure.square().sum(1))
            ),
            "force_quadrature_error": abs(self.drive_norm / self.drive_exact - 1),
        }


class Equilibrator:
    def __init__(self, meridian):
        self.grid = meridian
        sp = meridian.space
        self.sqrt_volume = meridian.weights.sqrt()[:, None]
        self.weighted_unit_force = (self.sqrt_volume * meridian.torque).reshape(-1)
        p = meridian.pressure
        gradients = dense_columns(p.op).real.contiguous()
        gradients *= sp.weight.sqrt().reshape(-1, 1, 1)
        gradients = gradients.reshape(-1, gradients.shape[-1]) * p.scale
        # One triangular solve per quadrature row. Bounded host tiles avoid the
        # large workspace ROCm's TRSM requests for tens of thousands of RHS.
        factor = p.chol.real.contiguous().cpu().numpy()
        self.pressure_frame = torch.empty_like(gradients)
        for start in range(0, len(gradients), 2048):
            section = slice(start, start + 2048)
            block = gradients[section].cpu().numpy()
            q = la.solve_triangular(factor, block.T, lower=True, check_finite=False)
            self.pressure_frame[section].copy_(torch.as_tensor(q.T))
        del gradients
        v, c, linear_columns = [
            dense_columns(o).real.contiguous() for o in meridian.ops
        ]
        n = linear_columns.shape[-1]
        linear = self.project(
            (self.sqrt_volume[:, :, None] * linear_columns).reshape(-1, n)
        )
        norms = torch.linalg.vector_norm(linear, dim=0)
        self.scale = 1 / norms.clamp(min=norms.max() * 1e-13)
        self.linear = linear * self.scale
        self.velocity = v * self.scale
        self.vorticity = c * self.scale
        self.anchor = meridian.swirl_weight @ self.velocity[:, 2, :]
        # The linearized toroidal equation is independent of pressure and the
        # poloidal velocity. A unit physical torque calibrates the force scale.
        mean = meridian.basis.groups[0]
        start = math.prod(s.count for s in mean["S"])
        count = math.prod(s.count for s in mean["T"])
        ti = slice(start, start + count)
        toroidal = self.linear.reshape(-1, 3, n)[:, 2, ti]
        self.stokes = sp.tensor(np.zeros(n))
        self.stokes[ti] = sp.tensor(
            la.lstsq(
                toroidal.cpu().numpy(),
                self.weighted_unit_force[2::3].cpu().numpy(),
                cond=1e-12,
            )[0]
        )
        response = float(self.anchor @ self.stokes)
        if not math.isfinite(response) or response <= 0:
            raise ValueError(
                "The linear torque has no resolved positive swirl response"
            )
        self.reference_force = meridian.figure_swirl / response

    def project(self, a):
        return a - self.pressure_frame @ (self.pressure_frame.mT @ a)

    def fit_figure(self):
        m = self.grid
        w = (m.observed / m.observed.sum()).sqrt()[:, None]
        X = (self.velocity * w[:, :, None]).reshape(-1, self.velocity.shape[-1])
        target = (m.figure * w).ravel()
        gram, rhs = X.mT @ X, X.mT @ target
        wcore = m.space.core_weight.reshape(-1, 1, 1).sqrt()
        Xcore = (self.velocity * wcore).reshape(-1, self.velocity.shape[-1])
        gram += 1e-6 * (Xcore.mT @ Xcore)
        diag = gram.diagonal().sqrt().clamp(min=1e-15)
        scaled = (gram / (diag[:, None] * diag)).cpu().numpy()
        x = la.lstsq(scaled, (rhs / diag).cpu().numpy(), cond=1e-11)[0]
        return m.space.tensor(x) / diag

    def coefficients(self, x):
        return (x * self.scale).cpu().numpy()

    def residual(self, x, force, derivative=False):
        u, c = self.velocity @ x, self.vorticity @ x
        adv = torch.linalg.cross(c, u)
        r = (
            self.linear @ x
            + self.project((self.sqrt_volume * adv).ravel())
            - force * self.weighted_unit_force
        )
        if not derivative:
            return r
        jac = torch.linalg.cross(self.vorticity.transpose(1, 2), u[:, None, :], dim=-1)
        jac += torch.linalg.cross(c[:, None, :], self.velocity.transpose(1, 2), dim=-1)
        jac = (jac * self.sqrt_volume[:, :, None]).transpose(1, 2).reshape(-1, len(x))
        return r, self.linear + self.project(jac)

    def solve(self, x, force, iterations):
        history, damping, reason = [], 1e-5, "iteration_budget"
        for k in range(iterations + 1):
            residual = self.residual(x, force)
            value = float(torch.linalg.vector_norm(residual))
            history.append({"iteration": k, "residual_h": value, "damping": damping})
            if k == iterations:
                break
            r, J = self.residual(x, force, True)
            colnorm = torch.linalg.vector_norm(J, dim=0)
            d = colnorm.clamp(min=colnorm.max() * 1e-13)
            G = ((J.mT @ J) / (d[:, None] * d)).cpu().numpy()
            grad = ((J.mT @ r) / d).cpu().numpy()
            eig, Q = la.eigh(G)
            eig = np.maximum(eig, 0)
            ng = Q.T @ grad
            if la.norm(grad, np.inf) <= 1e-10 * max(value, 1e-12):
                reason = "stationary_least_squares"
                break
            for _ in range(16):
                delta = -Q @ (ng / (eig + damping))
                predicted = -2 * (grad @ delta) - delta @ (G @ delta)
                trial = x + self.grid.space.tensor(delta) / d
                trial_h = float(torch.linalg.vector_norm(self.residual(trial, force)))
                gain = (value - trial_h) * (value + trial_h)
                ratio = gain / predicted if predicted > 0 else -1
                if ratio > 0.03 and gain > 0:
                    x = trial
                    if ratio > 0.75:
                        damping = max(1e-12, damping * 0.28)
                    elif ratio < 0.25:
                        damping *= 2
                    break
                damping *= 6
            else:
                reason = "stationary_least_squares"
                break
            if gain <= 1e-11 * max(value**2, 1e-30):
                history.append(
                    {"iteration": k + 1, "residual_h": trial_h, "damping": damping}
                )
                reason = "stationary_least_squares"
                break
        return x, history, reason


def save(path, coeff, cfg):
    atomic(path, lambda f: np.savez_compressed(f, c0=coeff, config=json.dumps(cfg)))


def read(path):
    with np.load(path, allow_pickle=False) as data:
        cfg, c = json.loads(str(data["config"])), data["c0"].copy()
    if cfg.get("kind") != "fixed_torque_axisymmetric":
        raise ValueError("Supply an axisymmetric fixed-torque checkpoint")
    basis = WaveBasis(**cfg["basis"])
    size = sum(r.count * a.count if r else 1 for r, a in basis.groups[0].values())
    if basis.harmonics or c.shape != (size,) or not np.isfinite(c).all():
        raise ValueError("Invalid axisymmetric velocity coefficients")
    return cfg, basis, c


def run(args, restored=None):
    device = torch.device(backend(args.device))
    basis = WaveBasis(
        degree=args.degree,
        harmonics=0,
        radius=args.radius,
        sigma_breaks=args.sigma_breaks,
        eta_breaks=args.eta_breaks,
    )

    def grid(n, extra):
        return MeanGrid(
            basis, n, device, args.g, args.drive_r, args.drive_z, args.profile, extra
        )

    main = grid(args.grid, 1)
    eq = Equilibrator(main)
    independent = [grid(n, 2) for n in args.audit_grids]
    independent.append(grid(args.audit_grids[-1], 3))
    force_scale = eq.reference_force
    if restored is not None:
        cfg, old, coeff = restored
        for key in ("g", "drive_r", "drive_z"):
            if cfg[key] != getattr(args, key):
                raise ValueError(f"Restart changes physical parameter {key}")
        coeff = lift_coefficients(old, basis, [coeff])[0] if old != basis else coeff
        x = main.space.tensor(np.asarray(coeff).real) / eq.scale
        force_scale = cfg["reference_force"]
    else:
        x = (
            eq.fit_figure()
            if args.seed == "figure"
            else eq.stokes * (args.fractions[0] * force_scale)
        )
    args.out.mkdir(parents=True, exist_ok=False)
    base = {
        "kind": "fixed_torque_axisymmetric",
        "basis": asdict(basis),
        "g": args.g,
        "nu": 1.0,
        "drive_r": args.drive_r,
        "drive_z": args.drive_z,
        "reference_force": force_scale,
        "search_grid": args.grid,
    }
    report = {
        **base,
        "seed": str(args.restart) if args.restart else args.seed,
        "device": str(device),
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "stages": [],
        "figure_swirl_mean": main.figure_swirl,
        "unknowns": len(x),
        "balance_tolerance": args.balance_tol,
        "status": "running",
    }
    for i, fraction in enumerate(args.fractions):
        force = fraction * force_scale
        x, history, reason = eq.solve(x, force, args.iterations)
        c = eq.coefficients(x)
        audit = [m.audit(c, force) for m in [main, *independent]]
        last = audit[-1]
        variation = max(
            abs(a["residual_over_force"] - last["residual_over_force"])
            for a in audit[1:]
        )
        passed = (
            max(a["residual_over_force"] for a in audit[1:]) <= args.balance_tol
            and variation <= 0.1 * args.balance_tol
        )
        entry = {
            "stage": i,
            "fraction": fraction,
            "force": force,
            "balance_passed": bool(passed),
            "reason": reason,
            "iterations": history,
            "audits": audit,
        }
        report["stages"].append(entry)
        cfg = {
            **base,
            "force": force,
            "fraction": fraction,
            "stage": i,
            "balance_passed": bool(passed),
        }
        save(args.out / f"stage-{i:02d}.npz", c, cfg)
        write_json(args.out / "axisymmetric.json", report)
        print(
            json.dumps(
                {k: v for k, v in entry.items() if k not in ("audits", "iterations")}
                | last
            ),
            flush=True,
        )
    report["status"] = "finished"
    write_json(args.out / "axisymmetric.json", report)


def audit_saved(args):
    cfg, basis, coeff = read(args.audit_only)
    device = torch.device(backend(args.device))
    measurements = []
    for n in [args.grid, *args.audit_grids]:
        for extra in (2, 3) if n == args.audit_grids[-1] else (2,):
            mesh = MeanGrid(
                basis,
                n,
                device,
                cfg["g"],
                cfg["drive_r"],
                cfg["drive_z"],
                args.profile,
                extra,
            )
            measurements.append(mesh.audit(coeff, cfg["force"]))
    comparison = None
    if args.compare:
        other, old, coarse = read(args.compare)
        for key in ("g", "nu", "drive_r", "drive_z", "force"):
            if not math.isclose(cfg[key], other[key], rel_tol=1e-12):
                raise ValueError(f"Field comparison changes {key}")
        sp = mesh.space
        mer = WaveMeridian(old, sp.rho / (sp.rho + old.radius), sp.meridian.eta)
        op = mer.operator(
            0, 0, cfg["g"], 0, (1 - cfg["g"]) / cfg["g"], "velocity", device
        )
        v = (
            mesh.ops[0]
            .apply(torch.as_tensor(coeff, device=device, dtype=torch.complex128))
            .real
        )
        oldv = op.apply(
            torch.as_tensor(coarse, device=device, dtype=torch.complex128)
        ).real
        delta, energy = (v - oldv).square().sum(-1).ravel(), v.square().sum(-1).ravel()

        def relative(w):
            return math.sqrt(float(w @ delta) / float(w @ energy))

        comparison = {
            "checkpoint": str(args.compare),
            "relative_core_velocity_difference": relative(sp.core_weight.ravel()),
            "relative_observed_velocity_difference": relative(mesh.observed),
        }
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(
        args.out / "audit.json",
        {
            "checkpoint": str(args.audit_only),
            "state": cfg,
            "measurements": measurements,
            "field_comparison": comparison,
        },
    )
    print(json.dumps(measurements[-1]), flush=True)


def apply_defaults(args, restored):
    cfg = restored[0] if restored is not None else {}
    basis = cfg.get("basis", {})
    defaults = {
        "degree": basis.get("degree", 3),
        "radius": basis.get("radius", 24),
        "sigma_breaks": basis.get("sigma_breaks", [0, 0.18, 0.36, 0.54, 0.73, 0.88, 1]),
        "eta_breaks": basis.get("eta_breaks", [0, 0.12, 0.30, 0.53, 0.76, 1]),
        "g": cfg.get("g", 0.502),
        "drive_r": cfg.get("drive_r", 19),
        "drive_z": cfg.get("drive_z", 21),
        "grid": cfg.get("search_grid", 128),
        "fractions": [cfg["fraction"]] if cfg else [1, 0.5, 0.25, 0.1, 0.04],
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.audit_grids is None:
        args.audit_grids = [math.ceil(1.25 * args.grid), math.ceil(1.5 * args.grid)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("rotating_figure_seed.npz"),
    )
    p.add_argument("--grid", type=int)
    p.add_argument("--audit-grids", type=int, nargs="+")
    p.add_argument("--degree", type=int)
    p.add_argument("--sigma-breaks", type=float, nargs="+")
    p.add_argument("--eta-breaks", type=float, nargs="+")
    p.add_argument("--radius", type=float)
    p.add_argument("--g", type=float)
    p.add_argument("--drive-r", type=float)
    p.add_argument("--drive-z", type=float)
    p.add_argument("--seed", choices=("figure", "stokes"), default="figure")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--restart", type=Path)
    source.add_argument(
        "--audit-only",
        type=Path,
        help="Measure an unchanged checkpoint on independent grids",
    )
    p.add_argument(
        "--compare",
        type=Path,
        help="Compare a second field at the same physical torque during --audit-only",
    )
    p.add_argument("--fractions", nargs="+", type=float)
    p.add_argument("--iterations", type=int, default=45)
    p.add_argument(
        "--balance-tol",
        type=float,
        default=0.03,
        help="Maximum forced momentum error in units of the applied force norm",
    )
    args = p.parse_args()
    restored = read(args.restart) if args.restart else None
    apply_defaults(args, restored)
    if args.compare and not args.audit_only:
        p.error("--compare requires --audit-only")
    if not (
        0 < args.g < 2 / 3
        and args.drive_r > 0
        and args.drive_z > 0
        and args.grid >= 128
        and args.audit_grids == sorted(set(args.audit_grids))
        and min(args.audit_grids) > args.grid
        and 0 < args.balance_tol < 1
        and args.iterations > 0
        and all(0 < f < math.inf for f in args.fractions)
    ):
        p.error("Invalid force, similarity exponent, iteration budget or quadrature")
    torch.set_num_threads(1)
    if args.audit_only:
        audit_saved(args)
    else:
        run(args, restored)


if __name__ == "__main__":
    main()
