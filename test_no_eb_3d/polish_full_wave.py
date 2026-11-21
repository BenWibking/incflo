"""Minimize the physical pressure-fitted momentum norm at fixed wave amplitude and torque."""

import argparse
import hashlib
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from probe_rotating_modes import FullSphereBasis
from scipy.optimize import minimize
from similarity_elements import PressureFit
from similarity_rotating import (
    WaveGeometry,
    WaveSpace,
    dense_columns,
    lift_coefficients,
)
from solve_axisymmetric import read as read_base
from solve_full_rotating_wave import (
    Background,
    audit,
    base_pressure_gradient,
    inflow_weight,
    packet,
    read,
    save,
)
from solve_similarity_orbit import backend, write_json


class PhysicalFit:
    def __init__(self, source, cfg, b, coeff, grid, degree, device):
        mesh = FullSphereBasis(
            sigma_breaks=source[1].sigma_breaks,
            eta_breaks=source[1].eta_breaks,
            harmonics=0,
        )
        ps = WaveSpace(mesh, grid, device, cfg["m"])
        self.sp = sp = WaveSpace(b, grid, device, cfg["m"], quadrature_basis=mesh)
        self.bg = bg = Background(sp, source)
        self.anchor_weight = (
            inflow_weight(sp) if cfg.get("anchor") == "inflow" else sp.core_weight
        )
        self.anchor_base_norm = float(
            (self.anchor_weight[..., None] * bg.fields[0].abs().square()).sum().sqrt()
        )
        self.omega, self.damping = cfg["omega"], cfg["damping"]
        self.force = cfg["torque_fraction"] * bg.force * bg.drive
        self.base_pressure = base_pressure_gradient(ps, bg, source[1])
        self.force2 = float((sp.weight[..., None] * self.force.abs().square()).sum())
        self.c = [
            torch.as_tensor(v, device=device, dtype=torch.complex128) for v in coeff
        ]
        self.nodes = (-1.5, -0.5, 0.5, 1.5)
        self.step = max(2.0, abs(self.omega) / 4)
        # The factored velocity, curl and linear operator are polynomials of degree
        # <=3 in frequency. Only the external radial phase is non-polynomial.
        self.geo = [
            WaveGeometry(sp, cfg["m"], bg.g, self.omega + n * self.step)
            for n in self.nodes
        ]
        self.pressures = [
            PressureFit(
                ps,
                1.0,
                0,
                j,
                degree - mesh.degree,
                angular_order=j * cfg["m"],
                radial_frequency=-1j * j * self.omega / (2 * bg.g),
            )
            for j in range(2 * b.harmonics + 1)
        ]
        ref = WaveGeometry(sp, cfg["m"], bg.g, self.omega)
        self.reference_wave = ref.velocity[1].apply(self.c[1])
        self.columns = []
        for j in range(b.harmonics + 1):
            v, c, l = (
                dense_columns(ops[j]) for ops in (ref.velocity, ref.curl, ref.linear)
            )
            h = (
                l
                + torch.linalg.cross(c, bg.fields[0].reshape(-1, 3, 1), dim=1)
                + torch.linalg.cross(bg.fields[1].reshape(-1, 3, 1), v, dim=1)
            )
            if j:
                h += self.damping * bg.window.reshape(-1, 1, 1) * v
            n = h.shape[-1]
            mat = (
                (h.reshape(-1, n).mH @ (sp.weight.reshape(-1, 1, 1) * h).reshape(-1, n))
                .cpu()
                .numpy()
            )
            s = 1 / np.sqrt(mat.diagonal().real)
            a = mat * s[:, None] * s
            factor = la.cholesky((a + a.conj().T) / 2 + 1e-7 * np.eye(n), lower=True)
            col = s[:, None] * la.solve_triangular(
                factor.conj().T, np.eye(n), lower=False
            )
            self.columns.append(
                torch.as_tensor(col * math.sqrt(self.force2), device=device)
            )
        self.size = 2 + sum(len(c) * (2 if j else 1) for j, c in enumerate(self.c))

    def fields(self, x, amplitude):
        sp = self.sp
        t = x[-2]
        omega = self.omega + self.step * t
        damping = self.damping + 2 * x[-1]
        weights = []
        for i, a in enumerate(self.nodes):
            f = 1.0
            for j, b in enumerate(self.nodes):
                if j != i:
                    f = f * (t - b) / (a - b)
            weights.append(f)
        coeff = []
        offset = 0
        for j, (initial, transform) in enumerate(
            zip(self.c, self.columns, strict=True)
        ):
            n = len(initial)
            real = x[offset : offset + n]
            offset += n
            imag = x[offset : offset + n] if j else torch.zeros_like(real)
            if j:
                offset += n
            coeff.append(initial + transform @ torch.complex(real, imag))
        fields = []
        for attr in ("velocity", "curl", "linear"):
            fields.append(
                [
                    sum(
                        w * getattr(geo, attr)[j].apply(c)
                        for w, geo in zip(weights, self.geo, strict=True)
                    )
                    for j, c in enumerate(coeff)
                ]
            )
        factor = (
            amplitude
            / (
                2 * (self.anchor_weight[..., None] * fields[0][1].abs().square()).sum()
            ).sqrt()
        )
        coeff[1] = coeff[1] * factor
        for f, bg in zip(fields, self.bg.fields, strict=True):
            f[0] = f[0] + bg
            f[1] = f[1] * factor
        clock = torch.exp(1j * (omega - self.omega) / (2 * self.bg.g) * sp.log_radius)
        phase = torch.angle(
            (
                self.anchor_weight[..., None]
                * self.reference_wave.conj()
                * fields[0][1]
                * clock
            ).sum()
        )
        for j in range(1, len(coeff)):
            rotation = torch.exp(-1j * j * phase)
            coeff[j] = coeff[j] * rotation
            for f in fields:
                f[j] = f[j] * rotation
        return coeff, fields, omega, damping

    def snapshot(self, y, amplitude):
        with torch.no_grad():
            cs, _, omega, damping = self.fields(self.sp.tensor(y), amplitude)
        return [v.cpu().numpy() for v in cs], float(omega), float(damping)

    def value(self, y, amplitude):
        sp = self.sp
        x = sp.tensor(y).requires_grad_(True)
        _, (u, c, l), omega, damping = self.fields(x, amplitude)
        nphi = 4 * sp.basis.harmonics + 1

        def synth(cs):
            return torch.fft.irfft(torch.stack(cs), n=nphi, dim=0, norm="forward")

        nonlinear = torch.fft.rfft(
            torch.linalg.cross(synth(c), synth(u)), dim=0, norm="forward"
        )
        total = 0.0
        for j, pr in enumerate(self.pressures):
            raw = nonlinear[j]
            if j < len(l):
                raw = raw + l[j]
                if j:
                    raw = raw + damping * self.bg.window * u[j]
            if j:
                raw = raw * torch.exp(
                    1j * j * (omega - self.omega) / (2 * self.bg.g) * sp.log_radius
                )
            else:
                raw = raw - self.base_pressure - self.force
            # The least-squares pressure is stationary in its coefficients; the
            # envelope theorem gives the velocity gradient with this fit held fixed.
            with torch.no_grad():
                gradp = pr.fit(raw.detach())
            total = (
                total
                + (2 if j else 1)
                * (sp.weight[..., None] * (raw - gradp).abs().square()).sum()
            )
        loss = 0.5 * total / self.force2
        grad = torch.autograd.grad(loss, x)[0]
        return float(loss.detach()), grad.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--base", type=Path, default=Path(__file__).with_name("axisymmetric_base.npz")
    )
    p.add_argument("--grid", type=int, default=160)
    p.add_argument("--pressure-degree", type=int, default=4)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--degree", type=int)
    p.add_argument("--harmonics", type=int)
    p.add_argument("--anchor", choices=["core", "inflow"])
    p.add_argument(
        "--packet",
        type=Path,
        help="Reinitialize the wave on the saved packet and restore the original base",
    )
    p.add_argument("--amplitudes", type=float, nargs="+", default=[0.03, 0.10])
    p.add_argument("--audit-grid", type=int, default=256)
    p.add_argument("--audit-degree", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    a = p.parse_args()
    cfg, b, c = read(a.checkpoint)
    if (
        cfg["base_sha256"] != hashlib.sha256(a.base.read_bytes()).hexdigest()
        or min(a.amplitudes) <= 0
        or not np.isfinite(a.amplitudes).all()
        or a.audit_grid <= a.grid
        or a.audit_degree <= a.pressure_degree
    ):
        p.error(
            "Use the original base, positive finite amplitudes and a finer independent audit"
        )
    source = read_base(a.base)
    device = backend(a.device)
    richer = replace(
        b, degree=a.degree or b.degree, harmonics=a.harmonics or b.harmonics
    )
    if richer.degree < b.degree or richer.harmonics < b.harmonics:
        p.error("Velocity refinement must retain the initial field")
    if richer != b:
        c = lift_coefficients(b, richer, c)
        b = richer
        cfg = {**cfg, "basis": asdict(b)}
    cfg = {**cfg, "anchor": a.anchor or cfg.get("anchor", "core")}
    if a.packet:
        mesh = FullSphereBasis(
            sigma_breaks=source[1].sigma_breaks,
            eta_breaks=source[1].eta_breaks,
            harmonics=0,
        )
        sp = WaveSpace(b, a.grid, device, cfg["m"], quadrature_basis=mesh)
        omega, target = packet(sp, a.packet, cfg["m"], cfg["base_sha256"])
        cfg.update(omega=omega, damping=0.0)
        mer = sp.meridian
        op = mer.operator(
            1,
            cfg["m"],
            cfg["g"],
            omega,
            (1 - cfg["g"]) / cfg["g"],
            "velocity",
            sp.device,
        )
        phase = torch.exp(-1j * omega / (2 * cfg["g"]) * sp.log_radius)
        D = dense_columns(op)
        n = D.shape[-1]
        W = sp.core_weight.reshape(-1, 1, 1)
        gram = (D.reshape(-1, n).mH @ (W * D).reshape(-1, n)).cpu().numpy()
        rhs = op.adjoint(sp.core_weight[..., None] * phase * target).cpu().numpy()
        s = 1 / np.sqrt(gram.diagonal().real)
        c = [np.zeros_like(v) for v in c]
        c[1] = s * la.solve(
            s[:, None] * gram * s + 1e-10 * np.eye(n), s * rhs, assume_a="her"
        )
        del D
    a.out.mkdir(parents=True, exist_ok=False)
    report = {
        "source": str(a.checkpoint),
        "packet": str(a.packet) if a.packet else None,
        "stages": [],
    }
    for i, ratio in enumerate(a.amplitudes):
        fit = PhysicalFit(source, cfg, b, c, a.grid, a.pressure_degree, device)
        amp = ratio * fit.anchor_base_norm
        x = np.zeros(fit.size)

        cfg = {
            **cfg,
            "amplitude": amp,
            "amplitude_ratio": ratio,
            "weak_converged": False,
            "method": "pressure-fitted H minimization",
            "search_quadrature": [
                len(fit.sp.wr),
                len(fit.sp.wa),
                "exact Fourier convolution",
            ],
        }
        initial, _, _ = fit.snapshot(x, amp)
        before = audit(source, cfg, b, initial, a.audit_grid, a.audit_degree, device)
        first = fit.value(x, amp)[0]
        # The lowest-H iterate keeps the requested amplitude by construction.
        sol = minimize(
            fit.value,
            x,
            args=(amp,),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": a.iterations,
                "maxcor": 15,
                "gtol": 1e-9,
                "ftol": 1e-13,
                "maxls": 22,
            },
        )
        c, omega, damping = fit.snapshot(sol.x, amp)
        cfg.update(omega=omega, damping=damping)
        save(a.out / f"stage-{i:02d}.npz", cfg, c)
        del fit
        after = audit(source, cfg, b, c, a.audit_grid, a.audit_degree, device)
        row = {
            "ratio": ratio,
            "omega": omega,
            "damping": damping,
            "iterations": sol.nit,
            "evaluations": sol.nfev,
            "stop": str(sol.message),
            "initial_training_h_over_force": math.sqrt(2 * first),
            "final_training_h_over_force": math.sqrt(2 * sol.fun),
            "initial_audit": before,
            "final_audit": after,
        }
        report["stages"].append(row)
        write_json(a.out / "physical.json", report)
        print(
            f"m={cfg['m']} ratio={ratio:g} independent wave H={after['wave_relative_h']:.5f} "
            f"core defect={after['corotating_core_defect']:.5f} damping={damping:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
