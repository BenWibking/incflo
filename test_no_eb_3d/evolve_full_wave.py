"""Independently time-march a nonlinear candidate for one rotation of its fundamental."""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from probe_rotating_modes import TAIL, FullSphereBasis
from similarity_rotating import WaveMeridian, WaveSpace, dense_columns
from solve_axisymmetric import read as read_base
from solve_full_rotating_wave import (
    MESHES,
    Background,
    TotalGeometry,
    inflow_weight,
    read,
)
from solve_similarity_orbit import backend, write_json


class Evolution:
    """Time-dependent, localized solenoidal corrections to the entire input field."""

    def __init__(self, sp, cfg, basis, coeff, source):
        self.sp, self.cfg = sp, cfg
        self.J = sp.basis.harmonics
        omega, g, m = cfg["omega"], cfg["g"], cfg["m"]
        self.bg = Background(sp, source)
        self.layer_weight = inflow_weight(sp)[..., None]
        mesh = FullSphereBasis(
            sigma_breaks=source[1].sigma_breaks,
            eta_breaks=source[1].eta_breaks,
            harmonics=0,
        )
        orig = WaveSpace(basis, sp.grid, sp.device, m, quadrature_basis=mesh)
        geo = TotalGeometry(orig, g, omega, self.bg)
        U, C, L = geo.fields([torch.as_tensor(c, device=sp.device) for c in coeff])
        pad = [torch.zeros_like(U[0]) for _ in range(self.J + 1 - len(U))]
        self.U = U + pad
        self.C = C + pad
        self.offset = L + pad
        self.offset = [
            self.offset[0] - cfg["torque_fraction"] * self.bg.force * self.bg.drive
        ] + [
            l + cfg["damping"] * self.bg.window * u
            for l, u in zip(self.offset[1:], self.U[1:], strict=True)
        ]
        self.vel = []
        self.curl = []
        self.mass = []
        self.lin = []
        self.scale = []
        sig = sp.rho / (sp.rho + sp.basis.radius)
        for j in range(self.J + 1):
            v, c, l = [
                sp.meridian.operator(j, m, g, omega, TAIL, f, sp.device)
                for f in ("velocity", "curl", "linear")
            ]
            self.vel.append(v)
            self.curl.append(c)
            M = v.gram(v, sp.wr, sp.wa)
            A = v.gram(l, sp.wr, sp.wa)
            n = len(M)
            extra = torch.zeros((n, n), dtype=torch.complex128, device=sp.device)
            # Implicitly advance viscosity, dilation, rotation, damping, and mean advection.
            for k in range(0, len(sig), 16):
                rows = slice(k, k + 16)
                local = WaveMeridian(sp.basis, sig[rows], sp.meridian.eta)
                V, W = [
                    dense_columns(local.operator(j, m, g, omega, TAIL, f, sp.device))
                    for f in ("velocity", "curl")
                ]
                adv = torch.linalg.cross(W, self.U[0][rows].reshape(-1, 3, 1), dim=1)
                adv += torch.linalg.cross(self.C[0][rows].reshape(-1, 3, 1), V, dim=1)
                if j:
                    adv += cfg["damping"] * self.bg.window[rows].reshape(-1, 1, 1) * V
                extra.addmm_(
                    V.reshape(-1, n).mH,
                    (sp.weight[rows].reshape(-1, 1, 1) * adv).reshape(-1, n),
                )
            s = 1 / np.sqrt(M.diagonal().real)
            self.scale.append(sp.tensor(s))
            self.mass.append(torch.as_tensor(s[:, None] * M * s, device=sp.device))
            scaled = torch.as_tensor(s[:, None] * A * s, device=sp.device)
            self.lin.append(scaled + self.scale[-1][:, None] * extra * self.scale[-1])

    def remainder(self, state):
        dU = [
            op.apply(s * x)
            for op, s, x in zip(self.vel, self.scale, state, strict=True)
        ]
        dC = [
            op.apply(s * x)
            for op, s, x in zip(self.curl, self.scale, state, strict=True)
        ]
        # Exact convolution of retained Fourier modes, including both signs.
        nphi = 4 * self.J + 1

        def synth(fixed, correction):
            return torch.fft.irfft(
                torch.stack([a + b for a, b in zip(fixed, correction, strict=True)]),
                n=nphi,
                dim=0,
                norm="forward",
            )

        adv = torch.fft.rfft(
            torch.linalg.cross(synth(self.C, dC), synth(self.U, dU)),
            dim=0,
            norm="forward",
        )
        result = []
        for j, op in enumerate(self.vel):
            explicit = (
                self.offset[j]
                + adv[j]
                - torch.linalg.cross(dC[j], self.U[0])
                - torch.linalg.cross(self.C[0], dU[j])
            )
            result.append(
                self.scale[j] * op.adjoint(self.sp.weight[..., None] * explicit)
            )
        return result

    def diagnostics(self, state):
        sp = self.sp
        updated = [
            fixed + op.apply(s * x)
            for fixed, op, s, x in zip(self.U, self.vel, self.scale, state, strict=True)
        ]
        w = sp.core_weight[..., None]
        first = torch.stack(self.U[1:])
        final = torch.stack(updated[1:])
        overlaps = (w * first.conj() * final).sum(dim=(1, 2, 3))
        e0 = (w * first.abs().square()).sum().real
        e1 = (w * final.abs().square()).sum().real
        angle = torch.angle(overlaps[0])
        orders = torch.arange(1, self.J + 1, device=sp.device)
        aligned = final * torch.exp(-1j * orders * angle).reshape(-1, 1, 1, 1)

        def change(x):
            return float(((w * (x - first).abs().square()).sum() / e0).sqrt())

        fund = (
            abs(overlaps[0])
            / (
                (w * first[0].abs().square()).sum()
                * (w * final[0].abs().square()).sum()
            ).sqrt()
        )
        iw = self.layer_weight
        layer0 = (iw * first.abs().square()).sum()
        layer_angle = torch.angle((iw * first[0].conj() * final[0]).sum())
        layer_aligned = final * torch.exp(-1j * orders * layer_angle).reshape(
            -1, 1, 1, 1
        )
        return {
            "layer_wave_gain": float(
                ((iw * final.abs().square()).sum() / layer0).sqrt()
            ),
            "layer_phase_aligned_change": float(
                ((iw * (layer_aligned - first).abs().square()).sum() / layer0).sqrt()
            ),
            "wave_gain": float((e1 / e0).sqrt()),
            "fixed_frame_relative_change": change(final),
            "phase_aligned_relative_change": change(aligned),
            "fundamental_overlap": float(fund),
            "fundamental_phase_drift": float(angle),
        }

    def run(self, steps, periods):
        duration = periods * 2 * math.pi / abs(self.cfg["omega"])
        dt = duration / steps
        state = [torch.zeros_like(s, dtype=torch.complex128) for s in self.scale]
        factors = [
            torch.linalg.lu_factor(M + 0.5 * dt * A)
            for M, A in zip(self.mass, self.lin, strict=True)
        ]
        right = [M - 0.5 * dt * A for M, A in zip(self.mass, self.lin, strict=True)]

        def advance(r, forcing):
            return [
                torch.linalg.lu_solve(F, p, (v - dt * q)[:, None])[:, 0]
                for (F, p), v, q in zip(factors, r, forcing, strict=True)
            ]

        history = [{"s": 0.0, **self.diagnostics(state)}]
        for it in range(1, steps + 1):
            rhs = [a @ x for a, x in zip(right, state, strict=True)]
            q = self.remainder(state)
            predicted = advance(rhs, q)
            q2 = self.remainder(predicted)
            state = advance(rhs, [(a + b) * 0.5 for a, b in zip(q, q2, strict=True)])
            if it == steps or it % max(1, steps // 8) == 0:
                diagnostics = self.diagnostics(state)
                if not all(map(math.isfinite, diagnostics.values())):
                    return {
                        "steps": steps,
                        "dt": dt,
                        "status": "nonfinite",
                        "stopped_at": it * dt,
                        "history": history,
                    }
                history.append({"s": it * dt, **diagnostics})
        return {"steps": steps, "dt": dt, "status": "completed", "history": history}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--base", type=Path, default=Path(__file__).with_name("axisymmetric_base.npz")
    )
    p.add_argument("--mesh", choices=MESHES, default="middle")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--grid", type=int, default=160)
    p.add_argument("--harmonics", type=int, default=3)
    p.add_argument("--steps", nargs="+", type=int, default=[96, 192])
    p.add_argument("--periods", type=float, default=1.0)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cfg, b, c = read(a.checkpoint)
    if (
        cfg["base_sha256"] != hashlib.sha256(a.base.read_bytes()).hexdigest()
        or a.harmonics <= b.harmonics
        or min(a.steps) < 1
        or a.periods <= 0
        or not np.isfinite(a.periods)
        or abs(cfg["omega"]) < 1e-8
    ):
        p.error(
            "Use the matching base, a rotating field, positive time and steps, and additional harmonics for the independent march"
        )
    source = read_base(a.base)
    sigma, eta = MESHES[a.mesh]
    basis = FullSphereBasis(
        degree=a.degree, sigma_breaks=sigma, eta_breaks=eta, harmonics=a.harmonics
    )
    mesh = FullSphereBasis(
        sigma_breaks=source[1].sigma_breaks,
        eta_breaks=source[1].eta_breaks,
        harmonics=0,
    )
    sp = WaveSpace(basis, a.grid, backend(a.device), cfg["m"], quadrature_basis=mesh)
    eq = Evolution(sp, cfg, b, c, source)
    report = {
        "checkpoint": str(a.checkpoint),
        "setup": cfg,
        "evolution_mesh": a.mesh,
        "evolution_degree": a.degree,
        "meridional_quadrature": [len(sp.wr), len(sp.wa)],
        "harmonics": a.harmonics,
        "periods": a.periods,
        "runs": [],
    }
    for steps in a.steps:
        row = eq.run(steps, a.periods)
        report["runs"].append(row)
        write_json(a.out / "evolution.json", report)
        print(
            f"m={cfg['m']} steps={steps} {row['status']} last={row['history'][-1]}",
            flush=True,
        )
    if not any(r["status"] == "completed" for r in report["runs"]):
        raise FloatingPointError("No step size reached the requested time")


if __name__ == "__main__":
    main()
