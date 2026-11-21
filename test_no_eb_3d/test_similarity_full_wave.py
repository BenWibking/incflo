"""Full-space torque, wave Jacobian, transfer and out-of-band momentum checks."""

import math
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from evolve_full_wave import Evolution
from polish_full_wave import PhysicalFit
from probe_rotating_modes import FullSphereBasis
from similarity_rotating import WaveBasis, WaveGeometry, WaveSpace, lift_coefficients
from solve_axisymmetric import read
from solve_full_rotating_wave import Background, FixedTorqueWave, audit
from solve_similarity_orbit import backend


class FullWaveTest(unittest.TestCase):
    def test_inflow_amplitude_agrees_with_independent_measurement(self):
        b = WaveBasis(harmonics=0, sigma_breaks=(0, 1), eta_breaks=(0, 1))
        n = sum(a.count * b.count if a else 1 for a, b in b.groups[0].values())
        source = (
            {"g": 0.502, "force": 3.5, "nu": 1.0, "drive_r": 19.0, "drive_z": 21.0},
            b,
            self.rng.normal(size=n),
        )
        cfg = {
            "g": 0.502,
            "omega": 8.0,
            "m": 14,
            "damping": -0.4,
            "torque_fraction": 1.0,
            "anchor": "inflow",
        }
        fit = PhysicalFit(source, cfg, self.b, self.c, 64, 4, self.device)
        cs, _, _ = fit.snapshot(np.zeros(fit.size), 0.10 * fit.anchor_base_norm)
        metrics = audit(source, cfg, self.b, cs, 96, 5, self.device)
        self.assertAlmostEqual(metrics["inflow_fundamental_over_base"], 0.10, places=5)
        self.assertLess(metrics["inflow_region_energy_fraction"], 1.0)

    def test_physical_objective_and_frequency_gradient(self):
        base = WaveBasis(harmonics=0, sigma_breaks=(0, 1), eta_breaks=(0, 1))
        n = sum(a.count * b.count if a else 1 for a, b in base.groups[0].values())
        source = (
            {"g": 0.502, "force": 3.5, "nu": 1.0, "drive_r": 19.0, "drive_z": 21.0},
            base,
            np.zeros(n),
        )
        cfg = {
            "g": 0.502,
            "omega": 8.0,
            "m": 14,
            "damping": -0.4,
            "torque_fraction": 1.0,
        }
        fit = PhysicalFit(source, cfg, self.b, self.c, 48, 4, self.device)
        x = np.zeros(fit.size)
        f, grad = fit.value(x, 0.7)
        cs, omega, damping = fit.snapshot(x, 0.7)
        row = audit(
            source,
            {**cfg, "omega": omega, "damping": damping},
            self.b,
            cs,
            48,
            4,
            self.device,
        )
        self.assertAlmostEqual(
            math.sqrt(2 * f), row["residual_h"] / row["force_h"], places=8
        )
        x[-2] = 0.13
        _, grad = fit.value(x, 0.7)
        direction = self.rng.normal(size=len(x))
        direction /= np.linalg.norm(direction)
        direction[-2:] = [0.4, -0.3]
        h = 1e-6
        diff = (
            fit.value(x + h * direction, 0.7)[0] - fit.value(x - h * direction, 0.7)[0]
        ) / (2 * h)
        np.testing.assert_allclose(grad @ direction, diff, rtol=2e-5, atol=2e-6)
        cs, omega, _ = fit.snapshot(x, 0.7)
        actual = WaveGeometry(fit.sp, 14, 0.502, omega).fields(
            [torch.as_tensor(c, device=self.device) for c in cs]
        )
        approx = fit.fields(fit.sp.tensor(x), 0.7)[1]
        for aa, bb in zip(actual, approx, strict=True):
            for a, b in zip(aa[1:], bb[1:], strict=True):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=5e-12)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = backend(os.environ.get("DSS_TEST_DEVICE", "cpu"))
        cls.b = FullSphereBasis(sigma_breaks=(0, 1), eta_breaks=(0, 1), harmonics=2)
        cls.sp = WaveSpace(cls.b, 32, cls.device, 14)
        cls.rng = np.random.default_rng(164)
        cls.c = [
            cls.rng.normal(
                size=sum(a.count * b.count if a else 1 for a, b in group.values())
            ).astype(complex)
            for group in cls.b.groups
        ]
        for x in cls.c[1:]:
            x.imag = cls.rng.normal(size=len(x))

    def test_affine_mean_and_jacobian(self):
        bg = Background(
            self.sp, read(Path(__file__).with_name("axisymmetric_base.npz"))
        )
        eq = FixedTorqueWave(self.sp, bg, 8.0, -0.7, self.c, 1.5)
        x = eq.retract(eq.start)
        u = eq.geometry(x).fields(eq.coefficients(x))[0]
        raw = [torch.zeros_like(a) for a in u]
        applied = eq.applied_terms(x, u, raw)
        torch.testing.assert_close(applied[0], -bg.force * bg.drive)
        torch.testing.assert_close(applied[2], -0.7 * bg.window * u[2])
        v = self.rng.normal(size=len(x))
        v /= np.linalg.norm(v)
        J = eq.jacobian(x, 0) @ v
        h = 1e-6
        diff = (eq.evaluate(x + h * v, 0) - eq.evaluate(x - h * v, 0)) / (2 * h)
        np.testing.assert_allclose(J, diff, rtol=3e-5, atol=4e-6)
        self.assertAlmostEqual(self.sp.wave_rms(u[1]), 1.5, places=9)

    def test_unrestricted_mean_and_wave_transfer_in_south(self):
        rich = FullSphereBasis(
            sigma_breaks=(0, 0.45, 1), eta_breaks=(0, 0.30, 1), harmonics=3
        )
        lifted = lift_coefficients(self.b, rich, self.c)
        a = WaveGeometry(
            WaveSpace(self.b, 48, self.device, 14, quadrature_basis=rich),
            14,
            0.502,
            8.0,
        )
        b = WaveGeometry(WaveSpace(rich, 48, self.device, 14), 14, 0.502, 8.0)
        for j, c in enumerate(self.c):
            va = a.velocity[j].apply(torch.as_tensor(c, device=self.device))
            vb = b.velocity[j].apply(torch.as_tensor(lifted[j], device=self.device))
            torch.testing.assert_close(va, vb, atol=2e-12, rtol=1e-10)
        self.assertEqual(np.linalg.norm(lifted[-1]), 0.0)

    def test_force_norm_and_quadratic_modes(self):
        mean = WaveBasis(harmonics=0, sigma_breaks=(0, 0.5, 1), eta_breaks=(0, 0.4, 1))
        size = sum(a.count * b.count if a else 1 for a, b in mean.groups[0].values())
        base = (
            {"g": 0.502, "force": 3.5, "nu": 1.0, "drive_r": 19.0, "drive_z": 21.0},
            mean,
            np.zeros(size),
        )
        cfg = {"m": 14, "omega": 8.0, "damping": 0.0, "torque_fraction": 1.0}
        c = [np.zeros_like(x) for x in self.c]
        blank = audit(base, cfg, self.b, c, 80, 4, self.device)
        exact = 3.5 * math.sqrt(math.pi**1.5 * math.exp(2) * 19**2 * 21 / 2)
        self.assertAlmostEqual(blank["residual_h"] / exact, 1.0, places=9)
        self.assertEqual(blank["wave_residual_h"], 0.0)
        c[2] = self.c[2]
        active = audit(base, cfg, self.b, c, 80, 4, self.device)
        self.assertGreater(active["modes"][4]["residual"], 0.0)
        self.assertAlmostEqual(
            active["outside_h"], active["modes"][4]["residual"], places=10
        )

    def test_time_march_split_and_step_doubling(self):
        mean = WaveBasis(harmonics=0, sigma_breaks=(0, 1), eta_breaks=(0, 1))
        n = sum(a.count * b.count if a else 1 for a, b in mean.groups[0].values())
        source = (
            {"g": 0.502, "force": 0.02, "nu": 1.0, "drive_r": 19.0, "drive_z": 21.0},
            mean,
            np.zeros(n),
        )
        b = FullSphereBasis(sigma_breaks=(0, 1), eta_breaks=(0, 1), harmonics=3)
        sp = WaveSpace(b, 28, self.device, 14)
        cfg = {
            "g": 0.502,
            "omega": 8.0,
            "m": 14,
            "damping": 0.4,
            "torque_fraction": 1.0,
        }
        eq = Evolution(sp, cfg, self.b, self.c, source)
        x = [
            torch.as_tensor(
                0.01 * self.rng.normal(size=len(s)),
                device=sp.device,
                dtype=torch.complex128,
            )
            for s in eq.scale
        ]
        remainder = eq.remainder(x)
        c = [s * y for s, y in zip(eq.scale, x, strict=True)]
        du = [o.apply(v) for o, v in zip(eq.vel, c, strict=True)]
        U = [a + d for a, d in zip(eq.U, du, strict=True)]
        C = [a + o.apply(v) for a, o, v in zip(eq.C, eq.curl, c, strict=True)]

        def physical(a):
            return torch.fft.irfft(torch.stack(a), n=40, dim=0, norm="forward")

        N = torch.fft.rfft(
            torch.linalg.cross(physical(C), physical(U)), dim=0, norm="forward"
        )
        for j, v in enumerate(c):
            lin = sp.meridian.operator(j, 14, 0.502, 8.0, 3.0, "linear", self.device)
            raw = eq.offset[j] + lin.apply(v) + N[j]
            if j:
                raw += 0.4 * eq.bg.window * du[j]
            projected = eq.scale[j] * eq.vel[j].adjoint(sp.weight[..., None] * raw)
            torch.testing.assert_close(
                eq.lin[j] @ x[j] + remainder[j], projected, rtol=2e-10, atol=1e-9
            )
        ends = [
            eq.run(steps, 0.08)["history"][-1]["fixed_frame_relative_change"]
            for steps in (4, 8, 16)
        ]
        self.assertLess(abs(ends[2] - ends[1]), 0.4 * abs(ends[1] - ends[0]))


if __name__ == "__main__":
    unittest.main()
