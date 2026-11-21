"""Independent checks for a reflection-symmetric steady mean under a fixed torque."""

import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import roots_legendre
from similarity_rotating import WaveBasis, lift_coefficients
from solve_axisymmetric import (
    Equilibrator,
    MeanGrid,
    apply_defaults,
    read,
    torque_shape,
)
from solve_similarity_orbit import backend


class AxisymmetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.basis = WaveBasis(
            harmonics=0, sigma_breaks=(0, 0.56, 1), eta_breaks=(0, 0.33, 1)
        )
        cls.device = backend(os.environ.get("DSS_TEST_DEVICE", "cpu"))
        cls.profile = Path(__file__).with_name("rotating_figure_seed.npz")
        cls.m = MeanGrid(cls.basis, 48, cls.device, 0.502, 19, 21, cls.profile, 1)
        cls.eq = Equilibrator(cls.m)

    def test_force_and_reflection(self):
        m = self.m
        self.assertAlmostEqual(torque_shape(19, 21, 19, 21), 1)
        self.assertAlmostEqual(torque_shape(19, -21, 19, 21), -1)
        self.assertLess(abs(m.drive_norm / m.drive_exact - 1), 1e-5)
        x = self.eq.fit_figure()
        c = self.eq.coefficients(x)
        north = m.probe(c, [7.0, 32.0], [15.0, 6.0])
        south = m.probe(c, [7.0, 32.0], [-15.0, -6.0])
        np.testing.assert_allclose(south, north * [1, -1, -1], atol=5e-10)
        zero = m.audit(np.zeros(len(x)), self.eq.reference_force)
        self.assertAlmostEqual(zero["residual_over_force"], 1)
        self.assertEqual(zero["unforced_h"], 0)

    def test_packaged_equilibrium_resumes_at_its_own_controls(self):
        state = read(Path(__file__).with_name("axisymmetric_base.npz"))
        cfg, basis, coefficients = state
        self.assertEqual(len(coefficients), 2804)
        self.assertLess(
            cfg["verification"]["measurements"][-1]["residual_over_force"], 0.03
        )
        opts = SimpleNamespace(
            **dict.fromkeys(
                (
                    "degree",
                    "radius",
                    "eta_breaks",
                    "sigma_breaks",
                    "drive_z",
                    "drive_r",
                    "g",
                    "grid",
                    "fractions",
                    "audit_grids",
                )
            )
        )
        opts.grid = 192
        apply_defaults(opts, state)
        self.assertEqual(opts.degree, basis.degree)
        self.assertEqual(tuple(opts.eta_breaks), basis.eta_breaks)
        self.assertEqual(tuple(opts.sigma_breaks), basis.sigma_breaks)
        self.assertEqual(opts.drive_z, cfg["drive_z"])
        self.assertEqual(opts.fractions, [cfg["fraction"]])
        self.assertEqual(opts.audit_grids, [240, 288])

    def test_strong_derivative_and_fixed_force_descent(self):
        eq, m = self.eq, self.m
        force = 0.35 * eq.reference_force
        x = 0.35 * eq.fit_figure()
        residual, jac = eq.residual(x, force, True)
        d = m.space.tensor(np.random.default_rng(741).normal(size=len(x)))
        d /= torch.linalg.vector_norm(d)
        # Nonlinearity is exactly quadratic, so a large symmetric increment is admissible.
        dx = 0.002 * max(1, float(torch.linalg.vector_norm(x)) / len(x) ** 0.5)
        fd = (eq.residual(x + dx * d, force) - eq.residual(x - dx * d, force)) / (
            2 * dx
        )
        self.assertLess(
            float(
                torch.linalg.vector_norm(jac @ d - fd) / torch.linalg.vector_norm(fd)
            ),
            2e-7,
        )
        native = m.audit(eq.coefficients(x), force)["residual_h"]
        self.assertAlmostEqual(
            native / float(torch.linalg.vector_norm(residual)), 1, places=9
        )
        y, hist, _ = eq.solve(x, force, 4)
        self.assertGreater(len(hist), 1)
        self.assertTrue(np.all(np.diff([s["residual_h"] for s in hist]) < 0))
        checked = m.audit(eq.coefficients(y), force)
        self.assertAlmostEqual(
            checked["residual_h"] / hist[-1]["residual_h"], 1, places=8
        )
        self.assertAlmostEqual(
            checked["force_h"] / native, abs(force) * m.drive_norm / native, places=10
        )

    def test_mean_transfer_preserves_physical_field(self):
        old = self.basis
        rich = replace(old, degree=4, sigma_breaks=(0, 0.27, 0.56, 1))
        c = self.eq.coefficients(self.eq.fit_figure())
        lifted = lift_coefficients(old, rich, [c])[0]
        refined = MeanGrid(rich, 48, self.device, 0.502, 19, 21, self.profile, 1)
        r, z = [0, 24, 42], [15, 0, 7]
        np.testing.assert_allclose(
            refined.probe(lifted, r, z), self.m.probe(c, r, z), atol=1e-8
        )

    def test_linear_swirl_against_exact_gaussian_response(self):
        # In (alpha - Laplacian + g*y.grad)^-1, write q=exp(-g*t).
        # Heat evolution widens the two Gaussian widths by (1-q^2)/g.
        g, rd, zd = 0.502, 19.0, 21.0
        r, z = np.array([9.0, 19.0, 27.0, 34.0]), np.array([8.0, 21.0, 16.0, 32.0])
        nodes, weights = roots_legendre(100)
        q = (1 + nodes) / 2
        spread = (1 - q * q) / g
        rr, zz = rd**2 + spread, zd**2 + spread
        weight = (
            weights
            / 2
            * q ** ((1 - g) / g + 1)
            * (rd**2 / rr) ** 2
            * (zd**2 / zz) ** 1.5
        )
        exact = (
            np.e
            * r
            * z
            / (g * rd * zd)
            * (
                np.exp(-0.5 * q**2 * (r[:, None] ** 2 / rr + z[:, None] ** 2 / zz))
                @ weight
            )
        )
        m = MeanGrid(
            replace(self.basis, degree=5), 48, self.device, g, rd, zd, self.profile, 1
        )
        eq = Equilibrator(m)
        swirl = m.probe(eq.coefficients(eq.stokes), r, z)[:, 1]
        self.assertLess(np.linalg.norm(swirl - exact) / np.linalg.norm(exact), 0.02)
