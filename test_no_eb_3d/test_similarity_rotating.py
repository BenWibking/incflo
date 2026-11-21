"""Cartesian, far-field and nonlinear-corrector checks for the reduced half-space problem."""

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from similarity_elements import PressureFit
from similarity_rotating import (
    RotatingWave,
    WaveBasis,
    WaveGeometry,
    WaveMeridian,
    WaveSpace,
    figure_seed,
    lift_coefficients,
)
from solve_rotating_wave import (
    audit_candidate,
    correct,
    curve_tangent,
    polish,
    read_state,
    save_state,
    trace,
)
from solve_similarity_orbit import backend


class HalfSpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.basis = WaveBasis(
            sigma_breaks=(0, 0.62, 1), eta_breaks=(0, 0.31, 1), harmonics=2
        )
        cls.k, cls.g, cls.omega = 18.4, 0.512, 15.2
        cls.space = WaveSpace(
            cls.basis, 40, backend(os.environ.get("DSS_TEST_DEVICE", "cpu")), cls.k
        )
        cls.geom = WaveGeometry(cls.space, cls.k, cls.g, cls.omega)
        cls.c = figure_seed(
            cls.space, cls.geom, Path(__file__).with_name("rotating_figure_seed.npz")
        )
        cls.eq = RotatingWave(cls.space, cls.g, cls.omega, cls.c)

    def values(self, points, s, field):
        r = np.linalg.norm(points, axis=1)
        unit = points / r[:, None]
        eta = unit[:, 2]
        phi = np.arctan2(points[:, 1], points[:, 0])
        sn = np.hypot(unit[:, 0], unit[:, 1])
        mer = WaveMeridian(self.basis, r / (self.basis.radius + r), eta)
        wind = (
            self.k * phi
            - self.omega * s
            + self.geom.twist * np.log1p((r / self.basis.phase_radius) ** 2)
        )
        sphere = np.zeros_like(points)
        for j, c in enumerate(self.c):
            op = mer.operator(
                j, self.k, self.g, self.omega, self.geom.tail, field, "cpu"
            )
            # Independent paired-point evaluation of the separated basis functions.
            cols = np.zeros_like(points, dtype=complex)
            i = 0
            for terms, shape in zip(op.parts, op.shapes, strict=True):
                k = np.prod(shape)
                coeff = c[i : i + k].reshape(shape)
                i += k
                for component, parts in enumerate(terms):
                    for R, A in parts:
                        cols[:, component] += ((R @ coeff) * A).sum(axis=1)
            sphere += (2 if j else 1) * (cols * np.exp(1j * j * wind)[:, None]).real
        polar = np.column_stack([eta * np.cos(phi), eta * np.sin(phi), -sn])
        az = np.column_stack([-np.sin(phi), np.cos(phi), 0 * phi])
        return (
            sphere[:, 0, None] * unit
            + sphere[:, 1, None] * polar
            + sphere[:, 2, None] * az
        )

    def test_cartesian_pde_and_slip(self):
        p = np.array([[19.0, 3.0, 12.0], [40.0, -7.0, 6.0], [7.0, 1.0, 18.0]])
        h = 1.5e-3
        s = 0.031
        u = self.values(p, s, "velocity")
        deriv = []
        lap = 0
        for e in np.eye(3) * h:
            plus, minus = (self.values(p + sign * e, s, "velocity") for sign in (1, -1))
            deriv.append((plus - minus) / (2 * h))
            lap += (plus + minus - 2 * u) / h**2
        D = np.stack(deriv, axis=2)
        np.testing.assert_allclose(np.trace(D, axis1=1, axis2=2), 0, atol=2e-4)
        curl = np.stack(
            [D[:, 2, 1] - D[:, 1, 2], D[:, 0, 2] - D[:, 2, 0], D[:, 1, 0] - D[:, 0, 1]],
            axis=1,
        )
        np.testing.assert_allclose(
            curl, self.values(p, s, "curl"), atol=3e-4, rtol=1e-4
        )
        ht = 2e-6
        ds = (
            self.values(p, s + ht, "velocity") - self.values(p, s - ht, "velocity")
        ) / (2 * ht)
        lin = ds + (1 - self.g) * u + self.g * np.einsum("nij,nj->ni", D, p) - lap
        np.testing.assert_allclose(
            lin, self.values(p, s, "linear"), atol=0.002, rtol=2e-4
        )
        p[:, 2] = 0
        np.testing.assert_allclose(self.values(p, s, "velocity")[:, 2], 0, atol=5e-9)
        # Mean azimuthal speed vanishes on the plane. The traveling wave permits tangential slip.
        phases = np.arange(7) * 2 * np.pi / (7 * self.omega)
        mean = np.mean([self.values(p, phase, "velocity") for phase in phases], axis=0)
        np.testing.assert_allclose(
            -mean[:, 0] * p[:, 1] + mean[:, 1] * p[:, 0], 0, atol=3e-8
        )

    def test_far_field_and_pressure(self):
        direction = np.array([[0.83, 0.2, 0.52]])
        direction /= np.linalg.norm(direction)

        # Remove the radial phase in the sampled angular coefficient at both radii.
        def ratio(r):
            s = (
                self.geom.twist
                / self.omega
                * np.log1p((r / self.basis.phase_radius) ** 2)
            )
            return np.linalg.norm(
                self.values(r * direction, s, "linear")
            ) / np.linalg.norm(self.values(r * direction, s, "velocity"))

        self.assertLess(ratio(5e5), 0.25 * ratio(5e4))
        sp = self.space
        rho = sp.rho[:, None]
        sig = sp.meridian.sig[:, None]
        eta = sp.meridian.eta[None, :]
        t = 1 - eta**2
        p = sig * (1 - sig) ** 2
        slope = 2 * rho**2 / (rho**2 + sp.basis.phase_radius**2)
        dr = sig * (1 - sig) ** 2 * (1 - 3 * sig) + 1j * self.geom.twist * slope * p
        grad = np.stack(
            [
                dr * t / rho,
                2 * p * eta * np.sqrt(t) / rho,
                1j * self.k * p * np.sqrt(t) / rho,
            ],
            axis=-1,
        )
        op = PressureFit(
            sp,
            1.0,
            0,
            1,
            1,
            angular_order=self.k,
            radial_frequency=-1j * self.geom.twist,
        )
        v = torch.as_tensor(grad, device=sp.device)
        np.testing.assert_allclose(op.fit(v).cpu().numpy(), grad, atol=1e-9)

    def test_augmented_jacobian_and_forcing(self):
        eq = self.eq
        x = eq.start
        np.testing.assert_allclose(eq.evaluate(x, 1.0), 0, atol=2e-11)
        np.testing.assert_allclose(eq.evaluate(x, 0.0)[:-2], eq.load)
        zero = x.copy()
        zero[:-2] = 0.0
        self.assertAlmostEqual(eq.evaluate(zero, 0.0)[-2], -0.5)
        J = eq.jacobian(x, 0.7)
        rng = np.random.default_rng(824)
        for scalar_only in (True, False):
            direction = rng.normal(size=len(x))
            if scalar_only:
                direction[:-2] = 0
            direction /= np.linalg.norm(direction)
            h = 1e-5
            diff = (
                eq.evaluate(x + h * direction, 0.7)
                - eq.evaluate(x - h * direction, 0.7)
            ) / (2 * h)
            np.testing.assert_allclose(J @ direction, diff, rtol=8e-5, atol=1e-6)
        normal = curve_tangent(eq, np.r_[x, 1.0], np.r_[np.zeros_like(x), 1.0])
        np.testing.assert_allclose(
            J @ normal[:-1], normal[-1] * np.r_[eq.load, 0, 0], atol=1e-9
        )
        target = np.r_[x, 1.0] - 0.005 * normal
        ans, ok, info = correct(eq, target, normal, 1e-7, 14)
        self.assertTrue(ok, info)
        self.assertLess(ans[-1], 1.0)
        self.assertLess(np.linalg.norm(eq.evaluate(ans[:-1], ans[-1])), 1e-7)
        self.assertGreater(abs(eq.parameters(ans[:-1])[0] - self.g), 1e-9)

    def test_retraction_and_nested_transfer(self):
        eq = self.eq
        old, g, omega = self.basis, self.g, self.omega
        rng = np.random.default_rng(184)
        cs = [c.copy() for c in self.c]
        cs[2] = 0.2 * (rng.normal(size=cs[2].shape) + 1j * rng.normal(size=cs[2].shape))
        rotated = [
            c * np.exp(0.61j * j) * (1.3 if j == 1 else 1) for j, c in enumerate(cs)
        ]
        restored = eq.retract(eq.coordinates(rotated, g, omega))
        np.testing.assert_allclose(eq.evaluate(restored, 0.0)[-2:], 0, atol=1e-11)
        for expected, actual in zip(cs, eq.coefficients(restored), strict=True):
            np.testing.assert_allclose(expected, actual.cpu().numpy(), atol=1e-8)
        rich = replace(old, degree=4, harmonics=3, sigma_breaks=(0, 0.27, 0.62, 1))
        lifted = lift_coefficients(old, rich, cs)
        sig, eta = np.array([0.13, 0.32, 0.93]), np.array([0.23, 0.54, 0.88])

        def field(basis, coeff, j):
            mer = WaveMeridian(basis, sig, eta)
            op = mer.operator(j, self.k, g, omega, self.geom.tail, "velocity", "cpu")
            return op.apply(torch.as_tensor(coeff[j])).numpy()

        for j in range(old.harmonics + 1):
            np.testing.assert_allclose(
                field(rich, lifted, j), field(old, cs, j), atol=3e-10
            )
        np.testing.assert_array_equal(lifted[-1], 0)

    def test_forced_checkpoint_cannot_start_unforced_family(self):
        eq = self.eq
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "wave.npz"
            point = np.r_[eq.start, 1.0]
            save_state(checkpoint, eq, point, {"stage": 0})
            cfg, b, coeff = read_state(checkpoint)
            np.testing.assert_allclose(
                eq.coordinates(coeff, cfg["g"], cfg["omega"]), eq.start, atol=2e-9
            )
            self.assertEqual(b, self.basis)
            self.assertEqual(cfg["forcing"], 1)
            with self.assertRaisesRegex(ValueError, "unforced checkpoint"):
                trace(SimpleNamespace(checkpoint=checkpoint))
            limits = SimpleNamespace(
                audit_grids=[48], tol=1e-7, strong_tol=0.01, h_tol=0.01
            )
            measured = audit_candidate(eq, point, limits)
            self.assertFalse(measured["accepted"])
            self.assertGreater(measured["measurements"][-1]["residual_h"], 0)

    def test_best_checkpoint_uses_physical_residual(self):
        eq = self.eq
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.npz"
            save_state(source, eq, np.r_[eq.start, 1.0], {"stage": 0})
            out = root / "run"
            args = SimpleNamespace(
                checkpoint=source,
                out=out,
                degree=None,
                harmonics=None,
                sigma_breaks=None,
                eta_breaks=None,
                kappa=None,
                amplitude=None,
                grid=40,
                device=os.environ.get("DSS_TEST_DEVICE", "cpu"),
                trust=0.05,
                iterations=3,
                tol=1e-7,
                strong_tol=0.01,
                h_tol=0.01,
                audit_grids=[48],
            )
            polish(args)
            data = json.loads((out / "polish.json").read_text())
            best = min(data["iterations"], key=lambda row: row["residual_h"])
            cfg, basis, coeff = read_state(out / "best-h.npz")
            self.assertEqual(cfg["iteration"], best["iteration"])
            self.assertEqual(cfg["forcing"], 0)
            self.assertEqual(data["best_h_audit"]["accepted"], cfg["audited"])
            sample = WaveSpace(basis, 40, self.space.device, cfg["kappa"]).audit(
                coeff, cfg["g"], cfg["omega"], 2
            )
            self.assertEqual(sample["quadrature"], [40, 40, 40])
            np.testing.assert_allclose(
                sample["residual_h"], best["residual_h"], rtol=1e-10
            )
            np.testing.assert_allclose(sample["wave_rms"], eq.amplitude, rtol=1e-11)
