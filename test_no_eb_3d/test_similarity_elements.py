"""Independent Cartesian, integral and checkpoint checks on the Appendix C.2 elements."""

import math
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from similarity_elements import (
    PaperBasis,
    PaperSpace,
    PressureFit,
    lift_paper,
    sample_paper,
)
from solve_similarity_orbit import (
    OrbitEquation,
    backend,
    checkpoint,
    checkpoint_basis,
    load_checkpoint,
)


class PaperElementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = backend(os.environ.get("DSS_TEST_DEVICE", "cpu"))
        cls.b = PaperBasis(
            degree=3,
            sigma_breaks=(0, 0.5, 1),
            eta_breaks=(0, 0.5, 1),
            azimuthal=1,
            temporal=1,
        )
        cls.sp = PaperSpace(cls.b, 48, cls.device)
        cls.v = cls.sp.seed(3.0, 0.7, 403, 0.16)
        cls.period = 0.371

    def test_cartesian_derivatives_origin_axis_and_both_rotations(self):
        b = replace(self.b, temporal=2, azimuthal=2)
        v = np.random.default_rng(125).normal(size=len(b.labels()))

        def field(p, phase=0.23, name="velocity"):
            return sample_paper(b, v, self.period, phase, p, name)

        p = np.array([[4.0, 2.2, 1.3], [-7.0, 1.0, -2.5], [4.0, -11.0, 3.0]])
        h = 2e-5

        def derivative(name):
            return np.stack(
                [
                    (field(p + h * e, name=name) - field(p - h * e, name=name))
                    / (2 * h)
                    for e in np.eye(3)
                ],
                axis=-1,
            )

        def curl(d):
            return np.column_stack(
                (
                    d[:, 2, 1] - d[:, 1, 2],
                    d[:, 0, 2] - d[:, 2, 0],
                    d[:, 1, 0] - d[:, 0, 1],
                )
            )

        d = derivative("velocity")
        np.testing.assert_allclose(d.trace(axis1=1, axis2=2), 0, atol=2e-8)
        np.testing.assert_allclose(field(p, name="curl"), curl(d), rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(
            field(p, name="dilation"),
            np.einsum("nij,nj->ni", d, p),
            rtol=2e-7,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            field(p, name="curlcurl"), curl(derivative("curl")), rtol=2e-7, atol=2e-8
        )
        flip = np.array([1.0, -1.0, -1.0])
        np.testing.assert_allclose(field(p * flip), field(p) * flip, atol=5e-13)
        angle = np.pi / 19
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        np.testing.assert_allclose(field(p @ rot.T, 0.73), field(p) @ rot.T, atol=5e-13)
        u = field([[0.0, 0.0, 0.0], [0.0, 0.0, -4.0], [0.0, 0.0, 4.0]])
        np.testing.assert_allclose(u[0], 0)
        np.testing.assert_allclose(u[:, 0:2], 0)
        np.testing.assert_allclose(u[1], -u[2])
        # Along a ray of constant unwound phase the h=1 coefficient tends to a limit.
        r = np.array([1e7, 2e7])
        ph = 0.2 + np.log1p((r / b.phase_radius) ** 2) / self.period
        tail = field(r[:, None] * np.array([0.8, 0, 0.6]), ph) * r[:, None]
        np.testing.assert_allclose(tail[0], tail[1], rtol=4e-5, atol=2e-5)

    def test_manufactured_pressure_fit_and_absolute_H_measure(self):
        sp = self.sp
        sig, eta = sp.meridian.sig[:, None], sp.meridian.eta[None, :]
        r = self.b.radius * sig / (1 - sig)
        t = 1 - eta**2
        for n, j in [(1, 1), (1, -1), (2, 0), (0, 0)]:
            # Differentiate a specified scalar independently of the production jets.
            a = t * (1 + eta)
            p = sig * (1 + sig) * (1 - sig) ** 2
            dp = (1 + 2 * sig) * (1 - sig) ** 2 - 2 * sig * (1 + sig) * (1 - sig)
            da = 1 - 2 * eta - 3 * eta**2
            grad = np.stack(
                (
                    (
                        dp * (1 - sig) ** 2 / self.b.radius
                        - 1j
                        * n
                        * (2 * np.pi / self.period)
                        * 2
                        * r
                        * p
                        / (r * r + self.b.phase_radius**2)
                    )
                    * a,
                    -np.sqrt(t) * p * da / r,
                    1j * (19 * j) * p * a / (r * np.sqrt(t)),
                ),
                axis=-1,
            )
            q = torch.as_tensor(grad, device=sp.device)
            fit = PressureFit(sp, self.period, n, j, 1)
            self.assertLess(sp.norm_squared(q - fit.fit(q)), 1e-20 * sp.norm_squared(q))
        # Axisymmetric swirl is orthogonal to every scalar gradient. This integral
        # tests the full sphere's 2*pi and the L^3 Jacobian, with time measure 1.
        torus = torch.zeros_like(q)
        torus[..., 2] = sp.tensor(sig * (1 - sig) ** 2 * np.sqrt(t))
        mixed = torus + q
        remain = mixed - fit.fit(mixed)
        self.assertLess(sp.norm_squared(remain - torus), 1e-20 * sp.norm_squared(q))
        self.assertAlmostEqual(
            2 * np.pi * sp.norm_squared(remain),
            8 * np.pi * self.b.radius**3 / 15,
            places=9,
        )

    def test_weak_equations_and_full_pointwise_residual_against_cartesian_integrals(
        self,
    ):
        sp = self.sp
        v, p, b = self.v, self.period, self.b
        test = np.random.default_rng(600).normal(size=len(v))
        lhs = float(sp.residual(sp.tensor(v), sp.factors(p)).cpu().numpy() @ test)
        report = sp.strong_norm(sp.tensor(v), p, 1, detail=True)
        self.assertGreater(report["outside_velocity_h"], 0.01)
        self.assertLess(report["residual_h"], report["raw_h"])
        refined = sp.strong_norm(sp.tensor(v), p, 2)
        self.assertLessEqual(refined["residual_h"], report["residual_h"] * (1 + 1e-12))
        self.assertAlmostEqual(
            sum(np.sum(row["cell_h_squared"]) for row in report["harmonics"]),
            report["residual_h"] ** 2,
            delta=1e-8,
        )
        sig, eta = sp.meridian.sig, sp.meridian.eta
        r = b.radius * sig / (1 - sig)
        rr, mu, phi = np.meshgrid(
            r, eta, 2 * np.pi * np.arange(sp.wedge) / (19 * sp.wedge), indexing="ij"
        )
        pts = np.stack(
            (
                rr * np.sqrt(1 - mu**2) * np.cos(phi),
                rr * np.sqrt(1 - mu**2) * np.sin(phi),
                rr * mu,
            ),
            axis=-1,
        ).reshape(-1, 3)
        weights = (
            np.broadcast_to(
                sp.wr[:, None, None] * sp.wa[None, :, None], rr.shape
            ).ravel()
            * 2
            * np.pi
            / (sp.nt * sp.wedge)
        )
        integral = square = 0.0
        for it in range(sp.nt):
            phase = it / sp.nt + np.log1p((rr.ravel() / b.phase_radius) ** 2) / p

            def sample(c, name="velocity", tail=1, t=phase):
                return sample_paper(b, c, p, t, pts, name, tail)

            u, w = sample(v), sample(v, "curl")
            dt = (sample(v, t=phase + 1e-5) - sample(v, t=phase - 1e-5)) / (2e-5 * p)
            force = dt + (u + sample(v, "dilation")) / 2 + np.cross(w, u)
            integral += weights @ np.sum(
                sample(test, tail=3) * force + sample(test, "curl", 3) * w, axis=1
            )
            square += weights @ np.sum((force + sample(v, "curlcurl")) ** 2, axis=1)
        np.testing.assert_allclose(lhs, integral, rtol=2e-7, atol=1e-7)
        np.testing.assert_allclose(report["raw_h"], math.sqrt(square), rtol=2e-7)

    def test_local_degrees_cells_joins_exact_lift_and_checkpoint(self):
        b = replace(
            self.b,
            cell_degrees=(
                (1, 1, 1, 1, -1, -1),
                (0, 0, 0, 0, -1, -1),
                (0, 0, 1, 0, 2, 2),
            ),
        )
        sp = PaperSpace(b, 48, self.device)
        v = sp.seed(3.0, 0.7, 9, 0.05)
        wave = np.array([m == 19 for _, _, m, *_ in sp.labels])
        self.assertLess(len(v), len(self.v))
        pts = np.array([[3.0, 2.0, 15.0], [3.0, 2.0, -15.0]])
        np.testing.assert_allclose(
            sample_paper(b, v * wave, self.period, 0.2, pts), 0, atol=1e-12
        )
        np.testing.assert_allclose(
            sample_paper(b, v * ~wave, self.period, 0.2, [[4.0, 2.0, 0.5]]),
            0,
            atol=1e-12,
        )
        axis = b.modes[0]
        band = b.expand(axis, next(b.coefficients(v)))
        pos = 0
        for name, (r, a) in b.families[axis].items():
            if name == "gamma":
                continue
            size = r.count * a.count
            c = band[pos : pos + size].reshape(r.count, a.count)
            # In the restricted ring the original cubic spaces reduce to biquadratics.
            np.testing.assert_allclose(
                r.spline([0.55, 0.8], nu=3) @ c @ a.spline([0.1, 0.4]).T, 0, atol=1e-9
            )
            np.testing.assert_allclose(
                r.spline([0.55, 0.8]) @ c @ a.spline([0.1, 0.4], nu=3).T, 0, atol=1e-9
            )
            pos += size
        for group in b.families.values():
            for name, (r, a) in group.items():
                if name == "gamma":
                    continue
                for sp1d, join, knot in (
                    (r, 2 if name in ("S", "A") else 1, 0.5),
                    (a, 2 if name in ("B", "S") else 1, 0.0),
                ):
                    for derivative in range(join + 1):
                        np.testing.assert_allclose(
                            sp1d.spline(knot - 1e-11, nu=derivative),
                            sp1d.spline(knot + 1e-11, nu=derivative),
                            atol=5e-8,
                        )
        finer = replace(
            self.b,
            degree=4,
            sigma_breaks=(0, 0.25, 0.5, 1),
            eta_breaks=(0, 0.3, 0.5, 1),
            temporal=2,
            azimuthal=2,
        )
        new = lift_paper(v, b, finer)
        pts = np.random.default_rng(13).normal(0, 12, size=(25, 3))
        np.testing.assert_allclose(
            sample_paper(finer, new, self.period, 0.17, pts),
            sample_paper(b, v, self.period, 0.17, pts),
            atol=2e-12,
        )
        with tempfile.TemporaryDirectory() as path:
            cfg = {
                "basis": asdict(b),
                "period": self.period,
                "grid": 128,
                "discretization": "paper",
            }
            target = Path(path) / "orbit.npz"
            checkpoint(
                target,
                cfg,
                v,
                np.r_[v / np.linalg.norm(v), 0.0, 1.0],
                np.empty(0),
                0.1,
                0,
            )
            loaded, seed, *_ = load_checkpoint(target)
            self.assertEqual(checkpoint_basis(loaded), b)
            np.testing.assert_equal(seed, v)

    def test_zero_field_linearization_on_default_elements(self):
        # At the reference period, the unforced velocity Jacobian at zero is
        # L_ref^{-1} L_ref. An explicit inverse lost percent-level accuracy here.
        sp = PaperSpace(PaperBasis(), 128, self.device)
        seed = sp.seed(20, 0.7, 19002, 0.05)
        eq = OrbitEquation(sp, seed, 0.36)
        zero = np.zeros(len(seed) + 2)
        jac = eq.jacobian(zero)
        dx = np.random.default_rng(81).normal(size=len(seed))
        recovered = jac.field(sp.tensor(dx)).detach().cpu().numpy()
        self.assertLess(np.linalg.norm(recovered - dx) / np.linalg.norm(dx), 1e-6)
        self.assertLess(np.linalg.norm(jac.period_column), 1e-14)

    def test_jacobian_and_transpose(self):
        b = replace(self.b, cell_degrees=((1, 1, 1, 1, -1, -1),))
        sp = PaperSpace(b, 40, self.device)
        v = sp.seed(5.0, 0.7, 702, 0.1)
        eq = OrbitEquation(sp, v, self.period)
        rng = np.random.default_rng(50)
        y = np.r_[eq.ref, 0.01, 0.8]
        d = rng.normal(size=len(y))
        dual = rng.normal(size=len(y) - 1)
        jac = eq.jacobian(y)
        h = 2e-6
        np.testing.assert_allclose(
            jac(d),
            (eq.residual(y + h * d) - eq.residual(y - h * d)) / (2 * h),
            rtol=1e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(jac.transpose(dual) @ d, dual @ jac(d), rtol=2e-9)
        self.assertEqual(sp.strong_norm(sp.tensor(0 * v), self.period)["residual_h"], 0)


if __name__ == "__main__":
    unittest.main()
