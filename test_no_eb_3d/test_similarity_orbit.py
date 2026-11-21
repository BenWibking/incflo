"""Independent Cartesian and integral checks of the similarity-coordinate solver."""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.special import roots_legendre
from similarity_galerkin import Basis, OrbitSpace, sample_cartesian, transfer
from similarity_petsc import BorderedStokes, petsc_correct
from solve_similarity_orbit import (
    CorrectorSystem,
    OrbitEquation,
    PseudoTimePath,
    backend,
    checkpoint,
    correct,
    export_replay,
    load_checkpoint,
    parser,
    run_search,
)


class SimilarityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.basis = Basis(radial=2, degrees=2, azimuthal=1, temporal=1)
        cls.device = backend(os.environ.get("DSS_TEST_DEVICE", "cpu"))
        cls.space = OrbitSpace(cls.basis, 40, cls.device)
        cls.seed = cls.space.seed(20, 0.7, 23, 0.12)
        cls.period = 0.371
        cls.eq = OrbitEquation(cls.space, cls.seed, cls.period)

    def field(self, points, phase=0.2, name="velocity", **kwargs):
        return sample_cartesian(
            self.basis, self.seed, self.period, phase, points, name, **kwargs
        )

    def test_cartesian_derivatives_and_symmetries(self):
        p = np.array([[12.0, 3.0, 4.0], [-28.0, -14.0, 5.0], [7.0, -21.0, -11.0]])
        eps = 1e-4
        jac = np.stack(
            [
                (self.field(p + eps * axis) - self.field(p - eps * axis)) / (2 * eps)
                for axis in np.eye(3)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(jac.trace(axis1=1, axis2=2), 0, atol=1e-8)
        curl = np.stack(
            [
                jac[:, 2, 1] - jac[:, 1, 2],
                jac[:, 0, 2] - jac[:, 2, 0],
                jac[:, 1, 0] - jac[:, 0, 1],
            ],
            axis=1,
        )
        np.testing.assert_allclose(
            curl, self.field(p, name="curl"), rtol=5e-8, atol=5e-9
        )
        np.testing.assert_allclose(
            np.einsum("nij,nj->ni", jac, p),
            self.field(p, name="dilation"),
            rtol=5e-8,
            atol=5e-8,
        )
        for angle, shift in [(2 * np.pi / 19, 0), (np.pi / 19, 0.5)]:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            np.testing.assert_allclose(
                self.field(p @ rot.T, 0.2 + shift), self.field(p) @ rot.T, atol=2e-13
            )
        flip = np.array([1, -1, -1])
        np.testing.assert_allclose(
            self.field(flip * p), flip * self.field(p), atol=2e-13
        )
        np.testing.assert_allclose(self.field(p, 1.2), self.field(p), atol=2e-13)
        np.testing.assert_allclose(self.field([[0, 0, 0]]), 0)
        self.assertTrue(np.isfinite(self.field([[0, 0, -8], [0, 0, 8]])).all())

    def test_tail(self):
        direction = np.array([0.8, 0, 0.6])
        r = np.array([1e5, 2e5])
        phase = np.log1p((r / self.basis.phase_radius) ** 2) / self.period + 0.16
        far = self.field(r[:, None] * direction, phase) * r[:, None]
        self.assertGreater(np.linalg.norm(far), 0.1)
        np.testing.assert_allclose(far[0], far[1], atol=4e-4, rtol=2e-5)

    def test_weak_navier_stokes_in_cartesian_components(self):
        # Independent assembly of <V, W_s + (W+y.grad W)/2 + curl W x W> + <curl V, curl W>.
        # The production operator uses analytic linear blocks and spherical nonlinear projection.
        sp = self.space
        t = np.random.default_rng(7).normal(size=len(self.seed))
        lhs = (
            sp.residual(sp.tensor(self.seed), sp.factors(self.period)).cpu().numpy() @ t
        )
        g, nu, gamma = 0.502, 1.3, 0.2
        family_lhs = (
            sp.residual(
                sp.tensor(self.seed),
                sp.factors(self.period, dilation=g, viscosity=nu, damping=gamma),
            )
            .cpu()
            .numpy()
            @ t
        )
        nonaxis = np.array([m != 0 for _, _, m, *_ in sp.labels])
        mu, w = roots_legendre(sp.grid)
        a = np.pi * (mu + 1) / 4
        radius = self.basis.radius * np.tan(a)
        wr = w * np.pi / 4 * radius**2 / (self.basis.radius * np.cos(a)) ** 2
        phi = 2 * np.pi * np.arange(sp.wedge) / (19 * sp.wedge)
        rr, ct, ph = np.meshgrid(radius, mu, phi, indexing="ij")
        st = np.sqrt(1 - ct**2)
        pts = np.stack(
            (rr * st * np.cos(ph), rr * st * np.sin(ph), rr * ct), axis=-1
        ).reshape(-1, 3)
        weight = np.broadcast_to(
            wr[:, None, None] * w[None, :, None], rr.shape
        ).ravel() / (sp.nt * sp.wedge)
        h = np.log1p((rr.ravel() / self.basis.phase_radius) ** 2)
        integral = family_integral = 0.0
        eps = 2e-6
        for i in range(sp.nt):
            phase = i / sp.nt + h / self.period
            v = self.field(pts, phase)
            curl = self.field(pts, phase, "curl")
            vs = (self.field(pts, phase + eps) - self.field(pts, phase - eps)) / (
                2 * eps * self.period
            )
            dilation = (
                self.field(pts * (1 + eps), phase) - self.field(pts * (1 - eps), phase)
            ) / (2 * eps)
            f = vs + (v + dilation) / 2 + np.cross(curl, v)
            test = sample_cartesian(self.basis, t, self.period, phase, pts, tail=3)
            test_curl = sample_cartesian(
                self.basis, t, self.period, phase, pts, "curl", tail=3
            )
            integral += weight @ np.sum(test * f + test_curl * curl, axis=1)
            wave = sample_cartesian(
                self.basis, self.seed * nonaxis, self.period, phase, pts
            )
            family = f + (g - 0.5) * (dilation - v) + gamma * wave
            family_integral += weight @ np.sum(
                test * family + nu * test_curl * curl, axis=1
            )
        np.testing.assert_allclose(lhs, integral, atol=1e-7, rtol=8e-8)
        np.testing.assert_allclose(family_lhs, family_integral, atol=1e-7, rtol=8e-8)

    def test_bordered_jacobian_and_newton(self):
        eq = self.eq
        rng = np.random.default_rng(15)
        solution = np.r_[eq.ref, 0.0, 1.0]
        y = solution + rng.normal(size=solution.size) * 0.002
        v = rng.normal(size=y.size)
        eps = 2e-6
        fd = (eq.residual(y + eps * v) - eq.residual(y - eps * v)) / (2 * eps)
        np.testing.assert_allclose(eq.jacobian(y)(v), fd, rtol=2e-6, atol=2e-7)
        normal = np.zeros_like(y)
        normal[-1] = 1
        y[-1] = 1
        found, counts, converged = correct(eq, y, normal, 1e-9, 8, 40)
        self.assertTrue(converged)
        self.assertTrue(counts)
        np.testing.assert_allclose(found, solution, atol=2e-7)
        self.assertAlmostEqual(eq.metrics(found)["period"], self.period, places=7)

    def test_augmented_actions_and_bordered_stokes_inverse(self):
        eq = self.eq
        rng = np.random.default_rng(125)
        point = np.r_[eq.ref, 0.18, 0.4]
        direction, dual, normal = rng.normal(size=(3, len(point)))
        for deflate, shift in [(False, 0.0), (True, 0.0), (False, 7.0)]:
            if deflate:
                normal = np.zeros_like(point)
                normal[-1] = 1.0
                point[-1] = 0.0
            predictor = point.copy()
            predictor[:-2] += 0.001 * dual[:-2]
            system = CorrectorSystem(eq, predictor, normal, deflate, shift)
            jac = system.linearize(point, system.residual(point))
            eps = 2e-6
            fd = (
                system.residual(point + eps * direction)
                - system.residual(point - eps * direction)
            ) / (2 * eps)
            np.testing.assert_allclose(jac(direction), fd, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(
                jac.transpose(dual) @ direction,
                dual @ jac(direction),
                rtol=2e-12,
                atol=2e-12,
            )

            # Apply the approximate operator independently using the Galerkin blocks.
            # Recover an arbitrary input, including both scalar components.
            space = eq.space
            factors = space.factors(eq.period * np.exp(point[-2]))
            blocks = [
                (f[-1] @ c.unsqueeze(-1)).squeeze(-1)
                for f, c in zip(
                    factors, space.unpack(space.tensor(direction[:-2])), strict=True
                )
            ]
            field = space.scale_residual(space.pack(blocks), eq.inv).cpu().numpy()
            field += jac.orbit.period_column * direction[-2] - eq.force * direction[-1]
            rhs = np.r_[
                jac.weight * field
                + jac.fluid_residual * (jac.derivative @ direction)
                + shift * direction[:-2],
                eq.phase @ direction[:-2],
                normal @ direction,
            ]
            pc = BorderedStokes(jac)
            self.assertIsNotNone(pc.border_inverse)
            np.testing.assert_allclose(pc(rhs), direction, atol=3e-11, rtol=3e-11)

        duplicate = np.r_[eq.phase, 0.0, 0.0]
        system = CorrectorSystem(eq, point, duplicate, True)
        pc = BorderedStokes(system.linearize(point, system.residual(point)))
        self.assertIsNone(pc.border_inverse)
        self.assertTrue(np.isfinite(pc(dual)).all())

    @unittest.skipUnless(importlib.util.find_spec("petsc4py"), "petsc4py is optional")
    def test_snes_solvers_and_stopping_reasons(self):
        from petsc4py import PETSc

        eq = self.eq
        root = np.r_[eq.ref, 0.0, 1.0]
        predictor = root + np.random.default_rng(15).normal(size=root.size) * 0.002
        predictor[-1] = 1
        normal = np.zeros_like(predictor)
        normal[-1] = 1
        system = CorrectorSystem(eq, predictor, normal, False)
        for method, pc in [
            ("newtonls", "none"),
            ("newtonls", "bordered"),
            ("newtontr", "bordered"),
        ]:
            found, counts, converged, info = petsc_correct(
                system, 1e-9, 8, 40, method=method, preconditioner=pc
            )
            self.assertTrue(converged, info)
            self.assertEqual((info["snes"], info["pc"]), (method, pc))
            self.assertEqual(info["reason"], "CONVERGED_FNORM_ABS")
            self.assertTrue(all(i > 0 for i in counts))
            np.testing.assert_allclose(found, root, atol=2e-7)
        # A native positive stopping reason can still miss the requested equation tolerance.
        opts = PETSc.Options()
        name = "orbit_snes_rtol"
        prior = opts.getString(name, None) if opts.hasName(name) else None
        try:
            opts[name] = "0.99"
            _, _, ok, info = petsc_correct(system, 1e-11, 8, 40)
            self.assertEqual(info["reason"], "CONVERGED_FNORM_RELATIVE")
            self.assertFalse(ok)
            self.assertGreater(info["final_residual_norm"], 1e-11)
        finally:
            if prior is None:
                del opts[name]
            else:
                opts[name] = prior

    def test_lift_preserves_field_and_projection(self):
        fine = Basis(3, 3, 2, 2)
        rich = OrbitSpace(fine, 72, self.device)
        lifted = transfer(self.seed, self.basis.labels(), fine.labels())
        p = [[25.0, 2.0, 4.0], [-16.0, 11.0, 10.0]]
        np.testing.assert_allclose(
            sample_cartesian(fine, lifted, self.period, 0.2, p),
            self.field(p),
            atol=2e-14,
        )
        residual = (
            rich.residual(rich.tensor(lifted), rich.factors(self.period)).cpu().numpy()
        )
        native = (
            self.space.residual(
                self.space.tensor(self.seed), self.space.factors(self.period)
            )
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(
            transfer(residual, fine.labels(), self.basis.labels()),
            native,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_weak_defect_and_radial_conditioning(self):
        small = self.space
        bigger = OrbitSpace(Basis(4, 4, 2, 2), 96, self.device)
        lifted = transfer(self.seed, small.labels, bigger.labels)

        def measure(space, value):
            factors = space.factors(self.period)
            field = space.tensor(value)
            return space.weak_defect(
                field, space.residual(field, factors), factors, self.period
            )

        self.assertGreaterEqual(
            measure(bigger, lifted), measure(small, self.seed) - 1e-7
        )
        high = OrbitSpace(Basis(8, 4, 3, 3), 128, self.device)
        cond = max(
            torch.linalg.cond(f[-1]).max().item() for f in high.factors(self.period)
        )
        self.assertLess(cond, 1e4)

    def test_continue_restart_and_deflated_search(self):
        self.check_continue_restart_and_deflated_search("scipy")

    @unittest.skipUnless(importlib.util.find_spec("petsc4py"), "petsc4py is optional")
    def test_snes_continue_restart_and_deflated_search(self):
        self.check_continue_restart_and_deflated_search("petsc")

    def check_continue_restart_and_deflated_search(self, solver):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            common = [
                "--device",
                "gpu" if self.device == "cuda" else "cpu",
                "--solver",
                solver,
            ]

            def search(path, options):
                opts = parser().parse_args(
                    ["search", "--out", str(path), *common, *options]
                )
                run_search(opts)
                return json.loads((path / "search.json").read_text())

            first = search(
                root / "one",
                [
                    "--radial",
                    "2",
                    "--degrees",
                    "2",
                    "--azimuthal",
                    "1",
                    "--temporal",
                    "1",
                    "--step",
                    "0.05",
                    "--max-stages",
                    "1",
                ],
            )
            self.assertEqual(first["accepted"], 1)
            self.assertAlmostEqual(first["current"]["forcing"], 0.95, places=6)
            second = search(
                root / "two",
                [
                    "--restart",
                    str(root / "one/orbit.npz"),
                    "--step",
                    "0.05",
                    "--max-stages",
                    "1",
                ],
            )
            self.assertEqual(second["accepted"], 2)
            self.assertLess(second["current"]["forcing"], first["current"]["forcing"])
            direct = search(
                root / "attempt",
                ["--lift", str(root / "two/orbit.npz"), "--direct", "--newton", "1"],
            )
            self.assertEqual(direct["status"], "direct_unconverged")
            self.assertEqual(direct["current"]["forcing"], 0.0)
            self.assertGreater(direct["current"]["relative_spiral"], 0.05)
            self.assertNotEqual(direct["current"]["period"], 0.36)
            if solver == "petsc":
                self.assertEqual(
                    first["petsc_correctors"][-1]["reason"], "CONVERGED_FNORM_ABS"
                )
                self.assertEqual(
                    direct["petsc_correctors"][-1]["reason"], "DIVERGED_MAX_IT"
                )
                self.assertTrue(direct["direct_ksp_iterations"])

    def test_implicit_pseudo_time_and_restart(self):
        path = PseudoTimePath(decay_time=0.7, initial_viscosity=1.2)
        eq = OrbitEquation(self.space, self.seed, self.period, path)
        start = np.r_[eq.ref, 0.0, 1.0]
        np.testing.assert_allclose(eq.residual(start), 0.0, atol=1e-14)
        eq.set_strength(0.0)
        np.testing.assert_allclose(
            eq.fluid(eq.ref, 0.0), self.eq.fluid(eq.ref, 0.0), atol=1e-14
        )
        log_strength = 0.0
        for _ in range(80):
            log_strength = path.advance_clock(log_strength, path.decay_time)
        self.assertAlmostEqual(math.exp(log_strength) * 2**80, 1.0, places=12)

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fresh = [
                "--pseudo-transient",
                "--radial",
                "2",
                "--degrees",
                "2",
                "--azimuthal",
                "1",
                "--temporal",
                "1",
                "--ptc-decay-time",
                "0.7",
                "--ptc-initial-viscosity",
                "1.2",
                "--step",
                "0.1",
            ]

            def search(name, options):
                args = parser().parse_args(
                    [
                        "search",
                        "--out",
                        str(d / name),
                        "--device",
                        "gpu" if self.device == "cuda" else "cpu",
                        "--solver",
                        "petsc" if importlib.util.find_spec("petsc4py") else "scipy",
                        "--newton",
                        "10",
                        "--krylov",
                        "50",
                        *options,
                    ]
                )
                run_search(args)
                return json.loads((d / name / "search.json").read_text())

            one = search("one", [*fresh, "--max-stages", "1"])
            two = search(
                "resumed", ["--restart", str(d / "one/orbit.npz"), "--max-stages", "1"]
            )
            together = search("two", [*fresh, "--max-stages", "2"])
            self.assertEqual(two["status"], "pseudo_time_budget_reached")
            self.assertEqual(two["accepted"], 2)
            cfg, seed, y, old, _, _ = load_checkpoint(d / "resumed/orbit.npz")
            np.testing.assert_allclose(
                y, load_checkpoint(d / "two/orbit.npz")[2], rtol=2e-12, atol=2e-12
            )
            dt = two["stages"][-1]["dt"]
            self.assertAlmostEqual(
                (y[-1] - old[-1]) / dt, -y[-1] / path.decay_time, places=13
            )
            self.assertAlmostEqual(
                cfg["pseudo_time"], one["config"]["pseudo_time"] + dt
            )
            self.assertAlmostEqual(cfg["log_forcing"], math.log(y[-1]))
            self.assertEqual(
                two["config"]["pseudo_time"], together["config"]["pseudo_time"]
            )
            phys = OrbitSpace(Basis(**cfg["basis"]), cfg["grid"], self.device)
            step_eq = OrbitEquation(phys, seed, cfg["period"], path)
            step_eq.set_strength(y[-1])
            fluid = step_eq.residual(y)[:-1]
            np.testing.assert_allclose((y[:-2] - old[:-2]) / dt + fluid, 0, atol=1e-7)
            self.assertLess(two["stages"][-1]["implicit_residual"], 1e-7)
            self.assertGreater(two["current"]["unforced_relative_defect"], 0.1)
            self.assertGreater(np.linalg.norm(fluid), 0.01)
            diagnostic = search(
                "crossing", [*fresh, "--max-stages", "2", "--min-spiral", "0.9999"]
            )
            self.assertEqual(diagnostic["accepted"], 2)
            self.assertIn("first_below_amplitude_floor", diagnostic)
            # Large finite steps still give a positive forcing in the log-space clock.
            cutoff = PseudoTimePath().advance_clock(0.0, 1e200)
            self.assertTrue(math.isfinite(cutoff) and math.exp(cutoff) > 0)
            failed = search(
                "failed",
                [
                    *fresh,
                    "--max-stages",
                    "1",
                    "--step",
                    "0.4",
                    "--min-step",
                    "0.4",
                    "--tol",
                    "1e-12",
                    "--newton",
                    "1",
                ],
            )
            self.assertEqual(failed["status"], "pseudo_time_stalled")
            self.assertEqual(failed["accepted"], 0)
            self.assertEqual(failed["config"]["pseudo_time"], 0)
            self.assertEqual(failed["current"]["forcing"], 1.0)

    def test_checkpoint_and_native_replay(self):
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            y = np.r_[self.eq.ref, math.log(1.1), 0.43]
            cfg = {"basis": asdict(self.basis), "period": self.period, "grid": 128}
            checkpoint(d / "c.npz", cfg, self.seed, y, y * 0.9, 0.02, 4)
            a, b, c, _, ds, steps = load_checkpoint(d / "c.npz")
            np.testing.assert_equal(a, cfg)
            np.testing.assert_array_equal(b, self.seed)
            np.testing.assert_array_equal(c, y)
            self.assertEqual((ds, steps), (0.02, 4))
            args = parser().parse_args(
                [
                    "export",
                    str(d / "c.npz"),
                    "--out",
                    str(d / "native"),
                    "--grid",
                    "9",
                    "--steps",
                    "7",
                ]
            )
            export_replay(args)
            raw = np.fromfile(d / "native/velocity.bin").reshape(3, 9, 9, 9)
            coord = (np.arange(9) + 0.5) * (192 / 9) - 96
            expected = sample_cartesian(
                self.basis,
                self.seed,
                self.period * 1.1,
                0,
                [[coord[3], coord[5], coord[2]]],
            )[0]
            np.testing.assert_allclose(raw[:, 2, 5, 3], expected)
            deck = dict(
                line.split(" = ", 1)
                for line in (d / "native/inputs").read_text().splitlines()
            )
            self.assertAlmostEqual(
                float(deck["stop_time"]), -math.expm1(-self.period * 1.1)
            )
            self.assertAlmostEqual(
                7 * float(deck["incflo.fixed_dt"]), float(deck["stop_time"])
            )


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        sys.argv.remove("--gpu")
        os.environ["DSS_TEST_DEVICE"] = "gpu"
        print(
            f"Testing on {torch.cuda.get_device_name()} (HIP {torch.version.hip})",
            flush=True,
        )
    unittest.main()
