"""Check the optimization resolution and cell-centered periodic refinement."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from dss_resolution import (
    holdout_prediction,
    refine_velocity,
    refinement_transpose,
    richardson_return,
)
from optimize_dss import MIN_SEARCH_GRID, RichardsonCycle, search_seed


class SearchResolutionTest(unittest.TestCase):
    def test_rejects_subminimum_before_generating_a_seed(self):
        with self.assertRaisesRegex(ValueError, "at least 128"):
            search_seed(None, 127)
        self.assertEqual(MIN_SEARCH_GRID, 128)

    def test_joint_time_and_space_refinement(self):
        dummy = Path(__file__)
        with self.assertRaisesRegex(ValueError, "constant steps/grid"):
            RichardsonCycle(dummy, dummy, [128, 160, 192], [96, 119, 144], 2.0)


class RefinementTest(unittest.TestCase):
    def test_euclidean_transpose(self):
        rng = np.random.default_rng(170)
        for coarse, fine in ((8, 13), (8, 16), (9, 16), (9, 9)):
            with self.subTest(coarse=coarse, fine=fine):
                x = rng.normal(size=(3, coarse, coarse, coarse))
                y = rng.normal(size=(3, fine, fine, fine))
                np.testing.assert_allclose(
                    np.vdot(refine_velocity(x, fine), y),
                    np.vdot(x, refinement_transpose(y, coarse)),
                    rtol=2e-13,
                )

    def test_phase_and_amplitude(self):
        def velocity(n):
            axis = (np.arange(n) + 0.5) / n
            z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
            return np.stack(
                (
                    np.cos(2 * np.pi * (2 * x - 3 * y + z)),
                    np.sin(2 * np.pi * (x + 2 * z)),
                    np.ones_like(x),
                )
            )

        for coarse, fine in ((8, 13), (8, 16), (9, 16)):
            with self.subTest(coarse=coarse, fine=fine):
                np.testing.assert_allclose(
                    refine_velocity(velocity(coarse), fine), velocity(fine), atol=1e-13
                )

    def test_even_grid_nyquist(self):
        coarse, fine = 8, 13
        field = np.broadcast_to(
            (-1.0) ** np.arange(coarse), (3, coarse, coarse, coarse)
        )
        fine_centers = (np.arange(fine) + 0.5) / fine
        expected = np.cos(np.pi * coarse * (fine_centers - 0.5 / coarse))
        np.testing.assert_allclose(
            refine_velocity(field, fine),
            np.broadcast_to(expected, (3, fine, fine, fine)),
            atol=1e-13,
        )


class RichardsonTest(unittest.TestCase):
    grids = np.array([128, 160, 192])

    def test_unequal_ratios_and_exact_limit(self):
        errors = 0.3 + 2000.0 / self.grids**2
        result, _ = richardson_return(errors, self.grids)
        self.assertAlmostEqual(result["score"], 0.3)
        self.assertAlmostEqual(result["observed_order"], 2.0)
        self.assertTrue(result["order_consistent"])
        self.assertAlmostEqual(result["disagreement"], 0.0)
        first_order, _ = richardson_return(0.3 + 10.0 / self.grids, self.grids)
        self.assertAlmostEqual(first_order["observed_order"], 1.0)
        self.assertFalse(first_order["order_consistent"])

    def test_fourth_grid_prediction(self):
        errors = 0.3 + 2000.0 / self.grids**2
        metric, _ = richardson_return(errors, self.grids)
        guide = {
            "control_sha256": "abc",
            "trial": 2,
            "metrics": {"velocity_return": metric},
        }
        fine = {
            "control_sha256": "abc",
            "grid": 192,
            "steps": 144,
            "velocity_return": errors[-1],
        }
        holdout = {"grid": 224, "steps": 168, "velocity_return": 0.3 + 2000.0 / 224**2}
        result = holdout_prediction(guide, fine, holdout)["metrics"]["velocity_return"]
        self.assertAlmostEqual(result["prediction_error"], 0.0)
        self.assertGreater(result["finest_grid_error"], 0.0)
        with self.assertRaisesRegex(ValueError, "same timestep"):
            holdout_prediction(guide, fine, dict(holdout, steps=144))

    def test_turning_and_flat_sequences(self):
        for errors in ([0.40, 0.45, 0.43], [0.43, 0.43, 0.43], [0.0, 0.0, 0.0]):
            result, weights = richardson_return(errors, self.grids)
            self.assertEqual(result["correction"], 0.0)
            self.assertIsNone(result["observed_order"])
            self.assertGreaterEqual(result["score"], errors[-1])
            self.assertTrue(np.isfinite(weights).all())

    def test_chain_rule_on_native_objectives(self):
        direction = np.array([0.023, -0.012, 0.017])
        h = 1e-7
        for errors in ([0.40, 0.42, 0.429], [0.45, 0.43, 0.44], [0.56, 0.54, 0.53]):
            native = 0.5 * np.square(errors)
            _, grad = richardson_return(errors, self.grids)

            def objective(values):
                return (
                    0.5
                    * richardson_return(np.sqrt(2 * values), self.grids)[0]["score"]
                    ** 2
                )

            central = (
                objective(native + h * direction) - objective(native - h * direction)
            ) / (2 * h)
            np.testing.assert_allclose(grad @ direction, central, rtol=1e-6)


class CampaignRankingTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("rsync"), "the profile helper requires rsync")
    def test_guided_winner_with_incomplete_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(
                json.dumps({"mode": "richardson", "fine_grid": 192})
            )
            for trial, (fine, guided) in enumerate(
                ((0.2, 0.6), (0.3, 0.1), (0.01, None))
            ):
                directory = root / f"trial-{trial:05}"
                (directory / "coarse").mkdir(parents=True)
                (directory / "coarse/search.json").write_text("{}")
                (directory / "fine-score.json").write_text(
                    json.dumps({"trial": trial, "score": fine})
                )
                if guided is not None:
                    (directory / "richardson-score.json").write_text(
                        json.dumps({"trial": trial, "score": guided})
                    )
            rows = subprocess.check_output(
                [Path(__file__).with_name("best_dss_profile.sh"), "rank", root],
                text=True,
            ).splitlines()
            self.assertEqual([int(row.split("\t")[1]) for row in rows], [1, 0])
            self.assertEqual(rows[0].split("\t")[-1], "trial-00001/fine-score.json")


if __name__ == "__main__":
    unittest.main()
