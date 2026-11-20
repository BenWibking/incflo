"""Check the optimization resolution and cell-centered periodic refinement."""

import unittest

import numpy as np
from dss_vorticity import refine_velocity
from optimize_dss import MIN_SEARCH_GRID, search_seed


class SearchResolutionTest(unittest.TestCase):
    def test_rejects_subminimum_before_generating_a_seed(self):
        with self.assertRaisesRegex(ValueError, "at least 128"):
            search_seed(None, 127)
        self.assertEqual(MIN_SEARCH_GRID, 128)


class RefinementTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
