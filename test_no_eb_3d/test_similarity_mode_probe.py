"""Full-space linear wave propagation on a frozen similarity flow."""

import os
import unittest

import numpy as np
import torch
from compare_mode_probes import compare
from probe_rotating_modes import TAIL, FullSphereBasis, advection, packets, propagate
from similarity_rotating import WaveSpace
from solve_similarity_orbit import backend


class ModeProbeTest(unittest.TestCase):
    def test_field_alignment_removes_only_rotation(self):
        a = np.array([1 + 2j, -3j])
        b = 2 * np.exp(0.7j) * a
        d = compare(a, b, np.array([2.0, 3.0]))
        self.assertAlmostEqual(d["overlap"], 1.0)
        self.assertAlmostEqual(d["relative_change"], 0.5)

    def test_solenoidal_packets(self):
        m = 19

        def cylindrical(r, z):
            rho = np.hypot(r, z)
            eta, sn = z / rho, r / rho
            radial, polar, phi = packets(np.array([rho]), np.array([eta]), m)[0, 0]
            return np.stack((sn * radial + eta * polar, phi, eta * radial - sn * polar))

        h = 1e-4
        for r, z in [(31.0, -3.0), (40.0, 6.5), (51.0, 9.0)]:
            u = cylindrical(r, z)
            dr = (cylindrical(r + h, z) - cylindrical(r - h, z)) / (2 * h)
            dz = (cylindrical(r, z + h) - cylindrical(r, z - h)) / (2 * h)
            np.testing.assert_allclose(
                dr[0] + (u[0] + 1j * m * u[1]) / r + dz[2], 0, atol=1e-9
            )
        np.testing.assert_array_equal(cylindrical(1.0, 0.0), 0)

    def test_solid_rotation_jacobian(self):
        # Around U = e_z x y, the convective linearization in the spherical frame
        # is i*m*v + 2*e_z x v. This independently tests the weak curl formulation.
        m = 9
        b = FullSphereBasis(
            sigma_breaks=(0, 0.6, 1), eta_breaks=(0, 0.3, 1), harmonics=1
        )
        sp = WaveSpace(b, 48, backend(os.environ.get("DSS_TEST_DEVICE", "cpu")), m)
        e_z = torch.zeros(
            (*sp.weight.shape, 3), dtype=torch.complex128, device=sp.device
        )
        eta = sp.tensor(sp.meridian.eta)
        sn = (1 - eta**2).sqrt()
        e_z[..., 0], e_z[..., 1] = eta, -sn
        u = torch.zeros_like(e_z)
        u[..., 2] = sp.tensor(sp.rho)[:, None] * sn
        op = sp.meridian.operator(1, m, 0.502, 0, TAIL, "velocity", sp.device)
        n = sum(a * b for a, b in op.shapes)
        v = np.random.default_rng(123).normal(size=(n, 2)) @ np.array([1, 1j])
        samples = op.apply(torch.as_tensor(v, device=sp.device))
        expected = (
            op.adjoint(
                sp.weight[..., None]
                * (1j * m * samples + 2 * torch.linalg.cross(e_z, samples))
            )
            .cpu()
            .numpy()
        )
        actual = advection(sp, m, 0.502, u, 2 * e_z, n) @ v
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)

    def test_growth_and_rotation_convention(self):
        K = np.diag([0.3 - 4j, -0.6 + 2j])
        unit = np.eye(2)
        for t in (0.13, 0.36):
            row = propagate(K, -0.1 * unit, np.diag([2.0, 7.0]), unit, t)[0]
            self.assertAlmostEqual(row["end_core_amplitude_gain"], np.exp(0.3 * t))
            self.assertAlmostEqual(row["mean_phase_frequency"], 4.0)
            self.assertAlmostEqual(row["endpoint_shape_overlap"], 1.0)
            self.assertAlmostEqual(
                row["end_control_core_gain_same_packet"], np.exp(-0.1 * t)
            )


if __name__ == "__main__":
    unittest.main()
