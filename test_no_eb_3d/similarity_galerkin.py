"""Solenoidal space-time Galerkin discretization on R^3 for backward DSS flow."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.polynomial.legendre import Legendre
from scipy.special import eval_jacobi, gammaln, roots_legendre


@dataclass(frozen=True)
class Basis:
    radial: int = 8
    degrees: int = 8
    azimuthal: int = 3  # largest m/19
    temporal: int = 3  # largest temporal harmonic
    radius: float = 10.0  # radial map scale
    phase_radius: float = 60.0  # radial winding scale

    radial_family: str = "jacobi"

    def __post_init__(self):
        if self.radial_family not in ("jacobi", "legendre"):
            raise ValueError("Unknown radial polynomial family")
        if min(self.radial, self.degrees, self.azimuthal, self.temporal) < 1:
            raise ValueError("Positive mode counts are required")
        if not all(
            math.isfinite(x) and x > 0 for x in (self.radius, self.phase_radius)
        ):
            raise ValueError("Map radii must be positive and finite")

    def modes(self, n):
        # R_x(pi) leaves cosine for even ell+m and sine for odd ell+m.
        # R_z(pi/19) over half a period requires j and n of identical parity.
        return [
            (19 * j, ell)
            for j in range(n % 2, self.azimuthal + 1, 2)
            for ell in (
                range(2, 2 * self.degrees + 1, 2)
                if j == 0
                else range(19 * j, 19 * j + self.degrees)
            )
        ]

    def labels(self):
        return [
            (n, family, m, ell, k, part)
            for n in range(self.temporal + 1)
            for part in range(2 if n else 1)
            for family in range(2)
            for m, ell in self.modes(n)
            for k in range(self.radial)
        ]

    def coefficients(self, vector):
        offset = 0
        for n in range(self.temporal + 1):
            shape = (2, len(self.modes(n)), self.radial)
            size = math.prod(shape)
            c = vector[offset : offset + size].reshape(shape) + 0j
            offset += size
            if n:
                c = c + 1j * vector[offset : offset + size].reshape(shape)
                offset += size
            yield c


def angular(mu, phi, modes):
    """Real normalized Y, d_theta Y, (d_phi Y)/sin(theta) at paired points."""
    mu, phi = np.broadcast_arrays(mu, phi)
    sn = np.sqrt((1 - mu) * (1 + mu))
    denom_sin = np.where(sn > 0, sn, 1.0)
    rows = []
    for m, ell in modes:
        # Start with normalized P_m^m and recur without forming m! or a large P_l^m.
        logscale = (
            (gammaln(2 * m + 1) + np.log((2 * m + 1) / 2)) / 2
            - m * np.log(2)
            - gammaln(m + 1)
        )
        p = (-1) ** m * np.exp(logscale) * sn**m * (np.sqrt(2) if m else 1.0)
        prev = np.zeros_like(p)
        for degree in range(m + 1, ell + 1):
            denom = degree**2 - m**2
            a = np.sqrt((4 * degree**2 - 1) / denom)
            b = (
                np.sqrt(
                    (2 * degree + 1)
                    * ((degree - 1) ** 2 - m**2)
                    / ((2 * degree - 3) * denom)
                )
                if degree > m + 1
                else 0
            )
            prev, p = p, a * mu * p - b * prev
        h = (
            ell * mu * p
            - np.sqrt((2 * ell + 1) * (ell**2 - m**2) / (2 * ell - 1)) * prev
        ) / denom_sin
        h = np.where(sn > 0, h, 0.0)
        co, si = np.cos(m * phi), np.sin(m * phi)
        y, dy = (si, m * co) if (ell + m) % 2 else (co, -m * si)
        rows.append((p * y, h * y, p * dy / denom_sin))
    return np.stack(rows, axis=1).reshape(3, len(modes), -1)


def envelope(r, modes, basis, tail, poloidal):
    """Potential and two ordinary radial derivatives before applying the spiral phase."""
    r = np.asarray(r)[:, None, None]
    ell = np.array([l for _, l in modes])[None, :, None]
    scale = basis.radius
    x = r * r / (scale * scale + r * r)
    dx = 2 * r * scale**2 / (scale * scale + r * r) ** 2
    d2x = 2 * scale**2 * (scale * scale - 3 * r * r) / (scale * scale + r * r) ** 3
    power = tail - int(poloidal)
    # Writing 1-x this way also preserves accuracy far outside the core.
    base = x ** (ell / 2) * (scale**2 / (scale * scale + r * r)) ** (power / 2)
    lam = ell * (ell + 1)
    base *= scale / lam if poloidal else 1 / np.sqrt(lam)
    dlog = (ell - (ell + power) * x) / r
    db = base * dlog
    d2b = base * (dlog**2 - dlog / r - (ell + power) * dx / r)
    values, first, second = [], [], []
    # The trial/test radial mass has weight x^beta (1-x)^(-1/2).
    # Using its orthogonal polynomials keeps large-m blocks independent at high degree.
    alpha = -0.5
    beta = ell + 0.5 - int(poloidal)

    def log_norm(k):
        return (
            gammaln(k + alpha + 1)
            + gammaln(k + beta + 1)
            - np.log(2 * k + alpha + beta + 1)
            - gammaln(k + 1)
            - gammaln(k + alpha + beta + 1)
        )

    z = 2 * x - 1
    for k in range(basis.radial):
        if basis.radial_family == "legendre":
            p = Legendre.basis(k)
            v, vx, vxx = p(z), 2 * p.deriv()(z), 4 * p.deriv(2)(z)
        else:
            normalization = np.exp((log_norm(0) - log_norm(k)) / 2)
            t = k + alpha + beta + 1
            v = normalization * eval_jacobi(k, alpha, beta, z)
            vx = (
                normalization * t * eval_jacobi(k - 1, alpha + 1, beta + 1, z)
                if k
                else 0.0
            )
            vxx = (
                normalization * t * (t + 1) * eval_jacobi(k - 2, alpha + 2, beta + 2, z)
                if k >= 2
                else 0.0
            )
        dv = vx * dx
        d2v = vxx * dx**2 + vx * d2x
        values.append(base * v)
        first.append(db * v + base * dv)
        second.append(d2b * v + 2 * db * dv + base * d2v)
    return tuple(np.concatenate(arr, axis=-1) for arr in (values, first, second))


def components(r, lam, h1, h2, n_omega, potentials):
    """Radial coefficients in the Y e_r, grad_S Y, grad_S Y cross e_r basis."""

    def derivatives(pack):
        f, df, ddf = pack
        phase = -1j * n_omega * h1
        return (
            f + 0j,
            df + phase * f,
            ddf + 2 * phase * df + (phase**2 - 1j * n_omega * h2) * f,
        )

    g, dg, ddg = derivatives(potentials[0])
    f, df, _ = derivatives(potentials[1])
    # P[g] = curl(T[g]), curl(P[g]) = -T[Delta_ell g].
    velocity = (lam * g / r, dg + g / r, f)
    curl = (lam * f / r, df + f / r, -(ddg + 2 * dg / r - lam * g / r**2))
    dilation = (lam * (dg - g / r), r * ddg + dg - g / r, r * df)
    return velocity, curl, dilation


def phase_derivatives(r, rc):
    return 2 * r / (rc**2 + r * r), 2 * (rc**2 - r * r) / (rc**2 + r * r) ** 2


def pair(a, b, weight):
    """One radial matrix (test degree, trial degree) for each spherical harmonic."""
    return (a * weight).permute(1, 2, 0).conj() @ b.transpose(0, 1)


class Band:
    def __init__(self, n, modes, basis, r, wr, mu, wmu, phi, tensor):
        self.n = n
        self.nonaxisymmetric = tensor(
            np.array([m > 0 for m, _ in modes])[None, :, None]
        )
        self.r = tensor(r[:, None, None])
        self.weight = tensor(wr[:, None, None])
        self.lam = tensor(np.array([l * (l + 1) for _, l in modes])[None, :, None])
        self.h1, self.h2 = phase_derivatives(self.r, basis.phase_radius)
        angles = angular(mu[:, None], phi[None, :], modes)
        self.angles = tensor(angles).to(torch.complex128)
        weights = np.broadcast_to(wmu[:, None] / len(phi), (len(mu), len(phi))).ravel()
        self.dual = (self.angles * tensor(weights)).transpose(1, 2).contiguous()
        self.potentials = [
            [
                tuple(tensor(a) for a in envelope(r, modes, basis, tail, pol))
                for pol in (True, False)
            ]
            for tail in (1, 3)
        ]

    def factors(self, omega, viscosity, dilation=0.5, damping=0.0):
        w = self.n * omega
        trial, curl, drift = components(
            self.r, self.lam, self.h1, self.h2, w, self.potentials[0]
        )
        test, curl_test, _ = components(
            self.r, self.lam, self.h1, self.h2, w, self.potentials[1]
        )
        mass = 1 - dilation + 1j * w
        if damping:
            mass = mass + damping * self.nonaxisymmetric
        linear = [mass * v + dilation * d for v, d in zip(trial, drift)]
        angular_norm = self.lam
        tw = self.weight * (0.5 if self.n else 1.0)

        pp = pair(test[0], linear[0], tw) + pair(test[1], linear[1], tw * angular_norm)
        pp += viscosity * pair(curl_test[2], curl[2], tw * angular_norm)
        tt = pair(test[2], linear[2], tw * angular_norm)
        tt += viscosity * (
            pair(curl_test[0], curl[0], tw)
            + pair(curl_test[1], curl[1], tw * angular_norm)
        )
        return trial, curl, test, torch.stack((pp, tt))

    def gram(self, radial, radial_weight=1.0):
        a, b, t = radial
        w = self.weight * (0.5 if self.n else 1.0) * radial_weight
        return torch.stack(
            (pair(a, a, w) + pair(b, b, w * self.lam), pair(t, t, w * self.lam))
        )

    def synthesize(self, radial, coeff):
        y, h, g = self.angles
        a, b, t = [(f * c).sum(-1) for f, c in zip(radial, coeff)]
        return torch.stack((a @ y, b @ h + t @ g, b @ g - t @ h), dim=-1)

    def project(self, test, force):
        y, h, g = self.dual
        fr, ft, fp = force.unbind(-1)
        v = (fr @ y, ft @ h + fp @ g, ft @ g - fp @ h)
        a, b, t = [
            (f.conj() * self.weight * q.unsqueeze(-1)).sum(0) for f, q in zip(test, v)
        ]
        return torch.stack((a + b, t))


class OrbitSpace:
    def __init__(self, basis, grid, device="cpu", viscosity=1.0):
        if not math.isfinite(viscosity) or viscosity <= 0:
            raise ValueError("Viscosity must be positive")
        self.basis, self.grid, self.viscosity = basis, grid, viscosity
        self.device = torch.device(device)
        self.tensor = lambda x: torch.as_tensor(
            x, dtype=torch.float64, device=self.device
        )
        largest = max(l for n in range(basis.temporal + 1) for _, l in basis.modes(n))
        if grid <= 1.5 * (largest + 1) or grid < 2 * basis.radial + 2:
            raise ValueError(
                "Increase quadrature to resolve products of the requested modes"
            )
        nodes, weights = roots_legendre(grid)
        angle = (np.pi / 4) * (nodes + 1)
        r = basis.radius * np.tan(angle)
        wr = (
            weights
            * (np.pi / 4)
            * basis.radius
            * r
            * r
            / np.cos(angle) ** 2
            / basis.radius**3
        )
        self.wedge = max(3 * basis.azimuthal + 1, math.ceil(grid / 19))
        self.nt = 3 * basis.temporal + 1
        phi = 2 * np.pi * np.arange(self.wedge) / (19 * self.wedge)
        self.bands = [
            Band(n, basis.modes(n), basis, r, wr, nodes, weights, phi, self.tensor)
            for n in range(basis.temporal + 1)
        ]
        time = 2 * np.pi * np.arange(self.nt) / self.nt
        osc = np.exp(1j * np.arange(len(self.bands))[:, None] * time)
        self.clock = torch.as_tensor(osc, dtype=torch.complex128, device=self.device)
        self.labels = basis.labels()

    def unpack(self, vector):
        return self.basis.coefficients(vector)

    def pack(self, bands):
        parts = []
        for band, data in zip(self.bands, bands):
            parts.append(data.real.flatten())
            if band.n:
                parts.append(data.imag.flatten())
        return torch.cat(parts)

    def factors(self, period, *, dilation=0.5, viscosity=None, damping=0.0):
        if not math.isfinite(period) or period <= 0:
            raise ValueError("Period must be finite and positive")
        nu = self.viscosity if viscosity is None else viscosity
        return [
            b.factors(2 * np.pi / period, nu, dilation, damping) for b in self.bands
        ]

    def fields(self, vector, factors):
        vel, vort = [], []
        for band, coeff, (v, c, _, _) in zip(self.bands, self.unpack(vector), factors):
            p, t = coeff
            vel.append(band.synthesize(v, (p, p, t)))
            vort.append(band.synthesize(c, (t, t, p)))

        def collocate(harmonics):
            return (
                (torch.stack(harmonics, dim=-1) @ self.clock)
                .real.transpose(-1, -2)
                .contiguous()
            )

        return collocate(vel), collocate(vort)

    def residual(self, vector, factors, fields=None, background=None):
        v, w = self.fields(vector, factors) if fields is None else fields
        if background is None:
            adv = torch.linalg.cross(w, v)
        else:
            base_v, base_w = background
            adv = torch.linalg.cross(w, base_v) + torch.linalg.cross(base_w, v)
        # Re[a exp(i n chi)] gives the 1/2 for nonzero n through this time average.
        samples = adv.transpose(-1, -2)
        harmonics = (
            torch.complex(samples @ self.clock.real.T, -(samples @ self.clock.imag.T))
            / self.nt
        )
        blocks = []
        for band, coeff, (_, _, test, linear), f in zip(
            self.bands, self.unpack(vector), factors, harmonics.unbind(-1)
        ):
            blocks.append(
                (linear @ coeff.unsqueeze(-1)).squeeze(-1) + band.project(test, f)
            )
        return self.pack(blocks)

    def weak_defect(self, vector, raw, factors, period):
        """One-period L2 norm of the Riesz representative on divergence-free tests."""
        residual_sq = self.tensor(0.0)
        velocity_sq = self.tensor(0.0)
        for band, c, f, (v, _, test, _) in zip(
            self.bands, self.unpack(vector), self.unpack(raw), factors
        ):
            test_gram = band.gram(test)
            rep = torch.linalg.solve(test_gram, f.unsqueeze(-1)).squeeze(-1)
            residual_sq += (f.conj() * rep).real.sum()
            gram = band.gram(v, (1 + (band.r / 50) ** 2) ** -2)
            velocity_sq += (c.conj() * (gram @ c.unsqueeze(-1)).squeeze(-1)).real.sum()
        if velocity_sq.item() <= 0:
            raise ValueError("A nonzero velocity is required for the weak defect")
        return period * math.sqrt((residual_sq / velocity_sq).item())

    def linear_inverse(self, factors):
        return [torch.linalg.inv(entry[-1]) for entry in factors]

    def scale_residual(self, residual, inverse):
        return self.pack(
            [
                (matrix @ c.unsqueeze(-1)).squeeze(-1)
                for matrix, c in zip(inverse, self.unpack(residual))
            ]
        )

    def seed(self, amplitude, wave, random_seed=0, jitter=0.0):
        """Modeled forced start, counter-rotating waves above and below the equator."""
        a = np.zeros(len(self.labels))
        rng = np.random.default_rng(random_seed)
        for i, (n, fam, m, ell, k, part) in enumerate(self.labels):
            if (m, n, k) == (0, 0, 0) and ell == 2:
                a[i] = amplitude * (1 if fam else -0.6)
            if (m, n, k) == (19, 1, 0) and ell in (19, 20) and part == ell - 19:
                a[i] = amplitude * wave * (1 if fam else 0.7)
            if jitter:
                a[i] += amplitude * jitter * rng.normal() / np.sqrt(len(a))
        return a

    def rotate_time(self, coeff):
        return self.pack([1j * b.n * c for b, c in zip(self.bands, self.unpack(coeff))])


def transfer(coeff, old_labels, new_labels):
    values = dict(zip(old_labels, coeff, strict=True))
    return np.array([values.get(k, 0.0) for k in new_labels])


def sample_cartesian(basis, coeff, period, phase, points, field="velocity", tail=1):
    """Evaluate a velocity, its curl or r*d_r velocity at Cartesian points; phase is s/S."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    radius = np.linalg.norm(points, axis=1)
    r = np.where(radius > 0, radius, 1.0)
    phi = np.arctan2(points[:, 1], points[:, 0])
    mu = np.clip(points[:, 2] / r, -1.0, 1.0)
    sn = np.sqrt((1 - mu) * (1 + mu))
    cp, sp = np.cos(phi), np.sin(phi)
    er = np.stack((sn * cp, sn * sp, mu), axis=1)
    et = np.stack((mu * cp, mu * sp, -sn), axis=1)
    ep = np.stack((-sp, cp, np.zeros_like(cp)), axis=1)
    ans = np.zeros_like(points)
    radial_nodes, which = np.unique(r, return_inverse=True)
    h = np.log1p((r / basis.phase_radius) ** 2)
    bands = list(basis.coefficients(coeff))
    rr = radial_nodes[:, None, None]
    h1, h2 = phase_derivatives(rr, basis.phase_radius)
    for parity in (0, 1):
        modes = basis.modes(parity)
        y, g, t = angular(mu, phi, modes).transpose(0, 2, 1)
        lam = np.array([l * (l + 1) for _, l in modes])[None, :, None]
        potentials = [
            envelope(radial_nodes, modes, basis, tail, pol) for pol in (True, False)
        ]
        for n in range(parity, basis.temporal + 1, 2):
            vel, curl, drift = components(
                rr, lam, h1, h2, n * 2 * np.pi / period, potentials
            )
            factors = {"velocity": vel, "curl": curl, "dilation": drift}[field]
            p, tor = bands[n] if field != "curl" else bands[n][::-1]
            a, b, c = [(f * v).sum(-1)[which] for f, v in zip(factors, (p, p, tor))]
            components_xyz = (
                (a * y).sum(1)[:, None] * er
                + (b * g + c * t).sum(1)[:, None] * et
                + (b * t - c * g).sum(1)[:, None] * ep
            )
            oscillator = np.exp(2j * np.pi * n * (phase - h / period))
            ans += (oscillator[:, None] * components_xyz).real
    ans[radius == 0] = 0
    return ans
