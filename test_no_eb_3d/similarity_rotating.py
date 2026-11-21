"""Divergence-free Galerkin rotating waves on the northern similarity half-space.

Real fields are U_0 + 2 Re sum_j U_j exp(i j (kappa*phi - omega*s + b*log(1+r²/rc²))).
The amplitude of the fundamental and its azimuthal phase close the equations for g and omega.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from scipy.interpolate import RegularGridInterpolator
from similarity_elements import (
    Jet,
    Meridian,
    Operator,
    PressureFit,
    Splines,
    angular,
    breaks,
    gauss_elements,
    radial,
)


@dataclass(frozen=True)
class WaveBasis:
    degree: int = 3
    sigma_breaks: tuple = (0, 0.53, 0.76, 1)
    eta_breaks: tuple = (0, 0.22, 0.52, 1)
    harmonics: int = 3
    radius: float = 24.0
    phase_radius: float = 60.0

    def __post_init__(self):
        for key in ("sigma_breaks", "eta_breaks"):
            object.__setattr__(self, key, breaks(getattr(self, key)))
        if (
            self.degree < 3
            or self.harmonics < 0
            or not all(0 < x < math.inf for x in (self.radius, self.phase_radius))
        ):
            raise ValueError(
                "Invalid half-space element degree, harmonics or radial scales"
            )

    @property
    def full_eta(self):
        # PressureFit integrates only the northern domain for this basis.
        return self.eta_breaks

    @cached_property
    def groups(self):
        reflected = tuple(-v for v in self.eta_breaks[:0:-1]) + self.eta_breaks

        def pair(cr, ca, parity, axisymmetric):
            return (
                Splines(self.sigma_breaks, self.degree, cr, vanish_left=axisymmetric),
                Splines(reflected, self.degree, ca, parity),
            )

        mean = {
            "S": pair(2, 2, -1, True),
            "T": pair(1, 1, -1, True),
            "gamma": (None, None),
        }
        wave = {"A": pair(2, 1, 1, False), "B": pair(1, 2, -1, False)}
        return [mean] + [wave] * self.harmonics


class WaveMeridian(Meridian):
    def __init__(self, basis, sig, eta):
        super().__init__(basis, sig, eta)
        self.splines = {
            s: s.jet(points)
            for group in basis.groups
            for sr, se in group.values()
            if sr is not None
            for s, points in ((sr, sig), (se, eta))
        }

    def operator(self, j, kappa, g, omega, tail, field, device):
        # (1-sigma)^tail with analytic derivatives through curl curl.
        z = (1 - self.sig)[:, None]
        values = [z**tail]
        factor = 1.0
        for k in range(1, 4):
            factor *= -(tail - k + 1)
            values.append(factor * z ** (tail - k))
        envelope = self.x * Jet(tuple(values))
        dm = 1j * j * kappa
        freq = -1j * j * omega / (2 * g)
        sin_theta = np.sqrt(self.t.v[0])
        parts, shapes = [], []
        for name, (sr, se) in self.basis.groups[j].items():
            if sr is None:
                r, a = envelope, self.y
                shapes.append((1, 1))
            else:
                r, a = envelope * self.splines[sr], self.splines[se]
                shapes.append((sr.count, se.count))
            u, v, w = [], [], []
            if j:
                h = [(r * self.x, a * self.t * self.t)]
                if name == "A":
                    u = h
                    w = radial(u, lambda f: (-1 / dm) * (self.D(f, freq) + 2 * f))
                else:
                    v = h
                    w = angular(v, lambda f: (1 / dm) * (self.t * f).deriv())
            elif name == "T":
                w = [(r, a)]
            else:
                u = [(r, (-1) * (self.t * a).deriv())]
                v = [((-1) * (self.D(r, freq) + 2 * r), a)]
            fields = [u, v, w]
            if field in ("curl", "linear"):
                rot = self.curl(fields, freq, dm)
                if field == "curl":
                    fields = rot
                else:
                    cc = self.curl(rot, freq, dm)
                    fields = [
                        radial(
                            c,
                            lambda f: (
                                (1 - g - 1j * j * omega) * f + g * self.D(f, freq)
                            ),
                        )
                        + diff
                        for c, diff in zip(fields, cc, strict=True)
                    ]
            elif field != "velocity":
                raise ValueError(field)
            parts.append(
                [
                    [
                        (rr.v[0], aa.v[0] if i == 0 else sin_theta * aa.v[0])
                        for rr, aa in terms
                    ]
                    for i, terms in enumerate(fields)
                ]
            )
        return Operator(parts, shapes, device)


def dense_columns(operator):
    """Materialize a small modal block for the exact Newton derivative."""
    sample = next(
        pair for table in operator.tables for pair in table if pair is not None
    )
    nr, na = sample[0].shape[1], sample[1].shape[1]
    pieces = []
    for shape, table in zip(operator.shapes, operator.tables, strict=True):
        components = []
        for pair in table:
            if pair is None:
                col = sample[0].new_zeros((nr * na, math.prod(shape)))
            else:
                col = torch.einsum("tri,tak->raik", *pair).reshape(nr * na, -1)
            components.append(col)
        pieces.append(torch.stack(components, dim=1))
    return torch.cat(pieces, dim=2)


def signed(values, j):
    return values[j] if j >= 0 else values[-j].conj()


def pack(values):
    return np.concatenate(
        [
            b
            for j, v in enumerate(values)
            for b in ((v.real, v.imag) if j else (v.real,))
        ]
    )


def lift_coefficients(old, new, coeff):
    """Exact transfer into a nested half-space meridional and azimuthal space."""
    if (old.radius, old.phase_radius) != (
        new.radius,
        new.phase_radius,
    ) or new.harmonics < old.harmonics:
        raise ValueError(
            "Refinement must retain the original map, winding radius and angular modes"
        )

    def transfer(src, dst, edges):
        x, _ = gauss_elements(
            edges, (len(edges) - 1) * (max(old.degree, new.degree) + 3)
        )
        a, b = dst.spline(x), src.spline(x)
        P = la.lstsq(a, b)[0]
        if la.norm(a @ P - b) > 1e-10 * max(1.0, la.norm(b)):
            raise ValueError("The new space does not contain the original splines")
        return P

    if not set(old.sigma_breaks) <= set(new.sigma_breaks) or not set(
        old.eta_breaks
    ) <= set(new.eta_breaks):
        raise ValueError("Refinement must retain all original element boundaries")
    result = []
    for j, group in enumerate(new.groups):
        dest = []
        i = 0
        for name, (nr, na) in group.items():
            if j > old.harmonics:
                dest.extend(np.zeros(nr.count * na.count, complex))
            elif nr is None:
                dest.append(coeff[j][i])
                i += 1
            else:
                sr, sa = old.groups[j][name]
                n = sr.count * sa.count
                block = coeff[j][i : i + n].reshape(sr.count, sa.count)
                i += n
                dest.extend(
                    (
                        transfer(sr, nr, new.sigma_breaks)
                        @ block
                        @ transfer(sa, na, new.eta_breaks).T
                    ).ravel()
                )
        result.append(np.array(dest))
    return result


class WaveGeometry:
    def __init__(self, space, kappa, g, omega):
        self.space, self.kappa, self.g, self.omega = space, kappa, g, omega
        self.twist = omega / (2 * g)
        self.tail = (1 - g) / g
        if not (
            0 < self.tail < 200
            and self.tail > 0.5
            and np.isfinite(omega)
            and abs(omega) < 1e5
        ):
            raise ValueError(
                "Wave geometry exceeds the resolved integrable far-field range"
            )
        self.velocity, self.curl, self.linear = [], [], []
        for j in range(space.basis.harmonics + 1):
            for name in ("velocity", "curl", "linear"):
                getattr(self, name).append(
                    space.meridian.operator(
                        j, kappa, g, omega, self.tail, name, space.device
                    )
                )

    def fields(self, cs):
        return tuple(
            [op.apply(c) for op, c in zip(ops, cs, strict=True)]
            for ops in (self.velocity, self.curl, self.linear)
        )

    def raw(self, cs, all_modes=False):
        U, C, L = self.fields(cs)
        space = self.space
        n = space.phase_nodes

        # Use a real FFT in the single traveling phase. Quadratic products are dealiased.
        def samples(coeff):
            return torch.fft.irfft(torch.stack(coeff), n=n, dim=0, norm="forward")

        adv = torch.fft.rfft(
            torch.linalg.cross(samples(C), samples(U)), dim=0, norm="forward"
        )
        count = 2 * space.basis.harmonics + 1 if all_modes else len(U)
        raw = [adv[j] + L[j] if j < len(L) else adv[j] for j in range(count)]
        return U, C, L, raw


class WaveSpace:
    def __init__(self, basis, grid, device, kappa):
        if not math.isfinite(kappa) or kappa < 3:
            raise ValueError(
                "The reduced azimuthal wave number must be finite and at least three"
            )
        self.basis, self.grid, self.device, self.kappa = (
            basis,
            grid,
            torch.device(device),
            kappa,
        )
        sig, ws = gauss_elements(basis.sigma_breaks, grid)
        eta, wa = gauss_elements(basis.full_eta, grid)
        if (
            min(
                len(sig) // (len(basis.sigma_breaks) - 1),
                len(eta) // (len(basis.full_eta) - 1),
            )
            < basis.degree + 4
        ):
            raise ValueError("Too few meridional quadrature nodes per element")
        self.meridian = WaveMeridian(basis, sig, eta)
        self.wr = ws * basis.radius**3 * sig**2 / (1 - sig) ** 4
        self.wa = wa
        self.weight = self.tensor(self.wr[:, None] * wa)
        self.rho = basis.radius * sig / (1 - sig)
        self.log_radius = self.tensor(np.log1p((self.rho / basis.phase_radius) ** 2))[
            :, None, None
        ]
        self.core_weight = (
            self.weight
            * self.tensor(4 / (np.pi * 50**3) * (1 + (self.rho / 50) ** 2) ** -2)[
                :, None
            ]
        )
        # Sample all three coordinates within the repeating sector. The strong
        # residual also needs the quadratic products up to angular harmonic 2J.
        self.phase_nodes = max(4 * basis.harmonics + 1, grid)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float64)

    def audit(self, coefficients, g, omega, enrichment=1):
        geo = WaveGeometry(self, self.kappa, g, omega)
        cs = [
            torch.as_tensor(c, device=self.device, dtype=torch.complex128)
            for c in coefficients
        ]
        U, _, L, raw = geo.raw(cs, all_modes=True)
        residual2 = linear2 = nonlinear2 = outside2 = 0.0

        def norm2(j, v):
            return (
                (2 if j else 1)
                * 2
                * math.pi
                * float((self.weight[..., None] * v.abs().square()).sum())
            )

        for j, v in enumerate(raw):
            proj = PressureFit(
                self,
                1.0,
                0,
                j,
                enrichment,
                angular_order=self.kappa * j,
                radial_frequency=-1j * j * geo.twist,
            )
            err = norm2(j, v - proj.fit(v))
            residual2 += err
            if j < len(L):
                linear2 += norm2(j, L[j])
                nonlinear2 += norm2(j, v - L[j])
            else:
                outside2 += err
                nonlinear2 += norm2(j, v)
        h = math.sqrt(residual2)
        return {
            "grid": self.grid,
            "quadrature": [len(self.wr), len(self.wa), self.phase_nodes],
            "pressure_degree": self.basis.degree + enrichment,
            "residual_h": h,
            "relative_h": h / max(math.sqrt(linear2) + math.sqrt(nonlinear2), 1e-300),
            "outside_h": math.sqrt(outside2),
            "wave_rms": self.wave_rms(U[1]) if len(U) > 1 else 0.0,
        }

    def wave_rms(self, U1):
        return math.sqrt(
            2 * float((self.core_weight[..., None] * U1.abs().square()).sum())
        )


def figure_seed(space, geo, profile, mean_scale=1.0, wave_scale=1.0):
    """Fit the digitized northern mean and a completion of the first harmonic at phase zero."""
    mer = space.meridian
    rho = space.rho[:, None]
    eta = mer.eta[None, :]
    sn = np.sqrt(1 - eta**2)
    radius, z = rho * sn, rho * eta
    with np.load(Path(profile), allow_pickle=False) as data:
        rlim, zlim = data["r"][-1], data["z"][-1]
        ray = np.maximum.reduce([np.ones_like(radius), radius / rlim, z / zlim])
        coords = np.stack(
            [np.minimum(z / ray, zlim), np.minimum(radius / ray, rlim)], axis=-1
        )

        def sample(name):
            return (
                RegularGridInterpolator((data["z"], data["r"]), data[name])(coords)
                / ray**geo.tail
            )

        ur, uphi, uz = (
            sample(key) * mean_scale for key in ("mean_wr", "mean_wphi", "mean_wz")
        )
        mean = np.stack([sn * ur + eta * uz, eta * ur - sn * uz, uphi], axis=-1)
        env = sample("w19_modulus") * wave_scale
        slice_r = data["slice_r"]
        rflat = coords[..., 1]
        hr = np.interp(rflat, slice_r, data["slice_solenoidal_wr_19"])
        hp = np.interp(rflat, slice_r, data["slice_solenoidal_wphi_19"])
        envelope_slice = RegularGridInterpolator(
            (data["z"], data["r"]), data["w19_modulus"]
        )(np.stack([np.full_like(rflat, float(data["slice_z"])), rflat], axis=-1))
        horizontal = np.hypot(abs(hr), abs(hp))
        hscale = env / np.maximum.reduce(
            [horizontal, envelope_slice, np.full_like(env, 0.02)]
        )
        hr, hp = hr * hscale, hp * hscale
        # Vertical polarization is unknown. Give it the unused modulus, quadrature
        # phase relative to radial motion, and a smooth zero trace on the plane.
        hz = (
            1j
            * np.exp(1j * np.angle(hr))
            * np.sqrt(np.maximum(0, env**2 - abs(hr) ** 2 - abs(hp) ** 2))
            * np.tanh(z / 2)
        )
        wave = np.stack([sn * hr + eta * hz, eta * hr - sn * hz, hp], axis=-1)
    coeff = []
    for j, op in enumerate(geo.velocity):
        n = sum(math.prod(shape) for shape in op.shapes)
        if j > 1:
            coeff.append(np.zeros(n, complex))
            continue
        target = (
            mean
            if j == 0
            else wave
            * np.exp(-1j * geo.twist * np.log1p((rho / space.basis.phase_radius) ** 2))[
                ..., None
            ]
        )
        # Limit leverage of assumed exterior data. A ridge controls weakly observed columns.
        weights = (space.core_weight.cpu().numpy() / ray**3).ravel()
        X = dense_columns(op)
        w = space.tensor(np.sqrt(weights))[:, None]
        X = (X * w[..., None]).reshape(-1, n)
        y = (
            torch.as_tensor(
                target, device=space.device, dtype=torch.complex128
            ).reshape(-1, 3)
            * w
        )
        G = (X.mH @ X).cpu().numpy()
        rhs = (X.mH @ y.reshape(-1)).cpu().numpy()
        ridge = 1e-6 * np.max(np.linalg.eigvalsh(G))
        fitted = la.solve(G + ridge * np.eye(n), rhs, assume_a="pos")
        coeff.append(fitted.real.astype(complex) if not j else fitted)
    return coeff


class RotatingWave:
    """Fixed physical test functions and a trial space with the current far-field behavior."""

    def __init__(self, space, g, omega, coeff, amplitude=None):
        if space.basis.harmonics < 1:
            raise ValueError("A rotating wave requires a nonzero angular harmonic")
        self.space = space
        geo = WaveGeometry(space, space.kappa, g, omega)
        self.geo = geo
        self.reference_twist = geo.twist
        self.omega_scale = max(1.0, abs(omega))
        c = [
            torch.as_tensor(x.copy(), device=space.device, dtype=torch.complex128)
            for x in coeff
        ]
        initial = space.wave_rms(geo.velocity[1].apply(c[1]))
        self.amplitude = initial if amplitude is None else amplitude
        if initial <= 1e-12 or not 0 < self.amplitude < math.inf:
            raise ValueError("Supply a finite nonzero fundamental wave")
        c[1] *= self.amplitude / initial
        ref = [o.apply(x) for o, x in zip(geo.velocity, c, strict=True)]
        self.phase_reference = ref[1]
        scale = math.sqrt(
            sum(
                (2 if j else 1)
                * float((space.core_weight[..., None] * u.abs().square()).sum())
                for j, u in enumerate(ref)
            )
        )
        self.columns, self.projectors = [], []
        initial_state = []
        for j, (v, L, cj) in enumerate(zip(geo.velocity, geo.linear, c, strict=True)):
            test = space.meridian.operator(
                j, space.kappa, g, omega, 3, "velocity", space.device
            )
            V = dense_columns(v)
            gram = V.reshape(-1, V.shape[-1]).mH @ (
                V * space.core_weight.reshape(-1, 1, 1)
            ).reshape(-1, V.shape[-1])
            eig, Q = la.eigh(gram.cpu().numpy())
            col = scale * (Q / np.sqrt(np.maximum(eig, eig[-1] * 1e-12))) @ Q.conj().T
            linear = test.gram(L, space.wr, space.wa) @ col
            row_scale = 1 / la.norm(linear, axis=1)
            u, s, vh = la.svd(linear * row_scale[:, None])
            left = (
                (vh.conj().T / np.maximum(s, s[0] * 1e-10)) @ u.conj().T
            ) * row_scale
            T = dense_columns(test) * space.weight.reshape(-1, 1, 1)
            self.projectors.append(
                torch.as_tensor(left, device=space.device)
                @ T.reshape(-1, T.shape[-1]).mH
            )
            self.columns.append(torch.as_tensor(col, device=space.device))
            initial_state.append(la.solve(col, cj.cpu().numpy(), assume_a="her"))
        self.sizes = [len(cj) for cj in initial_state]
        self.start = np.r_[
            pack(initial_state), math.log(1 / g - 1.5), omega / self.omega_scale
        ]
        self.load = np.zeros(len(self.start) - 2)
        self.load = self.evaluate(self.start, 0.0)[:-2]

    def parameters(self, x):
        if not np.isfinite(x).all() or not -32 < x[-2] < 5.29:
            raise ValueError("Unresolved similarity tail")
        return 1 / (1.5 + math.exp(x[-2])), self.omega_scale * x[-1]

    def geometry(self, x):
        g, omega = self.parameters(x)
        if self.geo.g != g or self.geo.omega != omega:
            self.geo = WaveGeometry(self.space, self.space.kappa, g, omega)
        return self.geo

    def coefficients(self, x):
        out = []
        start = 0
        for j, (size, V) in enumerate(zip(self.sizes, self.columns, strict=True)):
            v = x[start : start + size] + 0j
            start += size
            if j:
                v += 1j * x[start : start + size]
                start += size
            out.append(V @ torch.as_tensor(v, device=self.space.device))
        return out

    def coordinates(self, coeff, g, omega):
        u = [
            la.solve(C.cpu().numpy(), c, assume_a="her")
            for C, c in zip(self.columns, coeff, strict=True)
        ]
        return np.r_[pack(u), math.log(1 / g - 1.5), omega / self.omega_scale]

    def retract(self, x):
        """Normalize the fundamental and fix its phase after a trial on the unforced manifold."""
        geo = self.geometry(x)
        U = geo.velocity[1].apply(self.coefficients(x)[1])
        rms = self.space.wave_rms(U)
        overlap = (
            (
                self.space.core_weight[..., None]
                * self.phase_reference.conj()
                * U
                * self.rotation(geo, 1)
            )
            .sum()
            .item()
        )
        if rms < 1e-12 * self.amplitude or abs(overlap) < 1e-14 * self.amplitude**2:
            raise ValueError("The phase slice or the fundamental amplitude is singular")
        angle = np.angle(overlap)
        y = x.copy()
        offset = self.sizes[0]
        for j, size in enumerate(self.sizes[1:], 1):
            c = x[offset : offset + size] + 1j * x[offset + size : offset + 2 * size]
            c *= np.exp(-1j * j * angle)
            if j == 1:
                c *= self.amplitude / rms
            y[offset : offset + size] = c.real
            y[offset + size : offset + 2 * size] = c.imag
            offset += 2 * size
        return y

    def rotation(self, geo, j):
        return torch.exp(
            (1j * j * (geo.twist - self.reference_twist)) * self.space.log_radius
        )

    def evaluate(self, x, forcing):
        geo = self.geometry(x)
        U, _, _, raw = geo.raw(self.coefficients(x))
        proj = [
            (P @ (self.rotation(geo, j) * raw[j]).reshape(-1)).cpu().numpy()
            for j, P in enumerate(self.projectors)
        ]
        w = 2 * self.space.core_weight[..., None] / self.amplitude**2
        amp = 0.5 * (float((w * U[1].abs().square()).sum()) - 1)
        phase = float(
            (w * self.phase_reference.conj() * U[1] * self.rotation(geo, 1)).sum().imag
        )
        return np.r_[pack(proj) - forcing * self.load, amp, phase]

    def jacobian(self, x, forcing):
        geo = self.geometry(x)
        U, C, _, _ = geo.raw(self.coefficients(x))
        U = [a.reshape(-1, 3) for a in U]
        C = [a.reshape(-1, 3) for a in C]

        def columns(ops):
            return [
                dense_columns(op) @ transform
                for op, transform in zip(ops, self.columns, strict=True)
            ]

        V, W = columns(geo.velocity), columns(geo.curl)
        J = self.space.basis.harmonics

        def block(j, mode, projector):
            k = j - mode
            width = self.sizes[abs(mode)]
            if abs(k) > J:
                return projector.new_zeros((self.sizes[j], width))
            # d(curl U x U) = curl(dU) x U + curl U x dU.
            t = torch.linalg.cross(
                signed(W, mode).transpose(1, 2), signed(U, k)[:, None, :], dim=-1
            )
            t += torch.linalg.cross(
                signed(C, k)[:, None, :], signed(V, mode).transpose(1, 2), dim=-1
            )
            h = t.transpose(1, 2)
            if mode == j:
                h = h + dense_columns(geo.linear[j]) @ self.columns[j]
            return projector @ h.reshape(-1, width)

        rows = []
        for j, P in enumerate(self.projectors):
            unit = (
                self.rotation(geo, j)
                .expand(len(self.space.wr), len(self.space.wa), 3)
                .reshape(-1)
            )
            projector = P * unit
            terms = [block(j, 0, projector)]
            for mode in range(1, J + 1):
                plus, minus = block(j, mode, projector), block(j, -mode, projector)
                terms.extend([plus + minus, 1j * (plus - minus)])
            row = torch.cat(terms, dim=1).cpu().numpy()
            rows.append(row.real)
            if j:
                rows.append(row.imag)
        N = len(x) - 2
        mat = np.zeros((N + 2, N + 2))
        mat[:N, :N] = np.vstack(rows)
        w = 2 * self.space.core_weight.reshape(-1, 1, 1) / self.amplitude**2
        a = (w * U[1].conj()[:, :, None] * V[1]).sum(dim=(0, 1)).cpu().numpy()
        physical_ref = self.phase_reference.conj() * self.rotation(geo, 1)
        p = (w * physical_ref.reshape(-1, 3, 1) * V[1]).sum(dim=(0, 1)).cpu().numpy()
        start = self.sizes[0]
        end = start + 2 * self.sizes[1]
        mat[-2, start:end] = np.r_[a.real, -a.imag]
        mat[-1, start:end] = np.r_[p.imag, p.real]
        for i in (-2, -1):
            dx = np.zeros_like(x)
            dx[i] = 1e-5 * max(1.0, abs(x[i]))
            mat[:, i] = (
                self.evaluate(x + dx, forcing) - self.evaluate(x - dx, forcing)
            ) / (2 * dx[i])
        return mat
