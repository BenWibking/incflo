"""Appendix B.1 meridional elements and Appendix C.2 similarity equations.

Fourier coefficients use Re(exp(i n chi) * (cos(m phi) or sin(m phi))),
chi = 2*pi*(theta - log(1+r**2/rc**2)/period). A radial phase shift at
each quadrature point preserves both angular integrals and mode products.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.special import roots_legendre


def breaks(values):
    values = tuple(float(x) for x in values)
    if (
        len(values) < 2
        or values[0] != 0
        or values[-1] != 1
        or not np.all(np.diff(values) > 0)
    ):
        raise ValueError("Element boundaries must increase strictly from 0 to 1")
    return values


def gauss_elements(edges, total):
    order = math.ceil(total / (len(edges) - 1))
    nodes, weights = roots_legendre(order)
    widths = np.diff(edges)[:, None]
    return (
        (np.asarray(edges[:-1])[:, None] + widths * (1 + nodes) / 2).ravel(),
        (widths * weights / 2).ravel(),
    )


class Splines:
    """A clamped piecewise-polynomial space, optionally restricted to a parity."""

    def __init__(self, edges, degree, continuity, parity=0, vanish_left=False):
        knots = np.repeat(
            edges,
            [degree + 1] + [degree - continuity] * (len(edges) - 2) + [degree + 1],
        )
        count = len(knots) - degree - 1
        transform = np.eye(count)
        if parity:
            # The negative-eta knots are the reflection of the positive-eta knots.
            transform = (transform + parity * transform[::-1])[
                :, : (count + (parity == 1)) // 2
            ]
            if count % 2 and parity == 1:
                transform[:, -1] /= 2
        if vanish_left:
            transform = transform[:, 1:]
        self.spline = BSpline(knots, transform, degree, extrapolate=False)
        self.count = transform.shape[1]

    def jet(self, points):
        return Jet(tuple(self.spline(points, nu=k) for k in range(4)))


@dataclass(frozen=True)
class Mode:
    n: int
    j: int
    sine: bool

    def parity(self, family):
        if self.j == 0:
            return -1
        return (-1 if self.sine else 1) * (1 if family == "A" else -1)

    def component_sine(self, component, curl=False):
        return self.j > 0 and bool(self.sine ^ (component == 2) ^ curl)

    @property
    def weight(self):
        return 2 * np.pi / ((2 if self.n else 1) * (2 if self.j else 1))


def cell_legendre(spline, interval, degree):
    """Map spline coefficients to local Legendre coefficients on one element."""
    x, w = roots_legendre(degree + 1)
    v = np.polynomial.legendre.legvander(x, degree)
    points = interval[0] + (interval[1] - interval[0]) * (x + 1) / 2
    return ((np.arange(degree + 1) + 0.5)[:, None] * v.T * w) @ spline(points)


def free_coordinates(rows, size):
    # Columns have a fixed order, independent of LAPACK pivoting or device choice.
    eps = 5e-12
    rows[np.abs(rows) < eps] = 0.0
    nonzero = np.max(np.abs(rows), axis=1) > 0
    a = rows[nonzero].copy()
    if not len(a):
        return np.arange(size), None
    a /= np.max(np.abs(a), axis=1)[:, None]
    pivots = []
    for col in range(size):
        k = len(pivots)
        if k == len(a):
            break
        pivot = k + np.argmax(np.abs(a[k:, col]))
        if abs(a[pivot, col]) <= eps:
            continue
        a[[k, pivot]] = a[[pivot, k]]
        a[k] /= a[k, col]
        elim = a[:, col].copy()
        elim[k] = 0.0
        a -= elim[:, None] * a[k]
        pivots.append(col)
    free = np.setdiff1d(np.arange(size), pivots)
    mat = np.eye(size)[:, free]
    mat[pivots, :] = -a[: len(pivots), free]
    if np.linalg.norm(rows @ mat, ord=np.inf) > 1e-8 * max(
        1, np.linalg.norm(rows, ord=np.inf)
    ):
        raise ValueError("Ill-conditioned element-degree constraints")
    return free, mat


@dataclass(frozen=True)
class PaperBasis:
    degree: int = 4
    sigma_breaks: tuple = (0, 0.25, 0.5, 0.75, 1)
    eta_breaks: tuple = (0, 0.25, 0.5, 0.75, 1)
    azimuthal: int = 2
    temporal: int = 2
    radius: float = 10.0
    phase_radius: float = 60.0
    cell_degrees: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "sigma_breaks", breaks(self.sigma_breaks))
        object.__setattr__(self, "eta_breaks", breaks(self.eta_breaks))
        entries = []
        seen = set()
        for record in self.cell_degrees:
            if len(record) != 6 or not all(type(v) is int for v in record):
                raise ValueError(
                    "Each cell-degree entry needs six integers [n,j,ring,interval,p_sigma,p_eta]"
                )
            n, j, i, k, pr, pa = record
            if not (
                0 <= n <= self.temporal
                and 0 <= j <= self.azimuthal
                and (n - j) % 2 == 0
                and 0 <= i < len(self.sigma_breaks) - 1
                and 0 <= k < len(self.eta_breaks) - 1
                and (
                    (pr, pa) == (-1, -1)
                    or (0 <= pr <= self.degree and 0 <= pa <= self.degree)
                )
            ):
                raise ValueError(
                    f"Invalid harmonic, cell or polynomial degrees in {record}"
                )
            if (n, j, i, k) in seen:
                raise ValueError("Duplicate harmonic and cell in degree map")
            seen.add((n, j, i, k))
            entries.append(tuple(record))
        object.__setattr__(self, "cell_degrees", tuple(sorted(entries)))
        if self.degree < 3 or min(self.azimuthal, self.temporal) < 1:
            raise ValueError("Use at least cubic elements and one traveling harmonic")
        if not all(
            math.isfinite(v) and v > 0 for v in (self.radius, self.phase_radius)
        ):
            raise ValueError(
                "The compactification and phase radii must be positive and finite"
            )

    @cached_property
    def full_eta(self):
        return tuple(-x for x in self.eta_breaks[:0:-1]) + self.eta_breaks

    @cached_property
    def full_modes(self):
        return [
            Mode(n, j, sine)
            for n in range(self.temporal + 1)
            for j in range(n % 2, self.azimuthal + 1, 2)
            for sine in ([False, True] if j else [False])
        ]

    @cached_property
    def modes(self):
        return [m for m in self.full_modes if len(self.free[m])]

    @cached_property
    def restrictions(self):
        # Identify independent full-spline coefficients with a deterministic elimination.
        # The free entries themselves are the persisted coordinates of the restricted space.
        out = {}
        for m, group in self.families.items():
            records = [row for row in self.cell_degrees if row[:2] == (m.n, m.j)]
            sizes = self.sizes[m]
            total = sum(sizes)
            if not records:
                out[m] = (np.arange(total), None)
                continue
            offsets = np.cumsum([0, *sizes])
            rows = []
            for _, _, i, k, pr, pa in records:
                for index, (name, (r, a)) in enumerate(group.items()):
                    if name == "gamma":
                        continue
                    xr, xe = self.sigma_breaks[i : i + 2], self.eta_breaks[k : k + 2]
                    cr = cell_legendre(r.spline, xr, self.degree)
                    ce = cell_legendre(a.spline, xe, self.degree)
                    forbidden = np.indices((self.degree + 1, self.degree + 1))
                    mask = (forbidden[0] > pr) | (forbidden[1] > pa)
                    if not mask.any():
                        continue
                    local = np.kron(cr, ce)[mask.ravel()]
                    block = np.zeros((len(local), total))
                    block[:, offsets[index] : offsets[index + 1]] = local
                    if name == "S" and pr == -1:
                        # An absent axisymmetric cell must eliminate the full stream
                        # function gamma*eta + S, including its origin-gradient term.
                        gamma = np.zeros(mask.shape)
                        gamma[0, :2] = (sum(xe) / 2, (xe[1] - xe[0]) / 2)
                        block[:, -1] = gamma[mask]
                    rows.append(block)
            out[m] = (
                free_coordinates(np.vstack(rows), total)
                if rows
                else (np.arange(total), None)
            )
        return out

    @cached_property
    def free(self):
        return {m: entry[0] for m, entry in self.restrictions.items()}

    def expand(self, mode, coeff):
        transform = self.restrictions[mode][1]
        return coeff if transform is None else transform @ coeff

    @cached_property
    def families(self):
        # The derivatives joined at each interior knot are exactly those in B.1.
        joins = {"A": (2, 1), "B": (1, 2), "S": (2, 2), "T": (1, 1)}
        families = {}
        for mode in self.full_modes:
            group = {}
            for name in ("A", "B") if mode.j else ("S", "T"):
                sr, se = joins[name]
                group[name] = (
                    Splines(self.sigma_breaks, self.degree, sr, vanish_left=not mode.j),
                    Splines(self.full_eta, self.degree, se, mode.parity(name)),
                )
            if not mode.j:
                group["gamma"] = (None, None)
            families[mode] = group
        return families

    @cached_property
    def sizes(self):
        return {
            m: [r.count * a.count if r is not None else 1 for r, a in group.values()]
            for m, group in self.families.items()
        }

    def labels(self):
        labels = []
        for m in self.modes:
            local = [
                (name, k)
                for name, size in zip(self.families[m], self.sizes[m], strict=True)
                for k in range(size)
            ]
            for part in range(2 if m.n else 1):
                for i in self.free[m]:
                    name, k = local[i]
                    labels.append((m.n, name, 19 * m.j, int(m.sine), k, part))
        return labels

    @property
    def block_frequencies(self):
        return [m.n for m in self.modes]

    def coefficients(self, vector):
        start = 0
        for mode in self.modes:
            size = len(self.free[mode])
            c = vector[start : start + size] + 0j
            start += size
            if mode.n:
                c = c + 1j * vector[start : start + size]
                start += size
            yield c


class Jet:
    """Ordinary derivatives of a univariate table through third order."""

    def __init__(self, values):
        self.v = values

    @classmethod
    def variable(cls, points):
        x = points[:, None]
        return cls((x, np.ones_like(x), np.zeros_like(x), np.zeros_like(x)))

    def __add__(self, other):
        if isinstance(other, Jet):
            return Jet(tuple(x + y for x, y in zip(self.v, other.v)))
        return Jet((self.v[0] + other, *self.v[1:]))

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Jet):
            return Jet(tuple(other * x for x in self.v))
        return Jet(
            tuple(
                sum(math.comb(k, j) * self.v[j] * other.v[k - j] for j in range(k + 1))
                for k in range(min(len(self.v), len(other.v)))
            )
        )

    __rmul__ = __mul__

    def inv(self):
        values = [1 / self.v[0]]
        for k in range(1, len(self.v)):
            values.append(
                -values[0]
                * sum(
                    math.comb(k, j) * self.v[j] * values[k - j] for j in range(1, k + 1)
                )
            )
        return Jet(tuple(values))

    def deriv(self):
        return Jet(self.v[1:])


# A term is the outer product of radial and meridional tables of basis functions.
# Keep them separated even after differentiation. A block never stores a 3D mesh per coefficient.
def radial(terms, f):
    return [(f(r), a) for r, a in terms]


def angular(terms, f):
    return [(r, f(a)) for r, a in terms]


class Meridian:
    def __init__(self, basis, sig, eta):
        self.basis = basis
        self.sig, self.eta = sig, eta
        self.x, self.y = Jet.variable(sig), Jet.variable(eta)
        self.z = 1 + (-1) * self.x
        self.t = 1 + (-1) * self.y * self.y
        self.dx = self.x * self.z
        self.ir = self.z * self.x.inv() * (1 / basis.radius)
        # r*d_r applied to the unwound harmonic contributes i*n*omega*phase_slope.
        a, b = self.x * basis.radius, self.z * basis.phase_radius
        self.phase_slope = (-2) * a * a * (a * a + b * b).inv()

    def D(self, r, frequency):
        return self.dx * r.deriv() + frequency * self.phase_slope * r

    def curl(self, fields, frequency, dm):
        u, v, w = fields
        # Components represent (u, sin(theta)*v, sin(theta)*w). The curl
        # exchanges the sine and cosine sectors. Imposing div=0 uses the same frame.
        c0 = angular(w, lambda a: (-1) * (self.t * a).deriv())
        c1 = radial(w, lambda r: (-1) * (self.D(r, frequency) + r))
        if dm:
            c0 += radial(v, lambda r: -dm * r)
            c1 += angular(u, lambda a: dm * a * self.t.inv())
        c2 = radial(v, lambda r: self.D(r, frequency) + r) + angular(
            u, lambda a: a.deriv()
        )
        return [radial(c, lambda r: self.ir * r) for c in (c0, c1, c2)]

    def columns(self, mode, name, period, tail, field, family=()):
        freq = 2j * np.pi * mode.n / period
        sign = 1 if mode.sine else -1
        spl_r, spl_a = self.basis.families[mode][name]
        if name == "gamma":
            r, a = self.x * 0 + 1, self.y  # g(sigma)=1 is an admissible B.8 choice
        else:
            r, a = spl_r.jet(self.sig), spl_a.jet(self.eta)
        r = r * self.x * self.z
        for _ in range(tail - 1):
            r = r * self.z
        u, v, w = [], [], []
        if mode.j:
            data = [(r * self.x, a * self.t * self.t)]
            if name == "A":
                u = data
                w = radial(
                    u, lambda q: (1 / (sign * 19 * mode.j)) * (self.D(q, freq) + 2 * q)
                )
            else:
                v = data
                w = angular(
                    v, lambda q: (-1 / (sign * 19 * mode.j)) * (self.t * q).deriv()
                )
        elif name == "T":
            w = [(r, a)]
        else:
            u = [(r, (-1) * (self.t * a).deriv())]
            v = [((-1) * (self.D(r, freq) + 2 * r), a)]
        velocity = [u, v, w]
        if field == "velocity":
            out = velocity
        elif field == "dilation":
            out = [radial(c, lambda q: self.D(q, freq)) for c in velocity]
        else:
            vort = self.curl(velocity, freq, sign * 19 * mode.j)
            if field == "curl":
                out = vort
            else:
                curlcurl = self.curl(vort, freq, -sign * 19 * mode.j)
                if field == "curlcurl":
                    out = curlcurl
                elif field == "linear":
                    g, nu, damping = family
                    mass = 1 - g + freq + (damping if mode.j else 0)
                    out = [
                        radial(c, lambda q: mass * q + g * self.D(q, freq))
                        + radial(k, lambda q: nu * q)
                        for c, k in zip(velocity, curlcurl)
                    ]
                else:
                    raise ValueError(f"Unknown field {field}")
        sn = np.sqrt(self.t.v[0])
        return [
            [(r.v[0], a.v[0] if i == 0 else sn * a.v[0]) for r, a in c]
            for i, c in enumerate(out)
        ]


def compressed(terms):
    """Combine terms with the same meridional factor before tensor contractions."""
    grouped = {}
    for r, a in terms:
        key = (a.shape, a.tobytes())
        if key in grouped:
            grouped[key] = (grouped[key][0] + r, a)
        else:
            grouped[key] = (r, a)
    return list(grouped.values())


class Operator:
    """Vector-valued tensor-product maps for a single angular harmonic."""

    def __init__(self, parts, shapes, device, transform=None):
        self.transform = transform
        self.embedding = (
            None
            if transform is None
            else torch.as_tensor(transform, device=device, dtype=torch.complex128)
        )
        self.parts = [[compressed(c) for c in field] for field in parts]
        self.shapes = shapes
        self.tables = []
        for field in self.parts:
            table = []
            for c in field:
                if c:
                    table.append(
                        tuple(
                            torch.as_tensor(
                                np.stack(arr), device=device, dtype=torch.complex128
                            )
                            for arr in zip(*c)
                        )
                    )
                else:
                    table.append(None)
            self.tables.append(table)

    def apply(self, vector):
        if self.embedding is not None:
            vector = self.embedding @ vector
        components = [None] * 3
        offset = 0
        for shape, table in zip(self.shapes, self.tables):
            size = math.prod(shape)
            c = vector[offset : offset + size].reshape(shape)
            offset += size
            for i, pair in enumerate(table):
                if pair:
                    r, a = pair
                    v = (r @ c @ a.mT).sum(dim=0)
                    components[i] = v if components[i] is None else components[i] + v
        template = next(v for v in components if v is not None)
        return torch.stack(
            [torch.zeros_like(template) if v is None else v for v in components], dim=-1
        )

    def adjoint(self, samples):
        coeff = []
        for shape, table in zip(self.shapes, self.tables):
            v = samples.new_zeros(shape)
            for i, pair in enumerate(table):
                if pair:
                    r, a = pair
                    v = v + (r.mH @ samples[..., i].contiguous() @ a.conj()).sum(dim=0)
            coeff.append(v.flatten())
        values = torch.cat(coeff)
        return values if self.embedding is None else self.embedding.mH @ values

    def gram(self, other, wr, wa):
        rows = []
        for test, shape in zip(self.parts, self.shapes):
            row = []
            for trial, trial_shape in zip(other.parts, other.shapes):
                b = np.zeros(
                    (math.prod(shape), math.prod(trial_shape)), dtype=np.complex128
                )
                for tc, uc in zip(test, trial):
                    for r, a in tc:
                        rt, at = r.conj().T * wr, a.conj().T * wa
                        for q, e in uc:
                            b += np.kron(rt @ q, at @ e)
                row.append(b)
            rows.append(row)
        mat = np.block(rows)
        if other.transform is not None:
            mat = mat @ other.transform
        if self.transform is not None:
            mat = self.transform.T @ mat
        return mat


def make_operator(meridian, mode, period, tail, field, device, family=()):
    group = meridian.basis.families[mode]
    shapes = [(r.count, a.count) if r else (1, 1) for r, a in group.values()]
    fields = [
        meridian.columns(mode, name, period, tail, field, family) for name in group
    ]
    return Operator(fields, shapes, device, meridian.basis.restrictions[mode][1])


class Factors(NamedTuple):
    velocity: Operator
    curl: Operator
    test: Operator
    strong: Operator
    linear: torch.Tensor


class EquilibratedBlock:
    """Apply a reference weak block inverse by a row- and column-scaled LU solve."""

    def __init__(self, matrix):
        self.rows = torch.linalg.vector_norm(matrix, dim=1).reciprocal()
        scaled = self.rows[:, None] * matrix
        self.columns = torch.linalg.vector_norm(scaled, dim=0).reciprocal()
        scaled = scaled * self.columns
        if not torch.isfinite(scaled).all():
            raise ValueError(
                "The element linear block has a zero or nonfinite row or column"
            )
        self.condition = float(torch.linalg.cond(scaled))
        self.lu, self.pivots = torch.linalg.lu_factor(scaled)

    def solve(self, rhs):
        rhs = (self.rows * rhs)[:, None]
        return self.columns * torch.linalg.lu_solve(self.lu, self.pivots, rhs)[:, 0]


class PaperSpace:
    krylov_type = "fgmres"
    pressure_enrichment = 1

    def __init__(self, basis, grid, device="cpu"):
        self.basis, self.grid, self.device = basis, grid, torch.device(device)
        sig, ws = gauss_elements(basis.sigma_breaks, grid)
        eta, wa = gauss_elements(basis.full_eta, grid)
        if (
            min(
                len(sig) // (len(basis.sigma_breaks) - 1),
                len(eta) // (len(basis.full_eta) - 1),
            )
            < basis.degree + 3
        ):
            raise ValueError("Increase quadrature to resolve each meridional rectangle")
        self.meridian = Meridian(basis, sig, eta)
        self.wr = ws * basis.radius**3 * sig**2 / (1 - sig) ** 4
        self.wa = wa
        self.weight = self.tensor(self.wr[:, None] * wa)
        self.labels = basis.labels()
        # Retain both signs of every Fourier mode of the quadratic product.
        self.nt = 4 * basis.temporal + 1
        self.wedge = max(4 * basis.azimuthal + 1, math.ceil(grid / 19))
        self._factors = {}

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float64, device=self.device)

    def pack(self, parts):
        out = []
        for mode, c in zip(self.basis.modes, parts, strict=True):
            out.append(c.real)
            if mode.n:
                out.append(c.imag)
        return torch.cat(out)

    def unpack(self, vector):
        return self.basis.coefficients(vector)

    def factors(self, period, *, dilation=0.5, viscosity=1.0, damping=0.0):
        key = (period, dilation, viscosity, damping)
        if key in self._factors:
            return self._factors[key]
        if not math.isfinite(period) or period <= 0:
            raise ValueError("Period must be positive and finite")
        blocks = []
        for mode in self.basis.modes:

            def op(tail, field, mode=mode):
                return make_operator(
                    self.meridian, mode, period, tail, field, self.device, key[1:]
                )

            v, w, test, strong = (
                op(1, "velocity"),
                op(1, "curl"),
                op(3, "velocity"),
                op(1, "linear"),
            )
            mat = mode.weight * test.gram(strong, self.wr, self.wa)
            blocks.append(
                Factors(v, w, test, strong, torch.as_tensor(mat, device=self.device))
            )
        if len(self._factors) >= 4:
            del self._factors[next(iter(self._factors))]
        self._factors[key] = blocks
        return blocks

    def spectrum(self, coeff, factors, field):
        result = torch.zeros(
            (self.nt, self.wedge, *self.weight.shape, 3),
            device=self.device,
            dtype=torch.complex128,
        )
        for mode, c, block in zip(
            self.basis.modes, self.unpack(coeff), factors, strict=True
        ):
            u = getattr(block, field).apply(c) / (2 if mode.n else 1)
            for j in [mode.j, -mode.j] if mode.j else [0]:
                if j:
                    s = 1 if j > 0 else -1
                    factors_phi = [
                        (-1j * s if mode.component_sine(i, field == "curl") else 1) / 2
                        for i in range(3)
                    ]
                    a = u * u.new_tensor(factors_phi)
                else:
                    a = u
                result[mode.n, j] = result[mode.n, j] + a
                if mode.n:
                    result[-mode.n, -j] = result[-mode.n, -j] + a.conj()
        return result

    def physical(self, spectrum):
        return torch.fft.ifft2(spectrum, dim=(0, 1), norm="forward").real.contiguous()

    def fields(self, vector, factors):
        return tuple(
            self.physical(self.spectrum(vector, factors, f))
            for f in ("velocity", "curl")
        )

    def nonlinear(self, fields, background=None):
        v, w = fields
        if background is None:
            adv = torch.linalg.cross(w, v)
        else:
            base_v, base_w = background
            adv = torch.linalg.cross(w, base_v) + torch.linalg.cross(base_w, v)
        return torch.fft.fft2(adv, dim=(0, 1), norm="forward")

    def residual(self, vector, factors, fields=None, background=None):
        adv = self.nonlinear(
            self.fields(vector, factors) if fields is None else fields, background
        )
        result = []
        for mode, c, block in zip(
            self.basis.modes, self.unpack(vector), factors, strict=True
        ):
            a = adv[mode.n, mode.j]
            if mode.j:
                b = adv[mode.n, -mode.j]
                a = torch.stack(
                    [
                        1j * (a[..., i] - b[..., i])
                        if mode.component_sine(i)
                        else a[..., i] + b[..., i]
                        for i in range(3)
                    ],
                    dim=-1,
                )
            test = block.test.adjoint(a * self.weight[..., None])
            result.append(block.linear @ c + (2 if mode.n else 1) * mode.weight * test)
        return self.pack(result)

    def linear_inverse(self, factors):
        return [EquilibratedBlock(b.linear) for b in factors]

    def scale_residual(self, residual, inverse):
        return self.pack(
            [
                block.solve(c)
                for block, c in zip(inverse, self.unpack(residual), strict=True)
            ]
        )

    def seed(self, amplitude, wave, random_seed=0, jitter=0):
        # Deterministic smooth coefficient patterns plus reproducible independent perturbations.
        # S, T already have odd parity. The cosine and sine sectors give distinct traveling arms.
        rng = np.random.default_rng(random_seed)
        out = []
        for m in self.basis.modes:
            v = []
            for name, (sr, se) in self.basis.families[m].items():
                if sr is None:
                    a = np.array([0.4])
                else:
                    rr = np.linspace(0, 1, sr.count)[:, None]
                    ee = np.linspace(0, 1, se.count)[None, :]
                    a = (
                        np.exp(-3 * (rr - 0.45) ** 2 - 3 * (ee - 0.6) ** 2)
                        * (-0.6 if name in ("S", "B") else 1)
                    ).ravel()
                if (m.n, m.j) == (1, 1):
                    a = a * wave * (1j if m.sine else 1)
                elif (m.n, m.j) != (0, 0):
                    a = 0 * a
                v.extend(a)
            c = np.array(v, dtype=complex)
            noise = rng.normal(size=c.size) + (
                1j * rng.normal(size=c.size) if m.n else 0
            )
            c = amplitude * (c + jitter * noise / math.sqrt(len(self.labels)))
            transform = self.basis.restrictions[m][1]
            if transform is not None:
                c = np.linalg.lstsq(transform, c, rcond=None)[0]
            out.extend([c.real, c.imag] if m.n else [c.real])
        return np.concatenate(out)

    def rotate_time(self, vector):
        return self.pack(
            [
                1j * m.n * c
                for m, c in zip(self.basis.modes, self.unpack(vector), strict=True)
            ]
        )

    def strong_norm(self, vector, period, enrichment=1, detail=False):
        """Fit each scalar pressure mode to the *pointwise* unforced C.1 residual."""
        if enrichment < 0:
            raise ValueError("Pressure enrichment must be nonnegative")
        factors = self.factors(period)
        spectrum = self.nonlinear(self.fields(vector, factors)) + self.spectrum(
            vector, factors, "strong"
        )
        raw_sq = residual_sq = outside_sq = 0.0
        rows = []
        rings, half = len(self.basis.sigma_breaks) - 1, len(self.basis.eta_breaks) - 1
        active = {(m.n, m.j) for m in self.basis.modes}
        omitted = {
            (n, j, i, k) for n, j, i, k, pr, pa in self.basis.cell_degrees if pr == -1
        }
        for n in range(2 * self.basis.temporal + 1):
            for j in range(-2 * self.basis.azimuthal, 2 * self.basis.azimuthal + 1):
                if (n - j) % 2 or (n == 0 and j < 0):
                    continue
                # Negative n (and negative j at n=0) are the complex conjugates.
                weight = 2 * np.pi * (2 if n or j else 1)
                q = spectrum[n, j]
                grad = PressureFit(self, period, n, j, enrichment)
                difference = q - grad.fit(q)
                density = weight * self.weight * difference.abs().square().sum(dim=-1)
                # Integrate each rectangle and its southern reflection.
                cells = density.reshape(
                    rings, len(self.wr) // rings, 2 * half, len(self.wa) // (2 * half)
                ).sum(dim=(1, 3))
                north = (
                    (cells[:, :half].flip((1,)) + cells[:, half:])
                    .detach()
                    .cpu()
                    .numpy()
                )
                square = float(north.sum())
                raw_sq += weight * self.norm_squared(q)
                residual_sq += square
                if (n, abs(j)) not in active:
                    outside_sq += square
                elif omitted:
                    outside_sq += sum(
                        north[i, k]
                        for nn, jj, i, k in omitted
                        if (nn, jj) == (n, abs(j))
                    )
                if detail:
                    rows.append(
                        {
                            "n": n,
                            "m": 19 * j,
                            "residual_h_squared": square,
                            "cell_h_squared": north.tolist(),
                        }
                    )
        result = {
            "residual_h": math.sqrt(residual_sq),
            "raw_h": math.sqrt(raw_sq),
            "outside_velocity_h": math.sqrt(outside_sq),
        }
        if detail:
            result.update(
                pressure_degree=self.basis.degree + enrichment, harmonics=rows
            )
        return result

    def norm_squared(self, samples):
        return float(torch.sum(self.weight[..., None] * samples.abs().square()))


class PressureFit:
    """Weighted least squares on continuous element pressures with square-integrable gradients."""

    def __init__(
        self,
        space,
        period,
        n,
        j,
        enrichment,
        *,
        angular_order=None,
        radial_frequency=None,
    ):
        self.space = space
        mer = space.meridian
        basis = space.basis
        angular_order = 19 * j if angular_order is None else angular_order
        radial_frequency = (
            2j * np.pi * n / period if radial_frequency is None else radial_frequency
        )
        degree = basis.degree + enrichment
        if (
            min(
                len(mer.sig) // (len(basis.sigma_breaks) - 1),
                len(mer.eta) // (len(basis.full_eta) - 1),
            )
            < degree + 1
        ):
            raise ValueError("Increase quadrature to resolve the enriched pressure fit")
        # Every pressure decays at infinity (fixing its constant). At the origin the
        # m=0 trace may be constant in eta; all other traces must vanish (B.1).
        sr = Splines(basis.sigma_breaks, degree, 0)
        se = Splines(basis.full_eta, degree, 0)
        pairs = [
            (
                sr.jet(mer.sig) * mer.x * mer.z * mer.z,
                se.jet(mer.eta) * (mer.t if j else 1),
            )
        ]
        if not j:
            pairs.append((mer.z * mer.z, mer.y * 0 + 1))
        parts = []
        sn = np.sqrt(mer.t.v[0])
        for r, a in pairs:
            ir = mer.ir.v[0]
            cols = [
                [(ir * mer.D(r, radial_frequency).v[0], a.v[0])],
                [(ir * r.v[0], -sn * a.deriv().v[0])],
                [(ir * r.v[0], 1j * angular_order * a.v[0] / sn)] if j else [],
            ]
            parts.append(cols)
        self.op = Operator(
            parts, [(r.v[0].shape[1], a.v[0].shape[1]) for r, a in pairs], space.device
        )
        gram = self.op.gram(self.op, space.wr, space.wa)
        self.scale = space.tensor(1 / np.sqrt(gram.diagonal().real))
        mat = (
            torch.as_tensor(gram, device=space.device)
            * self.scale[:, None]
            * self.scale
        )
        self.chol = torch.linalg.cholesky((mat + mat.mH) / 2)

    def fit(self, samples):
        rhs = self.scale * self.op.adjoint(samples * self.space.weight[..., None])
        coeff = self.scale * torch.cholesky_solve(rhs[:, None], self.chol)[:, 0]
        return self.op.apply(coeff)


def sample_paper(basis, coeff, period, phase, points, field="velocity", tail=1):
    """Cartesian evaluation independent of the search quadrature, including poles and origin."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    phase = np.broadcast_to(phase, (len(points),))
    result = np.zeros_like(points)
    for start in range(0, len(points), 2048):
        sub = slice(start, start + 2048)
        xyz = points[sub]
        radius = np.linalg.norm(xyz, axis=1)
        off = radius > 0
        rho = radius[off]
        if not len(rho):
            continue
        unit = xyz[off] / rho[:, None]
        eta = np.clip(unit[:, 2], -1, 1)
        phi = np.arctan2(unit[:, 1], unit[:, 0])
        chi = (
            2
            * np.pi
            * (phase[sub][off] - np.log1p((rho / basis.phase_radius) ** 2) / period)
        )
        spherical = np.zeros_like(unit)
        for mode, band in zip(basis.modes, basis.coefficients(coeff), strict=True):
            band = basis.expand(mode, band)
            active = np.abs(eta) < 1 if mode.j else np.ones_like(eta, dtype=bool)
            if not np.any(active):
                continue
            mer = Meridian(basis, (rho / (basis.radius + rho))[active], eta[active])
            offset = 0
            clock = np.exp(1j * mode.n * chi[active])
            angle = 19 * mode.j * phi[active]
            for name, (sr, se) in basis.families[mode].items():
                shape = (sr.count, se.count) if sr else (1, 1)
                size = math.prod(shape)
                c = band[offset : offset + size].reshape(shape)
                offset += size
                cols = mer.columns(mode, name, period, tail, field)
                for i, terms in enumerate(cols):
                    osc = (
                        np.sin(angle)
                        if mode.component_sine(i, field == "curl")
                        else np.cos(angle)
                    )
                    value = sum(
                        np.sum((r @ c) * a, axis=1) for r, a in compressed(terms)
                    )
                    spherical[active, i] += (clock * osc * value).real
        ct, st = np.cos(phi), np.sin(phi)
        polar = np.column_stack((eta * ct, eta * st, -np.hypot(unit[:, 0], unit[:, 1])))
        azimuth = np.column_stack((-st, ct, np.zeros_like(st)))
        result[sub][off] = (
            spherical[:, 0, None] * unit
            + spherical[:, 1, None] * polar
            + spherical[:, 2, None] * azimuth
        )
    return result


def lift_paper(vector, old, new):
    """Transfer represented piecewise polynomials exactly into a nested element space."""
    if (
        (old.radius, old.phase_radius) != (new.radius, new.phase_radius)
        or new.degree < old.degree
        or not (
            set(old.sigma_breaks) <= set(new.sigma_breaks)
            and set(old.eta_breaks) <= set(new.eta_breaks)
            and set(old.modes) <= set(new.modes)
        )
    ):
        raise ValueError(
            "A paper lift requires nested knots, harmonics and degrees with the same radii"
        )
    source = {
        m: old.expand(m, c)
        for m, c in zip(old.modes, old.coefficients(vector), strict=True)
    }

    def transform(a, b, edges):
        pts, _ = gauss_elements(edges, (len(edges) - 1) * (new.degree + 2))
        return np.linalg.lstsq(b.spline(pts), a.spline(pts), rcond=None)[0]

    result = []
    for m in new.modes:
        c = np.zeros(sum(new.sizes[m]), dtype=complex)
        if m in source:
            i = k = 0
            for name, (r, a) in old.families[m].items():
                R, A = new.families[m][name]
                if r is None:
                    c[k] = source[m][i]
                    k += 1
                    i += 1
                else:
                    size, new_size = r.count * a.count, R.count * A.count
                    old_c = source[m][i : i + size].reshape(r.count, a.count)
                    c[k : k + new_size] = (
                        transform(r, R, new.sigma_breaks)
                        @ old_c
                        @ transform(a, A, new.full_eta).T
                    ).ravel()
                    i += size
                    k += new_size
        reduced = c[new.free[m]]
        if not np.allclose(
            new.expand(m, reduced), c, rtol=2e-9, atol=2e-10 * max(1, np.linalg.norm(c))
        ):
            raise ValueError("The new cell degrees cannot represent the lifted field")
        result.extend([reduced.real, reduced.imag] if m.n else [reduced.real])
    return np.concatenate(result)
