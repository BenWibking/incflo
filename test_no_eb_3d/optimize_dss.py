"""Search for a nonzero return of incflo's physical-time DSS map."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import tempfile
import time as clock
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize

S = 0.36
T_END = -math.expm1(-S)
HALF_BOX = 96.0
MIN_SEARCH_GRID = 128
DEFAULT_SEARCH_STEPS = 96


def modeled_seed(n):
    """Average the discrete curl of the modeled vector potential onto cell centers."""
    dx = 2 * HALF_BOX / n
    q = (np.arange(n) + 0.5) * dx - HALF_BOX
    centers = (q[None, None, :], q[None, :, None], q[:, None, None])

    def bump(v):
        return np.exp(-v / np.maximum(1 - v, 1.0e-10)) * (v < 1)

    def potential(pos, d):
        x, y, z = pos
        r2 = x * x + y * y
        f = bump(r2 / 64**2 + z * z / 40**2)
        if d == 0:
            return -y * z * f
        if d == 1:
            return x * z * f
        dr = (np.sqrt(r2) - 36) / 14
        arm = bump(dr * dr + z * z / 20**2)
        return z * (160 * f + 6 * arm * np.cos(19 * np.arctan2(y, x) - dr * z / 5))

    def shift(pos, d, offset):
        return tuple(v + offset if k == d else v for k, v in enumerate(pos))

    def derivative(face, axis, component):
        return (
            potential(shift(face, axis, dx / 2), component)
            - potential(shift(face, axis, -dx / 2), component)
        ) / dx

    result = []
    for d in range(3):
        face = shift(centers, d, -dx / 2)
        a, b = (d + 1) % 3, (d + 2) % 3

        v = derivative(face, a, b) - derivative(face, b, a)
        result.append(0.5 * (v + np.roll(v, -1, axis=2 - d)))
    return np.stack(result)


def load_seed(path, n=None):
    if path.is_dir():
        import yt

        yt.set_log_level(40)
        ds = yt.load(str(path), hint="boxlib")
        if (
            ds.index.max_level
            or not np.allclose(ds.domain_left_edge, -HALF_BOX)
            or not np.allclose(ds.domain_right_edge, HALF_BOX)
        ):
            raise ValueError("Expected the single-level seed on [-96,96]^3")
        grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        fields = {name: (kind, name) for kind, name in ds.field_list}
        u = np.stack([np.asarray(grid[fields["velocity_" + c]]).T for c in "xyz"])
    else:
        u = np.load(path, allow_pickle=False)
    if (
        u.ndim != 4
        or u.shape[0] != 3
        or len(set(u.shape[1:])) != 1
        or not np.isfinite(u).all()
    ):
        raise ValueError("Expected finite (3,nz,ny,nx) cubic seed")
    if n and n != u.shape[1]:
        source_n = u.shape[1]
        j = (np.arange(n) + 0.5) * source_n / n - 0.5
        coords = np.meshgrid(j, j, j, indexing="ij")
        u = np.stack(
            [
                map_coordinates(v, coords, order=1, mode="grid-wrap", prefilter=False)
                for v in u
            ]
        )
    return np.ascontiguousarray(u, dtype=np.float64)


def search_seed(path, n):
    if n < MIN_SEARCH_GRID:
        raise ValueError(
            f"Optimization requires at least {MIN_SEARCH_GRID} cells per dimension"
        )
    return modeled_seed(n) if path is None else load_seed(path, n)


class CylindricalModes:
    """Bilinear cylindrical amplitudes with 19-fold symmetry and the equatorial rotation."""

    def __init__(self, seed, nr, nz, harmonics):
        n = seed.shape[1]
        z, y, x = np.indices((n, n, n), dtype=float)
        dx = 2 * HALF_BOX / n
        x, y, z = [(v + 0.5) * dx - HALF_BOX for v in (x, y, z)]
        r = np.hypot(x, y)
        self.mask = (r < 64) & (np.abs(z) < 40)
        self.core = x * x + y * y + z * z <= 50**2
        self.shape = seed.shape
        r, z, phi = r[self.mask], z[self.mask], np.arctan2(y[self.mask], x[self.mask])
        self.c, self.s = np.cos(phi), np.sin(phi)
        fr = r / 64 * (nr - 1)
        fz = np.abs(z) / 40 * (nz - 1)
        ir, iz = fr.astype(int), fz.astype(int)
        self.nodes = []
        for dr in (0, 1):
            for dz in (0, 1):
                w = (fr - ir if dr else 1 - fr + ir) * (fz - iz if dz else 1 - fz + iz)
                self.nodes.append(((iz + dz) * nr + ir + dr, w))
        taper = np.sin(np.pi * r / 64) ** 2 * (1 + np.cos(np.pi * z / 40)) / 2
        odd = z / np.sqrt(z * z + (40 / (nz - 1)) ** 2)
        amplitude = float(np.max(np.abs(seed)))
        self.w = []
        for m in [0, *harmonics]:
            for sine in [False, True] if m else [False]:
                ang = np.sin(m * phi) if sine else np.cos(m * phi)
                factor = amplitude * taper * ang
                # Under the half turn about x, phi,z -> -phi,-z and (ur,uphi,uz) -> (ur,-uphi,-uz).
                self.w.append(
                    tuple(
                        factor * (odd if ((comp != 0) ^ sine) else 1)
                        for comp in range(3)
                    )
                )
        self.nparam = (len(self.w), 3, nr * nz)
        self.anchors = []
        rotated = seed[:, :, ::-1, ::-1].copy()
        rotated[:2] *= -1
        # Fix separate core overlaps with the seed's even and odd azimuthal parts.
        # The odd part carries its spiral. The zero field cannot satisfy both constraints.
        for ref in ((seed + rotated) / 2, (seed - rotated) / 2):
            ref[:, ~self.core] = 0
            vec = self.project(self.transpose(ref))
            length = np.linalg.norm(vec)
            if length < 1.0e-12:
                raise ValueError("Seed must have both mean and spiral components")
            self.anchors.append(vec / length)

    def project(self, c):
        c = np.array(c, copy=True)
        for a in self.anchors:
            c -= a * np.vdot(a, c)
        return c

    def forward(self, c):
        c = self.project(c).reshape(self.nparam)
        cyl = np.zeros((3, self.c.size))
        for h, weights in enumerate(self.w):
            for a in range(3):
                for idx, w in self.nodes:
                    cyl[a] += weights[a] * w * c[h, a, idx]
        out = np.zeros(self.shape)
        out[:, self.mask] = [
            self.c * cyl[0] - self.s * cyl[1],
            self.s * cyl[0] + self.c * cyl[1],
            cyl[2],
        ]
        return out

    def transpose(self, g):
        gx, gy, gz = g[:, self.mask]
        cyl = (self.c * gx + self.s * gy, -self.s * gx + self.c * gy, gz)
        c = np.zeros(self.nparam)
        for h, weights in enumerate(self.w):
            for a in range(3):
                v = cyl[a] * weights[a]
                for idx, w in self.nodes:
                    c[h, a] += np.bincount(
                        idx, weights=w * v, minlength=self.nparam[-1]
                    )
        return c.ravel()


class Cycle:
    def __init__(self, exe, deck, n, steps):
        self.exe, self.deck = exe.resolve(), deck.resolve()
        self.n, self.steps = n, steps

    def run(self, u, work, gradient, plots=False):
        n = self.n
        raw = work / "velocity.bin"
        np.ascontiguousarray(u, dtype=np.float64).tofile(raw)
        cmd = [
            str(self.exe),
            str(self.deck),
            "dss.enabled=1",
            f"dss.gradient={int(gradient)}",
            "dss.control=velocity.bin",
            f"amr.n_cell={n} {n} {n}",
            f"amr.max_grid_size={n}",
            "amr.plot_per_exact=-1",
            f"amr.plot_int={self.steps if plots else -1}",
            "incflo.init_shrink=1",
            "incflo.initial_iterations=0",
            "incflo.verbose=0",
            f"incflo.fixed_dt={T_END / self.steps:.17g}",
            f"stop_time={T_END:.17g}",
            f"max_step={self.steps}",
        ]
        with (work / "incflo.log").open("w") as log:
            completed = subprocess.run(
                cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            tail = (work / "incflo.log").read_text(errors="replace").splitlines()[-30:]
            raise RuntimeError(
                f"incflo exited {completed.returncode} in {work}\n" + "\n".join(tail)
            )
        val, err, time, steps = np.loadtxt(work / "dss_result.txt")
        if steps != self.steps or abs(time - T_END) > 1.0e-12 or not np.isfinite(val):
            raise RuntimeError(f"Invalid or incomplete cycle in {work}")
        g = None
        if gradient:
            g = np.fromfile(work / "dss_gradient.bin", dtype=np.float64).reshape(
                u.shape
            )
            if not np.isfinite(g).all():
                raise RuntimeError("Nonfinite velocity gradient")
        return val, err, g


def check(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(319)
    n = 16
    u = 30 * gaussian_filter(
        rng.normal(size=(3, n, n, n)), sigma=(0, 1, 1, 1), mode="wrap"
    )
    cycle = Cycle(args.exe, args.inputs, n, 4)
    root = args.out.resolve()
    j, err, g = cycle.run(u, root, True)
    rows = []
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        for seed in (134, 135, 136):
            rng = np.random.default_rng(seed)
            d = gaussian_filter(
                rng.normal(size=u.shape), sigma=(0, 1, 1, 1), mode="wrap"
            )
            d /= np.sqrt(np.mean(d * d))
            ad = float(np.vdot(g, d))
            for h in (1.0e-5, 1.0e-6):
                plus = cycle.run(u + h * d, Path(tmp), False)[0]
                minus = cycle.run(u - h * d, Path(tmp), False)[0]
                fd = (plus - minus) / (2 * h)
                rows.append(
                    {
                        "seed": seed,
                        "h": h,
                        "adjoint": ad,
                        "central": fd,
                        "relative": abs(ad - fd) / max(abs(ad), abs(fd), 1.0e-14),
                    }
                )
    (root / "check.json").write_text(
        json.dumps({"objective": j, "return_error": err, "checks": rows}, indent=2)
        + "\n"
    )
    print(json.dumps(rows), flush=True)
    if max(r["relative"] for r in rows if r["h"] == 1.0e-6) > 5.0e-4:
        raise SystemExit("Directional derivative check failed")


def search(args):
    if min(args.nr, args.nz) < 3 or min(args.steps, args.iterations) < 1:
        raise ValueError(
            "Need at least three coefficient nodes and positive steps and iterations"
        )
    if len(set(args.harmonics)) != len(args.harmonics) or any(
        m <= 0 or m % 19 for m in args.harmonics
    ):
        raise ValueError("Choose distinct positive multiples of 19 as harmonics")
    seed = search_seed(args.seed, args.n)
    modes = CylindricalModes(seed, args.nr, args.nz, args.harmonics)
    cycle = Cycle(args.exe, args.inputs, args.n, args.steps)
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.out.resolve()
    rng = np.random.default_rng(args.random_seed)
    x0 = modes.project(rng.normal(size=np.prod(modes.nparam)))
    norm = np.linalg.norm(modes.forward(x0)[:, modes.core])
    x0 *= args.jitter * np.linalg.norm(seed[:, modes.core]) / norm
    first = seed + modes.forward(x0)
    initial_hash = hashlib.sha256(first.tobytes()).hexdigest()
    # Independently test the coefficient-space transpose, including the overlap constraints.
    t = rng.normal(size=seed.shape)
    b = rng.normal(size=x0.shape)
    if not np.isclose(
        np.vdot(modes.forward(b), t),
        np.vdot(b, modes.project(modes.transpose(t))),
        atol=1.0e-8,
        rtol=1.0e-11,
    ):
        raise RuntimeError("Cylindrical coefficient transpose test failed")
    rows = []
    with (
        tempfile.TemporaryDirectory(dir=root) as tmp,
        (root / "evaluations.csv").open("w") as stream,
    ):
        writer = csv.writer(stream)
        writer.writerow(
            ["evaluation", "objective", "relative_return", "gradient_l2", "seconds"]
        )

        def fun(c):
            started = clock.monotonic()
            j, err, g = cycle.run(seed + modes.forward(c), Path(tmp), True)
            grad = modes.project(modes.transpose(g))
            rows.append(
                (
                    len(rows) + 1,
                    j,
                    err,
                    np.linalg.norm(grad),
                    clock.monotonic() - started,
                )
            )
            writer.writerow(rows[-1])
            stream.flush()
            print(
                f"seed {args.random_seed}, evaluation {len(rows)}, return {err:.8g}",
                flush=True,
            )
            return j, grad

        result = minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": args.iterations,
                "maxls": 12,
                "maxcor": 8,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
            },
        )
    accepted = seed + modes.forward(result.x)
    final_dir = root / "candidate"
    final_dir.mkdir()
    j, err, _ = cycle.run(accepted, final_dir, False, plots=args.plots)
    if not np.isclose(j, result.fun, rtol=1.0e-7, atol=1.0e-10):
        raise RuntimeError("Independent forward replay disagrees with optimizer")
    np.savez(root / "coefficients.npz", initial=x0, final=result.x)
    report = {
        "random_seed": args.random_seed,
        "initial_sha256": initial_hash,
        "grid": args.n,
        "steps": args.steps,
        "nr": args.nr,
        "nz": args.nz,
        "harmonics": args.harmonics,
        "jitter": args.jitter,
        "coefficients": x0.size,
        "median_gradient_seconds": float(np.median([r[4] for r in rows])),
        "first_return": rows[0][2],
        "final_return": err,
        "candidate_norm_ratio": float(
            np.linalg.norm(accepted[:, modes.core])
            / np.linalg.norm(seed[:, modes.core])
        ),
        "iterations": int(result.nit),
        "evaluations": len(rows),
        "termination": str(result.message),
    }
    pending = root / "search.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(root / "search.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("output", type=Path)
    generate.add_argument("--n", type=int, default=MIN_SEARCH_GRID)
    pack = sub.add_parser("pack")
    pack.add_argument("plotfile", type=Path)
    pack.add_argument("output", type=Path)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--exe", type=Path, required=True)
    base.add_argument(
        "--inputs", type=Path, default=Path(__file__).with_name("inputs.dss_cycle")
    )
    base.add_argument("--out", type=Path, required=True)
    sub.add_parser("check", parents=[base])
    opt = sub.add_parser("search", parents=[base])
    opt.add_argument(
        "--seed",
        type=Path,
        help="Override the modeled seed generated on the search grid",
    )
    opt.add_argument("--random-seed", type=int, required=True)
    opt.add_argument("--n", type=int, default=MIN_SEARCH_GRID)
    opt.add_argument("--steps", type=int, default=DEFAULT_SEARCH_STEPS)
    opt.add_argument("--nr", type=int, default=9)
    opt.add_argument("--nz", type=int, default=7)
    opt.add_argument("--harmonics", type=int, nargs="+", default=[19])
    opt.add_argument("--iterations", type=int, default=5)
    opt.add_argument("--jitter", type=float, default=0.12)
    opt.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.action == "generate":
        if args.n < 8:
            parser.error("Use at least 8 cells per dimension")
        with args.output.open("xb") as out:
            np.save(out, modeled_seed(args.n))
    elif args.action == "pack":
        with args.output.open("xb") as out:
            np.save(out, load_seed(args.plotfile))
    elif args.action == "check":
        check(args)
    else:
        search(args)
