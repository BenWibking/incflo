"""Continue space-time Galerkin solutions to the unforced backward similarity equations."""

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, gmres
from similarity_galerkin import Basis, OrbitSpace, sample_cartesian, transfer


def host(a):
    return a.detach().cpu().numpy()


@dataclass(frozen=True)
class PseudoTimePath:
    """The modeled fixed-m=19 path through the forced similarity equations of C.3."""

    decay_time: float = 1.0
    initial_g: float = 0.502
    initial_viscosity: float = 1.0
    initial_damping: float = 0.1

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()) or not (
            self.decay_time > 0
            and self.initial_viscosity > 0
            and self.initial_damping >= 0
        ):
            raise ValueError(
                "Invalid pseudo-time decay scale, dilation, viscosity or damping"
            )

    def parameters(self, strength):
        return {
            "dilation": 0.5 + (self.initial_g - 0.5) * strength,
            "viscosity": (1.0 - strength) + self.initial_viscosity * strength,
            "damping": self.initial_damping * strength,
        }

    def advance_clock(self, log_strength, step):
        # Eliminate the backward-Euler row for d(lambda)/d(tau) = -lambda / decay_time.
        log_ratio = math.log(step) - math.log(self.decay_time)
        return log_strength - float(np.logaddexp(0.0, log_ratio))


class OrbitEquation:
    def __init__(self, space, seed, period, path=None):
        self.space, self.period = space, period
        self.path = path
        self.family = path.parameters(1.0) if path else {}
        self.scale = np.linalg.norm(seed)
        if not 0 < self.scale < np.inf:
            raise ValueError("The reference field must be finite and nonzero")
        self.ref = seed / self.scale
        tangent = host(space.rotate_time(space.tensor(self.ref)))
        if not np.isfinite(tangent).all() or np.linalg.norm(tangent) < 1e-12:
            raise ValueError(
                "The reference needs a time-dependent wave to fix its phase"
            )
        self.phase = tangent / np.linalg.norm(tangent)
        self.wave = np.array([n > 0 and m == 19 for n, _, m, _, _, _ in space.labels])
        if np.linalg.norm(self.ref[self.wave]) == 0:
            raise ValueError("Provide a nonzero time-dependent m=19 spiral")
        factors = space.factors(period)
        self.linear_condition = max(
            float(torch.linalg.cond(f[-1]).max()) for f in factors
        )
        if not self.linear_condition < 1 / np.finfo(float).eps:
            raise ValueError(
                "The radial linear block is numerically singular; reduce its degree or use Jacobi modes"
            )
        self.reference_linear = [host(f[-1]) for f in factors]
        self.inv = space.linear_inverse(factors)
        self.force = self.fluid(self.ref, 0.0)
        if path:
            axis = np.array([n == 0 and m == 0 for n, _, m, _, _, _ in space.labels])
            # A manufactured steady axisymmetric base and a finite-amplitude wave imprint.
            self.axis_force = np.where(axis, self.fluid(self.ref * axis, 0.0), 0.0)
            self.wave_force = self.force - self.axis_force

    def set_strength(self, strength):
        self.family = self.path.parameters(strength)
        self.force = self.axis_force + strength * self.wave_force

    def factors(self, period):
        return self.space.factors(period, **self.family)

    def scaled(self, raw):
        return host(self.space.scale_residual(raw, self.inv)) / self.scale

    def fluid(self, x, log_period):
        factors = self.factors(self.period * math.exp(log_period))
        return self.scaled(
            self.space.residual(self.space.tensor(self.scale * x), factors)
        )

    def residual(self, y):
        x, log_s, forcing = y[:-2], y[-2], y[-1]
        return np.r_[
            self.fluid(x, log_s) - forcing * self.force, self.phase @ (x - self.ref)
        ]

    def jacobian(self, y):
        return OrbitJacobian(self, y)

    def metrics(self, y, raw=None, target_raw=None):
        x = y[:-2]
        raw = self.fluid(x, y[-2]) if raw is None else raw
        if target_raw is None:
            if self.path:
                factors = self.space.factors(self.period * math.exp(y[-2]))
                target_raw = self.scaled(
                    self.space.residual(self.space.tensor(self.scale * x), factors)
                )
            else:
                target_raw = raw
        return {
            "period": self.period * math.exp(y[-2]),
            "forcing": float(y[-1]),
            "projected_residual": float(np.linalg.norm(raw - y[-1] * self.force)),
            "unforced_projected_residual": float(np.linalg.norm(target_raw)),
            "unforced_relative_defect": float(
                np.linalg.norm(target_raw) / max(np.linalg.norm(x), 1e-300)
            ),
            "relative_field": float(np.linalg.norm(x)),
            "relative_spiral": float(
                np.linalg.norm(x[self.wave]) / np.linalg.norm(self.ref[self.wave])
            ),
            "phase_error": float(abs(self.phase @ (x - self.ref))),
        }


class OrbitJacobian:
    """Derivative of the reference-scaled equations, including the free period."""

    def __init__(self, eq, point):
        self.eq = eq
        x, log_s = point[:-2], point[-2]
        self.factors = eq.factors(eq.period * math.exp(log_s))
        self.background = eq.space.fields(eq.space.tensor(eq.scale * x), self.factors)
        eps = 1e-5
        self.period_column = (eq.fluid(x, log_s + eps) - eq.fluid(x, log_s - eps)) / (
            2 * eps
        )

    def field(self, dx):
        eq = self.eq
        raw = eq.space.residual(eq.scale * dx, self.factors, background=self.background)
        return eq.space.scale_residual(raw, eq.inv) / eq.scale

    def __call__(self, dy):
        eq = self.eq
        fluid = host(self.field(eq.space.tensor(dy[:-2])))
        return np.r_[
            fluid + self.period_column * dy[-2] - eq.force * dy[-1], eq.phase @ dy[:-2]
        ]

    def transpose(self, rhs):
        # Only methods requesting J^T pay for a reverse pass through the linear field map.
        eq = self.eq
        with torch.enable_grad():
            dx = eq.space.tensor(np.zeros_like(eq.ref)).requires_grad_()
            (dual,) = torch.autograd.grad(self.field(dx), dx, eq.space.tensor(rhs[:-1]))
        return np.r_[
            host(dual) + eq.phase * rhs[-1],
            self.period_column @ rhs[:-1],
            -eq.force @ rhs[:-1],
        ]


class CorrectorSystem:
    """The same augmented nonlinear equations for every corrector backend."""

    def __init__(self, eq, predictor, normal, deflate, shift=0.0):
        self.shift = shift
        self.eq, self.predictor, self.normal, self.deflate = (
            eq,
            predictor,
            normal,
            deflate,
        )

    def multiplier(self, point):
        derivative = np.zeros_like(point)
        if not self.deflate:
            return 1.0, derivative
        part = point[:-2][self.eq.wave]
        norm2 = part @ part
        strength = (self.eq.ref[self.eq.wave] @ self.eq.ref[self.eq.wave]) / max(
            norm2, 1e-28
        )
        if norm2 > 1e-28:
            derivative[:-2][self.eq.wave] = -2 * strength / norm2 * part
        return 1 + strength, derivative

    def residual(self, point):
        residual = self.eq.residual(point)
        residual[:-1] *= self.multiplier(point)[0]
        if self.shift:
            residual[:-1] += self.shift * (point[:-2] - self.predictor[:-2])
        return np.r_[residual, self.normal @ (point - self.predictor)]

    def linearize(self, point, residual):
        weight, derivative = self.multiplier(point)
        steady = residual[:-2]
        if self.shift:
            steady = steady - self.shift * (point[:-2] - self.predictor[:-2])
        return CorrectorJacobian(
            self.eq.jacobian(point),
            self.normal,
            weight,
            derivative,
            steady / weight,
            self.shift,
        )


class CorrectorJacobian:
    def __init__(self, orbit, normal, weight, derivative, fluid_residual, shift=0.0):
        self.shift = shift
        self.orbit, self.normal = orbit, normal
        self.weight, self.derivative, self.fluid_residual = (
            weight,
            derivative,
            fluid_residual,
        )

    def __call__(self, dy):
        delta = self.orbit(dy)
        delta[:-1] = self.weight * delta[:-1] + self.fluid_residual * (
            self.derivative @ dy
        )
        if self.shift:
            delta[:-1] += self.shift * dy[:-2]
        return np.r_[delta, self.normal @ dy]

    def transpose(self, rhs):
        result = self.orbit.transpose(np.r_[self.weight * rhs[:-2], rhs[-2]])
        if self.shift:
            result[:-2] += self.shift * rhs[:-2]
        return (
            result
            + self.derivative * (self.fluid_residual @ rhs[:-2])
            + self.normal * rhs[-1]
        )


def correct(eq, predictor, normal, tolerance, iterations, krylov, *, deflate=False):
    return correct_system(
        CorrectorSystem(eq, predictor, normal, deflate), tolerance, iterations, krylov
    )


def correct_system(system, tolerance, iterations, krylov):
    y = system.predictor.copy()
    f = system.residual(y)
    linear_counts = []
    for iteration in range(iterations + 1):
        norm = np.linalg.norm(f)
        if np.isfinite(norm) and norm <= tolerance:
            return y, linear_counts, True
        if iteration == iterations or not np.isfinite(norm):
            break
        jac = system.linearize(y, f)
        op = LinearOperator((y.size, y.size), matvec=jac, dtype=np.float64)
        preconditioner = None
        if system.shift:
            from similarity_petsc import BorderedStokes

            preconditioner = LinearOperator(
                op.shape, matvec=BorderedStokes(jac), dtype=np.float64
            )
        history = []
        step, info = gmres(
            op,
            -f,
            M=preconditioner,
            atol=norm * min(0.08, max(1e-5, math.sqrt(norm) * 0.15)),
            restart=min(y.size, krylov),
            maxiter=1,
            callback=history.append,
            callback_type="pr_norm",
        )
        linear_counts.append(len(history))
        # An inexact Krylov step is usable if backtracking reduces the full residual.
        if info < 0 or not np.isfinite(step).all():
            break
        damping = min(1.0, 1.0 / max(abs(step[-2]), 1.0))
        for _ in range(12):
            trial = y + damping * step
            ahead = system.residual(trial)
            if (
                np.isfinite(ahead).all()
                and np.linalg.norm(ahead) <= (1 - 1e-4 * damping) * norm
            ):
                y, f = trial, ahead
                break
            damping *= 0.5
        else:
            break
    return y, linear_counts, False


def atomic(path, write, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb" if binary else "w", dir=path.parent, delete=False
    ) as tmp:
        try:
            write(tmp)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda f: json.dump(value, f, indent=2, allow_nan=False), binary=False)


def checkpoint(path, config, seed, y, previous, step, accepted):
    atomic(
        path,
        lambda f: np.savez(
            f,
            schema=2,
            config=json.dumps(config),
            seed=seed,
            state=y,
            previous=previous,
            step=step,
            accepted=accepted,
        ),
    )


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as a:
        if int(a["schema"]) not in (1, 2):
            raise ValueError("Unsupported orbit checkpoint")
        cfg = json.loads(str(a["config"]))
        if int(a["schema"]) == 1:
            cfg["basis"].setdefault("radial_family", "legendre")
        basis = Basis(**cfg["basis"])
        count = sum(
            2 * len(basis.modes(n)) * basis.radial * (2 if n else 1)
            for n in range(basis.temporal + 1)
        )
        seed, state, prev = (a[k].copy() for k in ("seed", "state", "previous"))
        step, accepted = float(a["step"]), int(a["accepted"])
        if (
            seed.shape != (count,)
            or state.shape != (count + 2,)
            or prev.shape not in ((0,), state.shape)
            or not all(np.isfinite(v).all() for v in (seed, state, prev))
            or not 0 < step < np.inf
            or not 0 < cfg["period"] < np.inf
            or accepted < 0
        ):
            raise ValueError("Malformed orbit checkpoint")
        return cfg, seed, state, prev, step, accepted


def backend(device):
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("The GPU backend requires a visible ROCm or CUDA device")
        return "cuda"
    return "cpu"


def run_search(args):
    source = (
        load_checkpoint(args.restart or args.lift)
        if args.restart or args.lift
        else None
    )
    saved_path = source[0].get("ptc_path") if args.restart else None
    use_ptc = args.pseudo_transient or saved_path is not None
    requested_step = args.step
    args.step = 0.2 if args.step is None else args.step
    args.solver = args.solver or ("petsc" if use_ptc else "scipy")
    path_values = {k: getattr(args, "ptc_" + k) for k in asdict(PseudoTimePath())}
    supplied = {k: v for k, v in path_values.items() if v is not None}
    if not use_ptc and (supplied or args.ptc_max_step is not None):
        raise ValueError("Pseudo-time parameters require --pseudo-transient")
    path = None
    if use_ptc:
        if args.direct:
            raise ValueError(
                "Use --lift to start a separate direct solve from a pseudo-time checkpoint"
            )
        if args.restart and saved_path is None:
            raise ValueError(
                "Use --lift with a Newton checkpoint to initialize a new pseudo-time path"
            )
        path = PseudoTimePath(**{**(saved_path or {}), **supplied})
        if saved_path is not None and asdict(path) != saved_path:
            raise ValueError(
                "Use --lift to change the physics of a saved pseudo-time path"
            )
        if args.ptc_max_step is None:
            args.ptc_max_step = source[0]["ptc_max_step"] if args.restart else 100.0
        if not math.isfinite(args.ptc_max_step) or args.ptc_max_step < args.min_step:
            raise ValueError(
                "The maximum pseudo-time step must be finite and at least --min-step"
            )
    original = Basis(**source[0]["basis"]) if source else Basis()
    basis = Basis(
        **{
            k: getattr(args, k) if getattr(args, k) is not None else v
            for k, v in asdict(original).items()
        }
    )
    args.grid = (
        args.grid if args.grid is not None else (source[0]["grid"] if source else 128)
    )
    if args.grid < 128:
        raise ValueError("The minimum search quadrature is 128 per spatial direction")
    if (
        not 0 < args.min_step <= args.step
        or args.max_stages < 1
        or args.newton < 1
        or args.krylov < 1
        or not 0 < args.tol < 1
        or not 0 < args.min_spiral < 1
    ):
        raise ValueError("Invalid continuation or Newton settings")
    if args.solver == "petsc":
        from similarity_petsc import petsc_correct, petsc_runtime

        petsc_runtime()
    elif args.snes_type or args.pc:
        raise ValueError("--snes-type and --pc require --solver petsc")
    dev = backend(args.device)
    if args.restart:
        cfg, seed, y, previous, step, accepted = source
        if args.grid != cfg["grid"] or basis != original:
            raise ValueError("Use --lift to continue on a different grid or basis")
        space = OrbitSpace(Basis(**cfg["basis"]), args.grid, dev)
        if use_ptc:
            if requested_step is not None:
                step = requested_step
        else:
            step = min(step, args.step)
    else:
        if (
            not 0 < args.period < np.inf
            or not 0 < args.amplitude < np.inf
            or not 0 < args.wave < np.inf
            or not 0 <= args.jitter < np.inf
        ):
            raise ValueError("Invalid seed amplitudes or period")
        space = OrbitSpace(basis, args.grid, dev)
        cfg = {
            "basis": asdict(basis),
            "grid": args.grid,
            "period": args.period,
            "viscosity": 1.0,
            "amplitude": args.amplitude,
            "wave": args.wave,
            "random_seed": args.seed,
            "jitter": args.jitter,
        }
        seed = space.seed(args.amplitude, args.wave, args.seed, args.jitter)
        if args.lift:
            old_cfg, old_seed, old_y, *_ = source
            old_basis = Basis(**old_cfg["basis"])
            if (old_basis.radius, old_basis.phase_radius) != (
                basis.radius,
                basis.phase_radius,
            ):
                raise ValueError(
                    "A modal lift requires the original radial map and phase radius"
                )
            if basis.radial_family != old_basis.radial_family:
                raise ValueError(
                    "A coefficient lift requires the same radial polynomial family"
                )
            if not set(old_basis.labels()).issubset(space.labels):
                raise ValueError(
                    "A modal lift must retain every original harmonic and radial degree"
                )
            seed = transfer(
                old_y[:-2] * np.linalg.norm(old_seed), old_basis.labels(), space.labels
            )
            cfg["period"] = old_cfg["period"] * math.exp(old_y[-2])
            cfg["lifted_from"] = str(args.lift)
            for unused in ("amplitude", "wave", "random_seed", "jitter"):
                del cfg[unused]
        y, previous, step, accepted = (
            np.r_[seed / np.linalg.norm(seed), 0.0, 1.0],
            np.empty(0),
            args.step,
            0,
        )
    if path:
        cfg["ptc_path"] = asdict(path)
        cfg["ptc_max_step"] = args.ptc_max_step
        if not args.restart:
            cfg.update(pseudo_time=0.0, log_forcing=0.0)
        if not (
            math.isfinite(cfg["pseudo_time"])
            and cfg["pseudo_time"] >= 0
            and math.isfinite(cfg["log_forcing"])
            and cfg["log_forcing"] <= 0
            and math.isclose(
                y[-1], math.exp(cfg["log_forcing"]), rel_tol=1e-12, abs_tol=0.0
            )
        ):
            raise ValueError("Inconsistent pseudo-time clock in checkpoint")
        step = min(step, args.ptc_max_step)
    eq = OrbitEquation(space, seed, cfg["period"], path)
    if path:
        eq.set_strength(y[-1])
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "config": cfg,
        "unknown_field_coefficients": len(seed),
        "max_linear_block_condition": eq.linear_condition,
        "quadrature": [args.grid, args.grid, space.wedge * 19],
        "phase_nodes": space.nt,
        "device": dev,
        "hardware": torch.cuda.get_device_name() if dev == "cuda" else "CPU",
        "hip": torch.version.hip,
        "solver": args.solver,
        "stages": [],
        "rejected_steps": 0,
    }
    initial_accepted = accepted

    def correct_point(predictor, normal, deflate, shift=0.0):
        system = CorrectorSystem(eq, predictor, normal, deflate, shift)
        if args.solver == "scipy":
            return correct_system(system, args.tol, args.newton, args.krylov)

        def monitor(it, norm, linear_steps):
            print(
                f"SNES iteration={it} residual={norm:.8g} total_ksp={linear_steps}",
                flush=True,
            )

        candidate, counts, converged, info = petsc_correct(
            system,
            args.tol,
            args.newton,
            args.krylov,
            method=args.snes_type or "newtonls",
            preconditioner=args.pc or "bordered",
            monitor=monitor,
        )
        info.update(starting_forcing=float(predictor[-1]), deflated=bool(deflate))
        if shift:
            info["mass_shift"] = shift
        report.setdefault("petsc_correctors", []).append(info)
        return candidate, counts, converged

    def save(status, metrics=None):
        report.update(
            status=status,
            accepted=accepted,
            elapsed_seconds=time.monotonic() - start,
            current=eq.metrics(y) if metrics is None else metrics,
        )
        checkpoint(args.out / "orbit.npz", cfg, seed, y, previous, step, accepted)
        write_json(args.out / "search.json", report)

    save("running")
    if path:
        normal = np.zeros_like(y)
        normal[-1] = 1.0
        metrics = report["current"]

        def limit_status():
            if (
                cfg["log_forcing"] > math.log(args.tol)
                or math.hypot(
                    metrics["unforced_projected_residual"], metrics["phase_error"]
                )
                > args.tol
            ):
                return None
            if (
                min(metrics["relative_spiral"], metrics["relative_field"])
                < args.min_spiral
            ):
                return "limit_below_amplitude_floor"
            if metrics["unforced_relative_defect"] <= args.tol:
                return "unforced_projected_root"
            return None

        while accepted - initial_accepted < args.max_stages and limit_status() is None:
            h = step
            log_next = path.advance_clock(cfg["log_forcing"], h)
            strength = math.exp(log_next)
            next_time = cfg["pseudo_time"] + h
            if strength == 0 or not cfg["pseudo_time"] < next_time < math.inf:
                save("pseudo_time_precision_limit", metrics)
                return
            predictor = y.copy()
            predictor[-1] = strength
            eq.set_strength(strength)
            candidate, counts, converged = correct_point(
                predictor, normal, False, shift=1.0 / h
            )
            if converged:
                candidate[-1] = strength
                # Verify the implicit step after enforcing the independently solved clock row.
                regularized = CorrectorSystem(eq, predictor, normal, False, 1.0 / h)
                be_norm = float(np.linalg.norm(regularized.residual(candidate)))
                met = eq.metrics(candidate)
                converged = be_norm <= args.tol and all(
                    math.isfinite(v) for v in met.values()
                )
            if not converged:
                report["rejected_steps"] += 1
                report["last_rejection"] = {
                    "pseudo_time": cfg["pseudo_time"] + h,
                    "dt": h,
                    "reason": "implicit_solve",
                }
                eq.set_strength(y[-1])
                if h / 2 < args.min_step:
                    save("pseudo_time_stalled", metrics)
                    return
                step = h / 2
                continue
            previous, y = y, candidate
            accepted += 1
            cfg["pseudo_time"] = next_time
            cfg["log_forcing"] = log_next
            metrics = met
            if min(met["relative_field"], met["relative_spiral"]) < args.min_spiral:
                report.setdefault(
                    "first_below_amplitude_floor",
                    {
                        "pseudo_time": next_time,
                        "forcing": strength,
                        "relative_field": met["relative_field"],
                        "relative_spiral": met["relative_spiral"],
                    },
                )
            step = min(
                args.ptc_max_step,
                h * (1.5 if len(counts) <= max(2, (args.newton + 1) // 2) else 1.0),
            )
            report["stages"].append(
                {
                    "stage": accepted,
                    "pseudo_time": cfg["pseudo_time"],
                    "dt": h,
                    "implicit_residual": be_norm,
                    "krylov_iterations": counts,
                    **eq.family,
                    **met,
                }
            )
            print(
                f"pseudo_time={cfg['pseudo_time']:.6g} dt={h:.5g} forcing={strength:.6g} "
                f"unforced_defect={met['unforced_relative_defect']:.6g} spiral={met['relative_spiral']:.5g}",
                flush=True,
            )
            checkpoint(
                args.out / "stages" / f"stage-{accepted:04d}.npz",
                cfg,
                seed,
                y,
                previous,
                step,
                accepted,
            )
            save("running", metrics)
        save(limit_status() or "pseudo_time_budget_reached", metrics)
        return
    if args.direct:
        predictor = y.copy()
        predictor[-1] = 0.0
        report["direct_start"] = eq.metrics(predictor)
        normal = np.zeros_like(y)
        normal[-1] = 1.0
        candidate, counts, converged = correct_point(predictor, normal, True)
        candidate[-1] = 0.0
        met = eq.metrics(candidate)
        if args.solver == "scipy":
            report["direct_gmres_iterations"] = counts
        else:
            report["direct_ksp_iterations"] = counts
        if min(met["relative_spiral"], met["relative_field"]) < args.min_spiral:
            report["rejected_candidate"] = met
            save("lost_nonzero_branch")
        else:
            previous, y = np.empty(0), candidate
            root = converged and met["unforced_relative_defect"] <= args.tol
            save("unforced_projected_root" if root else "direct_unconverged")
        return
    while accepted - initial_accepted < args.max_stages and y[-1] != 0.0:
        tangent = np.zeros_like(y)
        tangent[-1] = -1.0
        if previous.size:
            tangent = y - previous
            tangent /= np.linalg.norm(tangent)
        predictor = y + step * tangent
        normal = tangent
        target = predictor[-1] <= 0 and tangent[-1] < 0
        if target:
            predictor = y - y[-1] / tangent[-1] * tangent
            predictor[-1] = 0.0
            normal = np.zeros_like(tangent)
            normal[-1] = 1.0
        candidate, counts, converged = correct_point(predictor, normal, target)
        if converged and not target and candidate[-1] <= 0:
            # A curved correction crossed the endpoint. Solve at exactly zero forcing.
            predictor = candidate.copy()
            predictor[-1] = 0
            normal = np.zeros_like(tangent)
            normal[-1] = 1
            candidate, extra, converged = correct_point(predictor, normal, True)
            counts += extra
            target = True
        if not converged:
            report["rejected_steps"] += 1
            if step / 2 < args.min_step:
                save("correction_stalled")
                return
            step /= 2
            continue
        if target:
            candidate[-1] = 0.0
        met = eq.metrics(candidate)
        if (
            met["relative_spiral"] < args.min_spiral
            or met["relative_field"] < args.min_spiral
        ):
            report["rejected_candidate"] = met
            save("lost_nonzero_branch")
            return
        previous, y = y, candidate
        accepted += 1
        count_key = "gmres_iterations" if args.solver == "scipy" else "ksp_iterations"
        report["stages"].append(
            dict(stage=accepted, arclength_step=step, **{count_key: counts}, **met)
        )
        print(
            f"stage={accepted} forcing={y[-1]:.6g} period={met['period']:.7g} "
            f"unforced_defect={met['unforced_relative_defect']:.5g}",
            flush=True,
        )
        step = min(args.step, step * 1.4)
        checkpoint(
            args.out / "stages" / f"stage-{accepted:04d}.npz",
            cfg,
            seed,
            y,
            previous,
            step,
            accepted,
        )
        save("running", met)
    final = report["current"]
    if y[-1] == 0:
        status = (
            "unforced_projected_root"
            if final["unforced_relative_defect"] <= args.tol
            and final["phase_error"] <= args.tol
            else "unforced_tolerance_missed"
        )
    else:
        status = "stage_budget_reached"
    save(status)


def audit(args):
    cfg, seed, y, *_ = load_checkpoint(args.checkpoint)
    basis = Basis(**cfg["basis"])
    old_labels = None
    device = backend(args.device)
    grids = sorted(set(args.grids))
    if not grids or min(grids) < 128:
        raise ValueError("Use audit grids of at least 128")
    rows = []
    for grid, enrich in [(g, False) for g in grids] + [(max(grids), True)]:
        test = (
            replace(
                basis,
                radial=basis.radial + 1,
                degrees=basis.degrees + 1,
                azimuthal=basis.azimuthal + 1,
                temporal=basis.temporal + 1,
            )
            if enrich
            else basis
        )
        sp = OrbitSpace(test, grid, device)
        if old_labels is None:
            old_labels = sp.labels
        ref = transfer(seed, old_labels, sp.labels)
        state = np.r_[transfer(y[:-2], old_labels, sp.labels), y[-2:]]
        path = PseudoTimePath(**cfg["ptc_path"]) if "ptc_path" in cfg else None
        eq = OrbitEquation(sp, ref, cfg["period"], path)
        if path:
            eq.set_strength(y[-1])
        period = cfg["period"] * math.exp(state[-2])
        factors = sp.factors(period)
        field = sp.tensor(state[:-2] * eq.scale)
        raw = sp.residual(field, factors)
        white = eq.scaled(raw)
        met = eq.metrics(state, target_raw=white) if path else eq.metrics(state, white)
        weak = sp.weak_defect(field, raw, factors, period)
        retained = set(old_labels)
        new_modes = np.array([label not in retained for label in sp.labels])
        rows.append(
            dict(
                grid=grid,
                effective_azimuth=19 * sp.wedge,
                phase_nodes=sp.nt,
                enriched=enrich,
                coefficients=len(ref),
                max_linear_block_condition=eq.linear_condition,
                normalized_weak_defect=weak,
                new_mode_unforced_defect=float(
                    np.linalg.norm(white[new_modes]) / np.linalg.norm(state[:-2])
                ),
                **met,
            )
        )
        del eq, sp
    report = {
        "checkpoint": str(args.checkpoint),
        "forcing": float(y[-1]),
        "measurements": rows,
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


def export_replay(args):
    cfg, seed, y, *_ = load_checkpoint(args.checkpoint)
    basis = Basis(**cfg["basis"])
    period = cfg["period"] * math.exp(y[-2])
    end = -math.expm1(-period)
    n = args.grid
    if (
        n < 4
        or args.steps < 1
        or not 50 < args.half_box < np.inf
        or not 0 < end < 1
        or not math.isfinite(args.phase)
    ):
        raise ValueError(
            "Replay requires a finite phase, a box containing the diagnostic core and a representable cycle"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    axis = (np.arange(n) + 0.5) * (2 * args.half_box / n) - args.half_box
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coeff = np.linalg.norm(seed) * y[:-2]

    def write_velocity(f):
        shape = (3, n, n, n)
        f.truncate(math.prod(shape) * 8)
        u = np.memmap(f.name, mode="r+", dtype=np.float64, shape=shape)
        for iz, z in enumerate(axis):
            points = np.stack((xx, yy, np.full_like(xx, z)), axis=-1)
            plane = sample_cartesian(basis, coeff, period, args.phase, points)
            u[:, iz] = plane.T.reshape(3, n, n)
        u.flush()

    atomic(args.out / "velocity.bin", write_velocity)
    template = Path(__file__).with_name("inputs.dss_cycle")
    params = dict(
        line.split("=", 1)
        for line in (
            v.split("#", 1)[0].strip() for v in template.read_text().splitlines()
        )
        if "=" in line
    )
    params = {k.strip(): v.strip() for k, v in params.items()}
    del params["incflo.initial_velocity_plotfile"]
    params.update(
        {
            "amr.n_cell": " ".join([str(n)] * 3),
            "amr.max_grid_size": str(n),
            "geometry.prob_lo": " ".join([str(-args.half_box)] * 3),
            "geometry.prob_hi": " ".join([str(args.half_box)] * 3),
            "max_step": str(args.steps),
            "stop_time": format(end, ".17g"),
            "incflo.fixed_dt": format(end / args.steps, ".17g"),
            "incflo.init_shrink": "1",
            "incflo.initial_iterations": "0",
            "amr.plot_per_exact": "-1",
            "amr.plot_int": "-1",
            "dss.enabled": "1",
            "dss.gradient": "0",
            "dss.control": "velocity.bin",
        }
    )
    atomic(
        args.out / "inputs",
        lambda f: f.writelines(f"{k} = {v}\n" for k, v in params.items()),
        binary=False,
    )
    write_json(
        args.out / "replay.json",
        {
            "checkpoint": str(args.checkpoint),
            "period": period,
            "phase_fraction": args.phase,
            "forcing_at_checkpoint": float(y[-1]),
            "end_time": end,
            "steps": args.steps,
            "grid": n,
            "half_box": args.half_box,
        },
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    s = cmd.add_parser("search", help="Follow the forced branch toward zero forcing")
    s.add_argument("--out", type=Path, required=True)
    start = s.add_mutually_exclusive_group()
    start.add_argument(
        "--restart", type=Path, help="Resume the saved branch and arclength tangent"
    )
    start.add_argument(
        "--lift",
        type=Path,
        help="Start a finer continuation from a saved field and period",
    )
    s.add_argument("--grid", type=int)
    s.add_argument("--radial", type=int)
    s.add_argument("--degrees", type=int)
    s.add_argument("--azimuthal", type=int)
    s.add_argument("--temporal", type=int)
    s.add_argument("--radius", type=float)
    s.add_argument("--phase-radius", type=float)
    s.add_argument("--radial-family", choices=("jacobi", "legendre"))
    s.add_argument("--period", type=float, default=0.36)
    s.add_argument("--amplitude", type=float, default=20)
    s.add_argument("--wave", type=float, default=0.7)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--jitter", type=float, default=0)
    s.add_argument(
        "--direct",
        action="store_true",
        help="Apply deflated Newton at exactly zero forcing",
    )
    s.add_argument(
        "--pseudo-transient",
        action="store_true",
        help="March the forced orbit family implicitly in artificial time",
    )
    s.add_argument(
        "--ptc-decay-time",
        type=float,
        help="Exponential forcing decay time (default 1)",
    )
    s.add_argument(
        "--ptc-initial-g",
        type=float,
        help="Initial similarity dilation (default 0.502)",
    )
    s.add_argument(
        "--ptc-initial-viscosity", type=float, help="Initial viscosity (default 1)"
    )
    s.add_argument(
        "--ptc-initial-damping",
        type=float,
        help="Initial damping of non-axisymmetric modes (default 0.1)",
    )
    s.add_argument(
        "--ptc-max-step",
        type=float,
        help="Largest backward Euler pseudo-time step (default 100)",
    )
    s.add_argument(
        "--step",
        type=float,
        help="Initial step (default 0.2). A pseudo-time restart uses its saved step unless overridden",
    )
    s.add_argument("--min-step", type=float, default=0.001)
    s.add_argument("--max-stages", type=int, default=24)
    s.add_argument(
        "--solver",
        choices=("scipy", "petsc"),
        help="Default is PETSc for pseudo-time and SciPy otherwise",
    )
    s.add_argument("--snes-type", help="PETSc SNES method (default newtonls)")
    s.add_argument(
        "--pc",
        choices=("bordered", "none"),
        help="PETSc Krylov preconditioner (default bordered)",
    )
    s.add_argument("--newton", type=int, default=7)
    s.add_argument("--krylov", type=int, default=40)
    s.add_argument("--tol", type=float, default=1e-7)
    s.add_argument("--min-spiral", type=float, default=0.05)
    a = cmd.add_parser("audit", help="Check quadrature and unresolved field harmonics")
    a.add_argument("checkpoint", type=Path)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--grids", type=int, nargs="+", default=[128, 160, 192])
    e = cmd.add_parser(
        "export",
        help="Sample a field and write an independent physical-time incflo replay",
    )
    e.add_argument("checkpoint", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--grid", type=int, default=128)
    e.add_argument("--steps", type=int, default=96)
    e.add_argument("--phase", type=float, default=0)
    e.add_argument("--half-box", type=float, default=96)
    for sub in (s, a):
        sub.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    return p


def main():
    args = parser().parse_args()
    if args.command == "search":
        run_search(args)
    elif args.command == "audit":
        audit(args)
    else:
        export_replay(args)


if __name__ == "__main__":
    main()
