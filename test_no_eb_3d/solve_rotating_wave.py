"""Search and continue upper-half-space rotating waves with a free dilation and rotation rate."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from similarity_petsc import petsc_runtime, reason_name
from similarity_rotating import (
    RotatingWave,
    WaveBasis,
    WaveGeometry,
    WaveSpace,
    figure_seed,
    lift_coefficients,
)
from solve_similarity_orbit import atomic, backend, write_json


def correct(problem, predictor, normal, tol, max_it):
    """SNES corrector for a fixed load or a pseudo-arclength hyperplane."""
    P = petsc_runtime()
    n = len(predictor)
    state = predictor.copy()
    x = P.Vec().createWithArray(state, comm=P.COMM_SELF)
    f = x.duplicate()
    mat = P.Mat().createDense([n, n], comm=P.COMM_SELF)
    mat.setUp()
    indices = np.arange(n, dtype=P.IntType)
    solver = P.SNES().create(comm=P.COMM_SELF)

    def residual(point):
        return np.r_[
            problem.evaluate(point[:-1], point[-1]), normal @ (point - predictor)
        ]

    def function(snes, v, out):
        try:
            a = residual(v.array_r)
        except (ValueError, OverflowError, FloatingPointError):
            a = np.full(n, np.inf)
        out.array[:] = a if np.isfinite(a).all() else np.full(n, np.inf)

    def jacobian(snes, v, A, B):
        y = v.array_r
        J = np.empty((n, n))
        J[:-1, :-1] = problem.jacobian(y[:-1], y[-1])
        J[:-1, -1] = np.r_[-problem.load, 0.0, 0.0]
        J[-1] = normal
        A.setValues(indices, indices, J)
        A.assemble()

    try:
        solver.setOptionsPrefix("rotating_")
        solver.setType("newtonls")
        solver.setFunction(function, f)
        solver.setJacobian(jacobian, mat, mat)
        solver.setTolerances(rtol=0, atol=tol, stol=0, max_it=max_it)
        ksp = solver.getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        solver.setFromOptions()
        solver.solve(None, x)
        candidate = x.array_r.copy()
        reason = solver.getConvergedReason()
        try:
            error = float(la.norm(residual(candidate)))
        except (ValueError, OverflowError, FloatingPointError):
            error = math.inf
        info = {
            "iterations": solver.getIterationNumber(),
            "reason": reason_name(P.SNES.ConvergedReason, reason),
            "evaluations": solver.getFunctionEvaluations(),
            "residual": error if math.isfinite(error) else None,
        }
        return candidate, bool(reason > 0 and error <= tol), info
    finally:
        solver.destroy()
        mat.destroy()
        f.destroy()
        x.destroy()


def save_state(path, problem, point, info):
    g, omega = problem.parameters(point[:-1])
    cfg = dict(
        basis=asdict(problem.space.basis),
        grid=problem.space.grid,
        kappa=problem.space.kappa,
        g=g,
        omega=omega,
        forcing=float(point[-1]),
        amplitude=problem.amplitude,
        **info,
    )
    c = {
        f"c{j}": v.detach().cpu().numpy()
        for j, v in enumerate(problem.coefficients(point[:-1]))
    }
    atomic(path, lambda f: np.savez_compressed(f, config=json.dumps(cfg), **c))


def read_state(path):
    with np.load(path, allow_pickle=False) as data:
        cfg = json.loads(str(data["config"]))
        basis = WaveBasis(**cfg["basis"])
        coeff = [data[f"c{j}"].copy() for j in range(basis.harmonics + 1)]
    for group, c in zip(basis.groups, coeff, strict=True):
        size = sum(a.count * b.count if a else 1 for a, b in group.values())
        if c.shape != (size,) or not np.isfinite(c).all():
            raise ValueError("Invalid rotating-wave checkpoint")
    return cfg, basis, coeff


def audit_candidate(problem, point, args):
    g, omega = problem.parameters(point[:-1])
    coeff = [v.detach().cpu().numpy() for v in problem.coefficients(point[:-1])]
    grids = sorted({problem.space.grid, *args.audit_grids})
    if (
        len(grids) < 2
        or grids[0] < problem.space.grid
        or grids[-1] <= problem.space.grid
    ):
        raise ValueError("Audit requires an independent finer quadrature")
    rows = []
    for grid in grids:
        space = WaveSpace(
            problem.space.basis, grid, problem.space.device, problem.space.kappa
        )
        for enrichment in [1, 2] if grid == grids[-1] else [1]:
            rows.append(space.audit(coeff, g, omega, enrichment))
    weak = problem.evaluate(point[:-1], 0.0)
    # Compare both quadrature and pressure refinement. An amplitude condition
    # selects the family member; it does not replace a physical equation.
    final = rows[-1]
    error = max(abs(r["relative_h"] - final["relative_h"]) for r in rows)
    ok = bool(
        point[-1] == 0
        and la.norm(weak) <= args.tol
        and abs(omega) > 1e-7
        and all(
            abs(r["wave_rms"] / problem.amplitude - 1) < 1e-4
            and r["relative_h"] <= args.strong_tol
            and r["residual_h"] <= args.h_tol
            for r in rows
        )
        and error <= max(1e-7, 0.05 * final["relative_h"])
    )
    return {
        "accepted": ok,
        "forcing": float(point[-1]),
        "weak_residual": float(la.norm(weak)),
        "relative_h_tolerance": args.strong_tol,
        "absolute_h_tolerance": args.h_tol,
        "measurements": rows,
    }


def contact(problem, y, normal, args, tag, report):
    z, ok, info = correct(problem, y, normal, args.tol, args.newton)
    report["correctors"].append(dict(tag=tag, **info))
    return z, ok, info


def curve_tangent(eq, z, direction):
    n = len(z)
    border = np.empty((n, n))
    border[:-1, :-1] = eq.jacobian(z[:-1], z[-1])
    border[:-1, -1] = np.r_[-eq.load, 0.0, 0.0]
    border[-1] = direction
    rhs = np.zeros(n)
    rhs[-1] = 1
    tangent = la.lstsq(border, rhs, cond=1e-13)[0]
    return tangent / la.norm(tangent)


def search(args):
    b = WaveBasis(
        degree=args.degree,
        harmonics=args.harmonics,
        radius=args.radius,
        phase_radius=args.phase_radius,
        sigma_breaks=args.sigma_breaks,
        eta_breaks=args.eta_breaks,
    )
    if args.restart:
        cfg, b, cs = read_state(args.restart / "reference.npz")
        args.grid, args.kappa, args.g, args.omega, args.amplitude = (
            cfg[k] for k in ("grid", "kappa", "g", "omega", "amplitude")
        )
    sp = WaveSpace(b, args.grid, backend(args.device), args.kappa)
    if not args.restart:
        geo = WaveGeometry(sp, args.kappa, args.g, args.omega)
        cs = figure_seed(sp, geo, args.profile, args.mean_scale, args.wave_scale)
    eq = RotatingWave(sp, args.g, args.omega, cs, args.amplitude)
    args.out.mkdir(parents=True, exist_ok=False)
    z = np.r_[eq.start, 1.0]
    save_state(args.out / "reference.npz", eq, z, {"stage": 0})
    normal = np.zeros_like(z)
    normal[-1] = 1.0
    report = {
        "status": "running",
        "basis": asdict(b),
        "kappa": args.kappa,
        "wave_rms_target": eq.amplitude,
        "unknowns": len(eq.start),
        "device": str(sp.device),
        "hardware": torch.cuda.get_device_name() if sp.device.type == "cuda" else "CPU",
        "quadrature": [len(sp.wr), len(sp.wa), sp.phase_nodes],
        "initial_g": args.g,
        "initial_omega": args.omega,
        "profile": str(args.profile),
        "mean_scale": args.mean_scale,
        "wave_scale": args.wave_scale,
        "grid": args.grid,
        "correctors": [],
        "stages": [],
        "best_abs_forcing": None,
    }

    def record(point, stage):
        g, omega = eq.parameters(point[:-1])
        r = eq.evaluate(point[:-1], point[-1])
        data = {
            "stage": stage,
            "forcing": float(point[-1]),
            "g": g,
            "omega": omega,
            "weak_residual": float(la.norm(r)),
            "unforced_weak_residual": float(la.norm(eq.evaluate(point[:-1], 0.0)[:-2])),
            "wave_rms": eq.amplitude * math.sqrt(max(0, 1 + 2 * r[-2])),
        }
        report["stages"].append(data)
        save_state(args.out / "current.npz", eq, point, {"stage": stage})
        if (
            report["best_abs_forcing"] is None
            or abs(point[-1]) <= report["best_abs_forcing"]
        ):
            report["best_abs_forcing"] = abs(float(point[-1]))
            save_state(args.out / "closest.npz", eq, point, {"stage": stage})
        write_json(args.out / "search.json", report)
        print(json.dumps(data), flush=True)

    def try_zero(start, stage):
        y = start.copy()
        y[-1] = 0.0
        trial, ok, _ = contact(eq, y, normal, args, "zero", report)
        if ok:
            trial[-1] = 0.0
            audit = audit_candidate(eq, trial, args)
            report["audit"] = audit
            save_state(
                args.out / "unforced.npz",
                eq,
                trial,
                {"stage": stage, "audited": audit["accepted"]},
            )
            record(trial, stage)
            report["status"] = (
                "audited_unforced_wave"
                if audit["accepted"]
                else "projected_root_needs_refinement"
            )
            return True
        return False

    direction = -normal
    step = args.step
    if args.restart:
        with np.load(
            args.restart / "continuation.npz", allow_pickle=False
        ) as checkpoint:
            z = checkpoint["point"].copy()
            direction = checkpoint["tangent"].copy()
            step = float(checkpoint["step"])
        if la.norm(eq.evaluate(z[:-1], z[-1])) > max(5 * args.tol, 1e-6):
            raise ValueError("Continuation checkpoint fails the restored residual")
        report["parent"] = str(args.restart)
    record(z, 0)
    done = args.direct_first and try_zero(z, 0)
    if not done:
        tangent = curve_tangent(eq, z, direction)

        def checkpoint():
            atomic(
                args.out / "continuation.npz",
                lambda f: np.savez(f, point=z, tangent=tangent, step=step),
            )

        checkpoint()
        for stage in range(1, args.stages + 1):
            pred = z + step * tangent
            try:
                y, ok, info = contact(eq, pred, tangent, args, "arclength", report)
            except (ValueError, OverflowError, FloatingPointError):
                y, ok, info = None, False, {"reason": "predictor_out_of_domain"}
            if not ok:
                step *= 0.5
                report["last_failure"] = info
                checkpoint()
                write_json(args.out / "search.json", report)
                if step < args.min_step:
                    report["status"] = "continuation_stalled"
                    break
                continue
            dz = y - z
            distance = la.norm(dz)
            if distance == 0 or not np.isfinite(distance):
                report["status"] = "continuation_stalled"
                break
            prev, z = z, y
            record(z, stage)
            crossed = prev[-1] * z[-1] <= 0
            if crossed or (abs(z[-1]) < 0.08 and stage % 4 == 0):
                guess = (
                    z
                    if not crossed
                    else prev + (z - prev) * (-prev[-1] / (z[-1] - prev[-1]))
                )
                if try_zero(guess, stage):
                    break
            tangent = curve_tangent(eq, z, dz / distance)
            step = min(
                args.max_step, step * (1.35 if info["iterations"] <= 5 else 0.85)
            )
            checkpoint()
        if report["status"] == "running":
            report["status"] = "search_budget_reached"
    write_json(args.out / "search.json", report)
    print(report["status"], flush=True)


def trace(args):
    cfg, b, c = read_state(args.checkpoint)
    if cfg["forcing"] != 0:
        raise ValueError(
            "An unforced checkpoint is required to trace the unforced family"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "points": [], "correctors": []}
    start_k, start_a = cfg["kappa"], cfg["amplitude"]
    previous = (start_k, start_a, cfg["g"], cfg["omega"], c)

    def solve_at(origin, k, a):
        _, _, g, omega, coeff = origin
        space = WaveSpace(b, args.grid, backend(args.device), k)
        eq = RotatingWave(space, g, omega, coeff, a)
        initial = np.r_[eq.start, 0.0]
        normal = np.zeros_like(initial)
        normal[-1] = 1.0
        y, ok, _ = contact(eq, initial, normal, args, "unforced-trace", report)
        if not ok:
            return None
        y[-1] = 0.0
        audit = audit_candidate(eq, y, args)
        ng, nw = eq.parameters(y[:-1])
        n = len(report["points"])
        save_state(
            args.out / f"point-{n:03d}.npz", eq, y, {"audited": audit["accepted"]}
        )
        report["points"].append(
            {"kappa": k, "amplitude": a, "g": ng, "omega": nw, "audit": audit}
        )
        write_json(args.out / "trace.json", report)
        if not audit["accepted"]:
            return None
        return (
            k,
            a,
            ng,
            nw,
            [v.detach().cpu().numpy() for v in eq.coefficients(y[:-1])],
        )

    previous = solve_at(previous, start_k, start_a)
    if previous is None:
        report["status"] = "starting_wave_failed_audit"
    else:
        for target in args.values:
            goal = np.array(
                [target, previous[1]]
                if args.coordinate == "kappa"
                else [previous[0], target]
            )
            for _ in range(30):
                origin = np.array(previous[:2])
                delta = goal - origin
                if la.norm(delta) < 1e-10:
                    break
                # Limit the first correction so a distant target does not jump branches.
                coordinate = 0 if args.coordinate == "kappa" else 1
                step = delta * min(
                    1.0, args.max_fraction * origin[coordinate] / abs(delta[coordinate])
                )
                p = None
                for _ in range(9):
                    p = solve_at(previous, *(origin + step))
                    if p is not None:
                        break
                    step *= 0.5
                if p is None:
                    report["status"] = "trace_stalled"
                    break
                if (previous[2] - 0.5) * (p[2] - 0.5) <= 0:
                    lo, hi = previous, p
                    for _ in range(24):
                        if min(abs(lo[2] - 0.5), abs(hi[2] - 0.5)) < args.g_tol:
                            break
                        mid = solve_at(lo, (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
                        if mid is None:
                            break
                        if (lo[2] - 0.5) * (mid[2] - 0.5) <= 0:
                            hi = mid
                        else:
                            lo = mid
                    report["bracket"] = [list(lo[:4]), list(hi[:4])]
                    report["status"] = (
                        "crossing_refined"
                        if min(abs(lo[2] - 0.5), abs(hi[2] - 0.5)) < args.g_tol
                        else "crossing_bracketed"
                    )
                    break
                previous = p
            else:
                report["status"] = "trace_budget_reached"
            if report["status"] != "running":
                break
        if report["status"] == "running":
            report["status"] = "trace_completed"
    write_json(args.out / "trace.json", report)


def polish(args):
    """Solve at F=0 with amplitude and phase enforced throughout the trust-region iterations."""
    source, b, c = read_state(args.checkpoint)
    rich = WaveBasis(
        **{
            **asdict(b),
            **{
                k: getattr(args, k)
                for k in ("degree", "harmonics", "sigma_breaks", "eta_breaks")
                if getattr(args, k) is not None
            },
        }
    )
    c = lift_coefficients(b, rich, c) if rich != b else c
    k = source["kappa"] if args.kappa is None else args.kappa
    sp = WaveSpace(rich, args.grid, backend(args.device), k)
    eq = RotatingWave(
        sp, source["g"], source["omega"], c, args.amplitude or source["amplitude"]
    )
    x = eq.retract(eq.start)
    args.out.mkdir(parents=True, exist_ok=False)
    radius = args.trust
    best_h = math.inf
    best_x = x.copy()
    report = {
        "status": "running",
        "source": str(args.checkpoint),
        "source_forcing": source["forcing"],
        "applied_forcing": 0.0,
        "quadrature": [len(sp.wr), len(sp.wa), sp.phase_nodes],
        "kappa": k,
        "amplitude": eq.amplitude,
        "basis": asdict(rich),
        "iterations": [],
    }
    for iteration in range(args.iterations + 1):
        res = eq.evaluate(x, 0.0)
        norm = la.norm(res)
        g, omega = eq.parameters(x)
        physical = sp.audit(eq.coefficients(x), g, omega, enrichment=2)
        if physical["residual_h"] < best_h:
            best_x, best_h = x.copy(), physical["residual_h"]
            report["best_h_iteration"] = iteration
            save_state(
                args.out / "best-h.npz",
                eq,
                np.r_[x, 0.0],
                {
                    "iteration": iteration,
                    "weak_residual": float(norm),
                    "residual_h": best_h,
                },
            )
        report["iterations"].append(
            {
                "iteration": iteration,
                "residual": norm,
                "g": g,
                "omega": omega,
                "wave_rms": eq.amplitude * math.sqrt(1 + 2 * res[-2]),
                "radius": radius,
                "residual_h": physical["residual_h"],
                "relative_h": physical["relative_h"],
            }
        )
        print(json.dumps(report["iterations"][-1]), flush=True)
        save_state(
            args.out / "trial.npz",
            eq,
            np.r_[x, 0.0],
            {"audited": False, "weak_residual": norm},
        )
        write_json(args.out / "polish.json", report)
        if norm <= args.tol or iteration == args.iterations:
            break
        jac = eq.jacobian(x, 0.0)
        Z = la.qr(jac[-2:].T, mode="full")[0][:, 2:]
        M = jac[:-2] @ Z
        u, s, vh = la.svd(M, full_matrices=False)
        if s[0] == 0:
            break
        projected = -s * (u.T @ res[:-2])
        squared = np.maximum(s, s[0] * 1e-13) ** 2
        accepted = False
        for _ in range(18):
            # Solve the Euclidean trust-region normal equations by scalar bisection.
            p = projected / squared
            if la.norm(p) > radius:
                lo, hi = (
                    0.0,
                    max(float(la.norm(projected) / radius), np.finfo(float).tiny),
                )
                for _ in range(65):
                    mu = (lo + hi) / 2
                    if la.norm(projected / (squared + mu)) > radius:
                        lo = mu
                    else:
                        hi = mu
                p = projected / (squared + hi)
            delta = Z @ (vh.T @ p)
            denom = norm**2 - la.norm(res[:-2] + jac[:-2] @ delta) ** 2
            try:
                trial = eq.retract(x + delta)
                trial_norm = la.norm(eq.evaluate(trial, 0.0))
            except (ValueError, OverflowError, FloatingPointError):
                trial_norm = math.inf
            ratio = (norm**2 - trial_norm**2) / denom if denom > 0 else -1.0
            if ratio > 0.05:
                x = trial
                if ratio > 0.75 and la.norm(p) > 0.8 * radius:
                    radius = min(2 * radius, 8.0)
                elif ratio < 0.25:
                    radius *= 0.5
                accepted = True
                break
            radius *= 0.3
            if radius < 1e-10:
                break
        if not accepted:
            break
    point = np.r_[x, 0.0]
    audit = audit_candidate(eq, point, args)
    best_point = np.r_[best_x, 0.0]
    best_audit = (
        audit if np.array_equal(x, best_x) else audit_candidate(eq, best_point, args)
    )
    report["best_h_audit"] = best_audit
    save_state(
        args.out / "best-h.npz",
        eq,
        best_point,
        {
            "iteration": report["best_h_iteration"],
            "audited": best_audit["accepted"],
            "weak_residual": best_audit["weak_residual"],
            "residual_h": best_audit["measurements"][-1]["residual_h"],
        },
    )
    weak = audit["weak_residual"] <= args.tol
    report.update(
        status="audited_unforced_wave"
        if audit["accepted"]
        else (
            "projected_root_needs_refinement" if weak else "unforced_trial_incomplete"
        ),
        audit=audit,
    )
    if weak:
        save_state(
            args.out / "unforced.npz",
            eq,
            point,
            {"audited": audit["accepted"], "weak_residual": audit["weak_residual"]},
        )
    write_json(args.out / "polish.json", report)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="action", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--out", type=Path, required=True)
    shared.add_argument("--grid", type=int, default=128)
    shared.add_argument("--audit-grids", type=int, nargs="+", default=[160, 192])
    shared.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    shared.add_argument("--tol", type=float, default=1e-7)
    shared.add_argument("--strong-tol", type=float, default=0.01)
    shared.add_argument(
        "--h-tol",
        type=float,
        default=0.01,
        help="Absolute pressure-fitted half-space H residual threshold",
    )
    shared.add_argument("--newton", type=int, default=20)
    s = subs.add_parser("search", parents=[shared])
    s.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).with_name("rotating_figure_seed.npz"),
    )
    s.add_argument("--kappa", type=float, default=18.5)
    s.add_argument("--g", type=float, default=0.502)
    s.add_argument("--omega", type=float, default=2 * math.pi / 0.36)
    s.add_argument("--amplitude", type=float)
    s.add_argument("--mean-scale", type=float, default=1.0)
    s.add_argument("--wave-scale", type=float, default=1.0)
    for name in (
        "degree",
        "harmonics",
        "radius",
        "phase_radius",
        "sigma_breaks",
        "eta_breaks",
    ):
        v = getattr(WaveBasis(), name)
        s.add_argument(
            "--" + name.replace("_", "-"),
            default=v,
            **(
                {"nargs": "+", "type": float}
                if isinstance(v, tuple)
                else {"type": type(v)}
            ),
        )
    s.add_argument("--stages", type=int, default=50)
    s.add_argument("--step", type=float, default=0.2)
    s.add_argument("--max-step", type=float, default=1.0)
    s.add_argument("--min-step", type=float, default=1e-4)
    s.add_argument(
        "--direct-first", action=argparse.BooleanOptionalAction, default=True
    )
    s.add_argument(
        "--restart",
        type=Path,
        help="Resume the same load and pseudo-arclength curve from a run directory",
    )
    p = subs.add_parser("polish", parents=[shared])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--kappa", type=float)
    p.add_argument("--amplitude", type=float)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--trust", type=float, default=0.5)
    p.add_argument("--degree", type=int)
    p.add_argument("--harmonics", type=int)
    p.add_argument("--sigma-breaks", type=float, nargs="+")
    p.add_argument("--eta-breaks", type=float, nargs="+")
    t = subs.add_parser("trace", parents=[shared])
    t.add_argument("checkpoint", type=Path)
    t.add_argument("--coordinate", choices=("kappa", "amplitude"), required=True)
    t.add_argument("--values", nargs="+", type=float, required=True)
    t.add_argument("--g-tol", type=float, default=1e-6)
    t.add_argument(
        "--max-fraction",
        type=float,
        default=0.05,
        help="Largest relative change of the selected coordinate per family step",
    )
    args = cli.parse_args()
    if (
        args.grid < 128
        or min(args.audit_grids) < args.grid
        or max(args.audit_grids) <= args.grid
    ):
        cli.error("Use at least 128 meridional quadrature nodes and a finer audit")
    if not (
        0 < args.tol < 1
        and 0 < args.strong_tol < 1
        and 0 < args.h_tol < math.inf
        and args.newton > 0
    ):
        cli.error("Invalid corrector or audit tolerance")
    if args.action == "search" and not (
        0 < args.min_step <= args.step <= args.max_step
        and args.stages > 0
        and 0 < args.g < 2 / 3
    ):
        cli.error("Invalid initial dilation, stages or continuation steps")
    torch.set_num_threads(1)
    if args.action == "polish" and not (
        args.iterations > 0
        and args.trust > 0
        and (args.amplitude is None or args.amplitude > 0)
    ):
        cli.error("Invalid trust radius, iterations or wave amplitude")
    if args.action == "trace" and not (
        0 < args.max_fraction <= 1
        and 0 < args.g_tol < 1
        and all(
            math.isfinite(v) and (v >= 3 if args.coordinate == "kappa" else v > 0)
            for v in args.values
        )
    ):
        cli.error("Invalid family targets or continuation step")
    {"search": search, "trace": trace, "polish": polish}[args.action](args)


if __name__ == "__main__":
    main()
