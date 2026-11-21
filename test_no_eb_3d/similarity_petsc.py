"""Single-rank PETSc SNES corrector for the space-time Galerkin orbit search."""

import math

import numpy as np


class BorderedStokes:
    """Invert the augmented Jacobian with advection omitted from the field block."""

    def __init__(self, jac):
        eq = jac.orbit.eq
        self.basis = eq.space.basis
        self.weight = jac.weight
        # The equations already carry L_ref^-1 on the left, so B^-1 = L(S)^-1 L_ref.
        self.blocks = [
            np.linalg.solve(factor[-1].detach().cpu().numpy(), ref)
            for factor, ref in zip(jac.orbit.factors, eq.reference_linear, strict=True)
        ]
        self.rows = np.stack((eq.phase, jac.normal[:-2], jac.derivative[:-2]))
        # The third auxiliary unknown is the directional derivative of the deflator.
        self.columns = np.column_stack(
            [
                self.field_inverse(jac.weight * jac.orbit.period_column),
                self.field_inverse(-jac.weight * eq.force),
                self.field_inverse(jac.fluid_residual),
            ]
        )
        schur = self.rows @ self.columns
        schur[1, :2] -= jac.normal[-2:]
        schur[2, 2] += 1
        # A degenerate phase/period coupling can occur near a steady field. Keep an
        # invertible field-only preconditioner in that case and record the fallback.
        self.border_inverse = None
        if np.isfinite(schur).all() and np.linalg.cond(schur) < 1 / (
            100 * np.finfo(float).eps
        ):
            self.border_inverse = np.linalg.inv(schur)

    def field_inverse(self, rhs):
        parts = []
        for n, (block, c) in enumerate(
            zip(self.blocks, self.basis.coefficients(rhs), strict=True)
        ):
            x = (block @ c[..., None])[..., 0]
            parts.append(x.real.ravel())
            if n:
                parts.append(x.imag.ravel())
        return np.concatenate(parts) / self.weight

    def __call__(self, rhs):
        x = self.field_inverse(rhs[:-2])
        if self.border_inverse is None:
            return np.r_[x, rhs[-2:]]
        reduced = self.rows @ x
        reduced[:2] -= rhs[-2:]
        aux = self.border_inverse @ reduced
        return np.r_[x - self.columns @ aux, aux[:2]]


def petsc_runtime():
    try:
        from petsc4py import PETSc
    except ImportError as exc:
        raise RuntimeError(
            "--solver petsc requires petsc4py in the Python environment running the search"
        ) from exc
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise RuntimeError(
            "The real coefficient vectors require a real double-precision PETSc build"
        )
    if PETSc.COMM_WORLD.getSize() != 1:
        raise RuntimeError(
            "Run one MPI rank per search and give each independent search its own output directory"
        )
    return PETSc


class OrbitShell:
    """SNES callbacks and matrix/PC shell contexts for a single corrector."""

    def __init__(self, system):
        self.system = system
        self.point = self.residual = self.jac = self.pc = None
        self.fallbacks = 0

    def evaluate(self, point):
        if self.point is None or not np.array_equal(point, self.point):
            self.point = point.copy()
            eq = self.system.eq
            try:
                period = eq.period * math.exp(point[-2])
            except OverflowError:
                period = math.inf
            if np.isfinite(point).all() and 0 < period < np.inf:
                self.residual = self.system.residual(point)
            else:
                self.residual = np.full_like(point, np.inf)
        return self.residual

    def function(self, snes, x, f):
        f.array[:] = self.evaluate(x.array_r)

    def jacobian(self, snes, x, mat, pmat):
        point = x.array_r
        self.jac = self.pc = None
        self.jac = self.system.linearize(point, self.evaluate(point))
        mat.assemble()

    def mult(self, mat, x, y):
        y.array[:] = self.jac(x.array_r.copy())

    def multTranspose(self, mat, x, y):
        y.array[:] = self.jac.transpose(x.array_r.copy())

    def apply(self, pc, x, y):
        if self.pc is None:
            self.pc = BorderedStokes(self.jac)
            self.fallbacks += int(self.pc.border_inverse is None)
        y.array[:] = self.pc(x.array_r)


def reason_name(enum, code):
    return next(
        (key for key, value in vars(enum).items() if key.isupper() and value == code),
        str(code),
    )


def petsc_correct(
    system,
    tolerance,
    iterations,
    krylov,
    *,
    method="newtonls",
    preconditioner="bordered",
    monitor=None,
):
    P = petsc_runtime()
    comm = P.COMM_SELF
    shell = OrbitShell(system)
    state = system.predictor.copy()
    n = state.size
    x = P.Vec().createWithArray(state, comm=comm)
    f = x.duplicate()
    matrix = P.Mat().createPython((n, n), shell, comm=comm)
    snes = P.SNES().create(comm)
    try:
        matrix.setUp()
        snes.setOptionsPrefix("orbit_")
        snes.setType(method)
        snes.setFunction(shell.function, f)
        snes.setJacobian(shell.jacobian, matrix, matrix)
        snes.setTolerances(rtol=0, atol=tolerance, stol=0, max_it=iterations)
        ksp = snes.getKSP()
        ksp.setType("gmres")
        ksp.setGMRESRestart(min(n, krylov))
        ksp.setTolerances(rtol=0.03, max_it=krylov)
        ksp.setPCSide(P.PC.Side.RIGHT)
        ksp.setNormType(P.KSP.NormType.UNPRECONDITIONED)
        pc = ksp.getPC()
        if preconditioner == "bordered":
            pc.setType("python")
            pc.setPythonContext(shell)
        else:
            pc.setType("none")
        snes.setFromOptions()
        linear_counts, linear_reasons, norms = [], [], []

        def end_linear(solver, rhs, solution):
            linear_counts.append(solver.getIterationNumber())
            linear_reasons.append(
                reason_name(P.KSP.ConvergedReason, solver.getConvergedReason())
            )

        def iteration(solver, it, norm):
            norms.append(float(norm) if math.isfinite(norm) else None)
            if monitor is not None:
                monitor(it, norm, sum(linear_counts))

        ksp.setPostSolve(end_linear)
        snes.setMonitor(iteration)
        snes.solve(None, x)
        result = x.array_r.copy()
        reason = snes.getConvergedReason()
        norm = np.linalg.norm(shell.evaluate(result))
        converged = bool(reason > 0 and np.isfinite(norm) and norm <= tolerance)
        info = {
            "petsc_version": ".".join(map(str, P.Sys.getVersion())),
            "snes": snes.getType(),
            "ksp": ksp.getType(),
            "pc": "bordered"
            if pc.getType() == "python" and pc.getPythonContext() is shell
            else pc.getType(),
            "reason": reason_name(P.SNES.ConvergedReason, reason),
            "iterations": snes.getIterationNumber(),
            "function_evaluations": snes.getFunctionEvaluations(),
            "residual_norms": norms,
            "final_residual_norm": float(norm) if np.isfinite(norm) else None,
            "ksp_iterations": linear_counts,
            "ksp_reasons": linear_reasons,
            "border_fallbacks": shell.fallbacks,
            "residual_tolerance_met": converged,
        }
        return result, linear_counts, converged, info
    finally:
        snes.destroy()
        matrix.destroy()
        f.destroy()
        x.destroy()
