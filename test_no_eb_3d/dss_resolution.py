"""Matched periodic controls and three-grid recurrence estimates."""

import math

import numpy as np
from scipy.fft import irfft, rfft
from scipy.optimize import brentq


def refine_velocity(velocity, n):
    """Sample the same periodic trigonometric interpolant at finer cell centers."""
    source_n = velocity.shape[-1]
    if n < source_n:
        raise ValueError("The replay grid must be at least as fine as the saved grid")
    if n == source_n:
        return velocity
    modes = np.arange(source_n // 2 + 1)
    shift = np.exp(1j * np.pi * modes * (1 / n - 1 / source_n))
    if source_n % 2 == 0:
        # Split the real coarse Nyquist cosine between its two fine-grid modes.
        shift[-1] *= 0.5
    for axis in (1, 2, 3):
        spectrum = rfft(velocity, axis=axis, norm="forward")
        shape = [1] * velocity.ndim
        shape[axis] = len(shift)
        spectrum *= shift.reshape(shape)
        velocity = irfft(spectrum, n=n, axis=axis, norm="forward")
    return velocity


def refinement_transpose(gradient, coarse_n):
    """Euclidean transpose of the cell-centered Fourier interpolation."""
    fine_n = gradient.shape[-1]
    if fine_n == coarse_n:
        return gradient
    if fine_n < coarse_n:
        raise ValueError("The supplied gradient must be on the finer grid")
    k = np.arange(coarse_n // 2 + 1)
    phase = np.exp(-1j * np.pi * k * (1 / fine_n - 1 / coarse_n))
    for axis in (3, 2, 1):
        spectrum = rfft(gradient, axis=axis, norm="forward")
        selection = [slice(None)] * gradient.ndim
        selection[axis] = slice(0, k.size)
        shape = [1] * gradient.ndim
        shape[axis] = k.size
        # irfft on the coarse even grid takes the real part of its Nyquist
        # coefficient. Its single real degree of freedom already includes both fine modes.
        transformed = spectrum[tuple(selection)] * phase.reshape(shape)
        gradient = (fine_n / coarse_n) * irfft(
            transformed, n=coarse_n, axis=axis, norm="forward"
        )
    return gradient


def observed_order(errors, grids):
    first, last = np.diff(errors)
    tol = 64 * np.finfo(float).eps * max(np.max(np.abs(errors)), 1e-16)
    if min(abs(first), abs(last)) <= tol or np.sign(first) != np.sign(last):
        return None
    left, right = np.diff(np.log(np.asarray(grids, dtype=float)))
    target = first / last

    def mismatch(p):
        return math.expm1(p * left) / -math.expm1(-p * right) - target

    if mismatch(1e-6) * mismatch(20.0) >= 0:
        return None
    return float(brentq(mismatch, 1e-6, 20))


def richardson_return(errors, grids, order=2.0):
    """Return the guided error and d(half its square)/d(native half-squared errors)."""
    errors = np.asarray(errors, dtype=float)
    grids = np.asarray(grids, dtype=float)
    if (
        errors.shape != (3,)
        or grids.shape != (3,)
        or np.any(np.diff(grids) <= 0)
        or np.any(grids <= 0)
        or not np.isfinite(grids).all()
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
        or not math.isfinite(order)
        or order <= 0
    ):
        raise ValueError(
            "Need three nonnegative finite errors, increasing grids and positive finite order"
        )
    cm, mf = (grids[1:] / grids[:-1]) ** order
    w_fine = np.array([0.0, -1.0, 1.0]) / (mf - 1)
    w_middle = np.array([-1.0, 1.0, 0.0]) / ((cm - 1) * mf)
    # Both corrections estimate the leading truncation error at the finest h.
    fine = float((errors[2] - errors[1]) / (mf - 1))
    middle = float((errors[1] - errors[0]) / ((cm - 1) * mf))
    weights = np.array([0.0, 0.0, 1.0])
    correction = 0.0
    monotone = fine * middle > 0
    if monotone:
        total = fine + middle
        correction = 2 * (fine / total) * middle
        weights += 2 * ((middle / total) ** 2 * w_fine + (fine / total) ** 2 * w_middle)
    estimate = float(errors[-1] + correction)
    difference = fine - middle
    score = math.hypot(estimate, difference)
    derivative = estimate * weights + difference * (w_fine - w_middle)
    native_weights = np.divide(derivative, errors, out=np.zeros(3), where=errors > 0)
    p = observed_order(errors, grids)
    return {
        "score": score,
        "signed_estimate": estimate,
        "correction": correction,
        "disagreement": abs(difference),
        "fine_pair_estimate": float(errors[-1] + fine),
        "middle_pair_estimate": float(errors[1] + mf * middle),
        "trend": "monotone"
        if monotone
        else ("flat" if max(abs(fine), abs(middle)) == 0 else "turning or flat"),
        "observed_order": p,
        "assumed_order": float(order),
        "order_consistent": bool(
            p is not None and abs(p - order) <= 0.5 and estimate >= 0
        ),
    }, native_weights


def holdout_prediction(guide, finest, measured):
    """Compare the stored three-grid model with an independently computed finer grid."""
    n0, n = finest["grid"], measured["grid"]
    if (
        n <= n0
        or measured["steps"] * n0 != finest["steps"] * n
        or guide["control_sha256"] != finest["control_sha256"]
    ):
        raise ValueError(
            "The holdout must refine the same control at the same timestep-to-mesh ratio"
        )
    metrics = {}
    for name, model in guide["metrics"].items():
        prediction = finest[name] + model["correction"] * (
            1.0 - (n0 / n) ** model["assumed_order"]
        )
        metrics[name] = {
            "prediction": prediction,
            "measured": measured[name],
            "prediction_error": abs(prediction - measured[name]),
            "finest_grid_error": abs(finest[name] - measured[name]),
            "order_consistent": model["order_consistent"],
        }
    return {
        "trial": guide["trial"],
        "control_sha256": guide["control_sha256"],
        "grid": n,
        "steps": measured["steps"],
        "metrics": metrics,
    }
