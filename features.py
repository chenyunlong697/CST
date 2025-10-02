"""Derivation of geometric features from CST coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from cst import (
    airfoil_surfaces,
    cst_camber,
    cst_camber_derivative,
    cst_thickness,
    cst_thickness_derivative,
)


@dataclass
class FeatureResult:
    features: Dict[str, float]
    derived: Dict[str, float]


def find_extrema(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    idx = int(np.argmax(y))
    return float(y[idx]), float(x[idx])


def le_radius_fit(x: np.ndarray, half_thickness: np.ndarray) -> float:
    mask = (x >= 0.003) & (x <= 0.03)
    if mask.sum() < 2:
        mask = slice(0, min(5, x.size))
    sqrt_x = np.sqrt(x[mask])
    A = np.stack([sqrt_x, x[mask], np.ones_like(sqrt_x)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, half_thickness[mask], rcond=None)
    a = coeffs[0]
    return float(0.5 * a ** 2)


def monotonicity_penalty(x: np.ndarray, thickness: np.ndarray) -> float:
    mask = x >= 0.5
    dt_dx = np.gradient(thickness, x)
    if not np.any(mask):
        return 0.0
    penalty = np.maximum(0.0, dt_dx[mask])
    return float(np.trapz(penalty ** 2, x[mask]))


def compute_features_14(
    x: np.ndarray,
    b: Iterable[float],
    c: Iterable[float],
    dz_te: float,
) -> FeatureResult:
    x_dense = np.linspace(0.001, 0.999, 500)
    thickness = cst_thickness(x_dense, b, dz_te)
    camber = cst_camber(x_dense, c)
    thickness_deriv = cst_thickness_derivative(x_dense, b, dz_te)
    camber_deriv = cst_camber_derivative(x_dense, c)

    t_max, x_t = find_extrema(x_dense, thickness)
    idx_c = int(np.argmax(np.abs(camber)))
    f_max = float(camber[idx_c])
    x_f = float(x_dense[idx_c])

    half_thickness = 0.5 * thickness
    r_le_hat = le_radius_fit(x_dense, half_thickness)

    def _eval_at(x_point: float, values: np.ndarray) -> float:
        return float(np.interp(x_point, x_dense, values))

    feature_map = {
        "t_max": t_max,
        "x_t": x_t,
        "f_max": f_max,
        "x_f": x_f,
        "r_le_hat": r_le_hat,
        "dz_te": float(dz_te),
        "s_rec": _eval_at(0.6, thickness) - _eval_at(0.9, thickness),
        "t_015": _eval_at(0.15, thickness),
        "t_050": _eval_at(0.5, thickness),
        "t_075": _eval_at(0.75, thickness),
        "dt_080": _eval_at(0.8, thickness_deriv),
        "dz_005": _eval_at(0.05, camber_deriv),
        "dz_090": _eval_at(0.9, camber_deriv),
        "r_fx": _eval_at(0.9, camber)
        - (
            _eval_at(0.7, camber)
            + 0.20 * (_eval_at(0.95, camber) - _eval_at(0.7, camber))
        ),
    }

    derived = {
        "t_distribution_mean": float(np.mean(thickness)),
        "z_distribution_mean": float(np.mean(camber)),
    }

    return FeatureResult(features=feature_map, derived=derived)


def health_checks(x: np.ndarray, thickness: np.ndarray, dz_te: float) -> Dict[str, float]:
    results: Dict[str, float] = {}
    min_margin = np.min(thickness - dz_te)
    if min_margin < 0:
        results["thickness_margin"] = float(min_margin)

    penalty = monotonicity_penalty(x, thickness)
    if penalty > 0:
        results["monotonicity_penalty"] = float(penalty)
    return results

