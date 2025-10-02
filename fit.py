"""Least-squares fitting of CST coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cst import bernstein_matrix, class_function


@dataclass
class ThicknessFitResult:
    coeffs: np.ndarray
    dz_te: float
    residual: float


@dataclass
class CamberFitResult:
    coeffs: np.ndarray
    residual: float


def fit_cst_thickness(
    x: np.ndarray,
    thickness: np.ndarray,
    degree: int = 5,
    regularization: float = 1e-6,
) -> ThicknessFitResult:
    """Fit CST thickness coefficients using Tikhonov-regularized least squares."""

    basis = bernstein_matrix(degree, x)
    cfn = class_function(x)
    A = cfn[:, None] * basis
    A = np.concatenate([A, x[:, None]], axis=1)
    lhs = A.T @ A + regularization * np.eye(degree + 2)
    rhs = A.T @ thickness
    coeffs = np.linalg.solve(lhs, rhs)
    b = coeffs[:-1]
    dz_te = float(coeffs[-1])
    residual = float(np.sqrt(np.mean((A @ coeffs - thickness) ** 2)))
    return ThicknessFitResult(coeffs=b, dz_te=dz_te, residual=residual)


def fit_cst_camber(
    x: np.ndarray,
    camber: np.ndarray,
    degree: int = 3,
    regularization: float = 1e-6,
) -> CamberFitResult:
    basis = bernstein_matrix(degree, x)
    cfn = class_function(x)
    A = cfn[:, None] * basis
    lhs = A.T @ A + regularization * np.eye(degree + 1)
    rhs = A.T @ camber
    coeffs = np.linalg.solve(lhs, rhs)
    residual = float(np.sqrt(np.mean((A @ coeffs - camber) ** 2)))
    return CamberFitResult(coeffs=coeffs, residual=residual)

