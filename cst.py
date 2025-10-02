"""CST basis utilities for thickness and camber representations."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


def bernstein(n: int, i: int, x: np.ndarray) -> np.ndarray:
    """Evaluate the Bernstein polynomial of degree ``n`` and index ``i``."""

    if not 0 <= i <= n:
        raise ValueError("Bernstein index must satisfy 0 <= i <= n.")
    from math import comb

    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))


def bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """Construct a matrix of Bernstein basis evaluations."""

    return np.stack([bernstein(n, i, x) for i in range(n + 1)], axis=1)


def class_function(x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """Return the CST class function C(x) = x^n1 * (1 - x)^n2."""

    return (x ** n1) * ((1 - x) ** n2)


def class_function_derivative(x: np.ndarray, n1: float = 0.5, n2: float = 1.0) -> np.ndarray:
    """Derivative of the CST class function with respect to x."""

    cx = class_function(x, n1, n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.zeros_like(x)
        positive = (x > 0) & (x < 1)
        dlog[positive] = n1 / x[positive] - n2 / (1 - x[positive])
        dlog[~positive] = 0.0
    return cx * dlog


@lru_cache(maxsize=32)
def _cached_bernstein_matrix(n: int, num: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, num=num)
    return bernstein_matrix(n, grid)


def cst_thickness(x: np.ndarray, coeffs: Iterable[float], dz_te: float) -> np.ndarray:
    """Evaluate the CST thickness distribution at points x."""

    coeffs = np.asarray(list(coeffs), dtype=float)
    n = coeffs.size - 1
    basis = bernstein_matrix(n, x)
    return class_function(x) * (basis @ coeffs) + x * dz_te


def cst_thickness_derivative(x: np.ndarray, coeffs: Iterable[float], dz_te: float) -> np.ndarray:
    """Derivative of thickness with respect to x."""

    coeffs = np.asarray(list(coeffs), dtype=float)
    n = coeffs.size - 1
    basis = bernstein_matrix(n, x)
    dbasis = derivative_bernstein_matrix(n, x)
    return (
        class_function_derivative(x) * (basis @ coeffs)
        + class_function(x) * (dbasis @ coeffs)
        + dz_te
    )


def cst_camber(x: np.ndarray, coeffs: Iterable[float]) -> np.ndarray:
    coeffs = np.asarray(list(coeffs), dtype=float)
    n = coeffs.size - 1
    basis = bernstein_matrix(n, x)
    return class_function(x) * (basis @ coeffs)


def cst_camber_derivative(x: np.ndarray, coeffs: Iterable[float]) -> np.ndarray:
    coeffs = np.asarray(list(coeffs), dtype=float)
    n = coeffs.size - 1
    basis = bernstein_matrix(n, x)
    dbasis = derivative_bernstein_matrix(n, x)
    return class_function_derivative(x) * (basis @ coeffs) + class_function(x) * (
        dbasis @ coeffs
    )


def derivative_bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """Return the derivative matrix of Bernstein polynomials of degree n."""

    if n == 0:
        return np.zeros((x.size, 1))

    mat = np.zeros((x.size, n + 1))
    for i in range(n + 1):
        term1 = bernstein(n - 1, i - 1, x) if i > 0 else 0.0
        term2 = bernstein(n - 1, i, x) if i < n else 0.0
        mat[:, i] = n * (term1 - term2)
    return mat


def airfoil_surfaces(
    x: np.ndarray, b: Iterable[float], c: Iterable[float], dz_te: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return upper and lower surface ordinates."""

    thickness = cst_thickness(x, b, dz_te)
    camber = cst_camber(x, c)
    yu = camber + 0.5 * thickness
    yl = camber - 0.5 * thickness
    return yu, yl

