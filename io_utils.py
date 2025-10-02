"""Utility functions for loading and preprocessing airfoil coordinate files."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def load_airfoil(path: str | Path) -> np.ndarray:
    """Load an airfoil coordinate file.

    Supports Selig .dat format (x, y points following the standard ordering) and
    CSV files containing either a single (x, y) column pair or upper/lower
    surface coordinates in (xu, yu, xl, yl) columns.

    Args:
        path: Path to the input file.

    Returns:
        Array of shape (N, 2) containing ordered airfoil coordinates.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Airfoil file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        return _load_airfoil_csv(file_path)

    return _load_airfoil_selig(file_path)


def _load_airfoil_selig(path: Path) -> np.ndarray:
    points: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            points.append((x, y))

    if len(points) < 10:
        raise ValueError("Not enough coordinate points detected in Selig file.")

    coords = np.array(points, dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Invalid (NaN/inf) coordinates in airfoil file.")
    return coords


def _load_airfoil_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.lower() for name in reader.fieldnames or []]
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    def _get_column(name: str) -> Iterable[float]:
        idx = fieldnames.index(name)
        for row in rows:
            value = row[reader.fieldnames[idx]]
            yield float(value)

    lower_fields = {"xu", "yu", "xl", "yl"}
    if lower_fields.issubset(fieldnames):
        xu = np.fromiter(_get_column("xu"), dtype=float)
        yu = np.fromiter(_get_column("yu"), dtype=float)
        xl = np.fromiter(_get_column("xl"), dtype=float)
        yl = np.fromiter(_get_column("yl"), dtype=float)
        upper = np.stack([xu, yu], axis=1)
        lower = np.stack([xl[::-1], yl[::-1]], axis=1)
        return np.concatenate([upper, lower], axis=0)

    if {"x", "y"}.issubset(fieldnames):
        x = np.fromiter(_get_column("x"), dtype=float)
        y = np.fromiter(_get_column("y"), dtype=float)
        return np.stack([x, y], axis=1)

    raise ValueError(
        "CSV file must contain either columns (x,y) or (xu,yu,xl,yl)."
    )


def normalize_chord(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates so that the chord length becomes 1.

    The normalization aligns the chord with the x-axis, places the leading edge
    at x=0, and the trailing edge at x=1.

    Args:
        coords: Array of shape (N, 2).

    Returns:
        Normalized coordinates with chord length equal to 1 and monotonically
        increasing x.
    """

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be of shape (N, 2).")

    # Determine trailing edge midpoint using first and last points.
    te = 0.5 * (coords[0] + coords[-1])
    shifted = coords - te

    # Leading edge is the point farthest from the trailing edge.
    distances = np.linalg.norm(shifted, axis=1)
    le_index = int(np.argmax(distances))
    le_vector = shifted[le_index]

    chord_length = np.linalg.norm(le_vector)
    if chord_length <= 0:
        raise ValueError("Invalid airfoil geometry: chord length is zero.")

    # Rotate so that leading edge aligns with the negative x-axis.
    angle = math.atan2(le_vector[1], le_vector[0])
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = shifted @ rotation.T

    # Ensure leading edge lies on negative x side.
    if rotated[le_index, 0] > rotated[0, 0]:
        rotated[:, 1] *= -1

    scaled = rotated / chord_length

    # Shift so that leading edge at x=0 and trailing edge at x=1.
    scaled[:, 0] += 1.0

    # Enforce monotonicity by sorting by x coordinate and deduplicating.
    order = np.argsort(scaled[:, 0])
    normalized = scaled[order]

    _, unique_indices = np.unique(np.round(normalized[:, 0], decimals=8), return_index=True)
    normalized = normalized[np.sort(unique_indices)]

    return normalized


def resample_cosine(coords: np.ndarray, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample upper and lower surfaces on a cosine-spaced grid.

    Args:
        coords: Normalized coordinates (x, y).
        n_points: Number of cosine-spaced points between (0, 1).

    Returns:
        Tuple (x, yu, yl) of arrays representing the resampled surfaces.
    """

    x = coords[:, 0]
    y = coords[:, 1]

    # Identify upper and lower surfaces using a simple split around the leading edge.
    le_index = int(np.argmin(x))
    upper = coords[: le_index + 1]
    lower = coords[le_index:]

    # Ensure upper surface goes from trailing edge to leading edge.
    if upper[0, 0] < upper[-1, 0]:
        upper = upper[::-1]

    if lower[0, 0] > lower[-1, 0]:
        lower = lower[::-1]

    beta = np.linspace(0.0, math.pi, n_points, endpoint=False)[1:]
    x_target = 0.5 * (1 - np.cos(beta))

    yu = np.interp(x_target, upper[::-1, 0], upper[::-1, 1])
    yl = np.interp(x_target, lower[:, 0], lower[:, 1])

    return x_target, yu, yl

