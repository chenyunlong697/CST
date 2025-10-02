"""Sensitivity analysis utilities with optional SALib support."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from SALib.analyze import sobol
    from SALib.sample import saltelli

    HAS_SALIB = True
except Exception:  # pragma: no cover - SALib optional
    HAS_SALIB = False


DEFAULT_BOUNDS = {
    "t_max": (0.9, 1.1, "ratio"),
    "x_t": (-0.05, 0.05, "delta"),
    "f_max": (0.6, 1.4, "ratio"),
    "x_f": (-0.08, 0.08, "delta"),
    "r_le_hat": (0.9, 1.2, "scale"),
    "dz_te": (0.002, 0.004, "abs"),
    "s_rec": (-0.03, 0.03, "delta"),
    "t_015": (-0.015, 0.015, "delta"),
    "t_050": (-0.02, 0.02, "delta"),
    "t_075": (-0.02, 0.02, "delta"),
    "dt_080": (-0.3, -0.05, "fixed"),
    "dz_005": (0.0, 0.12, "abs"),
    "dz_090": (-0.10, 0.05, "abs"),
    "r_fx": (-0.004, 0.004, "delta"),
}


def build_ranges(
    features: Dict[str, float],
    overrides: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, Tuple[float, float]]:
    ranges: Dict[str, Tuple[float, float]] = {}
    for name, base in features.items():
        if name not in DEFAULT_BOUNDS:
            continue
        lb, ub, mode = DEFAULT_BOUNDS[name]
        if overrides and name in overrides:
            ranges[name] = overrides[name]
            continue
        if mode == "ratio":
            ranges[name] = (base * lb, base * ub)
        elif mode == "delta":
            ranges[name] = (base + lb, base + ub)
        elif mode == "scale":
            ranges[name] = (base * lb, base * ub)
        elif mode == "abs":
            ranges[name] = (lb, ub)
        elif mode == "fixed":
            ranges[name] = (lb, ub)
    return ranges


def lhs_sample(bounds: Dict[str, Tuple[float, float]], n_samples: int) -> np.ndarray:
    items = list(bounds.items())
    dim = len(items)
    rng = np.random.default_rng(42)
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(size=(n_samples, dim))
    a = cut[:n_samples, None]
    b = cut[1 : n_samples + 1, None]
    rdpoints = u * (b - a) + a
    H = np.zeros_like(rdpoints)
    for j in range(dim):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, j]
    for i, (_, (lower, upper)) in enumerate(items):
        H[:, i] = lower + H[:, i] * (upper - lower)
    return H


def sample_parameters(
    bounds: Dict[str, Tuple[float, float]],
    n_base: int,
) -> Tuple[np.ndarray, List[str], str, Dict[str, object] | None]:
    names = list(bounds.keys())
    if HAS_SALIB:
        problem = {
            "num_vars": len(names),
            "names": names,
            "bounds": [list(bounds[name]) for name in names],
        }
        sample = saltelli.sample(problem, n_base, calc_second_order=False)
        mode = "saltelli"
        return sample, names, mode, problem
    else:
        sample = lhs_sample(bounds, n_base)
        mode = "lhs"
        return sample, names, mode, None


def geom_score(sampled: np.ndarray, base_features: Dict[str, float], names: List[str]) -> float:
    weights = {
        "s_rec": 2.0,
        "r_le_hat": 1.5,
        "t_max": 1.0,
        "r_fx": 0.5,
    }
    penalty = 0.0
    values = dict(zip(names, sampled))
    for key, weight in weights.items():
        if key in values:
            penalty -= weight * abs(values[key] - base_features.get(key, 0.0))

    if "dt_080" in values and values["dt_080"] > -0.05:
        penalty -= 5.0 * (values["dt_080"] + 0.05) ** 2
    return penalty


def run_sensitivity(
    samples: np.ndarray,
    names: List[str],
    evaluator: Callable[[np.ndarray], float],
    mode: str,
    problem: Dict[str, object] | None = None,
) -> Dict[str, Dict[str, float]]:
    scores = np.array([evaluator(row) for row in samples])
    if HAS_SALIB and mode == "saltelli" and problem is not None:
        analysis = sobol.analyze(problem, scores, calc_second_order=False)
        return {
            name: {
                "Si": float(analysis["S1"][i]),
                "ST": float(analysis["ST"][i]),
                "Si_conf": float(analysis["S1_conf"][i]),
                "ST_conf": float(analysis["ST_conf"][i]),
            }
            for i, name in enumerate(names)
        }

    mean_score = float(np.mean(scores))
    variances = np.var(samples, axis=0)
    contributions = np.zeros_like(variances)
    for i in range(samples.shape[1]):
        contributions[i] = np.cov(samples[:, i], scores)[0, 1]
    total = np.sum(np.abs(contributions)) + 1e-8
    sensitivities = contributions / total
    return {
        name: {
            "Si": float(sensitivities[i]),
            "ST": float(abs(sensitivities[i])),
            "Si_conf": 0.0,
            "ST_conf": 0.0,
        }
        for i, name in enumerate(names)
    }


def save_sensitivity_csv(
    path: str,
    results: Dict[str, Dict[str, float]],
    order_by: str = "ST",
) -> None:
    import pandas as pd

    rows = []
    for name, metrics in results.items():
        row = {"name": name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values(order_by, ascending=False, inplace=True)
    df.to_csv(path, index=False)


def plot_sensitivity(path: str, results: Dict[str, Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    names = list(results.keys())
    st = [results[name]["ST"] for name in names]
    si = [results[name]["Si"] for name in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = np.arange(len(names))
    ax.barh(pos + 0.2, st, height=0.4, label="ST")
    ax.barh(pos - 0.2, si, height=0.4, label="Si")
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Sensitivity")
    ax.legend()
    ax.set_title("Sobol Sensitivity" if HAS_SALIB else "Approximate Sensitivity")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

