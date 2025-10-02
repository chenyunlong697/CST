"""Command line interface for CST feature extraction and sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from cst import airfoil_surfaces, cst_thickness
from features import FeatureResult, compute_features_14, health_checks
from fit import fit_cst_camber, fit_cst_thickness
from io_utils import load_airfoil, normalize_chord, resample_cosine
from sensitivity import (
    build_ranges,
    geom_score,
    plot_sensitivity,
    run_sensitivity,
    sample_parameters,
    save_sensitivity_csv,
)


@dataclass
class AirfoilData:
    x: np.ndarray
    yu: np.ndarray
    yl: np.ndarray
    thickness: np.ndarray
    camber: np.ndarray
    thickness_fit: np.ndarray
    camber_fit: np.ndarray
    dz_te: float
    features: FeatureResult


def prepare_airfoil(path: Path, n_points: int) -> AirfoilData:
    raw = load_airfoil(path)
    normalized = normalize_chord(raw)
    x_res, yu_res, yl_res = resample_cosine(normalized, n_points=n_points)
    thickness = yu_res - yl_res
    camber = 0.5 * (yu_res + yl_res)

    thickness_fit = fit_cst_thickness(x_res, thickness)
    camber_fit = fit_cst_camber(x_res, camber)

    features = compute_features_14(
        x_res, thickness_fit.coeffs, camber_fit.coeffs, thickness_fit.dz_te
    )

    return AirfoilData(
        x=x_res,
        yu=yu_res,
        yl=yl_res,
        thickness=thickness,
        camber=camber,
        thickness_fit=thickness_fit.coeffs,
        camber_fit=camber_fit.coeffs,
        dz_te=thickness_fit.dz_te,
        features=features,
    )


def create_preview(path: Path, data: AirfoilData) -> None:
    import matplotlib.pyplot as plt

    yu_fit, yl_fit = airfoil_surfaces(data.x, data.thickness_fit, data.camber_fit, data.dz_te)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(data.x, data.yu, label="Upper")
    ax.plot(data.x, data.yl, label="Lower")
    ax.plot(data.x, yu_fit, "--", label="Upper (fit)")
    ax.plot(data.x, yl_fit, "--", label="Lower (fit)")
    ax.set_title("Airfoil Surfaces")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(data.x, data.thickness, label="Thickness")
    ax.set_title("Thickness Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.grid(True, linestyle=":")

    ax = axes[2]
    ax.plot(data.x, data.camber, label="Camber")
    ax.set_title("Camber Line")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def create_dimension_map(path: Path, data: AirfoilData, features: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    yu_fit, yl_fit = airfoil_surfaces(data.x, data.thickness_fit, data.camber_fit, data.dz_te)
    ax.fill_between(data.x, yu_fit, yl_fit, color="#87ceeb", alpha=0.3, label="Airfoil")
    ax.plot(data.x, yu_fit, color="#0b486b")
    ax.plot(data.x, yl_fit, color="#0b486b")

    def _thickness_at(x_val: float) -> float:
        return float(cst_thickness(np.array([x_val]), data.thickness_fit, data.dz_te)[0])

    annotations = {
        "t_max": (features["x_t"], features["t_max"] / 2),
        "x_t": (features["x_t"], 0.0),
        "f_max": (features["x_f"], features["f_max"]),
        "x_f": (features["x_f"], features["f_max"] * 0.2),
        "r_le_hat": (0.03, 0.5 * _thickness_at(0.03)),
        "t_015": (0.15, 0.5 * _thickness_at(0.15)),
        "t_050": (0.5, 0.5 * _thickness_at(0.5)),
        "t_075": (0.75, 0.5 * _thickness_at(0.75)),
        "dt_080": (0.8, 0.5 * _thickness_at(0.8)),
        "dz_005": (0.05, features["dz_005"]),
        "dz_090": (0.9, features["dz_090"]),
        "r_fx": (0.9, features["r_fx"]),
        "s_rec": (0.75, 0.5 * (_thickness_at(0.6) + _thickness_at(0.9))),
        "dz_te": (0.98, 0.5 * features["dz_te"]),
    }

    for key, (x_pos, y_pos) in annotations.items():
        ax.scatter([x_pos], [y_pos], s=40, label=key)
        ax.text(x_pos, y_pos, f" {key}", fontsize=8, ha="left", va="bottom")

    for name in ("x_t", "x_f"):
        ax.axvline(annotations[name][0], color="gray", linestyle="--", linewidth=0.8)

    ax.set_aspect(2.0, adjustable="box")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Feature Locations")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_resampled_csv(path: Path, data: AirfoilData) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "x": data.x,
            "yu": data.yu,
            "yl": data.yl,
            "t": data.thickness,
            "z": data.camber,
        }
    )
    df.to_csv(path, index=False)


def build_opt_config(
    features: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    sensitivity: Dict[str, Dict[str, float]] | None = None,
    top_k: int = 8,
) -> Dict[str, object]:
    recommended = [
        "t_max",
        "x_t",
        "f_max",
        "x_f",
        "r_le_hat",
        "dz_te",
        "s_rec",
        "t_050",
        "dt_080",
        "dz_090",
    ]

    top = []
    threshold = []
    if sensitivity:
        sorted_items = sorted(sensitivity.items(), key=lambda kv: kv[1]["ST"], reverse=True)
        top = [name for name, _ in sorted_items[:top_k]]
        threshold = [name for name, val in sorted_items if val["ST"] > 0.05]

    return {
        "opt_vars_recommended": recommended,
        "opt_vars_by_sensitivity": {
            "top_k": top,
            "threshold": threshold,
        },
        "bounds": {name: list(bounds[name]) for name in bounds},
        "notes": "Generated for downstream optimization workflows.",
    }


def save_cst_coeffs(path: Path, data: AirfoilData) -> None:
    coeffs = {
        **{f"b{i}": float(val) for i, val in enumerate(data.thickness_fit)},
        **{f"c{j}": float(val) for j, val in enumerate(data.camber_fit)},
        "dz_te": float(data.dz_te),
    }
    write_json(path, coeffs)


def compute_rmse(data: AirfoilData) -> float:
    yu_fit, yl_fit = airfoil_surfaces(data.x, data.thickness_fit, data.camber_fit, data.dz_te)
    err = np.mean((yu_fit - data.yu) ** 2 + (yl_fit - data.yl) ** 2)
    return float(math.sqrt(err / 2.0))


def cmd_extract(args: argparse.Namespace) -> None:
    path_in = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_airfoil(path_in, n_points=args.n_base)
    rmse = compute_rmse(data)
    features = data.features.features

    create_preview(out_dir / "preview.png", data)
    create_dimension_map(out_dir / "dimension_map.png", data, features)
    write_resampled_csv(out_dir / "coords_resampled.csv", data)
    write_json(
        out_dir / "features.json",
        {
            "features": features,
            "derived": data.features.derived,
            "fit_rmse": rmse,
        },
    )
    save_cst_coeffs(out_dir / "cst_coeffs.json", data)

    bounds = build_ranges(features)
    opt_cfg = build_opt_config(features, bounds)
    write_json(out_dir / "opt_config.json", opt_cfg)

    checks = health_checks(data.x, data.thickness, data.dz_te)
    if checks:
        print("[WARN] Health checks reported issues:")
        for key, value in checks.items():
            print(f"  - {key}: {value:.5f}")

    print(f"Extraction complete. RMSE={rmse:.4e}")


def load_ranges(path: Path | None) -> Dict[str, Tuple[float, float]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {key: tuple(value) for key, value in payload.items()}


def resolve_evaluator(
    spec: str,
    base_features: Dict[str, float],
    names: List[str],
) -> Callable[[np.ndarray], float]:
    if spec == "geom":
        return lambda sample: geom_score(sample, base_features, names)

    if spec.startswith("plugin:"):
        plugin_path = Path(spec.split(":", 1)[1])
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin evaluator not found: {plugin_path}")
        module = runpy.run_path(str(plugin_path))
        if "evaluate" not in module:
            raise ValueError("Plugin file must define an 'evaluate' function.")

        def _call(sample: np.ndarray) -> float:
            values = dict(zip(names, sample))
            result = module["evaluate"](values, base_features)
            return float(result)

        return _call

    raise ValueError(f"Unsupported evaluator specification: {spec}")


def cmd_sensitivity(args: argparse.Namespace) -> None:
    path_in = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_airfoil(path_in, n_points=args.n_base)
    features = data.features.features

    overrides = load_ranges(Path(args.ranges) if args.ranges else None)
    bounds = build_ranges(features, overrides=overrides)

    samples, names, mode, problem = sample_parameters(bounds, args.n_base)
    if mode == "lhs":
        print("[INFO] SALib not detected; using Latin Hypercube fallback.")
    evaluator = resolve_evaluator(args.evaluator, features, names)
    print(f"Running sensitivity using {mode.upper()} sampling on {len(samples)} samples...")

    results = run_sensitivity(samples, names, evaluator, mode, problem=problem)

    save_sensitivity_csv(out_dir / "sensitivity.csv", results)
    plot_sensitivity(out_dir / "sensitivity.png", results)

    opt_cfg = build_opt_config(features, bounds, sensitivity=results, top_k=args.topk)
    write_json(out_dir / "opt_config.json", opt_cfg)

    print("Sensitivity analysis complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CST extraction toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract CST coefficients and features")
    extract.add_argument("--in", dest="input", required=True, help="Input airfoil file")
    extract.add_argument("--out", dest="out", required=True, help="Output directory")
    extract.add_argument(
        "--n_base",
        dest="n_base",
        type=int,
        default=200,
        help="Number of resampling points",
    )
    extract.set_defaults(func=cmd_extract)

    sens = sub.add_parser("sens", help="Run sensitivity analysis")
    sens.add_argument("--in", dest="input", required=True, help="Input airfoil file")
    sens.add_argument("--out", dest="out", required=True, help="Output directory")
    sens.add_argument(
        "--n_base",
        dest="n_base",
        type=int,
        default=200,
        help="Base sample count for Saltelli/LHS",
    )
    sens.add_argument(
        "--evaluator",
        dest="evaluator",
        default="geom",
        help="Evaluator specification (geom or plugin:path.py)",
    )
    sens.add_argument(
        "--ranges",
        dest="ranges",
        help="Optional JSON file overriding parameter bounds",
    )
    sens.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=8,
        help="Number of top variables for recommendation",
    )
    sens.set_defaults(func=cmd_sensitivity)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

