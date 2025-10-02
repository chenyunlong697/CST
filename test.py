"""One-stop script to run CST extraction and sensitivity workflows without CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from aflow import (
    AirfoilData,
    build_opt_config,
    compute_rmse,
    create_dimension_map,
    create_preview,
    load_ranges,
    prepare_airfoil,
    save_cst_coeffs,
    write_json,
    write_resampled_csv,
)
from features import health_checks
from sensitivity import (
    build_ranges,
    geom_score,
    plot_sensitivity,
    run_sensitivity,
    sample_parameters,
    save_sensitivity_csv,
)


class PipelineConfig:
    """Configuration container for the automated workflow."""

    def __init__(
        self,
        input_path: str | Path = "data/NACA4412.dat",
        output_dir: str | Path = "outputs",
        n_base: int = 200,
        sensitivity_samples: int = 256,
        topk: int = 8,
        ranges_path: str | Path | None = None,
        evaluator: str = "geom",
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.n_base = int(n_base)
        self.sensitivity_samples = int(sensitivity_samples)
        self.topk = int(topk)
        self.ranges_path = Path(ranges_path) if ranges_path else None
        self.evaluator = evaluator


def resolve_evaluator(
    spec: str,
    base_features: Dict[str, float],
    names: list[str],
) -> Callable[[np.ndarray], float]:
    if spec == "geom":
        return lambda sample: geom_score(sample, base_features, names)
    raise ValueError(f"Unsupported evaluator specification: {spec}")


def run_extraction(config: PipelineConfig) -> tuple[AirfoilData, Dict[str, float]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading airfoil from {config.input_path}")
    data = prepare_airfoil(config.input_path, n_points=config.n_base)
    rmse = compute_rmse(data)
    features = data.features.features
    derived = data.features.derived

    print("[INFO] Writing visualization artifacts...")
    create_preview(config.output_dir / "preview.png", data)
    create_dimension_map(config.output_dir / "dimension_map.png", data, features)

    print("[INFO] Exporting processed datasets...")
    write_resampled_csv(config.output_dir / "coords_resampled.csv", data)
    write_json(
        config.output_dir / "features.json",
        {
            "features": features,
            "derived": derived,
            "fit_rmse": rmse,
        },
    )
    save_cst_coeffs(config.output_dir / "cst_coeffs.json", data)

    checks = health_checks(data.x, data.thickness, data.dz_te)
    if checks:
        print("[WARN] Health checks reported issues:")
        for key, value in checks.items():
            print(f"  - {key}: {value:.5f}")
    else:
        print("[INFO] Health checks passed without warnings.")

    print(f"[INFO] Extraction complete. RMSE={rmse:.4e}")
    return data, features


def execute_sensitivity(
    config: PipelineConfig,
    features: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    overrides: Dict[str, Tuple[float, float]] = load_ranges(config.ranges_path)
    bounds = build_ranges(features, overrides=overrides or None)
    if not bounds:
        raise RuntimeError("No parameter bounds available for sensitivity analysis.")

    print("[INFO] Generating samples for sensitivity analysis...")
    samples, names, mode, problem = sample_parameters(bounds, config.sensitivity_samples)
    if mode == "lhs":
        print("[INFO] SALib not detected; using Latin Hypercube fallback.")
    else:
        print("[INFO] Using Saltelli sampling via SALib.")

    evaluator = resolve_evaluator(config.evaluator, features, names)
    print(f"[INFO] Evaluating {len(samples)} samples ({mode.upper()})...")
    results = run_sensitivity(samples, names, evaluator, mode, problem=problem)

    save_sensitivity_csv(config.output_dir / "sensitivity.csv", results)
    plot_sensitivity(config.output_dir / "sensitivity.png", results)

    opt_cfg = build_opt_config(features, bounds, sensitivity=results, top_k=config.topk)
    write_json(config.output_dir / "opt_config.json", opt_cfg)

    print("[INFO] Sensitivity analysis complete.")
    return results


def main() -> None:
    config = PipelineConfig()
    data, features = run_extraction(config)
    execute_sensitivity(config, features)
    print(f"[INFO] Workflow finished. Outputs stored in {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
