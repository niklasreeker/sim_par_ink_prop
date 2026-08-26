#!/usr/bin/env python3
"""Build and use a first residual calibration field for the ink model.

The hybrid measurement model is

    y_hybrid(w, T) = y_physics(w, T) + A(w)

with separate residual fields for density and sound velocity.  Evaporation is
not learned as part of A(w).  Instead, the script reconstructs the effective
composition at every timestamp before calculating the residual.

Current evaporation assumption
------------------------------
Only IPA evaporates.  For a cumulative IPA loss E(t),

    m_IPA,eff(t) = m_IPA,nominal(t) - E(t)
    m_total,eff(t) = m_total,nominal(t) - E(t)

and all mass percentages are recalculated.  SL120 is interpreted as 20 wt-%
Al, 40 wt-% IPA and 40 wt-% PG.

The evaporation rate can be supplied from an independent gravimetric test or
estimated from within-phase changes of density and sound velocity.  During
rate estimation, each recipe phase receives a free residual intercept.  This
prevents a static model offset from being mistaken for evaporation.

The saved JSON file is portable and contains the residual nodes, IDW settings,
quality information, evaporation settings and fingerprints of the calculator
and parameter tables.  It can be loaded by the ``predict`` and ``evaluate``
subcommands or by another Python project.

Examples
--------
Build a field from sample 3 and estimate the IPA loss rate::

    python residual_calibration_field.py build \
      --input "measurement_data/Kennfeld_v2 (10) korrigiert.csv" \
      --samples 3 --evaporation-mode estimate

Build with a gravimetrically determined loss rate::

    python residual_calibration_field.py build \
      --input measurements.csv --samples 3 \
      --evaporation-mode fixed --evaporation-rate-g-h 1.50

Predict one point::

    python residual_calibration_field.py predict \
      --model results/residual_calibration/calibration_field.json \
      --al 1.8 --ipa 4.0 --pg 4.5 --temperature 23.0

Evaluate a saved field against another CSV::

    python residual_calibration_field.py evaluate \
      --model results/residual_calibration/calibration_field.json \
      --input new_measurements.csv --samples all \
      --evaporation-mode fixed --evaporation-rate-g-h 1.50

The signed calibration residual always uses ``measurement - physics`` so it
can be added directly to the InkCalculator result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
SCRIPT_VERSION = "2.0-interactive"
DEFAULT_INPUT = SCRIPT_DIR / "measurement_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "residual_calibration"
DEFAULT_TABLES = REPO_DIR / "tables_parameters"
DEFAULT_CALCULATOR = REPO_DIR / "ink_calculator.py"

SL120 = {"Al": 0.20, "IPA": 0.40, "PG": 0.40}
MASS_COLUMNS = ["m_SL120", "m_Wasser", "m_IPA", "m_PG", "m_MG"]
MEASUREMENT_COLUMNS = ["Rho_M", "C_M", "T_M"]
COMPOSITION_COLUMNS = ["Al_wt_pct_eff", "IPA_wt_pct_eff", "PG_wt_pct_eff"]


def finite_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError("Value must be finite.")
    return number


def non_negative_float(value: str) -> float:
    number = finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be greater than or equal to zero.")
    return number


def positive_float(value: str) -> float:
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return number


def parse_sample_tokens(tokens: list[str] | None) -> set[int] | None:
    if not tokens:
        return None
    flattened: list[str] = []
    for token in tokens:
        flattened.extend(part.strip() for part in str(token).split(","))
    if any(token.lower() == "all" for token in flattened):
        return None
    try:
        return {int(token) for token in flattened if token}
    except ValueError as exc:
        raise ValueError("--samples must contain integers or 'all'.") from exc


def add_common_calculator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calculator",
        type=Path,
        default=DEFAULT_CALCULATOR,
        help="Path to ink_calculator.py (default: repository root).",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=DEFAULT_TABLES,
        help="Path to tables_parameters (default: repository tables_parameters).",
    )


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more measurement CSV files or directories containing CSV files.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["all"],
        help="ProbeNr values, separated by spaces or commas; use 'all' for all samples.",
    )
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--time-column", default="UTC Time")
    parser.add_argument(
        "--phase-gap-min",
        type=non_negative_float,
        default=60.0,
        help="Start a new phase after a larger time gap (default: 60 min).",
    )


def add_evaporation_arguments(
    parser: argparse.ArgumentParser, default_mode: str = "estimate"
) -> None:
    parser.add_argument(
        "--evaporation-mode",
        choices=["estimate", "fixed", "none"],
        default=default_mode,
        help="Estimate the IPA loss rate, use a fixed rate, or disable evaporation.",
    )
    parser.add_argument(
        "--evaporation-rate-g-h",
        type=non_negative_float,
        default=1.50,
        help="Fixed rate or prior centre for estimation (default: 1.50 g/h).",
    )
    parser.add_argument(
        "--evaporation-rate-max-g-h",
        type=positive_float,
        default=5.0,
        help="Upper bound for data-driven rate estimation (default: 5 g/h).",
    )
    parser.add_argument(
        "--evaporation-prior-sigma-g-h",
        type=positive_float,
        default=0.50,
        help="Prior uncertainty around --evaporation-rate-g-h (default: 0.50 g/h).",
    )
    parser.add_argument(
        "--no-evaporation-prior",
        action="store_true",
        help="Estimate only from measurement drift, without the 1.5 g/h prior.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, use and evaluate an evaporation-corrected residual ink field."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build and save a calibration field.")
    add_input_arguments(build)
    add_common_calculator_arguments(build)
    add_evaporation_arguments(build, default_mode="estimate")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument(
        "--quality-mode",
        choices=["auto", "flags", "low-noise", "all"],
        default="auto",
        help="Measurement selection strategy (default: auto).",
    )
    build.add_argument(
        "--minimum-points-per-phase",
        type=int,
        default=5,
        help="Minimum preferred calibration points per phase (default: 5).",
    )
    build.add_argument(
        "--settling-fraction",
        type=float,
        default=0.30,
        help="Fraction at the beginning of a phase excluded by low-noise fallback.",
    )
    build.add_argument(
        "--low-noise-keep-fraction",
        type=float,
        default=0.60,
        help="Fraction of late low-noise candidates retained (default: 0.60).",
    )
    build.add_argument(
        "--idw-power",
        type=positive_float,
        default=2.0,
        help="Inverse-distance weighting power (default: 2).",
    )
    build.add_argument(
        "--idw-neighbors",
        type=int,
        default=4,
        help="Number of residual nodes used for a prediction; 0 uses all.",
    )

    predict = subparsers.add_parser("predict", help="Predict a single composition.")
    predict.add_argument("--model", type=Path, required=True)
    add_common_calculator_arguments(predict)
    predict.add_argument("--al", type=finite_float, required=True)
    predict.add_argument("--ipa", type=finite_float, required=True)
    predict.add_argument("--pg", type=finite_float, required=True)
    predict.add_argument("--temperature", type=finite_float, required=True)
    predict.add_argument(
        "--json", action="store_true", help="Print the prediction as JSON."
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate physics and hybrid predictions against another CSV."
    )
    evaluate.add_argument("--model", type=Path, required=True)
    add_input_arguments(evaluate)
    add_common_calculator_arguments(evaluate)
    add_evaporation_arguments(evaluate, default_mode="fixed")
    evaluate.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "evaluation"
    )

    return parser


def resolve_csv_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    unique = list(dict.fromkeys(files))
    if not unique:
        raise FileNotFoundError("No CSV input files were found.")
    return unique


def load_measurements(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in resolve_csv_files(paths):
        frame = pd.read_csv(path, comment="/", skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["Source_File"] = path.name
        frame["Source_Path"] = str(path)
        frame["Source_Row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
        print(f"Loaded {path.name}: {len(frame)} rows")
    return pd.concat(frames, ignore_index=True)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def filter_samples(df: pd.DataFrame, tokens: list[str] | None) -> pd.DataFrame:
    require_columns(df, ["ProbeNr"])
    wanted = parse_sample_tokens(tokens)
    if wanted is None:
        return df.copy()
    result = df[pd.to_numeric(df["ProbeNr"], errors="coerce").isin(wanted)].copy()
    found = set(pd.to_numeric(result["ProbeNr"], errors="coerce").dropna().astype(int))
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Requested ProbeNr values were not found: {missing}")
    return result


def add_timestamps_and_phases(
    df: pd.DataFrame, date_column: str, time_column: str, phase_gap_min: float
) -> pd.DataFrame:
    result = df.copy()
    require_columns(result, MASS_COLUMNS + ["ProbeNr", date_column, time_column])

    combined = (
        result[date_column].astype("string").str.strip()
        + " "
        + result[time_column].astype("string").str.strip()
    )
    result["Measurement_Time_UTC"] = pd.to_datetime(
        combined, errors="coerce", format="mixed", dayfirst=True, utc=True
    )
    if result["Measurement_Time_UTC"].isna().any():
        bad = int(result["Measurement_Time_UTC"].isna().sum())
        raise ValueError(f"Could not parse {bad} measurement timestamps.")

    for column in MASS_COLUMNS + MEASUREMENT_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result["Experiment_Key"] = (
        result["Source_File"].astype(str)
        + "|Probe="
        + result["ProbeNr"].astype(str)
    )
    result["Phase"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["Experiment_Elapsed_h"] = np.nan
    result["Phase_Elapsed_h"] = np.nan

    for _, indexes in result.groupby("Experiment_Key", sort=False).groups.items():
        part = result.loc[indexes].sort_values("Measurement_Time_UTC")
        first_time = part["Measurement_Time_UTC"].iloc[0]
        elapsed = (part["Measurement_Time_UTC"] - first_time).dt.total_seconds() / 3600.0
        gaps = part["Measurement_Time_UTC"].diff().dt.total_seconds() / 60.0
        recipe_change = part[MASS_COLUMNS].ne(part[MASS_COLUMNS].shift()).any(axis=1)
        gap_change = gaps.gt(phase_gap_min) if phase_gap_min > 0 else False
        new_phase = recipe_change | gap_change
        new_phase.iloc[0] = True
        phase = new_phase.cumsum().astype(int)
        phase_start = part.groupby(phase)["Measurement_Time_UTC"].transform("min")
        phase_elapsed = (
            part["Measurement_Time_UTC"] - phase_start
        ).dt.total_seconds() / 3600.0
        result.loc[part.index, "Experiment_Elapsed_h"] = elapsed.to_numpy()
        result.loc[part.index, "Phase_Elapsed_h"] = phase_elapsed.to_numpy()
        result.loc[part.index, "Phase"] = phase.to_numpy()

    return result


def add_nominal_component_masses(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    require_columns(result, MASS_COLUMNS)
    result["m_Al_nom_g"] = SL120["Al"] * result["m_SL120"]
    result["m_IPA_nom_g"] = SL120["IPA"] * result["m_SL120"] + result["m_IPA"]
    result["m_PG_nom_g"] = SL120["PG"] * result["m_SL120"] + result["m_PG"]
    result["m_Water_nom_g"] = result["m_Wasser"]
    result["m_MG_nom_g"] = result["m_MG"]
    result["m_Total_nom_g"] = result[MASS_COLUMNS].sum(axis=1, min_count=5)
    invalid = (
        result[["m_Al_nom_g", "m_IPA_nom_g", "m_PG_nom_g", "m_Water_nom_g"]]
        .lt(0)
        .any(axis=1)
        | result["m_Total_nom_g"].le(0)
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} rows with invalid component masses.")
    if result["m_MG_nom_g"].fillna(0).abs().gt(1e-12).any():
        raise ValueError(
            "m_MG is non-zero, but the current InkCalculator has no MG model."
        )
    return result


def apply_ipa_evaporation(
    df: pd.DataFrame, rates_by_experiment: dict[str, float]
) -> pd.DataFrame:
    result = df.copy()
    rates = result["Experiment_Key"].map(rates_by_experiment)
    if rates.isna().any():
        missing = sorted(result.loc[rates.isna(), "Experiment_Key"].unique())
        raise ValueError(f"Missing evaporation rates for: {missing}")

    result["IPA_Evaporation_Rate_g_h"] = rates.astype(float)
    result["IPA_Loss_g"] = rates * result["Experiment_Elapsed_h"]
    available = result["m_IPA_nom_g"]
    if (result["IPA_Loss_g"] >= available).any():
        row = result.loc[result["IPA_Loss_g"] >= available].iloc[0]
        raise ValueError(
            "Estimated IPA loss reaches the available IPA mass at "
            f"{row['Experiment_Key']}, source row {int(row['Source_Row'])}."
        )

    result["m_IPA_eff_g"] = result["m_IPA_nom_g"] - result["IPA_Loss_g"]
    result["m_Total_eff_g"] = result["m_Total_nom_g"] - result["IPA_Loss_g"]
    denominator = result["m_Total_eff_g"]
    result["Al_wt_pct_eff"] = 100.0 * result["m_Al_nom_g"] / denominator
    result["IPA_wt_pct_eff"] = 100.0 * result["m_IPA_eff_g"] / denominator
    result["PG_wt_pct_eff"] = 100.0 * result["m_PG_nom_g"] / denominator
    result["Water_wt_pct_eff"] = 100.0 * result["m_Water_nom_g"] / denominator
    result["Composition_Sum_wt_pct"] = result[
        ["Al_wt_pct_eff", "IPA_wt_pct_eff", "PG_wt_pct_eff", "Water_wt_pct_eff"]
    ].sum(axis=1)
    return result


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calculator(calculator_path: Path, tables_dir: Path):
    calculator_path = calculator_path.expanduser().resolve()
    tables_dir = tables_dir.expanduser().resolve()
    if not calculator_path.is_file():
        raise FileNotFoundError(f"Calculator not found: {calculator_path}")
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"Parameter table directory not found: {tables_dir}")

    module_name = f"ink_calculator_calibration_{hash(calculator_path)}"
    spec = importlib.util.spec_from_file_location(module_name, calculator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load calculator module: {calculator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    calculator_class = getattr(module, "InkCalculator", None)
    if calculator_class is None:
        raise ImportError(f"InkCalculator class not found in {calculator_path}")
    return calculator_class(tables_dir=str(tables_dir))


def simulate_rows(df: pd.DataFrame, calculator) -> pd.DataFrame:
    result = df.copy()
    result["Rho_Physics_kg_m3"] = np.nan
    result["C_Physics_m_s"] = np.nan
    result["Simulation_Status"] = "pending"
    for index, row in result.iterrows():
        try:
            arguments = {
                "al": float(row["Al_wt_pct_eff"]),
                "ipa": float(row["IPA_wt_pct_eff"]),
                "pg": float(row["PG_wt_pct_eff"]),
                "temperature": float(row["T_M"]),
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rho = 1000.0 * float(calculator.density(**arguments))
                sound = float(calculator.sound_velocity(**arguments))
            if not np.isfinite(rho) or not np.isfinite(sound):
                raise ValueError("Calculator returned a non-finite result.")
            result.at[index, "Rho_Physics_kg_m3"] = rho
            result.at[index, "C_Physics_m_s"] = sound
            result.at[index, "Simulation_Status"] = "ok"
        except Exception as exc:
            result.at[index, "Simulation_Status"] = (
                f"error: {type(exc).__name__}: {exc}"
            )
    result["A_Rho_kg_m3"] = result["Rho_M"] - result["Rho_Physics_kg_m3"]
    result["A_C_m_s"] = result["C_M"] - result["C_Physics_m_s"]
    return result


def _safe_numeric(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def select_calibration_rows(
    df: pd.DataFrame,
    quality_mode: str,
    minimum_points: int,
    settling_fraction: float,
    low_noise_keep_fraction: float,
) -> pd.DataFrame:
    if minimum_points < 2:
        raise ValueError("--minimum-points-per-phase must be at least 2.")
    if not 0 <= settling_fraction < 1:
        raise ValueError("--settling-fraction must be in [0, 1).")
    if not 0 < low_noise_keep_fraction <= 1:
        raise ValueError("--low-noise-keep-fraction must be in (0, 1].")

    result = df.copy()
    finite = result[MEASUREMENT_COLUMNS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    finite &= result["m_Total_nom_g"].gt(0)
    finite &= _safe_numeric(result, "N", 1).gt(0)
    finite &= _safe_numeric(result, "SensOK", 1).eq(1)
    result["Selected_For_Calibration"] = False
    result["Selection_Method"] = "not selected"

    group_columns = ["Experiment_Key", "Phase"]
    for _, indexes in result.groupby(group_columns, sort=False).groups.items():
        part = result.loc[indexes].sort_values("Measurement_Time_UTC")
        valid_indexes = part.index[finite.loc[part.index]]
        if len(valid_indexes) == 0:
            continue

        flag_mask = finite.loc[part.index].copy()
        for flag in ("Gueltig", "Stabil"):
            if flag in result.columns:
                flag_mask &= _safe_numeric(part, flag, 0).eq(1)
        flag_indexes = part.index[flag_mask]

        def low_noise_indexes() -> pd.Index:
            valid_part = part.loc[valid_indexes]
            cut = int(math.floor(len(valid_part) * settling_fraction))
            late = valid_part.iloc[cut:]
            if late.empty:
                late = valid_part
            rho_noise = _safe_numeric(late, "Rho_S", np.nan)
            c_noise = _safe_numeric(late, "C_S", np.nan)
            rho_scale = float(rho_noise.median()) if rho_noise.notna().any() else 1.0
            c_scale = float(c_noise.median()) if c_noise.notna().any() else 1.0
            rho_scale = max(rho_scale, np.finfo(float).eps)
            c_scale = max(c_scale, np.finfo(float).eps)
            score = rho_noise.fillna(rho_scale) / rho_scale + c_noise.fillna(c_scale) / c_scale
            keep = max(minimum_points, int(math.ceil(len(late) * low_noise_keep_fraction)))
            keep = min(keep, len(late))
            return score.nsmallest(keep).index

        if quality_mode == "all":
            chosen, method = valid_indexes, "all finite SensOK rows"
        elif quality_mode == "flags":
            chosen, method = flag_indexes, "Gueltig=Stabil=SensOK=1"
        elif quality_mode == "low-noise":
            chosen, method = low_noise_indexes(), "settled low-noise fallback"
        elif len(flag_indexes) >= minimum_points:
            chosen, method = flag_indexes, "quality flags"
        else:
            chosen, method = low_noise_indexes(), "automatic low-noise fallback"

        result.loc[chosen, "Selected_For_Calibration"] = True
        result.loc[chosen, "Selection_Method"] = method

    if not result["Selected_For_Calibration"].any():
        raise ValueError("Quality selection did not retain any calibration rows.")
    return result


def _row_uncertainty(df: pd.DataFrame, column: str, floor: float) -> np.ndarray:
    values = _safe_numeric(df, column, np.nan).to_numpy(float)
    finite = values[np.isfinite(values) & (values > 0)]
    replacement = float(np.median(finite)) if finite.size else floor
    values = np.where(np.isfinite(values) & (values > 0), values, replacement)
    return np.maximum(values, floor)


def huber_loss(values: np.ndarray, delta: float = 1.5) -> float:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return float(np.sum(0.5 * quadratic**2 + delta * linear))


def drift_objective(
    rate: float,
    experiment: pd.DataFrame,
    calculator,
    prior_rate: float,
    prior_sigma: float,
    use_prior: bool,
) -> float:
    keyed = {str(experiment["Experiment_Key"].iloc[0]): float(rate)}
    simulated = simulate_rows(apply_ipa_evaporation(experiment, keyed), calculator)
    simulated = simulated[
        simulated["Selected_For_Calibration"] & simulated["Simulation_Status"].eq("ok")
    ]
    if len(simulated) < 4:
        return np.inf

    objective = 0.0
    for _, phase in simulated.groupby("Phase"):
        if len(phase) < 2:
            continue
        for residual_column, sd_column, floor in (
            ("A_Rho_kg_m3", "Rho_S", 0.01),
            ("A_C_m_s", "C_S", 0.05),
        ):
            residual = phase[residual_column].to_numpy(float)
            sigma = _row_uncertainty(phase, sd_column, floor)
            base_weight = 1.0 / sigma**2
            centre = float(np.average(residual, weights=base_weight))
            standardized = (residual - centre) / sigma
            objective += huber_loss(standardized)
    if use_prior:
        objective += 0.5 * ((rate - prior_rate) / prior_sigma) ** 2
    return float(objective)


@dataclass
class EvaporationEstimate:
    experiment_key: str
    mode: str
    rate_g_h: float
    objective: float | None
    objective_at_zero: float | None
    relative_improvement_pct: float | None
    boundary_warning: bool


def determine_evaporation_rates(
    df: pd.DataFrame,
    calculator,
    mode: str,
    fixed_or_prior_rate: float,
    maximum_rate: float,
    prior_sigma: float,
    use_prior: bool,
) -> tuple[dict[str, float], list[EvaporationEstimate]]:
    rates: dict[str, float] = {}
    summaries: list[EvaporationEstimate] = []
    for key, experiment in df.groupby("Experiment_Key", sort=False):
        key = str(key)
        if mode == "none":
            rate = 0.0
            estimate = EvaporationEstimate(key, mode, rate, None, None, None, False)
        elif mode == "fixed":
            rate = fixed_or_prior_rate
            estimate = EvaporationEstimate(key, mode, rate, None, None, None, False)
        else:
            objective = lambda candidate: drift_objective(
                candidate,
                experiment,
                calculator,
                fixed_or_prior_rate,
                prior_sigma,
                use_prior,
            )
            optimized = minimize_scalar(
                objective,
                bounds=(0.0, maximum_rate),
                method="bounded",
                options={"xatol": 1e-4},
            )
            if not optimized.success or not np.isfinite(optimized.fun):
                raise RuntimeError(f"Evaporation optimization failed for {key}.")
            rate = float(optimized.x)
            at_zero = float(objective(0.0))
            improvement = (
                100.0 * (at_zero - float(optimized.fun)) / at_zero if at_zero > 0 else np.nan
            )
            boundary = rate < 0.01 * maximum_rate or rate > 0.99 * maximum_rate
            estimate = EvaporationEstimate(
                key,
                mode,
                rate,
                float(optimized.fun),
                at_zero,
                float(improvement),
                boundary,
            )
        rates[key] = rate
        summaries.append(estimate)
    return rates, summaries


def robust_location(values: np.ndarray, base_weights: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(base_weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    centre = float(np.median(values))
    for _ in range(30):
        deviations = values - centre
        scale = 1.4826 * float(np.median(np.abs(deviations)))
        scale = max(scale, np.std(values) * 0.1, 1e-12)
        robust = np.ones_like(values)
        large = np.abs(deviations) > 1.5 * scale
        robust[large] = 1.5 * scale / np.abs(deviations[large])
        combined = weights * robust
        updated = float(np.average(values, weights=combined))
        if abs(updated - centre) < 1e-10:
            centre = updated
            break
        centre = updated
    deviations = values - centre
    robust_sd = 1.4826 * float(np.median(np.abs(deviations)))
    effective_n = float(weights.sum() ** 2 / np.sum(weights**2))
    standard_error = robust_sd / math.sqrt(max(effective_n, 1.0))
    return centre, robust_sd, standard_error


def build_nodes(simulated: pd.DataFrame) -> pd.DataFrame:
    selected = simulated[
        simulated["Selected_For_Calibration"] & simulated["Simulation_Status"].eq("ok")
    ].copy()
    if selected.empty:
        raise ValueError("No selected rows were simulated successfully.")

    rows: list[dict[str, Any]] = []
    for (key, phase_number), phase in selected.groupby(
        ["Experiment_Key", "Phase"], sort=False
    ):
        rho_sigma = _row_uncertainty(phase, "Rho_S", 0.01)
        c_sigma = _row_uncertainty(phase, "C_S", 0.05)
        a_rho, sd_rho, se_rho = robust_location(
            phase["A_Rho_kg_m3"].to_numpy(float), 1.0 / rho_sigma**2
        )
        a_c, sd_c, se_c = robust_location(
            phase["A_C_m_s"].to_numpy(float), 1.0 / c_sigma**2
        )
        method = "; ".join(sorted(set(phase["Selection_Method"].astype(str))))
        rows.append(
            {
                "Experiment_Key": str(key),
                "Source_File": str(phase["Source_File"].iloc[0]),
                "ProbeNr": int(phase["ProbeNr"].iloc[0]),
                "Phase": int(phase_number),
                "N_Selected": int(len(phase)),
                "Selection_Method": method,
                "Al_wt_pct": float(np.average(phase["Al_wt_pct_eff"])),
                "IPA_wt_pct": float(np.average(phase["IPA_wt_pct_eff"])),
                "PG_wt_pct": float(np.average(phase["PG_wt_pct_eff"])),
                "Water_wt_pct": float(np.average(phase["Water_wt_pct_eff"])),
                "Temperature_Mean_C": float(np.average(phase["T_M"])),
                "Temperature_Min_C": float(phase["T_M"].min()),
                "Temperature_Max_C": float(phase["T_M"].max()),
                "IPA_Loss_Mean_g": float(np.average(phase["IPA_Loss_g"])),
                "A_Rho_kg_m3": a_rho,
                "A_Rho_Robust_SD_kg_m3": sd_rho,
                "A_Rho_SE_kg_m3": se_rho,
                "A_C_m_s": a_c,
                "A_C_Robust_SD_m_s": sd_c,
                "A_C_SE_m_s": se_c,
            }
        )
    nodes = pd.DataFrame(rows)
    if nodes.empty:
        raise ValueError("No residual nodes could be built.")
    return nodes


def node_scaler(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = nodes[["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct"]].to_numpy(float)
    centre = np.mean(x, axis=0)
    scale = np.ptp(x, axis=0)
    fallback = np.std(x, axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return centre, scale


def model_payload(
    nodes: pd.DataFrame,
    evaporation_summaries: list[EvaporationEstimate],
    args: argparse.Namespace,
) -> dict[str, Any]:
    centre, scale = node_scaler(nodes)
    table_hashes = {}
    tables_path = args.tables.expanduser().resolve()
    for name in (
        "ipa_density.csv",
        "pg_density.csv",
        "ipa_sound.csv",
        "pg_sound.csv",
    ):
        digest = sha256_file(tables_path / name)
        if digest:
            table_hashes[name] = digest

    return {
        "schema": "ink-residual-calibration-field",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "equation": "measurement = physics + A(w)",
        "residual_definition": "A = measurement - physics",
        "composition_axes": ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct"],
        "temperature_model": "InkCalculator only; A(w) has no learned temperature term",
        "evaporation_assumption": "Only IPA evaporates; loss is cumulative from the first timestamp per source file and ProbeNr.",
        "evaporation": [summary.__dict__ for summary in evaporation_summaries],
        "interpolation": {
            "method": "scaled_inverse_distance_weighting",
            "power": float(args.idw_power),
            "neighbors": int(args.idw_neighbors),
            "centre": centre.tolist(),
            "scale": scale.tolist(),
        },
        "calculator": {
            "filename": args.calculator.name,
            "sha256": sha256_file(args.calculator.expanduser().resolve()),
            "table_sha256": table_hashes,
        },
        "nodes": nodes.replace({np.nan: None}).to_dict(orient="records"),
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_model(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("schema") != "ink-residual-calibration-field":
        raise ValueError(f"Unsupported calibration model: {path}")
    if not model.get("nodes"):
        raise ValueError("Calibration model contains no residual nodes.")
    return model


def warn_on_provenance_mismatch(
    model: dict[str, Any], calculator_path: Path, tables_dir: Path
) -> None:
    """Warn when a saved A(w) field is used with different physics inputs."""
    provenance = model.get("calculator", {})
    expected_calculator = provenance.get("sha256")
    actual_calculator = sha256_file(calculator_path.expanduser().resolve())
    if expected_calculator and actual_calculator != expected_calculator:
        print(
            "WARNING: ink_calculator.py differs from the version used to build "
            "this calibration field. Rebuilding A(w) is recommended.",
            file=sys.stderr,
        )

    expected_tables = provenance.get("table_sha256", {})
    table_directory = tables_dir.expanduser().resolve()
    mismatched = [
        name
        for name, expected_hash in expected_tables.items()
        if sha256_file(table_directory / name) != expected_hash
    ]
    if mismatched:
        print(
            "WARNING: Parameter tables differ from the calibration build: "
            + ", ".join(sorted(mismatched))
            + ". Rebuilding A(w) is recommended.",
            file=sys.stderr,
        )


def idw_residual(
    model: dict[str, Any], al: float, ipa: float, pg: float
) -> dict[str, Any]:
    nodes = pd.DataFrame(model["nodes"])
    coordinates = nodes[["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct"]].to_numpy(float)
    target = np.array([al, ipa, pg], dtype=float)
    centre = np.asarray(model["interpolation"]["centre"], dtype=float)
    scale = np.asarray(model["interpolation"]["scale"], dtype=float)
    distances = np.linalg.norm((coordinates - target) / scale, axis=1)
    nearest = int(np.argmin(distances))
    requested_neighbors = int(model["interpolation"].get("neighbors", 4))
    neighbor_count = len(nodes) if requested_neighbors <= 0 else min(requested_neighbors, len(nodes))
    indexes = np.argsort(distances)[:neighbor_count]
    power = float(model["interpolation"].get("power", 2.0))

    if distances[nearest] < 1e-12:
        indexes = np.array([nearest])
        geometric = np.ones(1)
    else:
        geometric = 1.0 / np.maximum(distances[indexes], 1e-12) ** power

    output: dict[str, Any] = {}
    for value_column, se_column, key in (
        ("A_Rho_kg_m3", "A_Rho_SE_kg_m3", "A_Rho_kg_m3"),
        ("A_C_m_s", "A_C_SE_m_s", "A_C_m_s"),
    ):
        values = nodes.iloc[indexes][value_column].to_numpy(float)
        se = pd.to_numeric(nodes.iloc[indexes][se_column], errors="coerce").to_numpy(float)
        finite_se = se[np.isfinite(se) & (se >= 0)]
        fallback = float(np.median(finite_se)) if finite_se.size else 1.0
        se = np.where(np.isfinite(se) & (se >= 0), se, fallback)
        quality = 1.0 / np.maximum(se, max(fallback * 0.25, 1e-9)) ** 2
        weights = geometric * quality
        weights /= np.sum(weights)
        estimate = float(np.sum(weights * values))
        uncertainty = float(
            np.sqrt(np.sum(weights * ((values - estimate) ** 2 + se**2)))
        )
        output[key] = estimate
        output[key.replace("A_", "A_Uncertainty_")] = uncertainty

    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    outside_axes = [
        name
        for name, value, low, high in zip(
            ("Al", "IPA", "PG"), target, minimum, maximum
        )
        if value < low or value > high
    ]
    output.update(
        {
            "Nearest_Normalized_Distance": float(distances[nearest]),
            "Nearest_Node": int(nearest),
            "Neighbor_Count": int(len(indexes)),
            "Outside_Bounding_Box": bool(outside_axes),
            "Outside_Axes": outside_axes,
        }
    )
    return output


def physics_prediction(calculator, al: float, ipa: float, pg: float, temperature: float):
    if min(al, ipa, pg) < 0 or al + ipa + pg > 100:
        raise ValueError("Composition must be non-negative and sum to at most 100 wt-%.")
    arguments = {"al": al, "ipa": ipa, "pg": pg, "temperature": temperature}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rho = 1000.0 * float(calculator.density(**arguments))
        sound = float(calculator.sound_velocity(**arguments))
    return rho, sound


def predict_one(
    model: dict[str, Any], calculator, al: float, ipa: float, pg: float, temperature: float
) -> dict[str, Any]:
    rho_physics, c_physics = physics_prediction(calculator, al, ipa, pg, temperature)
    correction = idw_residual(model, al, ipa, pg)
    return {
        "Al_wt_pct": float(al),
        "IPA_wt_pct": float(ipa),
        "PG_wt_pct": float(pg),
        "Water_wt_pct": float(100.0 - al - ipa - pg),
        "Temperature_C": float(temperature),
        "Rho_Physics_kg_m3": rho_physics,
        "A_Rho_kg_m3": correction["A_Rho_kg_m3"],
        "Rho_Hybrid_kg_m3": rho_physics + correction["A_Rho_kg_m3"],
        "A_Rho_Uncertainty_kg_m3": correction["A_Uncertainty_Rho_kg_m3"],
        "C_Physics_m_s": c_physics,
        "A_C_m_s": correction["A_C_m_s"],
        "C_Hybrid_m_s": c_physics + correction["A_C_m_s"],
        "A_C_Uncertainty_m_s": correction["A_Uncertainty_C_m_s"],
        "Nearest_Normalized_Distance": correction["Nearest_Normalized_Distance"],
        "Outside_Bounding_Box": correction["Outside_Bounding_Box"],
        "Outside_Axes": correction["Outside_Axes"],
    }


def metric_row(measured: pd.Series, predicted: pd.Series, name: str) -> dict[str, Any]:
    pair = pd.DataFrame({"measured": measured, "predicted": predicted}).dropna()
    if pair.empty:
        return {"Model": name, "N": 0}
    error = pair["predicted"].to_numpy(float) - pair["measured"].to_numpy(float)
    measured_values = pair["measured"].to_numpy(float)
    denominator = np.abs(measured_values) > np.finfo(float).eps
    sst = float(np.sum((measured_values - np.mean(measured_values)) ** 2))
    sse = float(np.sum(error**2))
    return {
        "Model": name,
        "N": int(len(pair)),
        "Bias_ME": float(np.mean(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE_pct": float(np.mean(np.abs(error[denominator] / measured_values[denominator])) * 100),
        "R2": 1.0 - sse / sst if sst > 0 else np.nan,
    }


def make_diagnostic_plot(rows: pd.DataFrame, nodes: pd.DataFrame, path: Path) -> None:
    selected = rows[rows["Selected_For_Calibration"] & rows["Simulation_Status"].eq("ok")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for _, part in selected.groupby("Experiment_Key"):
        label = str(part["Experiment_Key"].iloc[0])
        axes[0, 0].plot(part["Experiment_Elapsed_h"], part["IPA_Loss_g"], ".-", label=label)
        axes[0, 1].scatter(part["Experiment_Elapsed_h"], part["A_Rho_kg_m3"], s=18, label=label)
        axes[1, 0].scatter(part["Experiment_Elapsed_h"], part["A_C_m_s"], s=18, label=label)
    axes[0, 0].set_title("Assumed cumulative IPA loss")
    axes[0, 0].set_ylabel("IPA loss [g]")
    axes[0, 1].set_title("Point residuals after composition correction")
    axes[0, 1].set_ylabel("A_rho [kg/m3]")
    axes[1, 0].set_ylabel("A_c [m/s]")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("Elapsed experiment time [h]")
        axis.grid(alpha=0.25)

    scatter = axes[1, 1].scatter(
        nodes["IPA_wt_pct"],
        nodes["PG_wt_pct"],
        c=nodes["A_C_m_s"],
        s=45 + 8 * nodes["N_Selected"],
        cmap="coolwarm",
        edgecolor="black",
    )
    axes[1, 1].set_title("Sound residual nodes")
    axes[1, 1].set_xlabel("IPA [wt-%]")
    axes[1, 1].set_ylabel("PG [wt-%]")
    axes[1, 1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1, 1], label="A_c [m/s]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_nodes(nodes: pd.DataFrame) -> None:
    columns = [
        "Source_File",
        "ProbeNr",
        "Phase",
        "N_Selected",
        "Al_wt_pct",
        "IPA_wt_pct",
        "PG_wt_pct",
        "A_Rho_kg_m3",
        "A_C_m_s",
    ]
    print("\nResidual calibration nodes A(w) = measurement - physics")
    print("-" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(nodes[columns].round(6).to_string(index=False))


def build_command(args: argparse.Namespace) -> None:
    raw = filter_samples(load_measurements(args.input), args.samples)
    require_columns(raw, MASS_COLUMNS + MEASUREMENT_COLUMNS)
    prepared = add_nominal_component_masses(
        add_timestamps_and_phases(raw, args.date_column, args.time_column, args.phase_gap_min)
    )
    prepared = select_calibration_rows(
        prepared,
        args.quality_mode,
        args.minimum_points_per_phase,
        args.settling_fraction,
        args.low_noise_keep_fraction,
    )
    calculator = load_calculator(args.calculator, args.tables)
    rates, evaporation_summaries = determine_evaporation_rates(
        prepared,
        calculator,
        args.evaporation_mode,
        args.evaporation_rate_g_h,
        args.evaporation_rate_max_g_h,
        args.evaporation_prior_sigma_g_h,
        not args.no_evaporation_prior,
    )
    corrected = apply_ipa_evaporation(prepared, rates)
    simulated = simulate_rows(corrected, calculator)
    failures = simulated["Simulation_Status"].ne("ok")
    if failures.any():
        examples = simulated.loc[failures, "Simulation_Status"].value_counts().head(3)
        print(f"WARNING: {int(failures.sum())} simulations failed:\n{examples}")
    nodes = build_nodes(simulated)
    payload = model_payload(nodes, evaporation_summaries, args)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "calibration_field.json"
    nodes_path = output / "calibration_nodes.csv"
    rows_path = output / "calibration_measurements.csv"
    evaporation_path = output / "evaporation_summary.csv"
    plot_path = output / "calibration_diagnostics.png"
    save_json(payload, model_path)
    nodes.to_csv(nodes_path, index=False)
    simulated.to_csv(rows_path, index=False)
    pd.DataFrame([summary.__dict__ for summary in evaporation_summaries]).to_csv(
        evaporation_path, index=False
    )
    make_diagnostic_plot(simulated, nodes, plot_path)

    print("\nEvaporation estimates")
    print("-" * 100)
    evaporation_frame = pd.DataFrame([summary.__dict__ for summary in evaporation_summaries])
    print(evaporation_frame.round(6).to_string(index=False))
    if any(summary.boundary_warning for summary in evaporation_summaries):
        print("WARNING: At least one rate is close to an optimization bound and is weakly identified.")
    if args.evaporation_mode == "estimate":
        print(
            "NOTE: Drift-derived rates are equivalent IPA-loss rates, not an independent "
            "chemical proof of evaporation. Prefer a gravimetric fixed rate when available."
        )
    print_nodes(nodes)
    print("\nSaved calibration field")
    print(f"  Model:       {model_path}")
    print(f"  Nodes:       {nodes_path}")
    print(f"  Measurements:{rows_path}")
    print(f"  Evaporation: {evaporation_path}")
    print(f"  Diagnostics: {plot_path}")


def predict_command(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    warn_on_provenance_mismatch(model, args.calculator, args.tables)
    calculator = load_calculator(args.calculator, args.tables)
    result = predict_one(model, calculator, args.al, args.ipa, args.pg, args.temperature)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return
    print("Hybrid prediction")
    print("-" * 72)
    print(f"Composition:  Al={args.al:.6f} %, IPA={args.ipa:.6f} %, PG={args.pg:.6f} %")
    print(f"Temperature:  {args.temperature:.3f} C")
    print(
        f"Density:      physics={result['Rho_Physics_kg_m3']:.6f}, "
        f"A={result['A_Rho_kg_m3']:+.6f}, hybrid={result['Rho_Hybrid_kg_m3']:.6f} kg/m3"
    )
    print(
        f"Sound:        physics={result['C_Physics_m_s']:.6f}, "
        f"A={result['A_C_m_s']:+.6f}, hybrid={result['C_Hybrid_m_s']:.6f} m/s"
    )
    print(f"Node distance:{result['Nearest_Normalized_Distance']:.6f}")
    if result["Outside_Bounding_Box"]:
        print(f"WARNING: Composition is outside calibration bounds for {result['Outside_Axes']}.")


def evaluate_command(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    warn_on_provenance_mismatch(model, args.calculator, args.tables)
    raw = filter_samples(load_measurements(args.input), args.samples)
    require_columns(raw, MASS_COLUMNS + MEASUREMENT_COLUMNS)
    prepared = add_nominal_component_masses(
        add_timestamps_and_phases(raw, args.date_column, args.time_column, args.phase_gap_min)
    )
    prepared["Selected_For_Calibration"] = True
    prepared["Selection_Method"] = "evaluation"
    calculator = load_calculator(args.calculator, args.tables)
    rates, summaries = determine_evaporation_rates(
        prepared,
        calculator,
        args.evaporation_mode,
        args.evaporation_rate_g_h,
        args.evaporation_rate_max_g_h,
        args.evaporation_prior_sigma_g_h,
        not args.no_evaporation_prior,
    )
    corrected = simulate_rows(apply_ipa_evaporation(prepared, rates), calculator)
    predictions: list[dict[str, Any]] = []
    for index, row in corrected.iterrows():
        if row["Simulation_Status"] != "ok":
            continue
        correction = idw_residual(
            model,
            float(row["Al_wt_pct_eff"]),
            float(row["IPA_wt_pct_eff"]),
            float(row["PG_wt_pct_eff"]),
        )
        predictions.append(
            {
                "index": index,
                "A_Rho_Field_kg_m3": correction["A_Rho_kg_m3"],
                "A_C_Field_m_s": correction["A_C_m_s"],
                "Calibration_Distance": correction["Nearest_Normalized_Distance"],
                "Calibration_Extrapolation": correction["Outside_Bounding_Box"],
            }
        )
    prediction_frame = pd.DataFrame(predictions).set_index("index") if predictions else pd.DataFrame()
    corrected = corrected.join(prediction_frame)
    corrected["Rho_Hybrid_kg_m3"] = corrected["Rho_Physics_kg_m3"] + corrected["A_Rho_Field_kg_m3"]
    corrected["C_Hybrid_m_s"] = corrected["C_Physics_m_s"] + corrected["A_C_Field_m_s"]

    summary_rows = []
    for measured, physics, hybrid, label in (
        ("Rho_M", "Rho_Physics_kg_m3", "Rho_Hybrid_kg_m3", "Density"),
        ("C_M", "C_Physics_m_s", "C_Hybrid_m_s", "Sound velocity"),
    ):
        physics_row = metric_row(corrected[measured], corrected[physics], f"{label} - physics")
        hybrid_row = metric_row(corrected[measured], corrected[hybrid], f"{label} - hybrid")
        summary_rows.extend([physics_row, hybrid_row])
    summary = pd.DataFrame(summary_rows)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison_path = output / "hybrid_model_comparison.csv"
    summary_path = output / "hybrid_accuracy_summary.csv"
    evaporation_path = output / "evaluation_evaporation_summary.csv"
    corrected.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([item.__dict__ for item in summaries]).to_csv(evaporation_path, index=False)

    print("\nEvaluation summary")
    print("-" * 100)
    print(summary.round(6).to_string(index=False))
    extrapolated = int(corrected["Calibration_Extrapolation"].fillna(False).sum())
    if extrapolated:
        print(f"WARNING: {extrapolated} rows are outside the calibration bounding box.")
    if args.evaporation_mode == "estimate":
        print(
            "WARNING: Evaporation was inferred from the evaluation measurements. "
            "For a strict external test, use a fixed independently measured rate."
        )
    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved summary:    {summary_path}")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _ask_choice(title: str, options: list[str], default: int = 1) -> int:
    print(f"\n{title}")
    for number, label in enumerate(options, start=1):
        suffix = " [default]" if number == default else ""
        print(f"  {number}) {label}{suffix}")
    while True:
        answer = input(f"Choice [{default}]: ").strip()
        if not answer:
            return default
        try:
            choice = int(answer)
        except ValueError:
            print("Please enter one of the displayed numbers.")
            continue
        if 1 <= choice <= len(options):
            return choice
        print("Please enter one of the displayed numbers.")


def _ask_float_value(
    prompt: str, default: float | None = None, minimum: float | None = None
) -> float:
    default_text = f" [{default:g}]" if default is not None else ""
    while True:
        answer = input(f"{prompt}{default_text}: ").strip().replace(",", ".")
        if not answer and default is not None:
            return float(default)
        try:
            value = float(answer)
        except ValueError:
            print("Please enter a number, for example 1.5.")
            continue
        if not np.isfinite(value):
            print("The number must be finite.")
            continue
        if minimum is not None and value < minimum:
            print(f"The value must be at least {minimum:g}.")
            continue
        return value


def _ask_path(prompt: str, default: Path | None = None) -> Path:
    default_text = f" [{_display_path(default)}]" if default is not None else ""
    while True:
        answer = input(f"{prompt}{default_text}: ").strip().strip('"')
        if not answer and default is not None:
            return default
        if answer:
            return Path(answer).expanduser()
        print("Please enter a path.")


def _discover_measurement_csvs() -> list[Path]:
    if not DEFAULT_INPUT.is_dir():
        return []
    return sorted(DEFAULT_INPUT.glob("*.csv"), key=lambda path: path.name.lower())


def _choose_csv_files() -> list[Path]:
    discovered = _discover_measurement_csvs()
    if not discovered:
        while True:
            path = _ask_path("Path to a CSV file or directory")
            try:
                return resolve_csv_files([path])
            except FileNotFoundError as exc:
                print(f"Could not use that path: {exc}")

    print("\nMeasurement CSV files found:")
    for number, path in enumerate(discovered, start=1):
        print(f"  {number}) {_display_path(path)}")
    print("  M) Enter another path manually")
    while True:
        answer = input("Select file number(s), e.g. 1 or 1,2 [1]: ").strip()
        if not answer:
            return [discovered[0]]
        if answer.lower() == "m":
            path = _ask_path("Path to a CSV file or directory")
            try:
                return resolve_csv_files([path])
            except FileNotFoundError as exc:
                print(f"Could not use that path: {exc}")
                continue
        try:
            numbers = [int(part.strip()) for part in answer.split(",")]
        except ValueError:
            print("Enter displayed numbers separated by commas, or M.")
            continue
        if numbers and all(1 <= number <= len(discovered) for number in numbers):
            return list(dict.fromkeys(discovered[number - 1] for number in numbers))
        print("At least one selected number is outside the displayed range.")


def _available_samples(paths: list[Path]) -> list[int]:
    samples: set[int] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path, comment="/", skipinitialspace=True, usecols=["ProbeNr"])
            values = pd.to_numeric(frame["ProbeNr"], errors="coerce").dropna().astype(int)
            samples.update(values.tolist())
        except (ValueError, OSError) as exc:
            print(f"WARNING: Could not inspect ProbeNr in {path.name}: {exc}")
    return sorted(samples)


def _ask_samples(paths: list[Path], prefer_sample_three: bool = True) -> list[str]:
    available = _available_samples(paths)
    if not available:
        print("No ProbeNr values could be read; all rows will be used.")
        return ["all"]
    print("\nProbeNr values found in the selected file(s):")
    print("  " + ", ".join(str(value) for value in available))
    default = "3" if prefer_sample_three and 3 in available else "all"
    while True:
        answer = input(
            f"ProbeNr to use, e.g. 3 or 1,3; enter 'all' for all [{default}]: "
        ).strip()
        if not answer:
            answer = default
        if answer.lower() == "all":
            return ["all"]
        try:
            chosen = [int(part.strip()) for part in answer.split(",")]
        except ValueError:
            print("Enter available ProbeNr values separated by commas, or 'all'.")
            continue
        unavailable = sorted(set(chosen) - set(available))
        if unavailable:
            print(f"These ProbeNr values are not present: {unavailable}")
            continue
        return [str(value) for value in dict.fromkeys(chosen)]


def _discover_models() -> list[Path]:
    results = SCRIPT_DIR / "results"
    if not results.is_dir():
        return []
    return sorted(
        results.glob("**/calibration_field.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _choose_model() -> Path:
    discovered = _discover_models()
    if not discovered:
        while True:
            path = _ask_path("Path to calibration_field.json")
            if path.is_file():
                return path
            print("That model file does not exist.")

    print("\nCalibration fields found (newest first):")
    for number, path in enumerate(discovered, start=1):
        print(f"  {number}) {_display_path(path)}")
    print("  M) Enter another path manually")
    while True:
        answer = input("Select calibration field [1]: ").strip()
        if not answer:
            return discovered[0]
        if answer.lower() == "m":
            path = _ask_path("Path to calibration_field.json")
            if path.is_file():
                return path
            print("That model file does not exist.")
            continue
        try:
            number = int(answer)
        except ValueError:
            print("Enter one of the displayed numbers, or M.")
            continue
        if 1 <= number <= len(discovered):
            return discovered[number - 1]
        print("Please enter one of the displayed numbers.")


def _interactive_evaporation_arguments(for_evaluation: bool) -> list[str]:
    if for_evaluation:
        options = [
            "Use a fixed, independently measured IPA loss rate (recommended)",
            "Estimate an equivalent IPA loss rate from this CSV",
            "Do not correct for evaporation",
        ]
        choice = _ask_choice("How should evaporation be handled?", options, default=1)
        modes = {1: "fixed", 2: "estimate", 3: "none"}
    else:
        options = [
            "Estimate an equivalent IPA loss rate from measurement drift",
            "Use a fixed, independently measured IPA loss rate",
            "Do not correct for evaporation",
        ]
        choice = _ask_choice("How should evaporation be handled?", options, default=1)
        modes = {1: "estimate", 2: "fixed", 3: "none"}

    mode = modes[choice]
    arguments = ["--evaporation-mode", mode]
    if mode == "fixed":
        rate = _ask_float_value("IPA loss rate in g/h", default=1.5, minimum=0.0)
        arguments.extend(["--evaporation-rate-g-h", str(rate)])
    elif mode == "estimate":
        prior_choice = _ask_choice(
            "Should the 1.5 +/- 0.5 g/h prior be used?",
            [
                "No, estimate only from measurement drift",
                "Yes, use the weak prior",
            ],
            default=1,
        )
        if prior_choice == 1:
            arguments.append("--no-evaporation-prior")
    return arguments


def interactive_arguments() -> list[str]:
    """Run a beginner-friendly wizard and return normal argparse tokens."""
    print("=" * 72)
    print(f"Residual calibration field {SCRIPT_VERSION} - guided mode")
    print("=" * 72)
    command_choice = _ask_choice(
        "What would you like to do?",
        [
            "Build a new calibration field",
            "Predict one composition with an existing field",
            "Evaluate an existing field against a measurement CSV",
        ],
        default=1,
    )

    if command_choice == 1:
        paths = _choose_csv_files()
        samples = _ask_samples(paths)
        evaporation = _interactive_evaporation_arguments(for_evaluation=False)
        quality_choice = _ask_choice(
            "Which measurements should be used?",
            [
                "Automatic quality selection (recommended)",
                "Only rows with valid/stable quality flags",
                "Late low-noise rows",
                "All finite SensOK rows",
            ],
            default=1,
        )
        quality_modes = {1: "auto", 2: "flags", 3: "low-noise", 4: "all"}
        label = "all" if samples == ["all"] else "_".join(samples)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = SCRIPT_DIR / "results" / f"calibration_probe_{label}_{timestamp}"
        output = _ask_path("Output directory", default=default_output)
        print("\nStarting calibration build with the selected settings ...")
        return (
            ["build", "--input"]
            + [str(path) for path in paths]
            + ["--samples"]
            + samples
            + evaporation
            + ["--quality-mode", quality_modes[quality_choice], "--output", str(output)]
        )

    if command_choice == 2:
        model = _choose_model()
        al = _ask_float_value("Al content in wt-%", minimum=0.0)
        ipa = _ask_float_value("IPA content in wt-%", minimum=0.0)
        pg = _ask_float_value("PG content in wt-%", minimum=0.0)
        while al + ipa + pg > 100:
            print("Al + IPA + PG must not exceed 100 wt-%. Please enter the values again.")
            al = _ask_float_value("Al content in wt-%", minimum=0.0)
            ipa = _ask_float_value("IPA content in wt-%", minimum=0.0)
            pg = _ask_float_value("PG content in wt-%", minimum=0.0)
        temperature = _ask_float_value("Temperature in deg C", default=25.0)
        print("\nCalculating the hybrid prediction ...")
        return [
            "predict",
            "--model",
            str(model),
            "--al",
            str(al),
            "--ipa",
            str(ipa),
            "--pg",
            str(pg),
            "--temperature",
            str(temperature),
        ]

    model = _choose_model()
    paths = _choose_csv_files()
    samples = _ask_samples(paths, prefer_sample_three=False)
    evaporation = _interactive_evaporation_arguments(for_evaluation=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = SCRIPT_DIR / "results" / f"evaluation_{timestamp}"
    output = _ask_path("Output directory", default=default_output)
    print("\nStarting external CSV evaluation ...")
    return (
        ["evaluate", "--model", str(model), "--input"]
        + [str(path) for path in paths]
        + ["--samples"]
        + samples
        + evaporation
        + ["--output", str(output)]
    )


def main() -> None:
    if len(sys.argv) == 1:
        try:
            sys.argv.extend(interactive_arguments())
        except (EOFError, KeyboardInterrupt):
            print("\nGuided mode cancelled.")
            raise SystemExit(130)
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "build":
            build_command(args)
        elif args.command == "predict":
            predict_command(args)
        elif args.command == "evaluate":
            evaluate_command(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()