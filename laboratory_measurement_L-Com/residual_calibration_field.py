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

and all mass percentages, including methyl gallate (MG), are recalculated.
SL120 is interpreted as 20 wt-% Al, 40 wt-% IPA and 40 wt-% PG. MG is read
from the ``m_MG`` column and is assumed not to evaporate.

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
      --al 1.8 --ipa 4.0 --pg 4.5 --mg 0.227 --temperature 23.0

Evaluate a saved field against another CSV::

    python residual_calibration_field.py evaluate \
      --model results/residual_calibration/calibration_field.json \
      --input new_measurements.csv --samples all \
      --evaporation-mode fixed --evaporation-rate-g-h 1.50

The signed calibration residual always uses ``measurement - physics`` so it
can be added directly to the InkCalculator result.

Output folders are automatic and never requested interactively:
    results/calibration_field/<samples>__<sources>__<timestamp>/
    results/calibrated_model_evaluation/<field>__<sources>__<samples>__<timestamp>/
Open evaluation_report.html for the phase-averaged graphical comparison.
Detailed rows remain available for auditing. Water/air reference rows are
saved separately and do not become ink calibration points.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import importlib.util
import inspect
import json
import math
import re
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
SCRIPT_VERSION = "3.0-phase-report"
DEFAULT_INPUT = SCRIPT_DIR / "measurement_data"
CALIBRATION_ROOT = SCRIPT_DIR / "results" / "calibration_field"
EVALUATION_ROOT = SCRIPT_DIR / "results" / "calibrated_model_evaluation"
DEFAULT_TABLES = REPO_DIR / "tables_parameters"
DEFAULT_CALCULATOR = REPO_DIR / "ink_calculator.py"

SL120 = {"Al": 0.20, "IPA": 0.40, "PG": 0.40}
MASS_COLUMNS = ["m_SL120", "m_Wasser", "m_IPA", "m_PG", "m_MG"]
MEASUREMENT_COLUMNS = ["Rho_M", "C_M", "T_M"]
COMPOSITION_COLUMNS = [
    "Al_wt_pct_eff",
    "IPA_wt_pct_eff",
    "PG_wt_pct_eff",
    "MG_wt_pct_eff",
]
FIELD_AXES = ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"]
LEGACY_FIELD_AXES = ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct"]


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
    predict.add_argument(
        "--mg",
        type=finite_float,
        default=0.0,
        help="Methyl gallate mass percentage (default: 0).",
    )
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
    evaluate.add_argument("--quality-mode", choices=["auto", "flags", "low-noise", "all"],
                          default="auto", help="Phase summary selection (default: auto).")
    evaluate.add_argument("--minimum-points-per-phase", type=int, default=5)
    evaluate.add_argument("--settling-fraction", type=float, default=0.30)
    evaluate.add_argument("--low-noise-keep-fraction", type=float, default=0.60)

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
        if "m_MG" not in frame.columns:
            frame["m_MG"] = 0.0
            print(f"Note: {path.name} has no m_MG column; MG was set to 0 g.")
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
        result["Source_Path"].astype(str)
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
        result[
            [
                "m_Al_nom_g",
                "m_IPA_nom_g",
                "m_PG_nom_g",
                "m_MG_nom_g",
                "m_Water_nom_g",
            ]
        ]
        .lt(0)
        .any(axis=1)
        | ~np.isfinite(result[MASS_COLUMNS]).all(axis=1)
        | result["m_Total_nom_g"].le(0)
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} rows with invalid component masses.")
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
    if (result["IPA_Loss_g"] > available).any():
        row = result.loc[result["IPA_Loss_g"] > available].iloc[0]
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
    result["MG_wt_pct_eff"] = 100.0 * result["m_MG_nom_g"] / denominator
    result["Water_wt_pct_eff"] = 100.0 * result["m_Water_nom_g"] / denominator
    result["Composition_Sum_wt_pct"] = result[
        [
            "Al_wt_pct_eff",
            "IPA_wt_pct_eff",
            "PG_wt_pct_eff",
            "MG_wt_pct_eff",
            "Water_wt_pct_eff",
        ]
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
    calculator = calculator_class(tables_dir=str(tables_dir))
    for method_name in ("density", "sound_velocity"):
        method = getattr(calculator, method_name, None)
        parameters = inspect.signature(method).parameters if method else {}
        if "mg" not in parameters:
            raise ImportError(
                f"{calculator_path.name} does not provide an 'mg' argument in "
                f"InkCalculator.{method_name}(). Use the MG-enabled calculator."
            )
    return calculator


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
                "mg": float(row["MG_wt_pct_eff"]),
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
    numeric_measurements = result[MEASUREMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric_measurements).all(axis=1)
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
                "MG_wt_pct": float(np.average(phase["MG_wt_pct_eff"])),
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
    x = nodes[FIELD_AXES].to_numpy(float)
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
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "equation": "measurement = physics + A(w)",
        "residual_definition": "A = measurement - physics",
        "composition_axes": FIELD_AXES,
        "temperature_model": "InkCalculator only; A(w) has no learned temperature term",
        "evaporation_assumption": (
            "Only IPA evaporates; MG is non-volatile in the mass balance. "
            "IPA loss is cumulative from the first timestamp per source file "
            "and ProbeNr."
        ),
        "methyl_gallate": {
            "source_mass_column": "m_MG",
            "effective_composition_column": "MG_wt_pct_eff",
            "residual_field_axis": "MG_wt_pct",
            "physics_argument": "mg",
        },
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


def model_composition_axes(model: dict[str, Any]) -> list[str]:
    """Return and validate the composition axes stored in a field."""
    axes = model.get("composition_axes", LEGACY_FIELD_AXES)
    if not isinstance(axes, list) or not axes:
        raise ValueError("Calibration model has invalid composition_axes metadata.")
    allowed = set(FIELD_AXES)
    unknown = [axis for axis in axes if axis not in allowed]
    if unknown:
        raise ValueError(f"Unsupported composition axes in calibration model: {unknown}")
    if len(set(axes)) != len(axes):
        raise ValueError("Calibration model contains duplicate composition axes.")
    return list(axes)


def idw_residual(
    model: dict[str, Any], al: float, ipa: float, pg: float, mg: float = 0.0
) -> dict[str, Any]:
    nodes = pd.DataFrame(model["nodes"])
    axes = model_composition_axes(model)
    missing = [axis for axis in axes if axis not in nodes.columns]
    if missing:
        raise ValueError(f"Calibration nodes are missing composition axes: {missing}")
    coordinates = nodes[axes].to_numpy(float)
    target_by_axis = {
        "Al_wt_pct": al,
        "IPA_wt_pct": ipa,
        "PG_wt_pct": pg,
        "MG_wt_pct": mg,
    }
    target = np.array([target_by_axis[axis] for axis in axes], dtype=float)
    scale = np.asarray(model["interpolation"]["scale"], dtype=float)
    if scale.shape != target.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(
            "Calibration interpolation scale does not match composition_axes."
        )
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
        axis.removesuffix("_wt_pct")
        for axis, value, low, high in zip(axes, target, minimum, maximum)
        if value < low - 1e-10 * max(1.0, abs(low), abs(high))
        or value > high + 1e-10 * max(1.0, abs(low), abs(high))
    ]
    output.update(
        {
            "Nearest_Normalized_Distance": float(distances[nearest]),
            "Nearest_Node": int(nearest),
            "Neighbor_Count": int(len(indexes)),
            "Outside_Bounding_Box": bool(outside_axes),
            "Outside_Axes": outside_axes,
            "Field_Composition_Axes": axes,
            "MG_Axis_Used": "MG_wt_pct" in axes,
        }
    )
    return output


def physics_prediction(
    calculator,
    al: float,
    ipa: float,
    pg: float,
    temperature: float,
    mg: float = 0.0,
):
    if min(al, ipa, pg, mg) < 0 or al + ipa + pg + mg > 100:
        raise ValueError("Composition must be non-negative and sum to at most 100 wt-%.")
    arguments = {
        "al": al,
        "ipa": ipa,
        "pg": pg,
        "mg": mg,
        "temperature": temperature,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rho = 1000.0 * float(calculator.density(**arguments))
        sound = float(calculator.sound_velocity(**arguments))
    return rho, sound


def predict_one(
    model: dict[str, Any],
    calculator,
    al: float,
    ipa: float,
    pg: float,
    temperature: float,
    mg: float = 0.0,
) -> dict[str, Any]:
    rho_physics, c_physics = physics_prediction(
        calculator, al, ipa, pg, temperature, mg=mg
    )
    correction = idw_residual(model, al, ipa, pg, mg=mg)
    return {
        "Al_wt_pct": float(al),
        "IPA_wt_pct": float(ipa),
        "PG_wt_pct": float(pg),
        "MG_wt_pct": float(mg),
        "Water_wt_pct": float(100.0 - al - ipa - pg - mg),
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
        "Field_Composition_Axes": correction["Field_Composition_Axes"],
        "MG_Axis_Used": correction["MG_Axis_Used"],
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
    for _, node in nodes[nodes["MG_wt_pct"].gt(1e-9)].iterrows():
        axes[1, 1].annotate(
            f"MG {node['MG_wt_pct']:.3f}%",
            (node["IPA_wt_pct"], node["PG_wt_pct"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
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
        "MG_wt_pct",
        "A_Rho_kg_m3",
        "A_C_m_s",
    ]
    print("\nResidual calibration nodes A(w) = measurement - physics")
    print("-" * 120)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(nodes[columns].round(6).to_string(index=False))


def safe_name(value: str, limit: int = 56) -> str:
    """Create a portable folder component; a hash preserves long-name identity."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unnamed"
    if len(cleaned) > limit:
        suffix = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]
        cleaned = cleaned[:limit - 9] + "_" + suffix
    return cleaned


def source_manifest(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for source, part in frame.groupby("Source_Path", sort=False):
        path = Path(source)
        records.append({"path": str(path), "filename": path.name,
                        "sha256": sha256_file(path),
                        "samples": sorted(pd.to_numeric(part["ProbeNr"]).astype(int).unique().tolist())})
    return records


def automatic_output(kind: str, frame: pd.DataFrame, model_path: Path | None = None) -> Path:
    """Allocate a new run directory without overwriting previous runs."""
    probes = "probe_" + "_".join(str(v) for v in sorted(
        pd.to_numeric(frame["ProbeNr"]).astype(int).unique()))
    sources = "__".join(Path(v).stem for v in frame["Source_Path"].unique())
    if kind == "build":
        root = CALIBRATION_ROOT
        label = safe_name(probes, 36) + "__" + safe_name(sources, 48)
    else:
        root = EVALUATION_ROOT
        field = model_path.parent.name if model_path.name == "calibration_field.json" else model_path.stem
        label = ("field_" + safe_name(field, 48) + "__data_" + safe_name(sources, 48)
                 + "__" + safe_name(probes, 28))
    root.mkdir(parents=True, exist_ok=True)
    label += "__" + datetime.now().strftime("%Y%m%d_%H%M%S")
    for counter in range(10000):
        path = root / (label if counter == 0 else f"{label}_{counter:03d}")
        try:
            path.mkdir()
            return path.resolve()
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique output directory.")


def load_ink_input(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep reference measurements separate from an ink residual field."""
    raw = filter_samples(load_measurements(args.input), args.samples)
    require_columns(raw, MASS_COLUMNS + MEASUREMENT_COLUMNS)
    numeric = raw[MASS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    solutes = numeric[["m_SL120", "m_IPA", "m_PG", "m_MG"]]
    reference = (pd.to_numeric(raw["ProbeNr"], errors="coerce").isin([100, 101])
                 | (solutes.notna().all(axis=1) & solutes.eq(0).all(axis=1)))
    excluded = raw.loc[reference].copy()
    if len(excluded):
        print(f"NOTE: {len(excluded)} water/air reference rows are excluded from the ink field "
              "and saved separately.")
    raw = raw.loc[~reference].copy()
    if raw.empty:
        raise ValueError("The selection contains only reference measurements, not ink.")
    return raw, excluded


def build_command(args: argparse.Namespace) -> None:
    raw, reference_rows = load_ink_input(args)
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
    payload["training_data"] = source_manifest(raw)
    payload["training_measurement_keys"] = measurement_keys(simulated.loc[
        simulated["Selected_For_Calibration"] & simulated["Simulation_Status"].eq("ok")]).tolist()
    output = automatic_output("build", raw)
    payload["field_name"] = output.name
    model_path = output / "calibration_field.json"
    nodes_path = output / "calibration_nodes.csv"
    rows_path = output / "calibration_measurements.csv"
    evaporation_path = output / "evaporation_summary.csv"
    plot_path = output / "calibration_diagnostics.png"
    save_json(payload, model_path)
    save_json({"operation": "build", "output_directory": str(output),
               "training_data": payload["training_data"],
               "reference_rows_excluded": len(reference_rows)}, output / "run_manifest.json")
    if len(reference_rows):
        reference_rows.to_csv(output / "excluded_reference_measurements.csv", index=False)
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
    result = predict_one(
        model,
        calculator,
        args.al,
        args.ipa,
        args.pg,
        args.temperature,
        mg=args.mg,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return
    print("Hybrid prediction")
    print("-" * 72)
    print(
        f"Composition:  Al={args.al:.6f} %, IPA={args.ipa:.6f} %, "
        f"PG={args.pg:.6f} %, MG={args.mg:.6f} %"
    )
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
    if args.mg > 0.0 and not result["MG_Axis_Used"]:
        print(
            "WARNING: Physics includes MG, but this legacy residual field has "
            "no MG composition axis."
        )


def measurement_keys(frame: pd.DataFrame) -> pd.Series:
    """Identify the same acquisition across differently named CSV snapshots."""
    def key(row):
        text = "|".join([str(int(row["ProbeNr"])), str(row["Measurement_Time_UTC"])]
                        + [f"{float(row[c]):.9g}" for c in MASS_COLUMNS])
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return frame.apply(key, axis=1)


def phase_evaluation_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Average paired predictions and observations, never predict at mean inputs.

    Contiguous phases are kept separate across files, samples and time gaps.
    SD describes variation among the retained timestamps, not a confidence
    interval or independent preparation repeatability.
    """
    records = []
    source_ids = {path: i + 1 for i, path in enumerate(rows["Source_Path"].unique())}
    for (_, phase), all_rows in rows.groupby(["Experiment_Key", "Phase"], sort=False):
        selected = all_rows.loc[all_rows["Selected_For_Evaluation"]]
        first = all_rows.iloc[0]
        probe = int(first["ProbeNr"])
        composition_rows = selected if len(selected) else all_rows
        record = {
            "Phase_ID": f"F{source_ids[first['Source_Path']]}-P{probe}-S{int(phase)}",
            "Source_File": first["Source_File"], "Source_Path": first["Source_Path"],
            "ProbeNr": probe, "Phase": int(phase), "N_Total": len(all_rows),
            "N_Selected": len(selected), "N_Excluded": len(all_rows) - len(selected),
            "Selection_Method": "; ".join(sorted(selected["Selection_Method"].unique())),
            "Start_UTC": str(all_rows["Measurement_Time_UTC"].min()),
            "End_UTC": str(all_rows["Measurement_Time_UTC"].max()),
            "N_Extrapolated": int(selected["Calibration_Extrapolation"].fillna(False).sum()),
            "N_Training_Overlap": int(selected["Training_Overlap"].sum()),
            "Temperature_Mean_C": float(composition_rows["T_M"].mean()),
            "Temperature_Min_C": float(composition_rows["T_M"].min()),
            "Temperature_Max_C": float(composition_rows["T_M"].max()),
        }
        for component in ("Al", "IPA", "PG", "MG", "Water"):
            values = composition_rows[f"{component}_wt_pct_eff"]
            record[f"{component}_wt_pct"] = float(values.mean())
            record[f"{component}_Min_wt_pct"] = float(values.min())
            record[f"{component}_Max_wt_pct"] = float(values.max())
        record.update({c: float(first[c]) for c in MASS_COLUMNS})
        for prefix, measured, physics, hybrid in (
            ("Rho", "Rho_M", "Rho_Physics_kg_m3", "Rho_Hybrid_kg_m3"),
            ("C", "C_M", "C_Physics_m_s", "C_Hybrid_m_s"),
        ):
            for label, column in (("Measured", measured), ("Physics", physics), ("Hybrid", hybrid)):
                values = selected[column]
                for stat, value in (("Mean", values.mean()), ("SD", values.std(ddof=1)),
                                    ("Min", values.min()), ("Max", values.max())):
                    record[f"{prefix}_{stat}_{label}"] = float(value)
            errors = selected[hybrid] - selected[measured]
            record[f"{prefix}_Mean_Error"] = float(errors.mean())
            record[f"{prefix}_SD_Error"] = float(errors.std(ddof=1))
            record[f"{prefix}_RMSE_Within_Phase"] = float(np.sqrt((errors ** 2).mean()))
        records.append(record)
    return pd.DataFrame(records)


def phase_accuracy_metrics(phases: pd.DataFrame) -> pd.DataFrame:
    """Weight every usable recipe phase once, regardless of recording length."""
    usable = phases.loc[phases["N_Selected"].gt(0)]
    records = []
    for prefix, name in (("Rho", "Density"), ("C", "Sound velocity")):
        for model in ("Physics", "Hybrid"):
            result = metric_row(usable[f"{prefix}_Mean_Measured"],
                                usable[f"{prefix}_Mean_{model}"], f"{name} - {model.lower()}")
            result["Weighting"] = "One equal weight per phase mean"
            records.append(result)
    return pd.DataFrame(records)


def create_phase_plots(phases: pd.DataFrame, output: Path) -> list[Path]:
    """Create paginated comparisons and a parity plot; bars are +/- one SD."""
    usable = phases.loc[phases["N_Selected"].gt(0)].copy()
    if usable.empty:
        return []
    paths = []
    specs = (("Rho", "Density", "kg/m3"), ("C", "Sound velocity", "m/s"))
    for page, start in enumerate(range(0, len(usable), 12), 1):
        part = usable.iloc[start:start + 12]
        x = np.arange(len(part))
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        for column, (prefix, name, unit) in enumerate(specs):
            axis = axes[0, column]
            for label, offset, color, marker in (("Measured", -0.10, "#1d4e89", "o"),
                                                  ("Hybrid", 0.10, "#bd4f22", "s")):
                axis.errorbar(x + offset, part[f"{prefix}_Mean_{label}"],
                              yerr=part[f"{prefix}_SD_{label}"].fillna(0),
                              fmt=marker, color=color, capsize=3, label=f"{label}: mean +/- SD")
            axis.set(title=name, ylabel=unit)
            axis.legend(fontsize=8)
            axis = axes[1, column]
            axis.axhline(0, color="#555555", linewidth=1)
            axis.errorbar(x, part[f"{prefix}_Mean_Error"],
                          yerr=part[f"{prefix}_SD_Error"].fillna(0), fmt="o",
                          color="#6a3d7d", capsize=3)
            axis.set(title="Paired error: hybrid - measured", ylabel=unit)
            for axis in axes[:, column]:
                axis.set_xticks(x)
                axis.set_xticklabels(part["Phase_ID"], rotation=45, ha="right", fontsize=8)
                axis.grid(axis="y", alpha=0.2)
        fig.suptitle(f"Recipe-phase comparison | page {page}\n"
                     "Predictions evaluated at each retained timestamp; SD is not a confidence interval",
                     fontsize=12)
        path = output / f"hybrid_phase_comparison_{page:02d}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, (prefix, name, unit) in zip(axes, specs):
        measured = usable[f"{prefix}_Mean_Measured"]
        predicted = usable[f"{prefix}_Mean_Hybrid"]
        groups = usable.groupby("ProbeNr", sort=True)
        for i, (probe, group) in enumerate(groups):
            axis.errorbar(group[f"{prefix}_Mean_Measured"], group[f"{prefix}_Mean_Hybrid"],
                          xerr=group[f"{prefix}_SD_Measured"].fillna(0),
                          yerr=group[f"{prefix}_SD_Hybrid"].fillna(0),
                          fmt="o", color=plt.get_cmap("tab10")(i % 10), alpha=0.85,
                          capsize=2, label=f"Probe {probe}")
        low, high = min(measured.min(), predicted.min()), max(measured.max(), predicted.max())
        padding = max(float(high - low) * 0.1, 0.05)
        axis.plot([low - padding, high + padding], [low - padding, high + padding],
                  "--", color="#555555", label="Ideal: prediction = measurement")
        axis.set(xlabel=f"Measured phase mean [{unit}]", ylabel=f"Hybrid phase mean [{unit}]",
                 title=name)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Hybrid parity | one point per phase, horizontal and vertical bars: +/- one SD")
    path = output / "hybrid_parity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.insert(0, path)
    return paths


def write_evaluation_report(phases: pd.DataFrame, metrics: pd.DataFrame,
                            metadata: dict[str, Any], plots: list[Path], output: Path) -> Path:
    """Write a portable HTML report with embedded plots and readable tables."""
    def table(frame):
        return frame.to_html(index=False, escape=True, border=0, na_rep="n/a",
                             float_format=lambda value: f"{value:.4f}")
    def heading(text):
        return f"<h2>{html.escape(text)}</h2>"
    overview_columns = ["Phase_ID", "Source_File", "ProbeNr", "N_Selected", "N_Excluded",
                        "Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct",
                        "Temperature_Min_C", "Temperature_Max_C", "N_Extrapolated"]
    sections = [heading("Accuracy of phase means (equal phase weighting)"), table(metrics),
                heading("Recipe phases and selection"), table(phases[overview_columns])]
    for prefix, label in (("Rho", "Density [kg/m3]"), ("C", "Sound velocity [m/s]")):
        cols = ["Phase_ID", "N_Selected", f"{prefix}_Mean_Measured", f"{prefix}_SD_Measured",
                f"{prefix}_Min_Measured", f"{prefix}_Max_Measured",
                f"{prefix}_Mean_Hybrid", f"{prefix}_SD_Hybrid",
                f"{prefix}_Mean_Error", f"{prefix}_SD_Error", f"{prefix}_RMSE_Within_Phase"]
        labels = {c: c.removeprefix(prefix + "_").replace("_", " ") for c in cols}
        sections += [heading(label), table(phases[cols].rename(columns=labels))]
    images = []
    for path in plots:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        images.append(f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{encoded}">')
    overlap = metadata["training_overlap_selected_rows"]
    caution = (f"{overlap} selected rows also occur in the calibration data. This is at least "
               "partly an in-sample comparison, not an independent validation." if overlap else
               "No matching training acquisition was detected. This alone does not prove independent validation.")
    if not metadata["training_overlap_check_available"]:
        caution = "This older field has no acquisition fingerprints; training-data overlap cannot be verified."
        if metadata["legacy_possible_overlap"]:
            caution += " Source filename and sample labels suggest possible overlap with training data."
    source_list = "".join(f"<li>{html.escape(item['path'])} | Probe {item['samples']}</li>"
                          for item in metadata["evaluation_data"])
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Hybrid ink model evaluation</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#172b3a;background:#f4f7fa;margin:0;padding:32px}
main{max-width:1440px;margin:auto;background:white;padding:28px;border-radius:12px}
h1,h2{color:#173d61}h2{margin-top:32px}img{width:100%;height:auto;margin:16px 0}
table{border-collapse:collapse;display:block;overflow-x:auto;font-size:12px;margin:16px 0}
th,td{padding:9px 12px;border-bottom:1px solid #dce3ea;text-align:right;white-space:nowrap}
th{background:#eaf0f6}tr:nth-child(even){background:#f7f9fb}li{overflow-wrap:anywhere}
.note{padding:14px;background:#fff5db;border-left:4px solid #c68613}
</style><main><h1>Hybrid model vs. measured ink</h1>"""
    document += f"<p><b>Calibration field:</b> {html.escape(metadata['model_path'])}</p><ul>{source_list}</ul>"
    document += f"<p><b>Selection:</b> {html.escape(metadata['quality_mode'])}; "
    document += f"<b>Evaporation:</b> {html.escape(metadata['evaporation_mode'])} (IPA only).</p>"
    document += f'<p class="note">{html.escape(caution)}</p>'
    document += """<p>Each point represents one contiguous recipe phase within one sample and source file.
Repeated phases and different days are kept separate. Measured and predicted means use exactly the
same selected timestamps. Temperature and effective composition are applied before averaging.
Bars show sample standard deviation (SD, ddof=1); for one retained timestamp SD is unavailable.
SD includes within-phase drift and is neither a confidence interval nor a model-uncertainty band.
Measured min/max and paired-error SD are listed below. Phase-mean metrics weight each phase equally;
they do not replace inspection of within-phase errors. Extrapolation counts are bounding-box flags,
not a guarantee that points inside the bounds are well supported.</p>"""
    document += "".join(images + sections) + "</main></html>"
    path = output / "evaluation_report.html"
    path.write_text(document, encoding="utf-8")
    return path


def evaluate_command(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    warn_on_provenance_mismatch(model, args.calculator, args.tables)
    raw, reference_rows = load_ink_input(args)
    prepared = add_nominal_component_masses(
        add_timestamps_and_phases(raw, args.date_column, args.time_column, args.phase_gap_min)
    )
    # Reuse the same quality rules; these flags select evaluation rows only.
    prepared = select_calibration_rows(
        prepared, args.quality_mode, args.minimum_points_per_phase,
        args.settling_fraction, args.low_noise_keep_fraction,
    )
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
            mg=float(row["MG_wt_pct_eff"]),
        )
        predictions.append(
            {
                "index": index,
                "A_Rho_Field_kg_m3": correction["A_Rho_kg_m3"],
                "A_C_Field_m_s": correction["A_C_m_s"],
                "Calibration_Distance": correction["Nearest_Normalized_Distance"],
                "Calibration_Extrapolation": correction["Outside_Bounding_Box"],
                "Calibration_Outside_Axes": ",".join(correction["Outside_Axes"]),
                "Calibration_MG_Axis_Used": correction["MG_Axis_Used"],
            }
        )
    if not predictions:
        raise ValueError("No measurement row could be simulated. Check the calculator and input data.")
    prediction_frame = pd.DataFrame(predictions).set_index("index")
    corrected = corrected.join(prediction_frame)
    corrected["Rho_Hybrid_kg_m3"] = corrected["Rho_Physics_kg_m3"] + corrected["A_Rho_Field_kg_m3"]
    corrected["C_Hybrid_m_s"] = corrected["C_Physics_m_s"] + corrected["A_C_Field_m_s"]
    paired_columns = ["Rho_M", "C_M", "Rho_Physics_kg_m3", "C_Physics_m_s",
                      "Rho_Hybrid_kg_m3", "C_Hybrid_m_s"]
    corrected["Selected_For_Evaluation"] = (corrected.pop("Selected_For_Calibration")
        & corrected["Simulation_Status"].eq("ok") & np.isfinite(corrected[paired_columns]).all(axis=1))
    training_keys = set(model.get("training_measurement_keys", []))
    corrected["Training_Overlap"] = measurement_keys(corrected).isin(training_keys)
    selected = corrected.loc[corrected["Selected_For_Evaluation"]]
    if selected.empty:
        raise ValueError("No selected paired observations remain for evaluation.")

    summary_rows = []
    for measured, physics, hybrid, label in (
        ("Rho_M", "Rho_Physics_kg_m3", "Rho_Hybrid_kg_m3", "Density"),
        ("C_M", "C_Physics_m_s", "C_Hybrid_m_s", "Sound velocity"),
    ):
        physics_row = metric_row(selected[measured], selected[physics], f"{label} - physics")
        hybrid_row = metric_row(selected[measured], selected[hybrid], f"{label} - hybrid")
        summary_rows.extend([physics_row, hybrid_row])
    summary = pd.DataFrame(summary_rows)

    phases = phase_evaluation_summary(corrected)
    phase_metrics = phase_accuracy_metrics(phases)
    output = automatic_output("evaluate", raw, args.model.resolve())
    comparison_path = output / "hybrid_model_comparison.csv"
    summary_path = output / "hybrid_accuracy_summary.csv"
    evaporation_path = output / "evaluation_evaporation_summary.csv"
    corrected.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([item.__dict__ for item in summaries]).to_csv(evaporation_path, index=False)
    phases.to_csv(output / "hybrid_evaluation_by_phase.csv", index=False)
    phase_metrics.to_csv(output / "hybrid_phase_accuracy_summary.csv", index=False)
    if len(reference_rows):
        reference_rows.to_csv(output / "excluded_reference_measurements.csv", index=False)
    metadata = {
        "operation": "evaluate", "script_version": SCRIPT_VERSION,
        "model_path": str(args.model.resolve()), "model_sha256": sha256_file(args.model.resolve()),
        "evaluation_data": source_manifest(raw), "output_directory": str(output),
        "quality_mode": args.quality_mode, "evaporation_mode": args.evaporation_mode,
        "minimum_points_per_phase": args.minimum_points_per_phase,
        "settling_fraction": args.settling_fraction,
        "low_noise_keep_fraction": args.low_noise_keep_fraction,
        "reference_rows_excluded": len(reference_rows),
        "training_overlap_selected_rows": int(selected["Training_Overlap"].sum()),
        "training_overlap_check_available": bool(training_keys),
        "legacy_possible_overlap": any(
            (str(node.get("Source_File")), int(node.get("ProbeNr", -1)))
            in set(zip(raw["Source_File"].astype(str), raw["ProbeNr"].astype(int)))
            for node in model["nodes"]
        ) if not training_keys else False,
        "calculator_path": str(args.calculator.resolve()),
        "calculator_sha256": sha256_file(args.calculator.resolve()),
    }
    save_json(metadata, output / "run_manifest.json")
    plots = create_phase_plots(phases, output)
    report = write_evaluation_report(phases, phase_metrics, metadata, plots, output)

    print("\nEvaluation summary: equal weight per phase mean")
    print("-" * 100)
    print(phase_metrics.round(6).to_string(index=False))
    extrapolated = int(corrected["Calibration_Extrapolation"].fillna(False).sum())
    if extrapolated:
        print(f"WARNING: {extrapolated} rows are outside the calibration bounding box.")
    mg_rows = corrected["MG_wt_pct_eff"].fillna(0.0).gt(0.0)
    if mg_rows.any() and "MG_wt_pct" not in model_composition_axes(model):
        print(
            "WARNING: Physics includes MG, but the residual field has no MG axis; "
            f"{int(mg_rows.sum())} MG-containing rows used legacy residual interpolation."
        )
    if args.evaporation_mode == "estimate":
        print(
            "WARNING: Evaporation was inferred from the evaluation measurements. "
            "For a strict external test, use a fixed independently measured rate."
        )
    print(f"\nSaved comparison: {comparison_path}")
    print(f"Saved summary:    {summary_path}")
    print(f"Open graphical report: {report}")
    print(f"Selected rows: {len(selected)} / {len(corrected)}; usable phases: "
          f"{int(phases['N_Selected'].gt(0).sum())}.")
    if metadata["training_overlap_selected_rows"]:
        print("WARNING: Evaluation includes training acquisitions; this is not an independent test.")
    elif not training_keys:
        print("NOTE: This older field cannot be checked reliably for training-data overlap.")


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
        print(f"\nOutput root (automatic run folder): {CALIBRATION_ROOT}")
        print("\nStarting calibration build with the selected settings ...")
        return (
            ["build", "--input"]
            + [str(path) for path in paths]
            + ["--samples"]
            + samples
            + evaporation
            + ["--quality-mode", quality_modes[quality_choice]]
        )

    if command_choice == 2:
        model = _choose_model()
        al = _ask_float_value("Al content in wt-%", minimum=0.0)
        ipa = _ask_float_value("IPA content in wt-%", minimum=0.0)
        pg = _ask_float_value("PG content in wt-%", minimum=0.0)
        mg = _ask_float_value("MG content in wt-%", default=0.0, minimum=0.0)
        while al + ipa + pg + mg > 100:
            print(
                "Al + IPA + PG + MG must not exceed 100 wt-%. "
                "Please enter the values again."
            )
            al = _ask_float_value("Al content in wt-%", minimum=0.0)
            ipa = _ask_float_value("IPA content in wt-%", minimum=0.0)
            pg = _ask_float_value("PG content in wt-%", minimum=0.0)
            mg = _ask_float_value("MG content in wt-%", default=0.0, minimum=0.0)
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
            "--mg",
            str(mg),
            "--temperature",
            str(temperature),
        ]

    model = _choose_model()
    paths = _choose_csv_files()
    samples = _ask_samples(paths, prefer_sample_three=False)
    evaporation = _interactive_evaporation_arguments(for_evaluation=True)
    print(f"\nOutput root (automatic run folder): {EVALUATION_ROOT}")
    print("Phase summaries use automatic quality selection; raw rows are retained.")
    print("\nStarting external CSV evaluation ...")
    return (
        ["evaluate", "--model", str(model), "--input"]
        + [str(path) for path in paths]
        + ["--samples"]
        + samples
        + evaporation
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