#!/usr/bin/env python3
"""Vergleicht L-Com-Messwerte mit den Ergebnissen aus ink_calculator.py.

Fuer jeden verwendbaren Messpunkt werden Dichte und Schallgeschwindigkeit
mit derselben Zusammensetzung und Temperatur simuliert. Das Programm erzeugt:

* model_comparison.csv       - Messwert, Simulation und Fehler je Messpunkt
* accuracy_summary.csv       - globale Kennzahlen fuer beide Messgroessen
* accuracy_by_sample.csv     - Kennzahlen getrennt nach ProbeNr
* accuracy_by_composition.csv- Kennzahlen je Rezeptur
* high_deviation_points.csv  - Messpunkte oberhalb der Fehlergrenzen
* time_analysis_by_sample.csv- Zeittrend der Residuen je Probe
* time_analysis_by_segment.csv- Zeittrend innerhalb konstanter Rezepturabschnitte
* model_accuracy.png         - Paritaets- und Fehlerdiagramme
* sample_plots/              - ein Vergleichsplot je ProbeNr
* time_plots/                - Residuen ueber der Zeit je ProbeNr

Aufruf aus dem Repository-Hauptverzeichnis:

    python laboratory_measurement_L-Com/evaluate_model_accuracy.py

Optionen zeigt ``--help``. Der signierte Fehler ist immer
``Simulation - Messung``. ``Rho_M`` wird als kg/m^3 interpretiert;
``InkCalculator.density`` liefert g/cm^3 und wird deshalb mit 1000
multipliziert.

Die Zeitanalyse verwendet den Zeitstempel aus ``Date`` und ``UTC Time``.
Ein negativer Dichtetrend zusammen mit einem positiven Schalltrend ist im
vorliegenden Stoffsystem mit IPA-Verlust vereinbar. Er ist jedoch ohne
Massenbilanz oder eine unabhaengige Konzentrationsmessung kein eindeutiger
Nachweis fuer IPA-Verdunstung.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = SCRIPT_DIR / "measurement_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "model_accuracy"
DEFAULT_TABLES = REPO_DIR / "tables_parameters"

# Massenanteile des Konzentrats SL120 laut Herstellerangabe.
SL120 = {"Al": 0.20, "IPA": 0.40, "PG": 0.40}
MASS_COLUMNS = ["m_SL120", "m_Wasser", "m_IPA", "m_PG", "m_MG"]
REQUIRED_COLUMNS = MASS_COLUMNS + ["Rho_M", "C_M", "T_M"]

# Dieselben grundlegenden Plausibilitaetsgrenzen wie in
# evaluation_laboratory_measurement.py.
VALUE_RANGES = {
    "Rho_M": (500.0, 2000.0),  # kg/m^3
    "C_M": (1000.0, 2200.0),  # m/s
    "T_M": (-10.0, 100.0),  # degC
}


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Wert muss groesser oder gleich 0 sein.")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Wert muss groesser oder gleich 0 sein.")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Messwerte von Dichte und Schallgeschwindigkeit mit "
        "ink_calculator.py vergleichen."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="CSV-Datei oder Verzeichnis mit CSV-Dateien (Standard: measurement_data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ausgabeverzeichnis (Standard: results/model_accuracy).",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=DEFAULT_TABLES,
        help="Verzeichnis tables_parameters fuer ink_calculator.py.",
    )
    parser.add_argument(
        "--require-quality-flags",
        action="store_true",
        help="Nur Zeilen verwenden, deren vorhandene Flags SensOK, Gueltig "
        "und Stabil jeweils 1 sind.",
    )
    parser.add_argument(
        "--allow-mg-as-water",
        action="store_true",
        help="MG-Anteil als Wasserrest behandeln. Ohne diese Option werden "
        "Rezepturen mit MG uebersprungen, da ink_calculator.py kein MG-Modell hat.",
    )
    parser.add_argument(
        "--rho-outlier-threshold-pct",
        type=non_negative_float,
        default=0.5,
        help="Grenze der absoluten relativen Dichteabweichung in %% (Standard: 0.5).",
    )
    parser.add_argument(
        "--c-outlier-threshold-pct",
        type=non_negative_float,
        default=1.0,
        help="Grenze der absoluten relativen Schallabweichung in %% (Standard: 1.0).",
    )
    parser.add_argument(
        "--max-console-outliers",
        type=non_negative_int,
        default=20,
        help="Maximal ausgegebene Punkte je Messgroesse; 0 zeigt alle (Standard: 20).",
    )
    parser.add_argument(
        "--date-column",
        default="Date",
        help="Spalte mit dem Messdatum fuer die Zeitanalyse (Standard: Date).",
    )
    parser.add_argument(
        "--time-column",
        default="UTC Time",
        help="Spalte mit der Messzeit fuer die Zeitanalyse (Standard: UTC Time).",
    )
    parser.add_argument(
        "--time-segment-gap-min",
        type=non_negative_float,
        default=60.0,
        help="Neue Zeitphase bei groesserer Messpause in Minuten; 0 deaktiviert "
        "die Trennung nach Pausen (Standard: 60). Rezepturwechsel erzeugen "
        "immer eine neue Phase.",
    )
    return parser.parse_args()


def resolve_csv_files(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if files:
            return files
        raise FileNotFoundError(f"Keine CSV-Datei in '{path}' gefunden.")
    raise FileNotFoundError(f"Eingabepfad '{path}' existiert nicht.")


def load_measurements(path: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for csv_path in resolve_csv_files(path):
        frame = pd.read_csv(csv_path, comment="/", skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(
                f"In '{csv_path.name}' fehlen Pflichtspalten: {', '.join(missing)}"
            )
        frame["Source_File"] = csv_path.name
        frame["Source_Row"] = np.arange(2, len(frame) + 2)
        parts.append(frame)
        print(f"  {csv_path.name}: {len(frame)} Messpunkte")

    return pd.concat(parts, ignore_index=True)


def add_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die Gesamtzusammensetzung aus den Einwaagen in Massen-%."""
    result = df.copy()
    for column in MASS_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    total_mass = result[MASS_COLUMNS].sum(axis=1, min_count=len(MASS_COLUMNS))
    safe_total = total_mass.where(total_mass > 0)
    result["Mass_Total_g"] = total_mass
    result["Al_wt_pct"] = 100.0 * SL120["Al"] * result["m_SL120"] / safe_total
    result["IPA_wt_pct"] = (
        100.0
        * (SL120["IPA"] * result["m_SL120"] + result["m_IPA"])
        / safe_total
    )
    result["PG_wt_pct"] = (
        100.0
        * (SL120["PG"] * result["m_SL120"] + result["m_PG"])
        / safe_total
    )
    result["MG_wt_pct"] = 100.0 * result["m_MG"] / safe_total
    result["Water_wt_pct"] = 100.0 * result["m_Wasser"] / safe_total
    result["Composition_Sum_wt_pct"] = result[
        ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct", "Water_wt_pct"]
    ].sum(axis=1, min_count=5)
    return result


def _time_group_columns(df: pd.DataFrame) -> list[str]:
    """Gruppenschluessel, damit gleiche ProbeNr aus mehreren CSVs nicht vermischt werden."""
    columns = [column for column in ("Source_File", "ProbeNr") if column in df.columns]
    return columns or ["Source_File"]


def add_time_information(
    df: pd.DataFrame,
    date_column: str,
    time_column: str,
    segment_gap_min: float,
) -> pd.DataFrame:
    """Liest Zeitstempel ein und erkennt Rezeptur-/Messphasen je Probe.

    Eine neue Phase beginnt bei einer geaenderten Einwaage oder - sofern die
    Grenze groesser null ist - nach einer groesseren Messpause. Die Phasen
    erlauben spaeter einen Trend innerhalb weitgehend konstanter Bedingungen.
    """
    result = df.copy()
    result["Measurement_Time_UTC"] = pd.NaT
    result["Elapsed_Time_h"] = np.nan
    result["Time_Delta_min"] = np.nan
    result["Time_Gap_Flag"] = False
    result["Time_Segment"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["Segment_Elapsed_h"] = np.nan

    if date_column not in result.columns or time_column not in result.columns:
        print(
            f"  WARNING: Zeitspalten '{date_column}' und/oder '{time_column}' fehlen; "
            "Zeitanalyse wird uebersprungen."
        )
        return result

    combined = (
        result[date_column].astype("string").str.strip()
        + " "
        + result[time_column].astype("string").str.strip()
    )
    timestamps = pd.to_datetime(
        combined,
        errors="coerce",
        format="mixed",
        dayfirst=True,
        utc=True,
    )
    result["Measurement_Time_UTC"] = timestamps.dt.tz_convert(None)

    group_columns = _time_group_columns(result)
    for _, group in result.groupby(group_columns, dropna=False, sort=False):
        valid = group.loc[group["Measurement_Time_UTC"].notna()].sort_values(
            "Measurement_Time_UTC", kind="stable"
        )
        if valid.empty:
            continue

        time_values = valid["Measurement_Time_UTC"]
        delta_min = time_values.diff().dt.total_seconds().div(60.0)
        elapsed_h = (time_values - time_values.iloc[0]).dt.total_seconds().div(3600.0)

        masses = valid[MASS_COLUMNS].to_numpy(float)
        composition_change = np.ones(len(valid), dtype=bool)
        if len(valid) > 1:
            composition_change[1:] = ~np.isclose(
                masses[1:], masses[:-1], rtol=0.0, atol=1e-9, equal_nan=True
            ).all(axis=1)
        gap_flag = delta_min.gt(segment_gap_min) if segment_gap_min > 0 else pd.Series(
            False, index=valid.index
        )
        new_segment = composition_change | gap_flag.to_numpy(bool)
        segment = np.cumsum(new_segment).astype(int)

        result.loc[valid.index, "Elapsed_Time_h"] = elapsed_h.to_numpy(float)
        result.loc[valid.index, "Time_Delta_min"] = delta_min.to_numpy(float)
        result.loc[valid.index, "Time_Gap_Flag"] = gap_flag.to_numpy(bool)
        result.loc[valid.index, "Time_Segment"] = segment

        segment_start = time_values.groupby(segment).transform("min")
        segment_elapsed = (time_values - segment_start).dt.total_seconds().div(3600.0)
        result.loc[valid.index, "Segment_Elapsed_h"] = segment_elapsed.to_numpy(float)

    invalid_count = int(result["Measurement_Time_UTC"].isna().sum())
    if invalid_count:
        print(f"  WARNING: {invalid_count} Zeitstempel konnten nicht gelesen werden.")
    return result


def _append_reason(reasons: pd.Series, mask: pd.Series, text: str) -> None:
    reasons.loc[mask.fillna(True)] += text + "; "


def flag_rows(
    df: pd.DataFrame,
    require_quality_flags: bool,
    allow_mg_as_water: bool,
) -> pd.DataFrame:
    """Kennzeichnet ungeeignete Zeilen, ohne sie aus der Detaildatei zu loeschen."""
    result = df.copy()
    reasons = pd.Series("", index=result.index, dtype=object)

    for column in ["Rho_M", "C_M", "T_M"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if "N" in result.columns:
        samples = pd.to_numeric(result["N"], errors="coerce")
        _append_reason(reasons, samples < 1, "N < 1")

    for column, (lower, upper) in VALUE_RANGES.items():
        values = result[column]
        _append_reason(
            reasons,
            values.isna() | ~values.between(lower, upper),
            f"{column} ausserhalb {lower:g}..{upper:g}",
        )

    # Das Dichtemodell in ink_calculator.py akzeptiert nur 20 bis 30 degC.
    _append_reason(
        reasons,
        result["T_M"].isna() | ~result["T_M"].between(20.0, 30.0),
        "T_M ausserhalb Modellbereich 20..30 degC",
    )

    invalid_mass = (
        result[MASS_COLUMNS].isna().any(axis=1)
        | (result[MASS_COLUMNS] < 0).any(axis=1)
        | (result["Mass_Total_g"] <= 0)
    )
    _append_reason(reasons, invalid_mass, "ungueltige Einwaage")

    if not allow_mg_as_water:
        _append_reason(
            reasons,
            result["MG_wt_pct"].fillna(np.inf).abs() > 1e-9,
            "MG wird von ink_calculator.py nicht modelliert",
        )

    if require_quality_flags:
        for flag in ("SensOK", "Gueltig", "Stabil"):
            if flag in result.columns:
                values = pd.to_numeric(result[flag], errors="coerce")
                _append_reason(reasons, values != 1, f"{flag} != 1")

    result["Evaluation_Reason"] = reasons.str.rstrip("; ")
    result["Evaluation_Usable"] = result["Evaluation_Reason"] == ""
    return result


def load_calculator(tables_dir: Path):
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from ink_calculator import InkCalculator

    return InkCalculator(tables_dir=str(tables_dir.expanduser().resolve()))


def simulate(df: pd.DataFrame, calculator) -> pd.DataFrame:
    result = df.copy()
    result["Rho_Sim_kg_m3"] = np.nan
    result["C_Sim_m_s"] = np.nan
    result["Simulation_Status"] = np.where(
        result["Evaluation_Usable"], "pending", "skipped"
    )

    for index, row in result.loc[result["Evaluation_Usable"]].iterrows():
        try:
            arguments = {
                "al": float(row["Al_wt_pct"]),
                "ipa": float(row["IPA_wt_pct"]),
                "pg": float(row["PG_wt_pct"]),
                "temperature": float(row["T_M"]),
            }
            # InkCalculator: g/cm^3 -> Messdaten: kg/m^3.
            rho_sim = 1000.0 * calculator.density(**arguments)
            # Einzelne Randspalten in pg_sound.csv enthalten nur einen Wert.
            # SciPy warnt beim Aufbau der ungenutzten Rand-Extrapolation, obwohl
            # das Ergebnis im Temperaturbereich 20..30 degC endlich ist.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in divide",
                    category=RuntimeWarning,
                    module=r"scipy\.interpolate.*",
                )
                c_sim = calculator.sound_velocity(**arguments)
            if not np.isfinite(rho_sim) or not np.isfinite(c_sim):
                raise ValueError("Simulation lieferte NaN oder unendlich")
            result.at[index, "Rho_Sim_kg_m3"] = rho_sim
            result.at[index, "C_Sim_m_s"] = c_sim
            result.at[index, "Simulation_Status"] = "ok"
        except Exception as exc:  # Zeilenweise weiterrechnen und Fehler dokumentieren.
            result.at[index, "Simulation_Status"] = (
                f"error: {type(exc).__name__}: {exc}"
            )

    ok = result["Simulation_Status"] == "ok"
    for measured, simulated, prefix in (
        ("Rho_M", "Rho_Sim_kg_m3", "Rho"),
        ("C_M", "C_Sim_m_s", "C"),
    ):
        signed = result[simulated] - result[measured]
        result[f"{prefix}_Error"] = signed.where(ok)
        result[f"{prefix}_Abs_Error"] = signed.abs().where(ok)
        denominator = result[measured].abs().replace(0.0, np.nan)
        result[f"{prefix}_Rel_Error_pct"] = (100.0 * signed / denominator).where(ok)
        result[f"{prefix}_Abs_Rel_Error_pct"] = (
            100.0 * signed.abs() / denominator
        ).where(ok)

    return result


def metric_row(
    measured: pd.Series,
    simulated: pd.Series,
    property_name: str,
    unit: str,
) -> dict[str, float | int | str]:
    pairs = pd.DataFrame({"measured": measured, "simulated": simulated}).dropna()
    if pairs.empty:
        return {
            "Property": property_name,
            "Unit": unit,
            "N": 0,
            "Mean_Measured": np.nan,
            "Mean_Simulated": np.nan,
            "Bias_ME": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_pct": np.nan,
            "Accuracy_100_minus_MAPE_pct": np.nan,
            "R2": np.nan,
            "Max_Abs_Error": np.nan,
        }

    y = pairs["measured"].to_numpy(float)
    y_hat = pairs["simulated"].to_numpy(float)
    error = y_hat - y
    nonzero = np.abs(y) > np.finfo(float).eps
    mape = float(np.mean(np.abs(error[nonzero] / y[nonzero])) * 100.0)
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    return {
        "Property": property_name,
        "Unit": unit,
        "N": len(pairs),
        "Mean_Measured": float(np.mean(y)),
        "Mean_Simulated": float(np.mean(y_hat)),
        "Bias_ME": float(np.mean(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE_pct": mape,
        "Accuracy_100_minus_MAPE_pct": max(0.0, 100.0 - mape),
        "R2": r2,
        "Max_Abs_Error": float(np.max(np.abs(error))),
    }


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df["Simulation_Status"] == "ok"
    return pd.DataFrame(
        [
            metric_row(
                df.loc[ok, "Rho_M"],
                df.loc[ok, "Rho_Sim_kg_m3"],
                "Density",
                "kg/m^3",
            ),
            metric_row(
                df.loc[ok, "C_M"],
                df.loc[ok, "C_Sim_m_s"],
                "Sound velocity",
                "m/s",
            ),
        ]
    )


def make_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet dieselben Modellkennzahlen separat fuer jede ProbeNr."""
    if "ProbeNr" not in df.columns:
        return pd.DataFrame()

    parts: list[pd.DataFrame] = []
    group_columns = _time_group_columns(df)
    for group_key, sample in df.groupby(group_columns, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        metrics = make_summary(sample)
        for column, value in reversed(list(zip(group_columns, key_values))):
            metrics.insert(0, column, value)
        parts.append(metrics)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def make_composition_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.loc[df["Simulation_Status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    composition = ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "Water_wt_pct"]
    # Einwaagen sind pro Rezeptur konstant; Rundung verhindert kuenstliche
    # Gruppen durch Gleitkomma-Rauschen.
    group_columns = []
    for column in composition:
        key = f"_{column}_group"
        ok[key] = ok[column].round(6)
        group_columns.append(key)

    rows: list[dict[str, float | int]] = []
    for _, group in ok.groupby(group_columns, dropna=False, sort=True):
        row: dict[str, float | int] = {
            column: float(group[column].iloc[0]) for column in composition
        }
        row["N"] = len(group)
        row["T_Min_C"] = float(group["T_M"].min())
        row["T_Max_C"] = float(group["T_M"].max())
        for measured, simulated, prefix in (
            ("Rho_M", "Rho_Sim_kg_m3", "Rho"),
            ("C_M", "C_Sim_m_s", "C"),
        ):
            error = group[simulated] - group[measured]
            row[f"{prefix}_Bias"] = float(error.mean())
            row[f"{prefix}_MAE"] = float(error.abs().mean())
            row[f"{prefix}_RMSE"] = float(np.sqrt(np.mean(error**2)))
            row[f"{prefix}_MAPE_pct"] = float(
                (100.0 * error.abs() / group[measured].abs().replace(0, np.nan)).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _linear_trend(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    """OLS-Zeittrend mit 95-%-Konfidenzintervall und Rangkorrelation."""
    pairs = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x", kind="stable")
    empty = {
        "N": len(pairs),
        "Slope_per_h": np.nan,
        "Intercept": np.nan,
        "Slope_CI95_Low": np.nan,
        "Slope_CI95_High": np.nan,
        "Slope_p_value": np.nan,
        "Time_R2": np.nan,
        "Spearman_rho": np.nan,
        "Spearman_p_value": np.nan,
        "Fitted_Change": np.nan,
    }
    if len(pairs) < 3 or pairs["x"].nunique() < 2:
        return empty

    regression = stats.linregress(pairs["x"], pairs["y"])
    dof = len(pairs) - 2
    critical = float(stats.t.ppf(0.975, dof))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman = stats.spearmanr(pairs["x"], pairs["y"], nan_policy="omit")
    duration = float(pairs["x"].iloc[-1] - pairs["x"].iloc[0])
    return {
        "N": len(pairs),
        "Slope_per_h": float(regression.slope),
        "Intercept": float(regression.intercept),
        "Slope_CI95_Low": float(regression.slope - critical * regression.stderr),
        "Slope_CI95_High": float(regression.slope + critical * regression.stderr),
        "Slope_p_value": float(regression.pvalue),
        "Time_R2": float(regression.rvalue**2),
        "Spearman_rho": float(spearman.statistic),
        "Spearman_p_value": float(spearman.pvalue),
        "Fitted_Change": float(regression.slope * duration),
    }


def _within_segment_trend(
    elapsed_h: pd.Series,
    error: pd.Series,
    segment: pd.Series,
) -> dict[str, float | int]:
    """Gemeinsamer Zeittrend nach Entfernen eines Offsets je Rezepturphase.

    Diese Fixed-Effects-Schaetzung reduziert die Verwechslung eines Zeittrends
    mit unterschiedlichen Modelloffsets nach einer Stoffzugabe.
    """
    values = pd.DataFrame(
        {"x": elapsed_h, "y": error, "segment": segment}
    ).dropna()
    empty = {
        "Within_Segment_N": len(values),
        "Within_Segment_Count": int(values["segment"].nunique()),
        "Within_Segment_Slope_per_h": np.nan,
        "Within_Segment_CI95_Low": np.nan,
        "Within_Segment_CI95_High": np.nan,
        "Within_Segment_p_value": np.nan,
        "Within_Segment_R2": np.nan,
    }
    if len(values) < 3 or values["segment"].nunique() < 1:
        return empty

    x_centered = values["x"] - values.groupby("segment")["x"].transform("mean")
    y_centered = values["y"] - values.groupby("segment")["y"].transform("mean")
    denominator = float(np.sum(x_centered**2))
    if denominator <= np.finfo(float).eps:
        return empty

    slope = float(np.sum(x_centered * y_centered) / denominator)
    residual = y_centered - slope * x_centered
    segment_count = int(values["segment"].nunique())
    dof = len(values) - segment_count - 1
    if dof <= 0:
        return empty
    residual_variance = float(np.sum(residual**2) / dof)
    standard_error = float(np.sqrt(residual_variance / denominator))
    t_value = slope / standard_error if standard_error > 0 else np.inf
    p_value = float(2.0 * stats.t.sf(abs(t_value), dof))
    critical = float(stats.t.ppf(0.975, dof))
    total = float(np.sum(y_centered**2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan
    return {
        "Within_Segment_N": len(values),
        "Within_Segment_Count": segment_count,
        "Within_Segment_Slope_per_h": slope,
        "Within_Segment_CI95_Low": slope - critical * standard_error,
        "Within_Segment_CI95_High": slope + critical * standard_error,
        "Within_Segment_p_value": p_value,
        "Within_Segment_R2": r2,
    }


def make_time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Zeittrend der Modellresiduen fuer jede Probe und Messgroesse."""
    if "Measurement_Time_UTC" not in df.columns:
        return pd.DataFrame()
    usable = df.loc[
        df["Simulation_Status"].eq("ok") & df["Measurement_Time_UTC"].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = _time_group_columns(usable)
    specs = (
        ("Density", "kg/m^3", "Rho_Error"),
        ("Sound velocity", "m/s", "C_Error"),
    )
    for group_key, sample in usable.groupby(group_columns, dropna=False, sort=True):
        sample = sample.sort_values("Measurement_Time_UTC", kind="stable")
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = dict(zip(group_columns, key_values))
        start = sample["Measurement_Time_UTC"].iloc[0]
        end = sample["Measurement_Time_UTC"].iloc[-1]
        duration_h = float((end - start).total_seconds() / 3600.0)
        max_gap_h = float(sample["Time_Delta_min"].max() / 60.0)
        if not np.isfinite(max_gap_h):
            max_gap_h = np.nan

        for property_name, unit, error_column in specs:
            trend = _linear_trend(sample["Elapsed_Time_h"], sample[error_column])
            within = _within_segment_trend(
                sample["Elapsed_Time_h"], sample[error_column], sample["Time_Segment"]
            )
            row: dict[str, object] = {
                **identifiers,
                "Property": property_name,
                "Unit": unit,
                "Start_Time_UTC": start,
                "End_Time_UTC": end,
                "Duration_h": duration_h,
                "Max_Gap_h": max_gap_h,
                "First_Error": float(sample[error_column].iloc[0]),
                "Last_Error": float(sample[error_column].iloc[-1]),
                "Raw_First_to_Last_Change": float(
                    sample[error_column].iloc[-1] - sample[error_column].iloc[0]
                ),
                **trend,
                **within,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def make_time_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Zeittrends innerhalb automatisch erkannter Rezeptur-/Messphasen."""
    if "Time_Segment" not in df.columns:
        return pd.DataFrame()
    usable = df.loc[
        df["Simulation_Status"].eq("ok")
        & df["Measurement_Time_UTC"].notna()
        & df["Time_Segment"].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = _time_group_columns(usable) + ["Time_Segment"]
    specs = (
        ("Density", "kg/m^3", "Rho_Error"),
        ("Sound velocity", "m/s", "C_Error"),
    )
    for group_key, segment in usable.groupby(group_columns, dropna=False, sort=True):
        segment = segment.sort_values("Measurement_Time_UTC", kind="stable")
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = dict(zip(group_columns, key_values))
        start = segment["Measurement_Time_UTC"].iloc[0]
        end = segment["Measurement_Time_UTC"].iloc[-1]
        base = {
            **identifiers,
            "Start_Time_UTC": start,
            "End_Time_UTC": end,
            "Duration_h": float((end - start).total_seconds() / 3600.0),
            "Al_wt_pct": float(segment["Al_wt_pct"].iloc[0]),
            "IPA_wt_pct": float(segment["IPA_wt_pct"].iloc[0]),
            "PG_wt_pct": float(segment["PG_wt_pct"].iloc[0]),
            "Water_wt_pct": float(segment["Water_wt_pct"].iloc[0]),
        }
        for property_name, unit, error_column in specs:
            rows.append(
                {
                    **base,
                    "Property": property_name,
                    "Unit": unit,
                    **_linear_trend(segment["Segment_Elapsed_h"], segment[error_column]),
                }
            )
    return pd.DataFrame(rows)


def identify_high_deviations(
    df: pd.DataFrame,
    rho_threshold_pct: float,
    c_threshold_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Markiert und extrahiert Punkte oberhalb der relativen Fehlergrenzen."""
    result = df.copy()
    specs = (
        {
            "property": "Density",
            "unit": "kg/m^3",
            "prefix": "Rho",
            "measured": "Rho_M",
            "simulated": "Rho_Sim_kg_m3",
            "threshold": rho_threshold_pct,
        },
        {
            "property": "Sound velocity",
            "unit": "m/s",
            "prefix": "C",
            "measured": "C_M",
            "simulated": "C_Sim_m_s",
            "threshold": c_threshold_pct,
        },
    )
    optional_identifiers = [
        column
        for column in ("ProbeNr", "Nr", "N", "Stabil", "SensOK", "Gueltig")
        if column in result.columns
    ]
    identifier_columns = ["Source_File", "Source_Row"] + optional_identifiers
    composition_columns = [
        "T_M",
        "Al_wt_pct",
        "IPA_wt_pct",
        "PG_wt_pct",
        "Water_wt_pct",
    ]
    standardized_columns = [
        "Property",
        "Unit",
        "Threshold_pct",
        *identifier_columns,
        *composition_columns,
        "Measured",
        "Simulated",
        "Error_Sim_minus_Meas",
        "Abs_Error",
        "Rel_Error_pct",
        "Abs_Rel_Error_pct",
    ]
    parts: list[pd.DataFrame] = []

    for spec in specs:
        prefix = spec["prefix"]
        flag_column = f"{prefix}_High_Deviation"
        mask = (
            result["Simulation_Status"].eq("ok")
            & result[f"{prefix}_Abs_Rel_Error_pct"].ge(spec["threshold"])
        )
        result[flag_column] = mask
        selected = result.loc[
            mask,
            identifier_columns
            + composition_columns
            + [
                spec["measured"],
                spec["simulated"],
                f"{prefix}_Error",
                f"{prefix}_Abs_Error",
                f"{prefix}_Rel_Error_pct",
                f"{prefix}_Abs_Rel_Error_pct",
            ],
        ].copy()
        if selected.empty:
            continue

        selected.insert(0, "Threshold_pct", spec["threshold"])
        selected.insert(0, "Unit", spec["unit"])
        selected.insert(0, "Property", spec["property"])
        selected = selected.rename(
            columns={
                spec["measured"]: "Measured",
                spec["simulated"]: "Simulated",
                f"{prefix}_Error": "Error_Sim_minus_Meas",
                f"{prefix}_Abs_Error": "Abs_Error",
                f"{prefix}_Rel_Error_pct": "Rel_Error_pct",
                f"{prefix}_Abs_Rel_Error_pct": "Abs_Rel_Error_pct",
            }
        )
        parts.append(selected[standardized_columns])

    if not parts:
        return result, pd.DataFrame(columns=standardized_columns)

    deviations = pd.concat(parts, ignore_index=True)
    deviations = deviations.sort_values(
        ["Property", "Abs_Rel_Error_pct"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
    return result, deviations


def _identity_limits(measured: np.ndarray, simulated: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([measured, simulated])
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    padding = 0.04 * (upper - lower) if upper > lower else max(abs(lower) * 0.01, 1.0)
    return lower - padding, upper + padding


def create_plot(
    df: pd.DataFrame,
    output_path: Path,
    figure_title: str = "Accuracy of ink_calculator.py against L-Com measurements",
) -> None:
    ok = df.loc[df["Simulation_Status"] == "ok"].copy()
    if ok.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs: list[tuple[str, str, str, str, str]] = [
        ("Rho_M", "Rho_Sim_kg_m3", "Rho_Error", "Density", "kg/m³"),
        ("C_M", "C_Sim_m_s", "C_Error", "Sound velocity", "m/s"),
    ]
    cmap = plt.get_cmap("viridis")

    for column, (measured_col, simulated_col, error_col, title, unit) in enumerate(specs):
        measured = ok[measured_col].to_numpy(float)
        simulated = ok[simulated_col].to_numpy(float)
        temperatures = ok["T_M"].to_numpy(float)

        ax = axes[0, column]
        scatter = ax.scatter(
            measured,
            simulated,
            c=temperatures,
            cmap=cmap,
            s=28,
            alpha=0.8,
            edgecolors="none",
        )
        lower, upper = _identity_limits(measured, simulated)
        ax.plot([lower, upper], [lower, upper], "--", color="#555555", lw=1.2)
        ax.set(xlim=(lower, upper), ylim=(lower, upper))
        ax.set_xlabel(f"Measured [{unit}]")
        ax.set_ylabel(f"Simulated [{unit}]")
        ax.set_title(f"{title}: measured vs. simulated")
        ax.grid(alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="Temperature [°C]")

        ax = axes[1, column]
        ax.scatter(temperatures, ok[error_col], s=25, alpha=0.75, color="#1f4e79")
        ax.axhline(0.0, color="#555555", ls="--", lw=1.2)
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel(f"Simulation - measurement [{unit}]")
        ax.set_title(f"{title}: error vs. temperature")
        ax.grid(alpha=0.25)

    fig.suptitle(figure_title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _probe_file_label(probe: object) -> str:
    """Erzeugt einen stabilen, dateisystemtauglichen Bezeichner fuer ProbeNr."""
    try:
        numeric = float(probe)
        if np.isfinite(numeric) and numeric.is_integer():
            return f"{int(numeric):03d}"
    except (TypeError, ValueError):
        pass
    label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(probe)
    ).strip("_")
    return label or "unknown"


def create_sample_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Erstellt fuer jede ProbeNr einen eigenen Messung-Simulation-Plot."""
    if "ProbeNr" not in df.columns:
        print("  WARNING: Spalte 'ProbeNr' fehlt; Einzelplots werden uebersprungen.")
        return []

    sample_dir = output_dir / "sample_plots"
    paths: list[Path] = []
    for probe, sample in df.groupby("ProbeNr", dropna=False, sort=True):
        usable = sample.loc[sample["Simulation_Status"] == "ok"]
        if usable.empty:
            print(f"  Probe {probe}: keine simulierten Messpunkte, Plot uebersprungen.")
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f"Sample_{_probe_file_label(probe)}_model_accuracy.png"
        create_plot(
            usable,
            output_path,
            figure_title=(
                f"Accuracy of ink_calculator.py against L-Com measurements "
                f"- sample {probe} ({len(usable)} points)"
            ),
        )
        paths.append(output_path)
    return paths


def create_time_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Stellt die signierten Modellabweichungen je Probe ueber der Zeit dar."""
    required = {"Measurement_Time_UTC", "Elapsed_Time_h", "Time_Segment"}
    if not required.issubset(df.columns):
        return []
    usable = df.loc[
        df["Simulation_Status"].eq("ok") & df["Measurement_Time_UTC"].notna()
    ].copy()
    if usable.empty:
        return []

    time_dir = output_dir / "time_plots"
    time_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    group_columns = _time_group_columns(usable)
    multiple_files = usable["Source_File"].nunique() > 1 if "Source_File" in usable else False
    specs = (
        ("Rho_Error", "Dichte", "Simulation - Messung [kg/m³]", "kg/m³"),
        ("C_Error", "Schallgeschwindigkeit", "Simulation - Messung [m/s]", "m/s"),
    )

    for group_key, sample in usable.groupby(group_columns, dropna=False, sort=True):
        sample = sample.sort_values("Measurement_Time_UTC", kind="stable")
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = dict(zip(group_columns, key_values))
        probe = identifiers.get("ProbeNr", "unknown")
        file_prefix = ""
        if multiple_files:
            source_stem = Path(str(identifiers.get("Source_File", "data"))).stem
            file_prefix = _probe_file_label(source_stem) + "_"
        output_path = time_dir / (
            f"{file_prefix}Sample_{_probe_file_label(probe)}_error_over_time.png"
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        segment_ids = sorted(sample["Time_Segment"].dropna().astype(int).unique())
        color_map = plt.get_cmap("tab20")
        for ax, (error_column, title, ylabel, unit) in zip(axes, specs):
            for color_index, segment_id in enumerate(segment_ids):
                part = sample.loc[sample["Time_Segment"].eq(segment_id)]
                ax.scatter(
                    part["Elapsed_Time_h"],
                    part[error_column],
                    s=29,
                    alpha=0.82,
                    color=color_map(color_index % 20),
                    label=f"Phase {segment_id}",
                )

            trend = _linear_trend(sample["Elapsed_Time_h"], sample[error_column])
            slope = float(trend["Slope_per_h"])
            if np.isfinite(slope):
                x_line = np.array(
                    [sample["Elapsed_Time_h"].min(), sample["Elapsed_Time_h"].max()]
                )
                y_line = float(trend["Intercept"]) + slope * x_line
                ax.plot(x_line, y_line, color="black", lw=1.6, ls="--", label="Gesamttrend")
                annotation = (
                    f"Trend: {slope:+.4g} {unit}/h\n"
                    f"p = {float(trend['Slope_p_value']):.3g}, R² = {float(trend['Time_R2']):.3f}"
                )
                ax.text(
                    0.015,
                    0.97,
                    annotation,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82},
                )
            ax.axhline(0.0, color="#666666", lw=1.0)
            ax.set_title(f"{title}: Modellabweichung ueber der Zeit")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)

        axes[-1].set_xlabel("Zeit seit erstem Zeitstempel der Probe [h]")
        if len(segment_ids) <= 10:
            axes[0].legend(loc="best", fontsize=8, ncol=2)
        start = sample["Measurement_Time_UTC"].min()
        end = sample["Measurement_Time_UTC"].max()
        fig.suptitle(
            f"Probe {probe}: Residuen ueber der Zeit ({start} bis {end} UTC)", fontsize=13
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)
    return paths


def write_outputs(
    df: pd.DataFrame,
    deviations: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = make_summary(df)
    sample_summary = make_sample_summary(df)
    composition_summary = make_composition_summary(df)
    time_summary = make_time_summary(df)
    time_segment_summary = make_time_segment_summary(df)
    detail_path = output_dir / "model_comparison.csv"
    summary_path = output_dir / "accuracy_summary.csv"
    sample_summary_path = output_dir / "accuracy_by_sample.csv"
    composition_path = output_dir / "accuracy_by_composition.csv"
    deviations_path = output_dir / "high_deviation_points.csv"
    time_summary_path = output_dir / "time_analysis_by_sample.csv"
    time_segment_path = output_dir / "time_analysis_by_segment.csv"
    plot_path = output_dir / "model_accuracy.png"

    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    sample_summary.to_csv(sample_summary_path, index=False)
    composition_summary.to_csv(composition_path, index=False)
    deviations.to_csv(deviations_path, index=False)
    time_summary.to_csv(time_summary_path, index=False)
    time_segment_summary.to_csv(time_segment_path, index=False)
    create_plot(df, plot_path)
    outputs = [
        detail_path,
        summary_path,
        sample_summary_path,
        composition_path,
        deviations_path,
        time_summary_path,
        time_segment_path,
    ]
    if plot_path.exists():
        outputs.append(plot_path)
    outputs.extend(create_sample_plots(df, output_dir))
    outputs.extend(create_time_plots(df, output_dir))
    return summary, time_summary, time_segment_summary, outputs


def print_summary(summary: pd.DataFrame) -> None:
    print("\nGesamtkennzahlen")
    print("-" * 96)
    columns = [
        "Property",
        "N",
        "Bias_ME",
        "MAE",
        "RMSE",
        "MAPE_pct",
        "Accuracy_100_minus_MAPE_pct",
        "R2",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 130):
        print(summary[columns].round(5).to_string(index=False))
    print("\nHinweis: 'Accuracy' ist 100 - MAPE. Der signierte Fehler ist Simulation - Messung.")


def _print_deviation_rows(group: pd.DataFrame, max_rows: int) -> None:
    """Gibt die Detailzeilen einer bereits ausgewaehlten Ausreissergruppe aus."""
    group = group.sort_values("Abs_Rel_Error_pct", ascending=False, kind="stable")
    shown = group if max_rows == 0 else group.head(max_rows)
    display_columns = [
        column
        for column in (
            "Source_Row",
            "Nr",
            "T_M",
            "Al_wt_pct",
            "IPA_wt_pct",
            "PG_wt_pct",
            "Measured",
            "Simulated",
            "Error_Sim_minus_Meas",
            "Abs_Rel_Error_pct",
            "Gueltig",
            "Stabil",
        )
        if column in shown.columns
    ]
    display = shown[display_columns].rename(
        columns={
            "Source_Row": "CSV_Row",
            "T_M": "T_C",
            "Al_wt_pct": "Al_pct",
            "IPA_wt_pct": "IPA_pct",
            "PG_wt_pct": "PG_pct",
            "Error_Sim_minus_Meas": "Error",
            "Abs_Rel_Error_pct": "AbsRel_pct",
        }
    )
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        180,
        "display.max_rows",
        None,
    ):
        print(display.round(5).to_string(index=False))
    if len(shown) < len(group):
        print(
            f"  ... {len(group) - len(shown)} weitere; vollstaendig in "
            "high_deviation_points.csv."
        )


def _matching_group(df: pd.DataFrame, identifiers: dict[str, object]) -> pd.DataFrame:
    """Filtert eine Ergebnistabelle mit denselben Probenschluesseln."""
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for column, value in identifiers.items():
        if column not in df.columns:
            continue
        mask &= df[column].isna() if pd.isna(value) else df[column].eq(value)
    return df.loc[mask]


def _print_time_analysis(sample_time: pd.DataFrame) -> None:
    print("\nZeitanalyse der Modellabweichung")
    print("-" * 132)
    if sample_time.empty:
        print("  Keine auswertbaren Zeitstempel oder zu wenige Messpunkte.")
        return

    display_columns = [
        "Property",
        "N",
        "Duration_h",
        "Slope_per_h",
        "Slope_CI95_Low",
        "Slope_CI95_High",
        "Slope_p_value",
        "Time_R2",
        "Fitted_Change",
        "Within_Segment_Slope_per_h",
        "Within_Segment_p_value",
    ]
    display = sample_time[display_columns].rename(
        columns={
            "Duration_h": "Dauer_h",
            "Slope_per_h": "Trend_pro_h",
            "Slope_CI95_Low": "KI95_unten",
            "Slope_CI95_High": "KI95_oben",
            "Slope_p_value": "p_Trend",
            "Time_R2": "R2_Zeit",
            "Fitted_Change": "Aenderung_fit",
            "Within_Segment_Slope_per_h": "Trend_innerhalb_Phase_pro_h",
            "Within_Segment_p_value": "p_innerhalb_Phase",
        }
    )
    with pd.option_context("display.max_columns", None, "display.width", 210):
        print(display.round(6).to_string(index=False))

    rho = sample_time.loc[sample_time["Property"].eq("Density")]
    sound = sample_time.loc[sample_time["Property"].eq("Sound velocity")]
    if rho.empty or sound.empty:
        return
    rho_slope = float(rho["Slope_per_h"].iloc[0])
    sound_slope = float(sound["Slope_per_h"].iloc[0])
    rho_p = float(rho["Slope_p_value"].iloc[0])
    sound_p = float(sound["Slope_p_value"].iloc[0])
    direction_matches = rho_slope < 0 and sound_slope > 0
    both_significant = rho_p < 0.05 and sound_p < 0.05
    if direction_matches and both_significant:
        verdict = (
            "Deutliche IPA-kompatible Signatur: Dichte-Residuum faellt und "
            "Schall-Residuum steigt; beide Gesamttrends sind signifikant (p < 0,05)."
        )
    elif direction_matches:
        verdict = (
            "Richtung mit IPA-Verlust vereinbar, aber nicht beide Gesamttrends "
            "sind signifikant. Das ist nur ein schwacher Hinweis."
        )
    else:
        verdict = (
            "Keine konsistente IPA-Signatur aus beiden Kanaelen: erwartet waeren "
            "fallendes Dichte- und steigendes Schall-Residuum."
        )
    print(f"\nVerdunstungsindikator: {verdict}")

    max_gap = float(sample_time["Max_Gap_h"].max())
    if np.isfinite(max_gap) and max_gap > 1.0:
        print(
            f"WARNUNG: Groesste Messpause = {max_gap:.2f} h. Gesamttrend und Dauer "
            "nur zusammen mit Versuchsprotokoll interpretieren."
        )


def _print_time_segment_analysis(sample_segments: pd.DataFrame) -> None:
    """Kompakte Konsolentabelle der Trends je konstanter Rezepturphase."""
    print("\nZeittrends innerhalb der automatisch erkannten Phasen")
    print("-" * 132)
    if sample_segments.empty:
        print("  Keine Phasen mit auswertbaren Zeitstempeln vorhanden.")
        return

    density = sample_segments.loc[sample_segments["Property"].eq("Density")].copy()
    sound = sample_segments.loc[
        sample_segments["Property"].eq("Sound velocity")
    ].copy()
    keys = ["Source_File", "ProbeNr", "Time_Segment"]
    keys = [column for column in keys if column in sample_segments.columns]
    base_columns = keys + [
        "N",
        "Duration_h",
        "Al_wt_pct",
        "IPA_wt_pct",
        "PG_wt_pct",
        "Slope_per_h",
        "Slope_p_value",
    ]
    density = density[base_columns].rename(
        columns={
            "N": "N_Rho",
            "Slope_per_h": "Rho_Trend_pro_h",
            "Slope_p_value": "Rho_p",
        }
    )
    sound = sound[keys + ["N", "Slope_per_h", "Slope_p_value"]].rename(
        columns={
            "N": "N_C",
            "Slope_per_h": "C_Trend_pro_h",
            "Slope_p_value": "C_p",
        }
    )
    display = density.merge(sound, on=keys, how="outer").rename(
        columns={
            "Time_Segment": "Phase",
            "Duration_h": "Dauer_h",
            "Al_wt_pct": "Al_pct",
            "IPA_wt_pct": "IPA_pct",
            "PG_wt_pct": "PG_pct",
        }
    )
    display_columns = [
        column
        for column in (
            "Phase",
            "N_Rho",
            "Dauer_h",
            "Al_pct",
            "IPA_pct",
            "PG_pct",
            "Rho_Trend_pro_h",
            "Rho_p",
            "C_Trend_pro_h",
            "C_p",
        )
        if column in display.columns
    ]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(display[display_columns].round(6).to_string(index=False))


def print_sample_analyses(
    df: pd.DataFrame,
    deviations: pd.DataFrame,
    time_summary: pd.DataFrame,
    time_segment_summary: pd.DataFrame,
    max_rows: int,
    rho_threshold_pct: float,
    c_threshold_pct: float,
) -> None:
    """Gibt Kennzahlen und Ausreisser getrennt fuer jede ProbeNr aus."""
    print("\n" + "#" * 132)
    print("AUSWERTUNG NACH PROBE")
    print("#" * 132)
    if "ProbeNr" not in df.columns:
        print("Spalte 'ProbeNr' fehlt; keine probenspezifische Auswertung moeglich.")
        return

    outlier_specs = (
        ("Density", rho_threshold_pct),
        ("Sound velocity", c_threshold_pct),
    )
    metric_columns = [
        "Property",
        "N",
        "Bias_ME",
        "MAE",
        "RMSE",
        "MAPE_pct",
        "Accuracy_100_minus_MAPE_pct",
        "R2",
    ]

    group_columns = _time_group_columns(df)
    for group_key, sample in df.groupby(group_columns, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = dict(zip(group_columns, key_values))
        probe = identifiers.get("ProbeNr", "unknown")
        source_note = (
            f" | Datei {identifiers['Source_File']}" if len(df["Source_File"].unique()) > 1 else ""
        )
        simulated = int(sample["Simulation_Status"].eq("ok").sum())
        skipped = int(sample["Simulation_Status"].eq("skipped").sum())
        failed = len(sample) - simulated - skipped
        print("\n" + "=" * 132)
        print(
            f"PROBE {probe}{source_note} | {simulated} simuliert, {skipped} uebersprungen, "
            f"{failed} Simulationsfehler"
        )
        print("=" * 132)

        metrics = make_summary(sample)
        print("\nGenauigkeitskennzahlen")
        print("-" * 96)
        with pd.option_context("display.max_columns", None, "display.width", 130):
            print(metrics[metric_columns].round(5).to_string(index=False))

        _print_time_analysis(_matching_group(time_summary, identifiers))
        _print_time_segment_analysis(
            _matching_group(time_segment_summary, identifiers)
        )

        print("\nAusreisser dieser Probe")
        sample_deviations = _matching_group(deviations, identifiers)

        for property_name, threshold in outlier_specs:
            group = sample_deviations.loc[
                sample_deviations["Property"].eq(property_name)
            ]
            print(
                f"\n{property_name}: {len(group)} Punkt(e) mit absoluter relativer "
                f"Abweichung >= {threshold:g} %"
            )
            if group.empty:
                print("  Keine Ausreisser oberhalb dieser Grenze.")
            else:
                _print_deviation_rows(group, max_rows)


def main() -> int:
    args = parse_args()
    try:
        print("Messdaten laden")
        measurements = load_measurements(args.input)
        evaluated = add_composition(measurements)
        evaluated = add_time_information(
            evaluated,
            date_column=args.date_column,
            time_column=args.time_column,
            segment_gap_min=args.time_segment_gap_min,
        )
        evaluated = flag_rows(
            evaluated,
            require_quality_flags=args.require_quality_flags,
            allow_mg_as_water=args.allow_mg_as_water,
        )

        calculator = load_calculator(args.tables)
        evaluated = simulate(evaluated, calculator)
        evaluated, deviations = identify_high_deviations(
            evaluated,
            rho_threshold_pct=args.rho_outlier_threshold_pct,
            c_threshold_pct=args.c_outlier_threshold_pct,
        )
        summary, time_summary, time_segment_summary, output_paths = write_outputs(
            evaluated, deviations, args.output
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2

    simulated = int((evaluated["Simulation_Status"] == "ok").sum())
    skipped = int((evaluated["Simulation_Status"] == "skipped").sum())
    failed = len(evaluated) - simulated - skipped
    print(
        f"\nAuswertung: {simulated} simuliert, {skipped} uebersprungen, "
        f"{failed} Simulationsfehler."
    )
    print_summary(summary)
    print_sample_analyses(
        evaluated,
        deviations,
        time_summary,
        time_segment_summary,
        max_rows=args.max_console_outliers,
        rho_threshold_pct=args.rho_outlier_threshold_pct,
        c_threshold_pct=args.c_outlier_threshold_pct,
    )
    print("\nErzeugte Dateien:")
    for path in output_paths:
        print(f"  {path}")
    return 0 if simulated > 0 and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
