#!/usr/bin/env python3
"""Vergleicht L-Com-Messwerte mit den Ergebnissen aus ink_calculator.py.

Fuer jeden verwendbaren Messpunkt werden Dichte und Schallgeschwindigkeit
mit derselben Zusammensetzung und Temperatur simuliert. Das Programm erzeugt:

* model_comparison.csv       - Messwert, Simulation und Fehler je Messpunkt
* accuracy_summary.csv       - globale Kennzahlen fuer beide Messgroessen
* accuracy_by_composition.csv- Kennzahlen je Rezeptur
* model_accuracy.png         - Paritaets- und Fehlerdiagramme

Aufruf aus dem Repository-Hauptverzeichnis:

    python laboratory_measurement_L-Com/evaluate_model_accuracy.py

Optionen zeigt ``--help``. Der signierte Fehler ist immer
``Simulation - Messung``. ``Rho_M`` wird als kg/m^3 interpretiert;
``InkCalculator.density`` liefert g/cm^3 und wird deshalb mit 1000
multipliziert.
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


def _identity_limits(measured: np.ndarray, simulated: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([measured, simulated])
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    padding = 0.04 * (upper - lower) if upper > lower else max(abs(lower) * 0.01, 1.0)
    return lower - padding, upper + padding


def create_plot(df: pd.DataFrame, output_path: Path) -> None:
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

    fig.suptitle("Accuracy of ink_calculator.py against L-Com measurements", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = make_summary(df)
    composition_summary = make_composition_summary(df)
    detail_path = output_dir / "model_comparison.csv"
    summary_path = output_dir / "accuracy_summary.csv"
    composition_path = output_dir / "accuracy_by_composition.csv"
    plot_path = output_dir / "model_accuracy.png"

    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    composition_summary.to_csv(composition_path, index=False)
    create_plot(df, plot_path)
    outputs = [detail_path, summary_path, composition_path]
    if plot_path.exists():
        outputs.append(plot_path)
    return summary, outputs


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


def main() -> int:
    args = parse_args()
    try:
        print("Messdaten laden")
        measurements = load_measurements(args.input)
        evaluated = add_composition(measurements)
        evaluated = flag_rows(
            evaluated,
            require_quality_flags=args.require_quality_flags,
            allow_mg_as_water=args.allow_mg_as_water,
        )

        calculator = load_calculator(args.tables)
        evaluated = simulate(evaluated, calculator)
        summary, output_paths = write_outputs(evaluated, args.output)
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
    print("\nErzeugte Dateien:")
    for path in output_paths:
        print(f"  {path}")
    return 0 if simulated > 0 and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
