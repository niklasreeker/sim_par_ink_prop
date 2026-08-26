#!/usr/bin/env python3
"""Estimate evaporation from paired batch-mass measurements.

The PLC export contains the mass immediately before (``m_vorher``) and
after (``m_nachher``) taking a representative laboratory sample.  The
difference inside one row is therefore sample removal, not evaporation.
Evaporation occurs between two rows and is calculated as

    previous mass after sampling + additions - current mass before sampling

The script keeps every input row, adds transparent quality/status columns and
writes a compact summary.  It deliberately does not infer an absolute mass
loss from density and sound velocity alone: that loss is confounded with the
physical-model residual that is to be learned later.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "measurement_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "evaporation"

MASS_COLUMNS = ["m_SL120", "m_Wasser", "m_IPA", "m_PG", "m_MG"]
COMPONENTS = ["Al", "IPA", "PG", "Water", "MG"]
SL120 = {"Al": 0.20, "IPA": 0.40, "PG": 0.40}


@dataclass(frozen=True)
class Settings:
    before_column: str = "m_vorher"
    after_column: str = "m_nachher"
    date_column: str = "Date"
    time_column: str = "UTC Time"
    group_column: str = "ProbeNr"
    negative_tolerance_g: float = 0.02
    ipa_share: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verdunstung aus m_vorher/m_nachher massenbilanziert abschaetzen."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="CSV-Datei oder Verzeichnis mit CSV-Dateien (Standard: measurement_data).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Ausgabeverzeichnis (Standard: results/evaporation).",
    )
    parser.add_argument(
        "--ipa-share", type=float, default=1.0,
        help="Angenommener IPA-Massenanteil am Verdunstungsverlust, 0..1 "
             "(Standard: 1.0 = konservative IPA-only-Annahme).",
    )
    parser.add_argument(
        "--negative-tolerance-g", type=float, default=0.02,
        help="Toleranz fuer kleine negative Bilanzverluste durch Waagenrauschen.",
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


def load_measurements(path: Path, settings: Settings) -> pd.DataFrame:
    required = set(MASS_COLUMNS + [settings.before_column, settings.after_column])
    parts: list[pd.DataFrame] = []
    for csv_path in resolve_csv_files(path):
        frame = pd.read_csv(csv_path, comment="/", skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"In '{csv_path.name}' fehlen Pflichtspalten: {', '.join(missing)}"
            )
        frame["Source_File"] = csv_path.name
        frame["Source_Row"] = np.arange(2, len(frame) + 2)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def add_timestamp(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    result = df.copy()
    if {settings.date_column, settings.time_column}.issubset(result.columns):
        raw = (
            result[settings.date_column].astype("string").str.strip()
            + " "
            + result[settings.time_column].astype("string").str.strip()
        )
        result["Measurement_Time"] = pd.to_datetime(
            raw, errors="coerce", format="mixed", dayfirst=True
        )
    else:
        result["Measurement_Time"] = pd.NaT
    return result


def recipe_component_masses(df: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative weighed component masses for every recipe row."""
    numeric = df[MASS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame(index=df.index)
    result["Al"] = SL120["Al"] * numeric["m_SL120"]
    result["IPA"] = SL120["IPA"] * numeric["m_SL120"] + numeric["m_IPA"]
    result["PG"] = SL120["PG"] * numeric["m_SL120"] + numeric["m_PG"]
    result["Water"] = numeric["m_Wasser"]
    result["MG"] = numeric["m_MG"]
    return result


def _ordered_group(group: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("Measurement_Time", "Nr", "Source_Row")
               if column in group.columns]
    return group.sort_values(columns, kind="stable") if columns else group


def estimate_evaporation(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add interval losses, cumulative losses and corrected composition.

    Corrected composition is propagated through sample removals, additions and
    evaporation.  ``ipa_share`` partitions total evaporation between IPA and
    water; the gravimetric data alone cannot identify that partition.
    """
    if not 0.0 <= settings.ipa_share <= 1.0:
        raise ValueError("ipa_share muss zwischen 0 und 1 liegen.")
    if settings.negative_tolerance_g < 0.0:
        raise ValueError("negative_tolerance_g darf nicht negativ sein.")

    result = add_timestamp(df, settings)
    before = pd.to_numeric(result[settings.before_column], errors="coerce")
    after = pd.to_numeric(result[settings.after_column], errors="coerce")
    recipe = recipe_component_masses(result)

    result["Mass_Data_Available"] = (before > 0.0) & (after > 0.0)
    result["Sample_Removed_g"] = np.nan
    result["Mass_Added_g"] = np.nan
    result["Evaporation_Step_g"] = np.nan
    result["Evaporation_Cumulative_g"] = np.nan
    result["Evaporation_Rate_g_h"] = np.nan
    result["Evaporation_Status"] = "not_processed"
    for component in COMPONENTS:
        result[f"Corrected_{component}_g"] = np.nan
        result[f"Corrected_{component}_wt_pct"] = np.nan

    group_columns = ["Source_File"]
    if settings.group_column in result.columns:
        group_columns.append(settings.group_column)

    for _, raw_group in result.groupby(group_columns, dropna=False, sort=False):
        group = _ordered_group(raw_group)
        previous_index: int | None = None
        previous_recipe: pd.Series | None = None
        state: pd.Series | None = None
        cumulative = 0.0

        for index in group.index:
            current_recipe = recipe.loc[index].astype(float)
            available = bool(result.at[index, "Mass_Data_Available"])
            if not available:
                result.at[index, "Evaporation_Status"] = "missing_mass_data"
                previous_index = None
                previous_recipe = None
                state = None
                cumulative = 0.0
                continue

            m_before = float(before.at[index])
            m_after = float(after.at[index])
            sample_removed = m_before - m_after
            result.at[index, "Sample_Removed_g"] = sample_removed

            if sample_removed < -settings.negative_tolerance_g:
                result.at[index, "Evaporation_Status"] = "mass_after_exceeds_before"
                previous_index = None
                previous_recipe = None
                state = None
                cumulative = 0.0
                continue
            sample_removed = max(sample_removed, 0.0)

            if previous_index is None or previous_recipe is None or state is None:
                total_recipe = float(current_recipe.sum())
                if not np.isfinite(total_recipe) or total_recipe <= 0.0:
                    result.at[index, "Evaporation_Status"] = "invalid_recipe"
                    continue
                state = current_recipe * (m_before / total_recipe)
                added = 0.0
                step = 0.0
                result.at[index, "Evaporation_Status"] = "group_start"
            else:
                component_additions = (current_recipe - previous_recipe).clip(lower=0.0)
                added = float(component_additions.sum())
                step_raw = float(after.at[previous_index]) + added - m_before
                if step_raw < -settings.negative_tolerance_g:
                    result.at[index, "Evaporation_Status"] = "negative_loss_check_addition_or_tare"
                    previous_index = None
                    previous_recipe = None
                    state = None
                    cumulative = 0.0
                    continue
                step = max(step_raw, 0.0)

                # The previous sample withdrawal is representative and changes
                # component masses, but not their fractions.
                previous_before = float(before.at[previous_index])
                withdrawal_factor = max(float(after.at[previous_index]), 0.0) / previous_before
                state = state * withdrawal_factor + component_additions
                ipa_loss = settings.ipa_share * step
                water_loss = (1.0 - settings.ipa_share) * step
                if ipa_loss > state["IPA"] + settings.negative_tolerance_g:
                    result.at[index, "Evaporation_Status"] = "ipa_loss_exceeds_available_mass"
                    previous_index = None
                    previous_recipe = None
                    state = None
                    cumulative = 0.0
                    continue
                if water_loss > state["Water"] + settings.negative_tolerance_g:
                    result.at[index, "Evaporation_Status"] = "water_loss_exceeds_available_mass"
                    previous_index = None
                    previous_recipe = None
                    state = None
                    cumulative = 0.0
                    continue
                state["IPA"] = max(state["IPA"] - ipa_loss, 0.0)
                state["Water"] = max(state["Water"] - water_loss, 0.0)
                cumulative += step
                result.at[index, "Evaporation_Status"] = "ok"

                t_now = result.at[index, "Measurement_Time"]
                t_previous = result.at[previous_index, "Measurement_Time"]
                if pd.notna(t_now) and pd.notna(t_previous):
                    hours = (t_now - t_previous).total_seconds() / 3600.0
                    if hours > 0.0:
                        result.at[index, "Evaporation_Rate_g_h"] = step / hours

            result.at[index, "Mass_Added_g"] = added
            result.at[index, "Evaporation_Step_g"] = step
            result.at[index, "Evaporation_Cumulative_g"] = cumulative
            state_total = float(state.sum())
            for component in COMPONENTS:
                result.at[index, f"Corrected_{component}_g"] = float(state[component])
                result.at[index, f"Corrected_{component}_wt_pct"] = (
                    100.0 * float(state[component]) / state_total
                )

            previous_index = int(index)
            previous_recipe = current_recipe

    return result.sort_index()


def make_summary(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    group_columns = ["Source_File"]
    if settings.group_column in df.columns:
        group_columns.append(settings.group_column)
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        available = group["Mass_Data_Available"].fillna(False)
        valid_loss = group["Evaporation_Step_g"].notna()
        row.update({
            "Rows": len(group),
            "Rows_With_Mass_Data": int(available.sum()),
            "Rows_With_Evaporation_Estimate": int(valid_loss.sum()),
            "Evaporation_Total_g": (
                float(group.loc[valid_loss, "Evaporation_Step_g"].sum())
                if valid_loss.any() else np.nan
            ),
            "Mean_Evaporation_Rate_g_h": (
                float(group["Evaporation_Rate_g_h"].mean())
                if group["Evaporation_Rate_g_h"].notna().any() else np.nan
            ),
            "IPA_Share_Assumption": settings.ipa_share,
            "Status": "ok" if available.any() else "not_identifiable_no_mass_data",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    settings = Settings(
        negative_tolerance_g=args.negative_tolerance_g,
        ipa_share=args.ipa_share,
    )
    measurements = load_measurements(args.input, settings)
    corrected = estimate_evaporation(measurements, settings)
    summary = make_summary(corrected, settings)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "evaporation_estimates.csv"
    summary_path = output / "evaporation_summary.csv"
    corrected.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nDetails: {detail_path}")
    print(f"Summary: {summary_path}")
    if not corrected["Mass_Data_Available"].any():
        print(
            "\nHINWEIS: m_vorher und m_nachher enthalten keine positiven "
            "Messpaare. Absolute Verdunstung ist mit dieser Datei nicht identifizierbar."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
