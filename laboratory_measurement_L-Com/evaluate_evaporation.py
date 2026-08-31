#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer model-consistent IPA/water losses from L-Com time series.

Run interactively from the project root:
    python laboratory_measurement_L-Com/evaluate_evaporation.py

Reproduce the supplied gravimetric experiment:
    python laboratory_measurement_L-Com/evaluate_evaporation.py --weighing

Only unchanged-recipe phases are fitted. Temperature is evaluated at every
timestamp. A separate constant offset is profiled out of each sensor channel,
so absolute calibration bias is not interpreted as evaporation. A composition-
or temperature-dependent model error, settling, sensor drift, or unrecorded
withdrawal can still be mistaken for evaporation. These estimates are NOT an
independent chemical measurement of the vapour composition.

Default: constant nonnegative rates within the selected time window. Optional
piecewise-constant rates (--segments N) maintain a continuous mass balance.
Al, PG and MG are assumed nonvolatile. MG is passed to the existing calculator.
Starting masses are nominal cumulative additions minus explicitly provided
prior losses; losses before this selected window are never silently inferred.

Dependencies: numpy, pandas, scipy, matplotlib and the user's MG-capable
ink_calculator.py with its tables_parameters directory. All comments and
docstrings in this file are in English. Application text and source code use
ASCII characters to prevent mojibake when transferred through legacy Windows
encodings. Calculations and numeric results are unchanged from version 1.0.0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Use ASCII minus signs on axes as well as ASCII-only application labels.
plt.rcParams["axes.unicode_minus"] = False


SCRIPT_DIR = Path(__file__).resolve().parent
MASS_COLUMNS = ["m_SL120", "m_Wasser", "m_IPA", "m_PG", "m_MG"]
COMPONENTS = ["Al", "IPA", "PG", "MG", "Water"]
COLORS = {"none": "#9AA4AE", "ipa": "#D17A22", "mixed": "#087F8C"}
LABELS = {"none": "Keine Verdunstung", "ipa": "Nur IPA", "mixed": "IPA + Wasser"}
VERSION = "1.0.1-ascii"


def number(value):
    """Accept either decimal separator in interactive numeric input."""
    return float(str(value).strip().replace(",", "."))


def nonnegative(value):
    x = number(value)
    if not np.isfinite(x) or x < 0:
        raise argparse.ArgumentTypeError("Wert muss endlich und >= 0 sein.")
    return x


def positive(value):
    x = nonnegative(value)
    if x == 0:
        raise argparse.ArgumentTypeError("Wert muss > 0 sein.")
    return x


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")[:100]


def save_figure(fig, path):
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def style_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def new_output(label, output_root=None):
    root = Path(output_root) if output_root else SCRIPT_DIR / "results" / "evaporation_analysis"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = root / f"{slug(label)}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def weighing_analysis(out):
    """Evaluate gross-mass differences without assigning them to a solvent."""
    times = ["10:00", "10:23", "11:01", "11:35", "12:11", "12:53",
             "13:38", "14:13", "14:57", "15:22", "15:50", "15:56"]
    gross = np.array([1635.85, 1634.13, 1632.53, 1630.86, 1629.42, 1627.86,
                      1626.22, 1625.17, 1623.74, 1622.96, 1621.84, 1621.47])
    minutes = np.array([60 * int(t[:2]) + int(t[3:]) for t in times], dtype=float)
    elapsed = (minutes - minutes[0]) / 60
    loss = gross[0] - gross
    dt = np.diff(elapsed)
    rates = -np.diff(gross) / dt
    overall = loss[-1] / elapsed[-1]
    after_first = (gross[1] - gross[-1]) / (elapsed[-1] - elapsed[1])
    slope, intercept = np.polyfit(elapsed, loss, 1)
    raw = pd.DataFrame({"Time_local": times, "Elapsed_h": elapsed,
                        "Gross_mass_g": gross, "Loss_from_first_g": loss})
    intervals = pd.DataFrame({"Start_local": times[:-1], "End_local": times[1:],
                             "Duration_min": np.diff(minutes), "Loss_g": -np.diff(gross),
                             "Loss_rate_g_h": rates})
    raw.to_csv(out / "weighing_measurements.csv", index=False)
    intervals.to_csv(out / "weighing_intervals.csv", index=False)
    summary = {"Nominal_ink_mass_g": 1211.70, "Duration_h": elapsed[-1],
               "Total_mass_loss_g": loss[-1], "Overall_rate_g_h": overall,
               "Rate_excluding_first_interval_g_h": after_first,
               "OLS_loss_slope_g_h": slope,
               "Interpretation": "Total mass loss, not independently measured IPA loss.",
               "Cautions": ["Weighing configuration changed in the first interval.",
                            "Stirrer speed changed; exact switch times are unknown.",
                            "Spills, retained liquid and configuration changes are not corrected.",
                            "Balance uncertainty was not provided; no uncertainty bars invented."]}
    write_json(out / "weighing_summary.json", summary)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 9.5), gridspec_kw={"height_ratios": [1, 1.15]})
    fig.subplots_adjust(top=.85, bottom=.245, hspace=.55, left=.09, right=.97)
    fig.suptitle("Verdunstungsversuch | gravimetrischer Masseverlust", x=.09,
                 ha="left", fontsize=19, fontweight="bold", y=.98)
    fig.text(.09, .925, f"14,38 g Verlust  |  5 h 56 min  |  Gesamtmittel: {overall:.3f} g/h".replace("2.424", "2,424"),
             fontsize=14, color=COLORS["mixed"])
    axes[0].plot(elapsed, loss, "o-", color=COLORS["mixed"], lw=2, label="Waegungen")
    axes[0].plot(elapsed, overall * elapsed, "--", color="#555C64", lw=1.3,
                 label=f"Verbindung Anfang-Ende: {overall:.3f} g/h")
    axes[0].axvspan(0, elapsed[1], color=COLORS["ipa"], alpha=.13)
    axes[0].set(ylabel="Kumulierter Masseverlust [g]", xlabel="Uhrzeit", ylim=(-.5, 16))
    ticks = list(range(10))+[11]
    axes[0].set_xticks(elapsed[ticks], [times[i] for i in ticks], rotation=35, ha="right")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].annotate("14,38 g", (elapsed[-1], loss[-1]), xytext=(-4, 10),
                     textcoords="offset points", ha="right", fontsize=11)
    positions = np.arange(len(rates))
    bar_colors = [COLORS["ipa"]] + [COLORS["mixed"]] * (len(rates)-1)
    bars = axes[1].bar(positions, rates, color=bar_colors, width=.70)
    bars[0].set_hatch("///")
    axes[1].axhline(overall, color="#333A42", ls="--", lw=1.5,
                    label=f"Gesamtmittel: {overall:.3f} g/h")
    for bar, rate in zip(bars, rates):
        axes[1].text(bar.get_x()+bar.get_width()/2, rate+.09,
                     f"{rate:.2f}".replace(".", ","), ha="center", fontsize=11)
    axes[1].set_xticks(positions, [f"{a}-\n{b}" for a, b in zip(times[:-1], times[1:])], fontsize=9)
    axes[1].set(ylabel="Mittlere Verlustrate je Intervall [g/h]", ylim=(0, 5.4),
                xlabel="Jeweils zwischen zwei Waegungen | Balkenbreiten sind nicht zeitproportional")
    axes[1].legend(frameon=False, loc="upper right")
    style_axes(axes)
    fig.text(.09, .035,
             "Erstes Intervall: Aufbau vor/nach Einsetzen am Ruehrer unterschiedlich - moeglicher Zusatzverlust.\n"
             "Drehzahl im Versuch reduziert: anfangs 800, spaeter ca. 700/500 und 480/450 1/min; Wechselzeiten unklar.\n"
             f"Ab 10:23 Uhr: {after_first:.3f} g/h. Die Waage trennt IPA und Wasser nicht; Waegeunsicherheit unbekannt.",
             fontsize=10, color="#505961", linespacing=1.5, va="bottom")
    save_figure(fig, out / "weighing_evaporation.png")
    print(intervals.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nGesamt: {loss[-1]:.2f} g / {elapsed[-1]:.6f} h = {overall:.6f} g/h")
    print(f"Ohne erstes Intervall: {after_first:.6f} g/h")
    return summary


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=convert,
                               allow_nan=False), encoding="utf-8")


def select_number(options, title, formatter=str):
    if not options:
        raise ValueError(f"Keine Auswahl verfuegbar: {title}")
    print(f"\n{title}")
    for i, option in enumerate(options, 1):
        print(f"  {i}: {formatter(option)}")
    while True:
        response = input("Nummer (q = Abbruch): ").strip()
        if response.lower() == "q":
            raise KeyboardInterrupt
        try:
            idx = int(response)
            if 1 <= idx <= len(options):
                return options[idx-1]
        except ValueError:
            pass
        print("Bitte eine der angezeigten Nummern eingeben.")


def is_measurement_csv(path):
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip() and not line.lstrip().startswith(("/", "#")):
                    return all(c in line for c in ["ProbeNr", "Rho_M", "C_M", "m_SL120"])
    except (OSError, UnicodeError):
        return False
    return False


def find_measurement_files(folder=None):
    folders = [Path(folder)] if folder else [SCRIPT_DIR / "measurement_data", SCRIPT_DIR,
                                            Path.cwd() / "laboratory_measurement_L-Com" / "measurement_data",
                                            Path.cwd() / "measurement_data", Path.cwd()]
    found = set()
    for root in folders:
        if root.is_dir():
            found.update(p.resolve() for p in root.glob("*.csv") if is_measurement_csv(p))
    return sorted(found, key=lambda p: (p.name.lower(), str(p)))


def read_measurements(path, max_gap_min=60):
    """Keep raw values and phase boundaries; never bridge a recipe change."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith(("/", "#")):
                sep = ";" if line.count(";") > line.count(",") else ","
                break
        else:
            raise ValueError("Leere Messdatei.")
    df = pd.read_csv(path, sep=sep, skipinitialspace=True, comment="/", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    required = ["Date", "UTC Time", "ProbeNr", "Rho_M", "C_M", "T_M"] + MASS_COLUMNS[:-1]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende CSV-Spalten: {sorted(missing)}")
    if "m_MG" not in df:
        df["m_MG"] = 0.0
    df["Source_record"] = np.arange(1, len(df)+1)
    for col in set(MASS_COLUMNS + ["ProbeNr", "Rho_M", "C_M", "T_M", "Rho_S", "C_S",
                                   "SensOK", "Stabil", "Gueltig", "N"]) & set(df.columns):
        df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Date"].astype(str).str.strip() + " " +
                                     df["UTC Time"].astype(str).str.strip(), format="mixed",
                                     dayfirst=False, utc=True, errors="coerce")
    if df["Timestamp"].isna().any() or df["ProbeNr"].isna().any():
        raise ValueError("Ungueltige Zeitstempel/Probennummern: vor einer Zeitauswertung korrigieren.")
    if df[MASS_COLUMNS].isna().any().any() or (df[MASS_COLUMNS] < 0).any().any():
        raise ValueError("Fehlende/negative Einwaagen. Fehlende MG-Spalte ist erlaubt, leere MG-Werte nicht.")
    pieces = []
    for _, sample in df.groupby("ProbeNr", sort=True):
        sample = sample.sort_values(["Timestamp", "Source_record"]).copy()
        change = (sample[MASS_COLUMNS].diff().abs() > 1e-5).any(axis=1)
        gap = sample["Timestamp"].diff().dt.total_seconds().div(60) > max_gap_min
        boundary = change | gap
        boundary.iloc[0] = True
        sample["Phase"] = boundary.cumsum().astype(int)
        pieces.append(sample)
    return pd.concat(pieces, ignore_index=True)


def phase_label(item):
    phase, frame = item
    first = frame.iloc[0]
    masses = ", ".join(f"{c[2:]}={first[c]:.2f} g" for c in MASS_COLUMNS)
    return (f"Komposition/Phase {phase} | {first['Timestamp'].strftime('%Y-%m-%d %H:%M')}-"
            f"{frame['Timestamp'].iloc[-1].strftime('%H:%M')} UTC | {len(frame)} Punkte | {masses}")


def prepare_window(phase, args):
    audit = phase.copy()
    audit["Phase_elapsed_min"] = (audit.Timestamp - audit.Timestamp.iloc[0]).dt.total_seconds()/60
    reasons = [[] for _ in range(len(audit))]
    def reject(mask, reason):
        for i in np.flatnonzero(np.asarray(mask)):
            reasons[i].append(reason)
    reject(audit.Phase_elapsed_min < args.start_min, "before selected window")
    if args.end_min is not None:
        reject(audit.Phase_elapsed_min > args.end_min, "after selected window")
    for c in ["Rho_M", "C_M", "T_M"]:
        reject(~np.isfinite(audit[c]), f"nonfinite {c}")
    reject(audit.Rho_M <= 0, "nonpositive density")
    reject(audit.C_M <= 0, "nonpositive sound velocity")
    if "SensOK" in audit:
        reject(audit.SensOK.ne(1), "SensOK != 1")
    if "N" in audit:
        reject(audit.N <= 0, "N <= 0")
    for c, limit in [("Rho_S", args.max_rho_sd), ("C_S", args.max_c_sd)]:
        if c in audit:
            reject(~np.isfinite(audit[c]) | (audit[c] < 0) | (audit[c] > limit), f"invalid/high {c}")
    if args.strict_flags:
        for c in ["Stabil", "Gueltig"]:
            if c in audit:
                reject(audit[c].ne(1), f"{c} != 1")
    reject(audit.Timestamp.duplicated(keep="first"), "duplicate timestamp")
    audit["Exclusion_reason"] = ["; ".join(r) for r in reasons]
    audit["Used"] = audit.Exclusion_reason.eq("")
    selected = audit[audit.Used].copy()
    if len(selected) < max(8, 4*args.segments+2):
        raise ValueError(f"Nur {len(selected)} brauchbare Punkte. Mindestens {max(8, 4*args.segments+2)} erforderlich; Zeitfenster/Filter/Segmentzahl pruefen.")
    selected["Elapsed_h"] = (selected.Timestamp-selected.Timestamp.iloc[0]).dt.total_seconds()/3600
    if selected.Elapsed_h.iloc[-1] <= 0:
        raise ValueError("Das ausgewaehlte Zeitfenster hat keine positive Dauer.")
    return selected, audit


def starting_masses(row, prior_ipa=0, prior_water=0):
    sl = float(row.m_SL120)
    mass = np.array([.2*sl, .4*sl + row.m_IPA - prior_ipa,
                     .4*sl + row.m_PG, row.m_MG, row.m_Wasser - prior_water], dtype=float)
    if np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError("Vorverluste ueberschreiten die eingewogene Stoffmasse oder Ansatzmasse ist null.")
    return mass


def load_calculator(calculator_path=None, tables=None):
    roots = [SCRIPT_DIR.parent, SCRIPT_DIR, Path.cwd()]
    candidates = [Path(calculator_path)] if calculator_path else [p / "ink_calculator.py" for p in roots]
    path = next((p.resolve() for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("ink_calculator.py nicht gefunden. --calculator PFAD verwenden.")
    spec = importlib.util.spec_from_file_location("_evap_ink_calculator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ["density", "sound_velocity"]:
        if "mg" not in inspect.signature(getattr(module.InkCalculator, name)).parameters:
            raise ValueError("MG-faehigen ink_calculator.py verwenden: Parameter mg fehlt.")
    candidates = [Path(tables)] if tables else [path.parent / "tables_parameters"] + [p / "tables_parameters" for p in roots]
    table_path = next((p.resolve() for p in candidates if p.is_dir()), None)
    if table_path is None:
        raise FileNotFoundError("tables_parameters nicht gefunden. --tables PFAD verwenden.")
    return module.InkCalculator(tables_dir=str(table_path)), path, table_path


@dataclass
class EvaporationModel:
    calculator: object
    masses0: np.ndarray
    time: np.ndarray
    temperature: np.ndarray
    segments: int = 1
    max_ipa_rate: float = 15.0
    max_water_rate: float = 30.0

    def __post_init__(self):
        self.time = np.asarray(self.time, float)
        self.temperature = np.asarray(self.temperature, float)
        self.masses0 = np.asarray(self.masses0, float)
        if self.time[-1] <= 0 or self.segments < 1:
            raise ValueError("Positive duration and segment count required.")
        self.edges = np.linspace(0, self.time[-1], self.segments+1)
        # Each row integrates a continuous piecewise-constant rate history.
        self.exposure = np.clip(self.time[:, None]-self.edges[:-1], 0, np.diff(self.edges))
        # Conservative per-segment bounds guarantee positive solvent stocks.
        self.ipa_cap = min(self.max_ipa_rate, .98*self.masses0[1]/self.time[-1])
        self.water_cap = min(self.max_water_rate, .98*self.masses0[4]/self.time[-1])

    def active(self, mode):
        return [(species, j) for species in (["IPA"] if mode == "ipa" else ["IPA", "Water"] if mode == "mixed" else [])
                for j in range(self.segments)
                if (self.ipa_cap if species == "IPA" else self.water_cap) > 0]

    def bounds(self, mode):
        return np.array([self.ipa_cap if s == "IPA" else self.water_cap for s, _ in self.active(mode)])

    def rates(self, params, mode):
        rates = np.zeros((2, self.segments))
        for val, (species, j) in zip(params, self.active(mode)):
            rates[0 if species == "IPA" else 1, j] = val
        return rates

    def states(self, params, mode):
        losses = self.exposure @ self.rates(params, mode).T
        masses = np.tile(self.masses0, (len(self.time), 1))
        masses[:, 1] -= losses[:, 0]
        masses[:, 4] -= losses[:, 1]
        if np.any(masses < -1e-9) or np.any(masses.sum(axis=1) <= 0):
            raise ValueError("Nonphysical remaining mass.")
        pct = 100*masses/masses.sum(axis=1)[:, None]
        return losses, masses, pct

    def predict(self, params, mode):
        _, _, pct = self.states(params, mode)
        result = np.empty((len(self.time), 2))
        for i, (w, t) in enumerate(zip(pct, self.temperature)):
            kw = dict(al=w[0], ipa=w[1], pg=w[2], mg=w[3], temperature=float(t))
            result[i] = (1000*self.calculator.density(**kw), self.calculator.sound_velocity(**kw))
        if not np.isfinite(result).all():
            raise ValueError("Calculator returned nonfinite predictions.")
        return result


def fit_model(model, observed, sigma, mode, start=None):
    """Profile constant channel offsets analytically in weighted least squares."""
    observed = np.asarray(observed, float)
    sigma = np.asarray(sigma, float)
    bounds = model.bounds(mode)
    def residual(params):
        diff = model.predict(params, mode)-observed
        return ((diff-diff.mean(axis=0))/sigma).ravel()
    def jacobian_at(params):
        # Absolute steps remain resolvable when a fitted rate approaches zero.
        # Relative steps alone can round to zero in density/sound predictions.
        base = residual(params)
        jac = np.empty((base.size, len(params)))
        for j in range(len(params)):
            h = min(1e-3, bounds[j]/10)
            if params[j]+h > bounds[j]:
                h = -h
            shifted = params.copy()
            shifted[j] += h
            jac[:, j] = (residual(shifted)-base)/h
        return jac
    if len(bounds):
        guess = (np.minimum(bounds*.1, .5) if start is None else
                 np.minimum(np.maximum(start, bounds*1e-4), bounds*(1-1e-4)))
        opt = least_squares(residual, guess, bounds=(np.zeros(len(bounds)), bounds),
                            jac=jacobian_at, ftol=1e-9, xtol=1e-9, gtol=1e-7, max_nfev=180)
        if not opt.success:
            raise RuntimeError(f"Fit did not converge ({mode}): {opt.message}")
        params = opt.x
        jacobian = opt.jac
    else:
        params = np.empty(0)
        jacobian = np.empty((observed.size, 0))
    physics = model.predict(params, mode)
    offset = (observed-physics).mean(axis=0)
    predicted = physics+offset
    errors = observed-predicted
    rates = model.rates(params, mode)
    duration = model.time[-1]
    average = rates @ np.diff(model.edges)/duration
    total = average.sum()
    norms = np.linalg.norm(jacobian, axis=0)
    if len(norms) >= 2 and np.all(norms > 1e-12):
        normalized = jacobian/norms
        sv = np.linalg.svd(normalized, compute_uv=False)
        condition = float(sv[0]/sv[-1]) if sv[-1] > 1e-14 else None
        correlation = normalized.T @ normalized
    else:
        condition = None
        correlation = None
    boundary = [f"{s} segment {j+1}" for x, upper, (s, j) in zip(params, bounds, model.active(mode))
                if x < max(1e-5, upper*1e-4) or upper-x < max(1e-5, upper*1e-4)]
    result = {"mode": mode, "params": params, "rates": rates, "average_rates_g_h": average,
              "total_rate_g_h": total, "ipa_mass_fraction_in_loss": float(average[0]/total) if total > 1e-6 else None,
              "offset": offset, "physics": physics, "predicted": predicted, "errors": errors,
              "rmse": np.sqrt(np.mean(errors**2, axis=0)),
              "weighted_sse": float(np.sum((errors/sigma)**2)),
              "normalized_jacobian_condition": condition, "sensitivity_correlations": correlation,
              "boundary_parameters": boundary, "bootstrap": None}
    return result


def bootstrap_fit(model, observed, sigma, result, repetitions=60, seed=184, block_length=None):
    """Paired moving-block residual bootstrap, conditional on the model/start."""
    if repetitions <= 0 or not len(result["params"]):
        return None
    n = len(observed)
    block = block_length or max(2, int(round(n**(1/3))))
    if not 1 <= block <= n:
        raise ValueError("Bootstrap-Blocklaenge muss zwischen 1 und Punktzahl liegen.")
    rng = np.random.default_rng(seed)
    averages, shares, parameter_sets = [], [], []
    failed = 0
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n/block)))
        idx = np.concatenate([(s+np.arange(block)) % n for s in starts])[:n]
        synthetic = result["predicted"]+result["errors"][idx]
        try:
            fitted = fit_model(model, synthetic, sigma, result["mode"], start=result["params"])
        except (ValueError, RuntimeError):
            failed += 1
            continue
        averages.append(fitted["average_rates_g_h"])
        parameter_sets.append(fitted["params"])
        if fitted["ipa_mass_fraction_in_loss"] is not None:
            shares.append(fitted["ipa_mass_fraction_in_loss"])
    if len(averages) < max(10, repetitions//2):
        return {"successful": len(averages), "failed": failed, "warning": "Too few successful bootstrap fits."}
    return {"successful": len(averages), "failed": failed, "block_length_points": block,
            "average_rates_ci95_g_h": np.quantile(averages, [.025, .975], axis=0),
            "ipa_share_ci95": np.quantile(shares, [.025, .975]) if len(shares) >= 10 else None,
            "parameters_ci95_g_h": np.quantile(parameter_sets, [.025, .975], axis=0),
            "interpretation": "Conditional residual-bootstrap range; not a complete metrological uncertainty."
            " Does not include starting-composition, model, temperature-calibration or weighing uncertainty."}


def diagnostics(model, results, selected, args):
    notices = ["Model-based estimates, not a direct measurement of vapour composition.",
               "Only losses after the first retained timestamp are fitted.",
               "Nominal additions minus entered prior losses define the starting composition.",
               "Al, PG and MG are assumed nonvolatile; no withdrawals, spills or unrecorded additions.",
               "A constant offset per sensor is fitted. Time-/composition-/temperature-dependent errors remain confounded.",
               "Bootstrap ranges are conditional on this calculator and the assumed initial composition.",
               "Stabil/Gueltig are not default filters because real evaporation itself can create a trend."]
    mixed = results["mixed"]
    condition = mixed["normalized_jacobian_condition"]
    if model.time[-1] < 1:
        notices.append("Window shorter than one hour: small drifts and settling can dominate evaporation estimates.")
    if condition is None or condition > 30:
        notices.append("WEAK SEPARATION: normalized sensitivity condition is >30 or undefined (heuristic threshold).")
    if mixed["boundary_parameters"]:
        notices.append("BOUNDARY SOLUTION: at least one mixed-model rate reaches zero/a limit; solvent split is not established.")
    if args.prior_ipa_loss is None or args.prior_water_loss is None:
        notices.append("Unknown prior evaporation: zero prior losses were assumed for at least one solvent.")
    if mixed["total_rate_g_h"] < .01:
        notices.append("Negligible fitted total loss: IPA/water ratio is not meaningful.")
    b = mixed["bootstrap"]
    if b and b.get("ipa_share_ci95") is not None and np.diff(b["ipa_share_ci95"])[0] > .5:
        notices.append("WIDE RATIO RANGE: conditional 95% bootstrap range spans >50 percentage points.")
    if len(selected) < 20:
        notices.append("Few timestamps: bootstrap uncertainty and residual diagnostics have limited reliability.")
    if args.segments > 1:
        notices.append("Piecewise rates use equal-duration bins; bin boundaries are user modelling choices, not detected physical events.")
    notices.append("A better two-solvent fit alone is not proof that water evaporated: the model has extra free parameters.")
    return notices


def calculator_support_warnings(model, results):
    """Retrieve diagnostic objects that scalar calculator methods omit."""
    sound = getattr(model.calculator, "sound_calc", None)
    if sound is None or not hasattr(sound, "calculate"):
        return ["Calculator does not expose detailed sound-table support diagnostics."]
    issues = set()
    for fit in results.values():
        _, _, pct = model.states(fit["params"], fit["mode"])
        for w, t in zip(pct, model.temperature):
            state = sound.calculate(w[0], w[1], w[2], float(t), pct_mg=w[3])
            issues.update(getattr(state, "warnings", []))
    return sorted(issues)


def plot_sensor_history(selected, model, results, out, title):
    minutes = model.time*60
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(top=.88, bottom=.10, hspace=.13, left=.12, right=.96)
    fig.suptitle(f"Messverlauf und Modellvergleich\n{title}", fontsize=15)
    for j, (col, label) in enumerate([("Rho_M", "Dichte [kg/m^3]"), ("C_M", "Schallgeschwindigkeit [m/s]")]):
        axes[j].plot(minutes, selected[col], "o", color="#222C38", ms=4, label="Messung")
        for mode in ["none", "ipa", "mixed"]:
            axes[j].plot(minutes, results[mode]["predicted"][:, j], color=COLORS[mode],
                         ls=":" if mode == "none" else "--" if mode == "ipa" else "-", lw=2, label=LABELS[mode])
        axes[j].set_ylabel(label)
    axes[0].legend(ncol=2, fontsize=9, frameon=False)
    axes[2].plot(minutes, selected.T_M, "o-", color="#6452A0", ms=4)
    axes[2].set(ylabel="Temperatur [degC]", xlabel="Zeit seit erstem verwendeten Messpunkt [min]")
    style_axes(axes)
    fig.text(.12, .025, "Alle Modelle bei gemessener Temperatur; je Messkanal konstanter Offset angepasst.\n"
             "Modellkurven sind Fits an diesen Daten, keine unabhaengigen Vorhersagen.", fontsize=10)
    save_figure(fig, out / "sensor_history.png")


def plot_losses(model, results, out, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=.87, bottom=.13, hspace=.34, wspace=.28)
    fig.suptitle(f"Modellbasierte Verdunstung | kein unabhaengiger Stoffnachweis\n{title}", fontsize=15)
    t = model.time*60
    for col, mode in enumerate(["ipa", "mixed"]):
        res = results[mode]
        losses, _, _ = model.states(res["params"], mode)
        axes[0, col].plot(t, losses[:, 0], label="IPA", color=COLORS["ipa"], lw=2)
        axes[0, col].plot(t, losses[:, 1], label="Wasser", color="#387DB5", lw=2)
        axes[0, col].plot(t, losses.sum(axis=1), label="Gesamt", color="#202C38", ls="--")
        axes[0, col].set(title=LABELS[mode], ylabel="Kumulierter Verlust [g]")
        axes[0, col].legend(frameon=False)
        for k, (name, color) in enumerate([("IPA", COLORS["ipa"]), ("Wasser", "#387DB5")]):
            axes[1, col].stairs(res["rates"][k], model.edges*60, baseline=None, label=name, color=color, lw=2)
        avg = res["average_rates_g_h"]
        axes[1, col].set(ylabel="Angenommene / gefittete Rate [g/h]", xlabel="Zeit [min]")
        axes[1, col].text(.03, .97, f"Zeitmittel: IPA {avg[0]:.3f} g/h\nWasser {avg[1]:.3f} g/h",
                          transform=axes[1, col].transAxes, va="top", fontsize=10,
                          bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
        axes[1, col].set_ylim(0, max(.1, res["rates"].max()*1.45))
    style_axes(axes)
    split = results["mixed"]["ipa_mass_fraction_in_loss"]
    ratio = f"Geschaetzter IPA-Massenanteil am Gesamtverlust: {100*split:.1f} %" if split is not None else "IPA/Wasser-Verhaeltnis nicht definiert (kein Gesamtverlust)."
    if results["mixed"]["boundary_parameters"]:
        ratio += " | Randloesung: Aufteilung nicht gesichert."
    fig.text(.1, .045, ratio+"\nRaten innerhalb jedes Zeitsegments konstant; Zusammensetzungen und Verluste sind rekonstruierte Modellzustaende.", fontsize=10)
    save_figure(fig, out / "evaporation_losses_and_rates.png")


def plot_compositions(model, results, out, title):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    fig.subplots_adjust(top=.9, bottom=.12, hspace=.28, wspace=.28)
    fig.suptitle(f"Zeitlicher Verlauf der Zusammensetzung (Modellrekonstruktion)\n{title}", fontsize=15)
    for mode in ["ipa", "mixed"]:
        _, masses, pct = model.states(results[mode]["params"], mode)
        for j, name in enumerate(COMPONENTS):
            axes.flat[j].plot(model.time*60, pct[:, j], color=COLORS[mode],
                              ls="--" if mode == "ipa" else "-", label=LABELS[mode], lw=2)
            axes.flat[j].set_ylabel(f"{name} [Massen-%]")
        axes.flat[5].plot(model.time*60, masses.sum(axis=1), color=COLORS[mode],
                         ls="--" if mode == "ipa" else "-", lw=2, label=LABELS[mode])
    axes.flat[5].set_ylabel("Verbleibende Tintenmasse [g]")
    axes[0, 0].legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Zeit [min]")
    style_axes(axes)
    fig.text(.1, .035, "Startzusammensetzung: Einwaagen abzueglich eingegebener Vorverluste. Keine automatische Rekonstruktion der Zeit davor.\n"
             "Al-, PG- und MG-Massen bleiben konstant; ihre Massenanteile steigen bei Loesungsmittelverlust.", fontsize=10)
    save_figure(fig, out / "composition_history.png")


def plot_diagnostics(selected, model, results, sigma, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(top=.87, bottom=.13, hspace=.38, wspace=.33)
    fig.suptitle("Fitdiagnostik | Restfehler und Empfindlichkeit", fontsize=16)
    for j, unit in enumerate(["kg/m^3", "m/s"]):
        for mode in ["none", "ipa", "mixed"]:
            axes[0, j].plot(model.time*60, results[mode]["errors"][:, j], "o-", ms=3,
                            color=COLORS[mode], label=LABELS[mode])
        axes[0, j].axhline(0, color="black", lw=.7)
        axes[0, j].set(xlabel="Zeit [min]", ylabel=f"Messung - Fit [{unit}]")
    axes[0, 0].legend(frameon=False, fontsize=9)
    mixed = results["mixed"]
    corr = mixed["sensitivity_correlations"]
    labels = [f"{s} {j+1}" for s, j in model.active("mixed")]
    if corr is not None:
        im = axes[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        axes[1, 0].set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        axes[1, 0].set_yticks(range(len(labels)), labels, fontsize=8)
        fig.colorbar(im, ax=axes[1, 0], fraction=.05, label="Normiertes Skalarprodukt")
        if len(labels) <= 6:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axes[1, 0].text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                                    color="white" if abs(corr[i,j]) > .6 else "black", fontsize=9)
    else:
        axes[1, 0].text(.5, .5, "Empfindlichkeit nicht bestimmbar", ha="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Aehnliche Empfindlichkeiten erschweren Trennung", fontsize=10)
    axes[1, 1].axis("off")
    c = mixed["normalized_jacobian_condition"]
    text = f"Normierte Jacobi-Konditionszahl: {c:.2f}" if c is not None else "Jacobi-Konditionszahl: nicht bestimmbar"
    text += f"\nGewichtung: sigma_rho={sigma[0]:.4g} kg/m^3\n             sigma_c={sigma[1]:.4g} m/s"
    text += "\n\nIPA + Wasser, Mittelraten:"
    b = mixed["bootstrap"]
    for k, name in enumerate(["IPA", "Wasser"]):
        text += f"\n{name}: {mixed['average_rates_g_h'][k]:.3f} g/h"
        if b and "average_rates_ci95_g_h" in b:
            ci = b["average_rates_ci95_g_h"][:, k]
            text += f"  [{ci[0]:.3f}; {ci[1]:.3f}]"
    text += ("\n\nKlammern: bedingter 95%-Bootstrapbereich.\nKeine vollstaendige Messunsicherheit."
             if b and "average_rates_ci95_g_h" in b else "\n\nKein Bootstrapbereich berechnet.")
    if mixed["boundary_parameters"]:
        text += "\nRANDLOeSUNG: Aufteilung nicht gesichert."
    axes[1, 1].text(0, 1, text, va="top", transform=axes[1, 1].transAxes, fontsize=10, linespacing=1.5)
    style_axes(axes[0])
    fig.text(.1, .025, "Modell-, Startmassen- und Temperaturkalibrierfehler sind im Bootstrap nicht enthalten.\n"
             "Eine unabhaengige Waegung im selben Zeitfenster ist die wichtigste zusaetzliche Pruefung.", fontsize=10)
    save_figure(fig, out / "fit_diagnostics.png")


def export_results(selected, audit, model, results, sigma, notices, args, out, paths):
    audit.to_csv(out / "measurement_selection.csv", index=False)
    records = selected.copy()
    for mode, fit in results.items():
        losses, masses, pct = model.states(fit["params"], mode)
        for k, name in enumerate(["IPA", "Water"]):
            records[f"{mode}_Loss_{name}_g"] = losses[:, k]
        for k, name in enumerate(COMPONENTS):
            records[f"{mode}_{name}_mass_g"] = masses[:, k]
            records[f"{mode}_{name}_wt_pct"] = pct[:, k]
        for k, name in enumerate(["Rho", "C"]):
            records[f"{mode}_{name}_physics"] = fit["physics"][:, k]
            records[f"{mode}_{name}_fit_with_offset"] = fit["predicted"][:, k]
            records[f"{mode}_{name}_residual"] = fit["errors"][:, k]
    records.to_csv(out / "evaporation_time_series.csv", index=False)
    summary, segments = [], []
    for mode, fit in results.items():
        entry = {"Model": mode, "IPA_rate_g_h": fit["average_rates_g_h"][0],
                 "Water_rate_g_h": fit["average_rates_g_h"][1], "Total_rate_g_h": fit["total_rate_g_h"],
                 "IPA_mass_fraction_in_loss": fit["ipa_mass_fraction_in_loss"],
                 "IPA_loss_g": fit["average_rates_g_h"][0]*model.time[-1],
                 "Water_loss_g": fit["average_rates_g_h"][1]*model.time[-1],
                 "RMSE_density_kg_m3": fit["rmse"][0], "RMSE_sound_m_s": fit["rmse"][1],
                 "Weighted_SSE": fit["weighted_sse"], "Fitted_parameters_including_offsets": len(fit["params"])+2,
                 "Normalized_Jacobian_condition": fit["normalized_jacobian_condition"],
                 "Boundary_parameters": "; ".join(fit["boundary_parameters"])}
        b = fit["bootstrap"]
        if b and "average_rates_ci95_g_h" in b:
            for k, name in enumerate(["IPA", "Water"]):
                entry[f"{name}_rate_conditional_CI95_low_g_h"] = b["average_rates_ci95_g_h"][0, k]
                entry[f"{name}_rate_conditional_CI95_high_g_h"] = b["average_rates_ci95_g_h"][1, k]
        summary.append(entry)
        for j in range(model.segments):
            segments.append({"Model": mode, "Segment": j+1, "Start_min": model.edges[j]*60,
                             "End_min": model.edges[j+1]*60, "IPA_rate_g_h": fit["rates"][0,j],
                             "Water_rate_g_h": fit["rates"][1,j]})
    pd.DataFrame(summary).to_csv(out / "evaporation_summary.csv", index=False)
    pd.DataFrame(segments).to_csv(out / "evaporation_rates_by_segment.csv", index=False)
    metadata = {"script_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(paths[0]), "calculator": str(paths[1]), "tables": str(paths[2]),
                "source_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                "calculator_sha256": hashlib.sha256(paths[1].read_bytes()).hexdigest(),
                "tables_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths[2].glob("*.csv"))},
                "arguments": vars(args), "sample": float(selected.ProbeNr.iloc[0]), "phase": int(selected.Phase.iloc[0]),
                "start_utc": selected.Timestamp.iloc[0].isoformat(), "end_utc": selected.Timestamp.iloc[-1].isoformat(),
                "start_masses_g": dict(zip(COMPONENTS, model.masses0)), "noise_scales": sigma,
                "notices": notices,
                "models": {mode: {k:v for k,v in fit.items() if k not in ["physics", "predicted", "errors"]} for mode, fit in results.items()}}
    write_json(out / "analysis_metadata.json", metadata)
    lines = ["# Evaporation analysis", "", f"Source: {paths[0].name}",
             f"Sample: {selected.ProbeNr.iloc[0]:g}; phase: {selected.Phase.iloc[0]}; duration: {model.time[-1]*60:.2f} min", "",
             "| Model | IPA [g/h] | Water [g/h] | Total [g/h] | Density RMSE [kg/m^3] | Sound RMSE [m/s] |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['Model']} | {row['IPA_rate_g_h']:.4f} | {row['Water_rate_g_h']:.4f} | {row['Total_rate_g_h']:.4f} | {row['RMSE_density_kg_m3']:.4f} | {row['RMSE_sound_m_s']:.4f} |")
    lines += ["", "## Interpretation and limitations", ""]+[f"- {s}" for s in notices]
    lines += ["", "The solvent ratio is a MASS ratio of the integrated inferred losses, not a volume or molar ratio.",
              "All losses start at the first retained timestamp. Do not extrapolate them back before that point.",
              "RMSE values refer to an in-sample fit with fitted channel offsets, not external validation.", "",
              "![Sensor history](sensor_history.png)", "![Losses](evaporation_losses_and_rates.png)",
              "![Composition](composition_history.png)", "![Diagnostics](fit_diagnostics.png)"]
    (out / "analysis_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(pd.DataFrame(summary)[["Model", "IPA_rate_g_h", "Water_rate_g_h", "Total_rate_g_h", "RMSE_density_kg_m3", "RMSE_sound_m_s"]].to_string(index=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weighing", action="store_true", help="Nur das mitgelieferte Wiegeprotokoll auswerten.")
    p.add_argument("--csv", type=Path, help="Messdatei; ohne Angabe nummerierte Dateiauswahl.")
    p.add_argument("--input-dir", type=Path, help="Ordner fuer die Dateiauswahl (nicht rekursiv).")
    p.add_argument("--sample", type=number, help="Tatsaechliche ProbeNr; ohne Angabe nummerierte Auswahl.")
    p.add_argument("--phase", type=int, help="Phasennummer innerhalb der Probe; ohne Angabe nummerierte Auswahl.")
    p.add_argument("--calculator", type=Path, help="Pfad zum MG-faehigen ink_calculator.py.")
    p.add_argument("--tables", type=Path, help="Pfad zu tables_parameters.")
    p.add_argument("--output-root", type=Path, help="Optionaler CLI-Pfad; sonst automatisch results/evaporation_analysis.")
    p.add_argument("--start-min", type=nonnegative, default=0, help="Fensterbeginn nach erstem Phasen-Zeitstempel.")
    p.add_argument("--end-min", type=positive, help="Fensterende nach erstem Phasen-Zeitstempel.")
    p.add_argument("--prior-ipa-loss", type=nonnegative, help="Bekannter kumulierter IPA-Vorverlust [g] bis zum ersten verwendeten Punkt.")
    p.add_argument("--prior-water-loss", type=nonnegative, help="Bekannter kumulierter Wasser-Vorverlust [g] bis zum ersten verwendeten Punkt.")
    p.add_argument("--segments", type=int, default=1, help="Anzahl gleich langer Abschnitte mit je konstanten Raten; Standard 1.")
    p.add_argument("--max-gap-min", type=positive, default=60, help="Groessere Aufzeichnungsluecken beginnen neue Phasen.")
    p.add_argument("--max-ipa-rate", type=positive, default=15, help="Obergrenze IPA [g/h]; zusaetzlich durch Vorrat beschraenkt.")
    p.add_argument("--max-water-rate", type=positive, default=30, help="Obergrenze Wasser [g/h]; zusaetzlich durch Vorrat beschraenkt.")
    p.add_argument("--max-rho-sd", type=positive, default=.05, help="Punkte mit Rho_S oberhalb dieser Grenze [kg/m^3] ausschliessen.")
    p.add_argument("--max-c-sd", type=positive, default=.20, help="Punkte mit C_S oberhalb dieser Grenze [m/s] ausschliessen.")
    p.add_argument("--sigma-rho", type=positive, help="Dichte-Gewichtungsskala [kg/m^3], sonst max(0.01, Median Rho_S).")
    p.add_argument("--sigma-c", type=positive, help="Schall-Gewichtungsskala [m/s], sonst max(0.05, Median C_S).")
    p.add_argument("--strict-flags", action="store_true", help="Zusaetzlich Stabil=Gueltig=1 verlangen; kann echte Verdunstungstrends selektieren.")
    p.add_argument("--bootstrap", type=int, default=60, help="Blockbootstrap-Wiederholungen (0=aus, >=200 fuer finale Auswertung empfohlen).")
    p.add_argument("--block-length", type=int, help="Bootstrap-Blocklaenge in aufeinanderfolgenden Messpunkten.")
    p.add_argument("--seed", type=int, default=184)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.segments < 1 or args.bootstrap < 0 or (args.end_min is not None and args.end_min <= args.start_min):
        raise ValueError("Segmentzahl/Bootstrapzahl/Zeitfenster ungueltig.")
    if args.weighing:
        out = new_output("weighing", args.output_root)
        weighing_analysis(out)
        print(f"\nErgebnisse: {out}")
        return out
    interactive = args.csv is None or args.sample is None or args.phase is None
    source = args.csv.resolve() if args.csv else select_number(find_measurement_files(args.input_dir), "Messdatei waehlen", lambda p: str(p))
    df = read_measurements(source, args.max_gap_min)
    samples = [x for x, g in df.groupby("ProbeNr") if x not in [100, 101] and (g.m_SL120 > 0).any()]
    sample = args.sample if args.sample is not None else select_number(samples, "Probe waehlen (Referenzen 100/101 ausgeschlossen)", lambda v: f"Probe {v:g}")
    if sample not in samples:
        raise ValueError("Probe nicht vorhanden oder keine Tintenprobe.")
    phases = list(df[df.ProbeNr.eq(sample)].groupby("Phase", sort=True))
    phase = next((g for n, g in phases if n == args.phase), None) if args.phase else select_number(phases, "Unveraenderte Komposition / zusammenhaengende Phase waehlen", phase_label)[1]
    if phase is None:
        raise ValueError("Phasennummer nicht vorhanden.")
    selected, audit = prepare_window(phase, args)
    print(f"\nVerwendet: {len(selected)}/{len(phase)} Punkte, {selected.Elapsed_h.iloc[-1]*60:.1f} min.")
    print(f"Start fuer die Massenbilanz: {selected.Timestamp.iloc[0]} (erster verwendeter Punkt)")
    print("Die Raten beziehen sich nur auf dieses ausgewaehlte Zeitfenster.")
    if selected.Elapsed_h.iloc[-1] < 1:
        print("KURZES ZEITFENSTER: weniger als eine Stunde, keine mehrstuendige Langzeitauswertung.")
    if interactive:
        for name in ["ipa", "water"]:
            attr = f"prior_{name}_loss"
            if getattr(args, attr) is None:
                entry = input(f"Bekannter {name.upper()}-Vorverlust bis zu diesem Zeitpunkt [g] (Enter = unbekannt, rechnerisch 0): ").strip()
                if entry:
                    setattr(args, attr, nonnegative(entry))
    masses = starting_masses(selected.iloc[0], args.prior_ipa_loss or 0, args.prior_water_loss or 0)
    calc, calculator_path, tables_path = load_calculator(args.calculator, args.tables)
    model = EvaporationModel(calc, masses, selected.Elapsed_h.to_numpy(), selected.T_M.to_numpy(),
                             args.segments, args.max_ipa_rate, args.max_water_rate)
    if args.segments > 1:
        bins = np.minimum(np.searchsorted(model.edges, model.time, side="right")-1, args.segments-1)
        if np.any(np.bincount(bins, minlength=args.segments) < 4):
            raise ValueError("Mindestens vier verwendete Zeitpunkte je Ratensegment erforderlich.")
    def scale(column, floor):
        med = selected[column].median() if column in selected else floor
        return max(floor, med) if np.isfinite(med) else floor
    sigma = np.array([args.sigma_rho or scale("Rho_S", .01), args.sigma_c or scale("C_S", .05)])
    observed = selected[["Rho_M", "C_M"]].to_numpy()
    results = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        for mode in ["none", "ipa", "mixed"]:
            print(f"Fit: {LABELS[mode]} ...", flush=True)
            results[mode] = fit_model(model, observed, sigma, mode)
            if mode != "none" and args.bootstrap:
                print(f"  {args.bootstrap} Blockbootstrap-Wiederholungen ...", flush=True)
                results[mode]["bootstrap"] = bootstrap_fit(model, observed, sigma, results[mode], args.bootstrap,
                                                            args.seed, args.block_length)
    notices = diagnostics(model, results, selected, args)
    notices += sorted(set(f"Calculator warning: {w.message}" for w in caught))
    notices += [f"Calculator support: {w}" for w in calculator_support_warnings(model, results)]
    out = new_output(f"{source.stem}_probe_{sample:g}_phase_{int(selected.Phase.iloc[0])}", args.output_root)
    title = f"{source.name} | Probe {sample:g} | Phase {int(selected.Phase.iloc[0])} | {model.time[-1]*60:.1f} min"
    plot_sensor_history(selected, model, results, out, title)
    plot_losses(model, results, out, title)
    plot_compositions(model, results, out, title)
    plot_diagnostics(selected, model, results, sigma, out)
    export_results(selected, audit, model, results, sigma, notices, args, out, (source, calculator_path, tables_path))
    print("\nWichtig: IPA/Wasser-Aufteilung ist modellabhaengig. Siehe analysis_report.md und fit_diagnostics.png.")
    print(f"Ergebnisse: {out}")
    return out


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        print(f"\nFehler: {error}", file=sys.stderr)
        sys.exit(2)
