#!/usr/bin/env python3
"""Create six plain Matplotlib diagrams for Probe 305.

Place this file in sim_par_ink_prop, next to ink_calculator.py, then run:
    python plot_probe305.py
    python plot_probe305.py --no-show

Dependencies: numpy, pandas, scipy, matplotlib; tzdata on Windows.
The four measured window summaries and the manual protocol values are embedded
below, so the original measurement CSV is not required. Optionally recalculate
the sensor summaries with --csv PATH. This does not change the manual weights
or dry-residue measurements. All times are local Europe/Berlin, 02 October 2026.

The composition inversion preserves Pigment:PG:MG = 1:2:0.125. P1 is anchored
to the known initial recipe. CONSTANT measurement-minus-model offsets are then
subtracted from every later measurement. Formal inverse solutions can violate
the evaporation-only assumption; these violations are reported, not removed.
Plot 6 independently inverts all four points with the calibration field,
without anchoring P1 to the recipe or subtracting an additional sensor offset.
The resulting compositions are pointwise fits, not a validated evaporation path.

Temperature correction, separately for density and sound velocity:
    y_corrected = y_measured + h(composition, T_reference) - h(composition, T_i)
T_reference is the unweighted mean of the FOUR point temperatures. Composition
is held at each point's InkCalculator inverse solution. The correction leaves
the constant offset in the measured signal and does not correct it twice.

Reference: GitHub niklasreeker/sim_par_ink_prop, commit b1c7bb792922992542dfbfe0f14c3e393bd15e65.
CSV: Messdaten_2026-10-03_12-42-45.csv; Notion protocol Probe 305.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


DATE = "2026-10-02"
FIELD_RELATIVE = Path("laboratory_measurement_L-Com/results/calibration_field/") / (
    "probe_9_10_11__Kennfeld_v2_23__korrigiert__20260928_110809/calibration_field.json"
)

# Sensor means and sample standard deviations (ddof=1), not total uncertainty.
# 'centre' is the midpoint between the first and last selected valid row.
POINTS = [
    dict(name="P1", sample="10:58", tank_time="10:58", tank_g=9005., samples_g=0.,
         start="11:00", end="11:05", centre="11:02:30.751", n=5,
         rho=1006.5914, rho_sd=.022700220263261406,
         sound=1547.223, sound_sd=.11773274820540433, temperature=24.131898,
         dry=[1.60, 1.77, 1.75]),
    dict(name="P2", sample="11:58", tank_time="12:15", tank_g=7915., samples_g=95.,
         start="12:12", end="12:17", centre="12:14:30.897", n=5,
         rho=1006.4272, rho_sd=.08560198595826493,
         sound=1540.954, sound_sd=.21346076922940685, temperature=22.964068,
         dry=[2.02, 2.11, 2.05]),
    dict(name="P3", sample="13:27", tank_time="13:37", tank_g=7470., samples_g=155.,
         start="13:37", end="13:42", centre="13:40:01.071", n=4,
         rho=1008.44425, rho_sd=.06412682745937838,
         sound=1538.25325, sound_sd=.17187277271281182, temperature=23.7224575,
         dry=[2.14, 2.16]),
    dict(name="P4", sample="15:54", tank_time="15:54", tank_g=6255., samples_g=285.,
         start="16:10", end="16:17", centre="16:13:38.779", n=7,
         rho=1012.8408571428571, rho_sd=.07199404737296881,
         sound=1534.2894285714285, sound_sd=.4614220256059323, temperature=24.269505714285714,
         dry=[2.39, 2.34, 2.45]),
]
STANDARD = np.array([180., 360., 360., 22.5, 9000.]) / 9922.5 * 100.
COMPONENTS = ["Pigment/Al", "IPA", "PG", "MG", "Wasser"]


def composition(pigment: float, ipa: float) -> np.ndarray:
    """Mass percentages; 'al' in InkCalculator denotes encapsulated pigment."""
    pg, mg = 2. * pigment, pigment / 8.
    return np.array([pigment, ipa, pg, mg, 100. - pigment - ipa - pg - mg])


def find_repo(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [
        *Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "ink_calculator.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("ink_calculator.py nicht gefunden. Skript in den Repository-Ordner legen oder --repo-root PFAD angeben.")


def load_calculator(repo: Path):
    # Use the actual local repository implementation, not copied formulas.
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("probe305_ink_calculator", repo / "ink_calculator.py")
    if spec is None or spec.loader is None:
        raise ImportError("Ink Calculator kann nicht geladen werden.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.InkCalculator(tables_dir=str(repo / "tables_parameters"))


class CalibrationField:
    """Saved field's scaled IDW, including its per-property quality weights.

    Implements idw_residual() from residual_calibration_field.py without
    importing that script's unrelated CLI/data-file dependencies.
    """
    def __init__(self, path: Path, repo: Path):
        self.model = json.loads(path.read_text(encoding="utf-8"))
        if self.model.get("schema") != "ink-residual-calibration-field":
            raise ValueError("Unbekanntes Kalibrierfeldformat.")
        provenance = self.model.get("calculator", {})
        hashes = {repo / "ink_calculator.py": provenance.get("sha256")}
        hashes.update({repo / "tables_parameters" / name: value
                       for name, value in provenance.get("table_sha256", {}).items()})
        for file, expected in hashes.items():
            # Git checkouts can change CRLF/LF without changing the model data.
            raw = file.read_bytes()
            lf = raw.replace(b"\r\n", b"\n")
            equivalent_hashes = {hashlib.sha256(data).hexdigest()
                                 for data in [raw, lf, lf.replace(b"\n", b"\r\n")]}
            if expected and expected not in equivalent_hashes:
                raise ValueError(f"Kalibrierfeld passt nicht zur aktuellen Datei {file.name}. Feld neu erstellen oder --no-field verwenden.")
        self.axes = self.model.get("composition_axes", ["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct"])
        self.nodes = pd.DataFrame(self.model["nodes"])
        self.coordinates = self.nodes[self.axes].to_numpy(float)
        self.scale = np.asarray(self.model["interpolation"]["scale"], dtype=float)
        if self.scale.shape != (len(self.axes),) or np.any(self.scale <= 0):
            raise ValueError("Ungültige Skalierung im Kalibrierfeld.")

    def residual(self, comp: np.ndarray) -> np.ndarray:
        lookup = dict(zip(["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"], comp[:4]))
        target = np.array([lookup[axis] for axis in self.axes])
        distances = np.linalg.norm((self.coordinates - target) / self.scale, axis=1)
        requested = int(self.model["interpolation"].get("neighbors", 4))
        count = len(distances) if requested <= 0 else min(requested, len(distances))
        indexes = np.argsort(distances)[:count]
        if distances[indexes[0]] < 1e-12:
            indexes = indexes[:1]
            geometric = np.ones(1)
        else:
            power = float(self.model["interpolation"].get("power", 2.))
            geometric = 1. / np.maximum(distances[indexes], 1e-12) ** power
        output = []
        for value_col, se_col in [("A_Rho_kg_m3", "A_Rho_SE_kg_m3"), ("A_C_m_s", "A_C_SE_m_s")]:
            values = self.nodes[value_col].to_numpy(float)[indexes]
            se = pd.to_numeric(self.nodes[se_col], errors="coerce").to_numpy(float)[indexes]
            good = se[np.isfinite(se) & (se >= 0)]
            fallback = float(np.median(good)) if len(good) else 1.
            se = np.where(np.isfinite(se) & (se >= 0), se, fallback)
            weights = geometric / np.maximum(se, max(fallback * .25, 1e-9)) ** 2
            weights /= weights.sum()
            output.append(float(weights @ values))
        return np.array(output)

    def outside_axes(self, comp: np.ndarray) -> list[str]:
        lookup = dict(zip(["Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"], comp[:4]))
        return [axis for j, axis in enumerate(self.axes)
                if lookup[axis] < self.coordinates[:, j].min() - 1e-10
                or lookup[axis] > self.coordinates[:, j].max() + 1e-10]


def properties(calculator, comp: np.ndarray, temperature: float, field=None) -> np.ndarray:
    arguments = dict(zip(["al", "ipa", "pg", "mg"], comp[:4]))
    result = np.array([1000. * calculator.density(**arguments, temperature=temperature),
                       calculator.sound_velocity(**arguments, temperature=temperature)])
    return result if field is None else result + field.residual(comp)


def invert(calculator, points: list[dict], field=None, *, anchor_recipe=True) -> tuple[np.ndarray, np.ndarray]:
    if anchor_recipe:
        offset = np.array([points[0]["rho"], points[0]["sound"]]) - properties(
            calculator, STANDARD, points[0]["temperature"], field)
        states = [STANDARD.copy()]  # P1 fixes the offset; it is not an independent prediction.
        selected_points = points[1:]
    else:
        offset = np.zeros(2)
        states = []
        selected_points = points
    for point in selected_points:
        target = np.array([point["rho"], point["sound"]]) - offset
        def residual(values):
            return (properties(calculator, composition(*values), point["temperature"], field) - target) / [.1, 1.]
        # Multiple starting values also detect discontinuities at IDW neighbors.
        solutions = [least_squares(residual, [al, ipa], bounds=([.8, 0.], [4., 8.]),
                                  xtol=1e-11, ftol=1e-11, gtol=1e-11)
                     for al in [1.7, 2., 2.4, 3.] for ipa in [.1, 1., 2., 3.5, 4.5]]
        best = min(solutions, key=lambda result: np.linalg.norm(result.fun))
        physical_residual = properties(calculator, composition(*best.x), point["temperature"], field) - target
        if np.max(np.abs(physical_residual)) > 1e-4:
            raise ValueError(f"Keine genaue gemeinsame Lösung für {point['name']}: Restfehler {physical_residual}. Keine Näherung wird als exakte Rückrechnung ausgegeben.")
        states.append(composition(*best.x))
    return np.array(states), offset


def read_sensor_csv(path: Path, points: list[dict]) -> None:
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    required = {"Date", "UTC Time", "ProbeNr", "Gueltig", "Rho_M", "C_M", "T_M"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV-Spalten fehlen: {sorted(required - set(frame.columns))}")
    for col in ["ProbeNr", "Gueltig", "Rho_M", "C_M", "T_M"]:
        frame[col] = pd.to_numeric(frame[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    frame["local"] = pd.to_datetime(frame["Date"].astype(str) + " " + frame["UTC Time"].astype(str), utc=True).dt.tz_convert("Europe/Berlin")
    for point in points:
        start = pd.Timestamp(f"{DATE} {point['start']}", tz="Europe/Berlin")
        end = pd.Timestamp(f"{DATE} {point['end']}", tz="Europe/Berlin")
        selected = frame[(frame.ProbeNr == 305) & (frame.Gueltig == 1)
                         & (frame.local >= start) & (frame.local < end)].dropna(subset=["Rho_M", "C_M", "T_M"]).sort_values("local")
        if len(selected) < 2:
            raise ValueError(f"Zu wenige gültige Messungen für {point['name']} im Fenster {point['start']}-{point['end']}.")
        for col, key in [("Rho_M", "rho"), ("C_M", "sound")]:
            point[key] = float(selected[col].mean())
            point[key + "_sd"] = float(selected[col].std(ddof=1))
        point["temperature"] = float(selected.T_M.mean())
        point["n"] = len(selected)
        centre = selected.local.min() + (selected.local.max() - selected.local.min()) / 2
        point["centre"] = centre.strftime("%H:%M:%S.%f")


def timestamp(time: str):
    # Naive datetimes here already represent local times, avoiding timezone tick shifts.
    return pd.Timestamp(f"{DATE} {time}").to_pydatetime()


def time_axis(axis, times, labels=None):
    axis.set_xticks(times)
    axis.set_xticklabels(labels if labels is not None else [time.strftime("%H:%M") for time in times])
    axis.set_xlabel("Uhrzeit am 02.10.2026 (MESZ)")
    axis.grid(True, alpha=.3)
    axis.margins(x=.08)


def create_plots(calculator, points, states, hybrid, output: Path, formats, show: bool,
                 field_states_without_offset=None):
    plt.style.use("default")  # Normal Matplotlib design: no theme or custom typography.
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    def save(figure, name):
        figure.tight_layout()
        for extension in formats:
            figure.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        figures.append(figure)
    sensor_times = [timestamp(p["centre"]) for p in points]
    rho = np.array([p["rho"] for p in points])
    sound = np.array([p["sound"] for p in points])
    temperatures = np.array([p["temperature"] for p in points])
    reference = float(temperatures.mean())  # Equal weight per point, not per minute row.
    correction = np.array([properties(calculator, state, reference) - properties(calculator, state, temp)
                           for state, temp in zip(states, temperatures)])
    normalized = np.column_stack([rho, sound]) + correction

    fig, ax = plt.subplots(figsize=(8, 4.8))
    times = [timestamp(p["tank_time"]) for p in points]
    tank = np.array([p["tank_g"] for p in points]); samples = np.array([p["samples_g"] for p in points])
    loss = tank[0] - tank - samples
    ax.plot(times, loss, "o-", label="Verlust ohne Probenentnahmen")
    ax.plot(times, tank[0] - tank, "o--", label="Tankverlust einschließlich Proben")
    ax.set(title="Massenverlust des Tanks", ylabel="Kumulativer Massenverlust [g]")
    time_axis(ax, times); ax.legend()
    save(fig, "01_Massenverlust")

    def sensor_plot(values, title, name, compare_raw=False):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for j, (ax, key, label) in enumerate(zip(axes, ["rho", "sound"], ["Dichte [kg/m³]", "Schallgeschwindigkeit [m/s]"])):
            if compare_raw:
                ax.plot(sensor_times, [p[key] for p in points], "o--", color="0.6", label="Unkorrigiert")
            ax.errorbar(sensor_times, values[:, j], yerr=[p[key + "_sd"] for p in points], fmt="o-", capsize=3,
                        label="Temperaturkorrigiert" if compare_raw else "Mittelwert ± s")
            ax.set_ylabel(label); time_axis(ax, sensor_times); ax.legend()
        axes[0].set_xlabel("")
        fig.suptitle(title)
        save(fig, name)
    sensor_plot(np.column_stack([rho, sound]), "Dichte und Schallgeschwindigkeit", "02_Dichte_Schallgeschwindigkeit")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    times = [timestamp(p["sample"]) for p in points]
    dry = np.array([np.mean(p["dry"]) for p in points])
    sd = np.array([np.std(p["dry"], ddof=1) for p in points])
    ax.errorbar(times, dry, yerr=sd, fmt="o-", capsize=3, label="Festkörpergehalt ± s")
    ax.plot(times[0], dry[0], "x", color="red", markersize=8, label="P1: auffällige Messung")
    ax.axhline(STANDARD[0] + STANDARD[3], color="0.4", linestyle="--", label="Rezeptur: Pigment + MG (2,04 %)")
    ax.set(title="Festkörpergehalt", ylabel="Festkörpergehalt [Masse-%]")
    time_axis(ax, times); ax.legend()
    save(fig, "03_Festkoerpergehalt")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for j, label in enumerate(COMPONENTS[:4]):
        line, = axes[0].plot(sensor_times, states[:, j], "o-", label=label)
        if hybrid is not None:
            axes[0].plot(sensor_times, hybrid[:, j], "--", color=line.get_color())
    axes[0].set(ylabel="Pigment, IPA, PG, MG [Masse-%]", title="Formale Rückrechnung aus Dichte und Schall", ylim=(0, 6))
    handles, labels = axes[0].get_legend_handles_labels()
    if hybrid is not None:
        handles.append(Line2D([], [], color="black", linestyle="--"))
        labels.append("Kalibrierfeld (gestrichelt)")
    axes[0].legend(handles, labels, ncol=2)
    axes[1].plot(sensor_times, states[:, 4], "o-", label="Wasser")
    if hybrid is not None:
        axes[1].plot(sensor_times, hybrid[:, 4], "--", label="Wasser: Kalibrierfeld")
    axes[1].set_ylabel("Wasser [Masse-%]"); axes[1].legend()
    ratio = states[:, [1, 4]] / states[:, [0]]
    flags = [False] + [bool(np.any(ratio[i] > ratio[0] + 1e-8)
                                  or np.any(ratio[i] > ratio[i - 1] + 1e-8)) for i in range(1, 4)]
    point_labels = [p["name"] + ("*" if flag else "") + "\n" + time.strftime("%H:%M")
                    for p, time, flag in zip(points, sensor_times, flags)]
    for ax in axes:
        time_axis(ax, sensor_times, point_labels)
    axes[1].set_xlabel("Uhrzeit am 02.10.2026 (MESZ); *: Bilanzabweichung")
    axes[0].set_xlabel("")
    save(fig, "04_Rueckgerechnete_Zusammensetzung")

    sensor_plot(normalized, f"Dichte und Schall, korrigiert auf {reference:.3f} °C", "05_Dichte_Schall_temperaturkorrigiert", compare_raw=True)
    # Error bars in the corrected plot retain the ORIGINAL within-window SD.
    # They do not include uncertainty of the model, the inferred composition or T.
    if field_states_without_offset is not None:
        direct = field_states_without_offset
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        for j, label in enumerate(COMPONENTS[:4]):
            axes[0].plot(sensor_times, direct[:, j], "o-", label=label)
        axes[0].set(ylabel="Pigment, IPA, PG, MG [Masse-%]",
                    title="Rückrechnung mit Kalibrierfeld\nOhne zusätzlichen Sensoroffset", ylim=(0, 6))
        axes[0].legend(ncol=2)
        axes[1].plot(sensor_times, direct[:, 4], "o-", label="Wasser")
        axes[1].set_ylabel("Wasser [Masse-%]")
        axes[1].legend()
        ratios = direct[:, [1, 4]] / direct[:, [0]]
        flags = [False] + [bool(np.any(ratios[i] > ratios[0] + 1e-8)
                               or np.any(ratios[i] > ratios[i - 1] + 1e-8))
                           for i in range(1, len(points))]
        labels = [point["name"] + ("*" if flag else "") + "\n" + time.strftime("%H:%M")
                  for point, time, flag in zip(points, sensor_times, flags)]
        for ax in axes:
            time_axis(ax, sensor_times, labels)
        axes[0].set_xlabel("")
        axes[1].set_xlabel("Uhrzeit am 02.10.2026 (MESZ); *: Bilanzabweichung")
        save(fig, "06_Zusammensetzung_Kalibrierfeld_ohne_Offset")
    summary = pd.DataFrame({"Punkt": [p["name"] for p in points], "T_C": temperatures,
                            "Rho_roh_kg_m3": rho, "C_roh_m_s": sound,
                            "Delta_Rho_T_kg_m3": correction[:, 0], "Delta_C_T_m_s": correction[:, 1],
                            "Rho_Tref_kg_m3": normalized[:, 0], "C_Tref_m_s": normalized[:, 1]})
    print(f"\nReferenztemperatur (Mittel der vier Punkte): {reference:.9f} °C")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("Fehlerbalken: ursprüngliche Fensterstreuung, keine gesamte Modell-/Korrekturunsicherheit.")
    print(f"\nDiagramme gespeichert in: {output.resolve()}")
    if show:
        plt.show()
    for fig in figures:
        plt.close(fig)
    return reference, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, help="Ordner mit ink_calculator.py und tables_parameters")
    parser.add_argument("--output-dir", type=Path, help="Standard: Skriptordner/results_Probe305")
    parser.add_argument("--csv", type=Path, help="Optional: Sensorfenster aus dieser Mess-CSV neu auswerten")
    parser.add_argument("--calibration-field", type=Path, help="Optional: anderes gespeichertes calibration_field.json")
    parser.add_argument("--no-field", action="store_true", help="Nur Ink Calculator verwenden")
    parser.add_argument("--no-show", action="store_true", help="Diagramme speichern, keine Fenster öffnen")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"], help="Standard: png; z.B. --formats png pdf")
    args = parser.parse_args()
    repo = find_repo(args.repo_root)
    calculator = load_calculator(repo)
    points = [dict(p, dry=list(p["dry"])) for p in POINTS]
    if args.csv:
        read_sensor_csv(args.csv, points)
    states, offset = invert(calculator, points)
    print(f"Ink Calculator Offset: Dichte {offset[0]:+.6f} kg/m³, Schall {offset[1]:+.6f} m/s")
    print(pd.DataFrame(states, index=[p["name"] for p in points], columns=COMPONENTS).to_string(float_format=lambda v: f"{v:.6f}"))
    for i in range(1, len(states)):
        ratio = states[i, [1, 4]] / states[i, 0]
        if np.any(ratio > states[0, [1, 4]] / states[0, 0] + 1e-8) or np.any(ratio > states[i - 1, [1, 4]] / states[i - 1, 0] + 1e-8):
            print(f"HINWEIS {points[i]['name']}: Wasser/Pigment oder IPA/Pigment steigt gegenüber dem Start oder dem vorherigen Punkt. Die formale Rückrechnung verletzt die reine Verdunstungsannahme.")
    hybrid = None
    field_states_without_offset = None
    field_path = args.calibration_field or repo / FIELD_RELATIVE
    if not args.no_field and field_path.is_file():
        field = CalibrationField(field_path, repo)
        hybrid, field_offset = invert(calculator, points, field)
        field_states_without_offset, _ = invert(calculator, points, field, anchor_recipe=False)
        print("\nPlot 6: punktweise Kalibrierfeld-Rückrechnung ohne zusätzlichen Sensoroffset")
        print(pd.DataFrame(field_states_without_offset, index=[p["name"] for p in points],
                           columns=COMPONENTS).to_string(float_format=lambda v: f"{v:.6f}"))
        ratios = field_states_without_offset[:, [1, 4]] / field_states_without_offset[:, [0]]
        for i, (point, comp) in enumerate(zip(points, field_states_without_offset)):
            outside = field.outside_axes(comp)
            if outside:
                print(f"HINWEIS Plot 6 {point['name']}: Kalibrierfeld außerhalb der Trainingsbereiche für {', '.join(outside)}.")
            if i and (np.any(ratios[i] > ratios[0] + 1e-8)
                      or np.any(ratios[i] > ratios[i - 1] + 1e-8)):
                print(f"HINWEIS Plot 6 {point['name']}: Bilanzabweichung; die punktweisen Lösungen bilden keinen reinen Verdunstungsverlauf.")
        print(f"Kalibrierfeld Offset: Dichte {field_offset[0]:+.6f} kg/m³, Schall {field_offset[1]:+.6f} m/s")
        for point, comp in zip(points, hybrid):
            outside = field.outside_axes(comp)
            if outside:
                print(f"HINWEIS {point['name']}: Kalibrierfeld außerhalb der Trainingsbereiche für {', '.join(outside)}.")
    elif args.calibration_field and not args.no_field:
        raise FileNotFoundError(field_path)
    elif not args.no_field:
        print("Kalibrierfeld nicht gefunden; Diagramm 4 verwendet nur den Ink Calculator.")
    if field_states_without_offset is None:
        print("Diagramm 6 wird ohne Kalibrierfeld nicht erstellt.")
    output = args.output_dir or Path(__file__).resolve().parent / "results_Probe305"
    create_plots(calculator, points, states, hybrid, output, args.formats, not args.no_show,
                 field_states_without_offset=field_states_without_offset)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError) as error:
        raise SystemExit(str(error)) from error
