#!/usr/bin/env python
"""Dichte-Schall-Kennfelder mit gefuehrter Eingabe erstellen.

Diese Datei im Ordner sim_par_ink_prop/plots ablegen und starten:
    python plot_messkennfeld.py

Pakete: numpy, pandas, scipy, matplotlib (Python >= 3.10).
    python -m pip install numpy pandas scipy matplotlib

Ohne Eingabedialog, z. B.:
    python plot_messkennfeld.py --model both --al-min 1.7 --al-max 2.2 \
        --ipa-min 2.8 --ipa-max 5.3 --temperature 23 --no-show

Im Eingabedialog kann das Kalibrierfeld aus den gespeicherten Feldern
ausgewaehlt werden (IDW oder Gaussian Kernel Regression). Mit
--calibration-field PFAD laesst sich direkt eine calibration_field.json oder
ihr Ordner angeben. Bei vollstaendig gesetzten Argumenten ohne Feldpfad
wird weiterhin das bisherige Standardfeld verwendet.

Standard: PG = 2 * Al; MG = Al / 8; Wasser = 100 - Al - PG - MG - IPA.
Alle Konzentrationen sind Massenprozente der fertigen Tinte. "Al" bezeichnet
den vollstaendigen Aluminiumpigmentanteil (inklusive Verkapselung), wie beim
InkCalculator. Das Verhaeltnis entspricht plots/plot_probe305.py.
Mit --pg-per-al und --mg-per-al lassen sich andere feste Verhaeltnisse setzen.
Fuer die gerundete Rezeptur des ersten Diagramms: --pg-per-al 2.0055248619
--mg-per-al 0.1215469613 (Al:PG:MG = 1.81:3.63:0.22).

Physik: direkte Aufrufe des vorhandenen InkCalculator, keine Ersatzformeln.
Hybrid: Physik + gespeicherte A(w)-Korrektur. Unterstuetzt werden die
qualitaetsgewichtete IDW (Schema v3) und Gaussian Kernel Regression ueber
alle Stuetzstellen (Schema v4). Die gespeicherten Verfahrensparameter gelten.
Die Korrektur wird bei jeder Zusammensetzung neu berechnet, nicht als
konstanter Offset. Temperatur wirkt im Physikmodell; A(w) hat keinen
zusaetzlich gelernten Temperaturterm. Ein Kennfeldplot beweist keine
eindeutige Umkehrbarkeit.

Ausgaben: PNG/SVG, berechnete Linienpunkte als CSV und Einstellungen mit
Quelldatei-Hashes als JSON, direkt in plots/messkennfeld/ neben dem Skript.
Ein Zeitstempel im Dateinamen unterscheidet die einzelnen Laeufe.
Das Skript veraendert weder Modelle noch Parameterdateien.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings


FIELD_RELATIVE = Path("laboratory_measurement_L-Com/results/calibration_field") / (
    "probe_9_10_11__Kennfeld_v2_23__korrigiert__20260928_110809/calibration_field.json"
)
IDW_METHOD = "scaled_inverse_distance_weighting"
GAUSSIAN_METHOD = "scaled_gaussian_kernel_regression"
SUPPORTED_FIELDS = {3: IDW_METHOD, 4: GAUSSIAN_METHOD}
TITLES = {"physics": "Ink Calculator (Physikmodell)",
          "hybrid": "Hybrides Kalibrierfeld: Physik + A(w)"}


@dataclass(frozen=True)
class Settings:
    model: str
    al_min: float
    al_max: float
    ipa_min: float
    ipa_max: float
    temperature: float
    pg_per_al: float = 2.0
    mg_per_al: float = 0.125
    lines: int = 7
    points: int = 151


def finite_number(value: str) -> float:
    try:
        result = float(value.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Bitte eine Zahl eingeben.") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("Die Zahl muss endlich sein.")
    return result


def ask_float(label: str, default: float, minimum: float | None = None) -> float:
    while True:
        answer = input(f"{label} [{default:g}]: ").strip()
        try:
            value = finite_number(answer) if answer else default
        except argparse.ArgumentTypeError as exc:
            print(exc)
            continue
        if minimum is not None and value < minimum:
            print(f"Der Wert muss mindestens {minimum:g} sein.")
            continue
        return value


def ask_model() -> str:
    print("\nWas moechtest du plotten?")
    print("  1) Ink Calculator (Physikmodell)")
    print("  2) Hybrides Kalibrierfeld")
    print("  3) Beides: Physikmodell und hybrides Kalibrierfeld")
    choices = {"1": "physics", "2": "hybrid", "3": "both"}
    while True:
        answer = input("Auswahl [3]: ").strip() or "3"
        if answer in choices:
            return choices[answer]
        print("Bitte 1, 2 oder 3 eingeben.")


def validate(settings: Settings) -> None:
    values = [settings.al_min, settings.al_max, settings.ipa_min, settings.ipa_max,
              settings.temperature, settings.pg_per_al, settings.mg_per_al]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Alle Eingaben muessen endlich sein.")
    if min(settings.al_min, settings.ipa_min, settings.pg_per_al, settings.mg_per_al) < 0:
        raise ValueError("Konzentrationen und Verhaeltnisse duerfen nicht negativ sein.")
    if settings.al_max <= settings.al_min or settings.ipa_max <= settings.ipa_min:
        raise ValueError("Jede obere Schranke muss groesser als die untere sein.")
    maximum_sum = settings.al_max * (1 + settings.pg_per_al + settings.mg_per_al) + settings.ipa_max
    if maximum_sum > 100.0:
        raise ValueError(f"Am oberen Rand ergeben sich {maximum_sum:g} Gew.-% ohne Wasser. "
                         "Schranken verkleinern: Al + PG + MG + IPA darf 100 nicht uebersteigen.")
    if settings.lines < 2 or settings.points < 2:
        raise ValueError("--lines und --points muessen mindestens 2 sein.")


def configure(args) -> Settings:
    prompted = args.model is None or any(getattr(args, name) is None for name in
        ("al_min", "al_max", "ipa_min", "ipa_max", "temperature"))
    if prompted:
        print("\nDichte-Schall-Messkennfeld")
        print("Alle Gehalte in Gew.-% der fertigen Tinte.")
        print(f"PG = {args.pg_per_al:g} * Al; MG = {args.mg_per_al:g} * Al; Wasser = Rest zu 100 %.")
        print("Enter uebernimmt jeweils den Wert in eckigen Klammern.")
    model = args.model or ask_model()
    names = ("al_min", "al_max", "ipa_min", "ipa_max", "temperature")
    labels = ("Aluminiumpigment: untere Schranke", "Aluminiumpigment: obere Schranke",
              "IPA: untere Schranke", "IPA: obere Schranke", "Temperatur [Grad C]")
    defaults = (1.7, 2.2, 2.8, 5.3, 23.0)
    values = {}
    for name, label, default in zip(names, labels, defaults):
        provided = getattr(args, name)
        values[name] = provided if provided is not None else ask_float(
            label, default, minimum=None if name == "temperature" else 0.0)
    while True:
        settings = Settings(model=model, **values, pg_per_al=args.pg_per_al,
                            mg_per_al=args.mg_per_al, lines=args.lines, points=args.points)
        try:
            validate(settings)
            return settings
        except ValueError as exc:
            if not prompted or args.pg_per_al < 0 or args.mg_per_al < 0 or min(args.lines, args.points) < 2:
                raise
            print(f"\nUngueltige Schranken: {exc}\nBitte die vier Schranken erneut eingeben.")
            for name, label in zip(names[:4], labels[:4]):
                values[name] = ask_float(label, values[name], minimum=0.0)


def find_repo(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [
        *Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if candidate is not None and (candidate / "ink_calculator.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("ink_calculator.py nicht gefunden. Skript in sim_par_ink_prop "
                            "oder plots/ ablegen; alternativ --repo-root PFAD verwenden.")


def load_calculator(repo: Path):
    path = repo / "ink_calculator.py"
    spec = importlib.util.spec_from_file_location("messkennfeld_ink_calculator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Ink Calculator konnte nicht geladen werden: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.InkCalculator(tables_dir=str(repo / "tables_parameters"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equivalent_hashes(path: Path) -> set[str]:
    # LF/CRLF and terminal blank lines do not alter these Python/CSV inputs.
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    variants = [raw, lf, lf.replace(b"\n", b"\r\n")]
    for ending in (b"", b"\n", b"\n\n"):
        normalized = lf.rstrip(b"\n") + ending
        variants.extend([normalized, normalized.replace(b"\n", b"\r\n")])
    return {hashlib.sha256(data).hexdigest() for data in variants}


class CalibrationField:
    """Evaluate saved IDW or Gaussian fields with their original weighting."""

    def __init__(self, path: Path, repo: Path):
        import numpy as np
        import pandas as pd

        self.path = path.resolve()
        self.model = json.loads(self.path.read_text(encoding="utf-8-sig"))
        interpolation = self.model.get("interpolation", {})
        self.method = interpolation.get("method")
        if (self.model.get("schema") != "ink-residual-calibration-field"
                or self.model.get("schema_version") not in SUPPORTED_FIELDS
                or self.method != SUPPORTED_FIELDS[self.model["schema_version"]]):
            raise ValueError("Unterstuetzt werden IDW-Felder (Schema 3) und "
                             "Gaussian-Kernel-Felder (Schema 4).")
        if not self.model.get("nodes"):
            raise ValueError("Das Kalibrierfeld enthaelt keine Knoten.")
        self.axes = self.model["composition_axes"]
        allowed = {"Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"}
        if not self.axes or len(set(self.axes)) != len(self.axes) or set(self.axes) - allowed:
            raise ValueError("Ungueltige Zusammensetzungsachsen im Kalibrierfeld.")
        self.nodes = pd.DataFrame(self.model["nodes"])
        self.coordinates = self.nodes[self.axes].to_numpy(float)
        self.minimum = self.coordinates.min(axis=0)
        self.maximum = self.coordinates.max(axis=0)
        self.scale = np.asarray(interpolation["scale"], dtype=float)
        if (self.scale.shape != (len(self.axes),) or np.any(~np.isfinite(self.scale))
                or np.any(self.scale <= 0)
                or not np.isfinite(self.coordinates).all()):
            raise ValueError("Ungueltige Kalibrierfeld-Skalierung oder Knotenkoordinaten.")
        if self.method == IDW_METHOD:
            self.power = float(interpolation.get("power", 2.0))
            requested = int(interpolation.get("neighbors", 4))
            self.count = len(self.nodes) if requested <= 0 else min(requested, len(self.nodes))
            if not math.isfinite(self.power) or self.power <= 0:
                raise ValueError("Ungueltiger IDW-Exponent.")
            self.description = f"IDW ({self.count} Stuetzstellen je Punkt)"
        else:
            self.bandwidth = float(interpolation["bandwidth"])
            if (not math.isfinite(self.bandwidth) or self.bandwidth <= 0
                    or interpolation.get("node_selection") != "all"
                    or interpolation.get("quality_weighting") != "fixed_inverse_squared_standard_error"
                    or interpolation.get("quality_floor_fraction") != 0.25
                    or interpolation.get("quality_absolute_floor") != 1e-9):
                raise ValueError("Ungueltige Gaussian-Kernel-Parameter.")
            self.count = len(self.nodes)
            self.description = f"Gaussian Kernel Regression (h={self.bandwidth:g}, alle Stuetzstellen)"
        self.properties = []
        for value_col, se_col in [("A_Rho_kg_m3", "A_Rho_SE_kg_m3"), ("A_C_m_s", "A_C_SE_m_s")]:
            values = self.nodes[value_col].to_numpy(float)
            se = pd.to_numeric(self.nodes[se_col], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nichtendliche Korrekturwerte: {value_col}")
            log_quality = None
            if self.method == GAUSSIAN_METHOD:
                # Keep quality weights fixed across all query compositions.
                finite = se[np.isfinite(se) & (se >= 0)]
                fallback = float(np.median(finite)) if finite.size else 1.0
                se = np.where(np.isfinite(se) & (se >= 0), se, fallback)
                log_quality = -2.0 * np.log(np.maximum(se, max(fallback * .25, 1e-9)))
            self.properties.append((values, se, log_quality))
        provenance = self.model.get("calculator", {})
        inputs = {repo / "ink_calculator.py": provenance.get("sha256")}
        inputs.update({repo / "tables_parameters" / name: expected
                       for name, expected in provenance.get("table_sha256", {}).items()})
        for file, expected in inputs.items():
            if expected and expected not in equivalent_hashes(file):
                raise ValueError(f"Kalibrierfeld passt nicht zur Datei {file.name}. "
                                 "Bitte die zugehoerige Modellversion verwenden oder A(w) neu kalibrieren.")

    def evaluate(self, al: float, ipa: float, pg: float, mg: float):
        import numpy as np

        lookup = dict(zip(("Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"), (al, ipa, pg, mg)))
        target = np.asarray([lookup[axis] for axis in self.axes])
        if not np.isfinite(target).all():
            raise ValueError("Die Zusammensetzung muss endlich sein.")
        if self.method == GAUSSIAN_METHOD:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                squared = np.sum(((self.coordinates - target) / self.scale)**2, axis=1)
            if not np.isfinite(squared).all():
                raise ValueError("Die skalierten Abstaende sind numerisch zu gross.")
            with np.errstate(over="ignore"):
                log_geometry = -0.5 * (squared - squared.min()) / self.bandwidth / self.bandwidth
        else:
            distances = np.linalg.norm((self.coordinates - target) / self.scale, axis=1)
            indexes = np.argsort(distances)[:self.count]
            if distances[indexes[0]] < 1e-12:
                indexes = indexes[:1]
                geometric = np.ones(1)
            else:
                geometric = 1.0 / np.maximum(distances[indexes], 1e-12) ** self.power
        estimates, uncertainties = [], []
        for values_all, se_all, log_quality in self.properties:
            if self.method == GAUSSIAN_METHOD:
                values, se = values_all, se_all
                log_weights = log_geometry + log_quality
                weights = np.exp(log_weights - log_weights.max())
            else:
                values, se = values_all[indexes], se_all[indexes]
                finite = se[np.isfinite(se) & (se >= 0)]
                fallback = float(np.median(finite)) if finite.size else 1.0
                se = np.where(np.isfinite(se) & (se >= 0), se, fallback)
                weights = geometric / np.maximum(se, max(fallback * .25, 1e-9)) ** 2
            weights /= weights.sum()
            estimate = float(np.sum(weights * values))
            estimates.append(estimate)
            uncertainties.append(float(np.sqrt(np.sum(weights * ((values - estimate) ** 2 + se**2)))))
        outside = [axis.removesuffix("_wt_pct") for axis, value, low, high in
                   zip(self.axes, target, self.minimum, self.maximum)
                   if value < low - 1e-10 * max(1., abs(low), abs(high))
                   or value > high + 1e-10 * max(1., abs(low), abs(high))]
        return estimates, uncertainties, outside


def discover_calibration_fields(repo: Path) -> list[tuple[Path, str]]:
    """List supported saved fields from both calibration workflows."""
    results = repo / "laboratory_measurement_L-Com" / "results"
    fields = []
    for path in results.rglob("calibration_field.json"):
        try:
            model = json.loads(path.read_text(encoding="utf-8-sig"))
            interpolation = model.get("interpolation", {})
            method = interpolation.get("method")
            if (model.get("schema") != "ink-residual-calibration-field"
                    or model.get("schema_version") not in SUPPORTED_FIELDS
                    or method != SUPPORTED_FIELDS[model["schema_version"]]
                    or not model.get("nodes")):
                continue
            label = (f"Gaussian Kernel Regression, h={float(interpolation['bandwidth']):g}"
                     if method == GAUSSIAN_METHOD else "IDW")
            fields.append((path.resolve(), f"{label}; {len(model['nodes'])} Stuetzstellen"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return sorted(fields, key=lambda item: item[0].stat().st_mtime, reverse=True)


def normalize_field_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        path /= "calibration_field.json"
    return path.resolve()


def field_path(args, repo: Path, interactive: bool) -> Path:
    if args.calibration_field is None and interactive:
        fields = discover_calibration_fields(repo)
        if fields:
            print("\nVerfuegbare Kalibrierfelder (neueste zuerst):")
            for number, (path, description) in enumerate(fields, start=1):
                print(f"  {number}) {path.parent.name}")
                print(f"     {description}")
                print(f"     {path.relative_to(repo)}")
            print("  M) Anderen Dateipfad oder Ordner eingeben")
            while True:
                answer = input("Kalibrierfeld auswaehlen [1]: ").strip() or "1"
                if answer.lower() == "m":
                    break
                if answer.isdecimal() and 1 <= int(answer) <= len(fields):
                    return fields[int(answer) - 1][0]
                print("Bitte eine der angezeigten Nummern oder M eingeben.")
        else:
            print("\nKeine gespeicherten IDW- oder Gaussian-Kernel-Felder gefunden.")
        while True:
            answer = input("Pfad zur calibration_field.json oder ihrem Ordner: ").strip().strip('"')
            if answer:
                path = normalize_field_path(Path(answer))
                if path.is_file():
                    return path
            print("Diese Kalibrierfelddatei wurde nicht gefunden.")
    path = normalize_field_path(args.calibration_field or repo / FIELD_RELATIVE)
    while not path.is_file():
        if not interactive:
            raise FileNotFoundError(f"Kalibrierfeld nicht gefunden: {path}")
        print(f"\nKalibrierfeld nicht gefunden: {path}")
        answer = input("Pfad zur calibration_field.json: ").strip().strip('"')
        path = normalize_field_path(Path(answer))
    return path.resolve()


def calculate_lines(calculator, field, settings: Settings):
    import numpy as np

    model_ids = ["physics", "hybrid"] if settings.model == "both" else [settings.model]
    data = {name: [] for name in model_ids}
    rows, cache, messages, outside_counts = [], {}, set(), {}

    def point(al, ipa):
        key = (float(al), float(ipa))
        if key not in cache:
            pg, mg = settings.pg_per_al * al, settings.mg_per_al * al
            arguments = dict(al=float(al), ipa=float(ipa), pg=float(pg), mg=float(mg),
                             temperature=settings.temperature)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    rho = 1000. * float(calculator.density(**arguments))
                    sound = float(calculator.sound_velocity(**arguments))
                messages.update(str(item.message) for item in caught)
                if not np.isfinite([rho, sound]).all():
                    raise ValueError("Das Modell liefert nichtendliche Werte.")
                corr, unc, outside = field.evaluate(al, ipa, pg, mg) if field else ([0., 0.], [0., 0.], [])
            except (ValueError, RuntimeError, ZeroDivisionError) as exc:
                raise ValueError(f"Berechnung bei Al={al:g} %, IPA={ipa:g} %, "
                                 f"T={settings.temperature:g} Grad C fehlgeschlagen: {exc}") from exc
            for axis in outside:
                outside_counts[axis] = outside_counts.get(axis, 0) + 1
            cache[key] = dict(Al_wt_pct=float(al), IPA_wt_pct=float(ipa), PG_wt_pct=float(pg),
                              MG_wt_pct=float(mg), Water_wt_pct=float(100. - al - ipa - pg - mg),
                              Temperature_C=settings.temperature, Rho_Physics_kg_m3=rho,
                              C_Physics_m_s=sound, A_Rho_kg_m3=corr[0], A_C_m_s=corr[1],
                              A_Rho_Uncertainty_kg_m3=unc[0], A_C_Uncertainty_m_s=unc[1],
                              Outside_Axes=",".join(outside))
        return cache[key]

    families = [
        ("al", np.linspace(settings.al_min, settings.al_max, settings.lines),
         np.linspace(settings.ipa_min, settings.ipa_max, settings.points)),
        ("ipa", np.linspace(settings.ipa_min, settings.ipa_max, settings.lines),
         np.linspace(settings.al_min, settings.al_max, settings.points)),
    ]
    for family, constants, varying in families:
        for constant in constants:
            values = [point(constant, value) if family == "al" else point(value, constant) for value in varying]
            for model_id in model_ids:
                density = [p["Rho_Physics_kg_m3"] + (p["A_Rho_kg_m3"] if model_id == "hybrid" else 0.) for p in values]
                sound = [p["C_Physics_m_s"] + (p["A_C_m_s"] if model_id == "hybrid" else 0.) for p in values]
                data[model_id].append(dict(family=family, constant=float(constant), density=density, sound=sound))
                for p, rho, c in zip(values, density, sound):
                    row = dict(Model=model_id, Line_Family=family, Constant_wt_pct=float(constant),
                               **p, Density_kg_m3=rho, Sound_m_s=c)
                    if model_id == "physics":
                        for key in ("A_Rho_kg_m3", "A_C_m_s", "A_Rho_Uncertainty_kg_m3", "A_C_Uncertainty_m_s", "Outside_Axes"):
                            row[key] = ""
                    rows.append(row)
    return data, rows, sorted(messages), outside_counts, len(cache)


def label_lines(axis, lines, colors) -> None:
    """Place every concentration directly on its curve, in display coordinates.

    Candidate positions follow actual polylines (also non-monotone IDW curves).
    Prefer separate parts of the two line families and avoid existing labels.
    """
    import numpy as np

    renderer = axis.figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    placed = []
    for line in lines:
        points = np.column_stack([line["density"], line["sound"]])
        display = axis.transData.transform(points)
        lengths = np.linalg.norm(np.diff(display, axis=0), axis=1)
        distance = np.concatenate([[0.], np.cumsum(lengths)])
        if distance[-1] <= 0:
            continue
        label = f"{'Al' if line['family'] == 'al' else 'IPA'} = {line['constant']:.3g} %"
        # Label Al near the low-IPA edge, IPA near the low-Al edge.
        fractions = (.18, .26, .34, .42, .58, .66, .74, .82)
        if line["family"] == "ipa":
            fractions = (.22, .30, .38, .46, .62, .70, .78, .86)
        best = None
        for preference, fraction in enumerate(fractions):
            location = distance[-1] * fraction
            index = int(np.clip(np.searchsorted(distance, location, side="right") - 1, 0, len(lengths) - 1))
            proportion = (location - distance[index]) / max(lengths[index], 1e-12)
            xy = display[index] + proportion * (display[index + 1] - display[index])
            # Use a longer tangent window to avoid an unstable angle at a kink.
            low, high = max(0, index - 2), min(len(display) - 1, index + 3)
            tangent = display[high] - display[low]
            angle = math.degrees(math.atan2(tangent[1], tangent[0]))
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            x, y = axis.transData.inverted().transform(xy)
            text = axis.text(x, y, label, color=colors[(line["family"], line["constant"])],
                             fontsize=9, ha="center", va="center", rotation=angle,
                             rotation_mode="anchor", zorder=5, clip_on=True,
                             bbox=dict(facecolor=axis.get_facecolor(), edgecolor="none", pad=.8))
            bounds = text.get_window_extent(renderer).expanded(1.06, 1.15)
            inside = (bounds.x0 >= frame.x0 + 3 and bounds.x1 <= frame.x1 - 3
                      and bounds.y0 >= frame.y0 + 3 and bounds.y1 <= frame.y1 - 3)
            overlap = sum(bounds.overlaps(previous) for previous in placed)
            score = (0 if inside else 1000) + 100 * overlap + preference
            text.remove()
            if best is None or score < best[0]:
                best = (score, x, y, angle, bounds)
            if inside and overlap == 0:
                break
        _, x, y, angle, bounds = best
        axis.text(x, y, label, color=colors[(line["family"], line["constant"])],
                  fontsize=9, ha="center", va="center", rotation=angle,
                  rotation_mode="anchor", zorder=5, clip_on=True,
                  bbox=dict(facecolor=axis.get_facecolor(), edgecolor="none", pad=.8))
        placed.append(bounds)


def create_plots(data, settings: Settings, output: Path, run_id: str, field=None):
    import matplotlib.pyplot as plt

    plt.style.use("default")
    figures, saved = [], []
    sample_lines = next(iter(data.values()))
    colors = {}
    for family, palette in (("al", plt.cm.Oranges), ("ipa", plt.cm.Blues)):
        lines = [line for line in sample_lines if line["family"] == family]
        for index, line in enumerate(lines):
            colors[(family, line["constant"])] = palette(.45 + .5 * index / max(1, len(lines) - 1))

    def draw(axis, name):
        for line in data[name]:
            axis.plot(line["density"], line["sound"],
                      color=colors[(line["family"], line["constant"])],
                      linestyle="--" if line["family"] == "al" else "-", linewidth=1.3)
        title = TITLES[name]
        if name == "hybrid" and field is not None:
            title += "\n" + field.description
        axis.set_title(title)
        axis.set_xlabel("Dichte [kg/m³]")
        axis.set_ylabel("Schallgeschwindigkeit [m/s]")
        axis.grid(True, alpha=.25)
        axis.ticklabel_format(axis="both", style="plain", useOffset=False)
        axis.margins(.06)

    def finish(fig, name, axes):
        fig.suptitle(f"Messkennfeld bei {settings.temperature:g} °C", fontsize=14)
        fig.text(.5, .93, f"Al:PG:MG = 1:{settings.pg_per_al:g}:{settings.mg_per_al:g}; Wasser = Rest zu 100 Gew.-%",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .91))
        fig.canvas.draw()
        for axis, model_id in axes:
            label_lines(axis, data[model_id], colors)
        for extension in ("png", "svg"):
            path = output / f"messkennfeld_{name}_{run_id}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path)
        figures.append(fig)

    for name in data:
        fig, axis = plt.subplots(figsize=(9, max(7.5, 4.5 + .32 * settings.lines)))
        draw(axis, name)
        finish(fig, name, [(axis, name)])
    if len(data) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 4.5 + .16 * settings.lines)), sharex=True, sharey=True)
        for axis, name in zip(axes, data):
            draw(axis, name)
        finish(fig, "vergleich", list(zip(axes, data)))
    return figures, saved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model", choices=("physics", "hybrid", "both"))
    for name in ("al-min", "al-max", "ipa-min", "ipa-max", "temperature"):
        result.add_argument(f"--{name}", type=finite_number)
    result.add_argument("--pg-per-al", type=finite_number, default=2.0)
    result.add_argument("--mg-per-al", type=finite_number, default=.125)
    result.add_argument("--lines", type=int, default=7, help="Linien je Familie (Standard: 7)")
    result.add_argument("--points", type=int, default=151, help="Stuetzpunkte je Linie (Standard: 151)")
    result.add_argument("--repo-root", type=Path)
    result.add_argument("--calibration-field", type=Path,
                        help="Gespeichertes IDW- oder Gaussian-Kernel-Feld (JSON oder Ordner)")
    result.add_argument("--output-dir", type=Path, help="Zielordner (Standard: messkennfeld neben dem Skript)")
    result.add_argument("--no-show", action="store_true", help="Dateien speichern, Plotfenster nicht oeffnen")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = configure(args)
        repo = find_repo(args.repo_root)
        if args.no_show:
            import matplotlib
            matplotlib.use("Agg")
        calculator = load_calculator(repo)
        field = None
        if settings.model in ("hybrid", "both"):
            interactive = any(getattr(args, name) is None for name in
                              ("model", "al_min", "al_max", "ipa_min", "ipa_max", "temperature"))
            field = CalibrationField(field_path(args, repo, interactive), repo)
            print(f"Kalibrierfeld: {field.path}")
            print(f"Verfahren: {field.description}")
        print(f"\nBerechne Al {settings.al_min:g}–{settings.al_max:g} %, "
              f"IPA {settings.ipa_min:g}–{settings.ipa_max:g} %, T={settings.temperature:g} Grad C ...")
        data, rows, messages, outside, unique_count = calculate_lines(calculator, field, settings)
        output = args.output_dir or Path(__file__).resolve().parent / "messkennfeld"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=True)
        figures, saved = create_plots(data, settings, output, run_id, field=field)
        csv_path = output / f"kennfeldwerte_{run_id}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files = [repo / "ink_calculator.py", *sorted((repo / "tables_parameters").glob("*.csv"))]
        if field:
            files.append(field.path)
        metadata = dict(settings=asdict(settings), repository=str(repo),
                        calibration_field=str(field.path) if field else None,
                        calibration_interpolation=field.model["interpolation"] if field else None,
                        source_sha256={str(path): sha256(path) for path in files},
                        unique_compositions=unique_count, outside_axis_counts=outside,
                        model_warnings=messages,
                        correction="physics(T,w) + A(w)" if field else "physics(T,w)",
                        uncertainty_note="A_*_Uncertainty beschreibt die Streuung der Kalibrierkorrektur, nicht die gesamte Messunsicherheit.")
        metadata_path = output / f"einstellungen_{run_id}.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if outside:
            print("Hinweis: Ausserhalb der Kalibrier-Knotenschranken (Achse: Anzahl eindeutiger Zusammensetzungen): "
                  + ", ".join(f"{axis}: {count}/{unique_count}" for axis, count in sorted(outside.items())))
            print("Diese Bereiche sind extrapoliert; ein Punkt innerhalb der Schranken ist nicht automatisch validiert.")
        for message in messages:
            print("Modellhinweis: " + message)
        print(f"\nGespeichert in: {output.resolve()}")
        for path in saved:
            print("  " + path.name)
        print("  " + csv_path.name + "\n  " + metadata_path.name)
        import matplotlib.pyplot as plt
        if not args.no_show:
            plt.show()
        for figure in figures:
            plt.close(figure)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nAbgebrochen.")
        return 130
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Benoetigte Pakete: python -m pip install numpy pandas scipy matplotlib", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
