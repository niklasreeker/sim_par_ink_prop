#!/usr/bin/env python3
"""Infer ink composition from manually entered sensor readings.

Place this standalone file in sim_par_ink_prop next to ink_calculator.py:
    python ink_composition_inversion.py

Dependencies (Python >= 3.10): numpy, pandas, scipy.
    python -m pip install numpy pandas scipy

The interactive workflow offers two modes:
  1. Absolute inversion of a single sensor reading.
  2. Fresh-ink reference calibration followed by one or more readings.
Density is entered in kg/m3, sound velocity in m/s, temperature in deg C.
Decimal commas are accepted. BOTH the physics and hybrid models are always
evaluated. The plotting script is not required.

Constraints, expressed as wt-% of the final ink:
    PG = 2 * Al; MG = Al / 8; Water = 100 - Al - PG - MG - IPA.
Al denotes the complete aluminum pigment fraction, including its coating.
Use --pg-per-al and --mg-per-al to change these fixed ratios.

Example without interactive prompts (replace the calibration field path):
    python ink_composition_inversion.py --density 1006.6 --sound 1547.2 \
        --temperature 24.1 --calibration-field PATH/calibration_field.json

For a calibrated measurement series, select mode 2 in the menu, or use:
    python ink_composition_inversion.py --mode series
Additional series arguments:
    --reference-al 1.814 --reference-ipa 3.628
    --reference-density 1006.6 --reference-sound 1547.2
    --reference-temperature 24.1
    --point 1007.0 1543.0 23.5 --point 1008.0 1538.0 24.0
Repeat --point for every follow-up reading; each triple is density, sound,
temperature. Supply --calibration-field PATH for a fully noninteractive run.

Reference calibration requires a KNOWN fresh-ink recipe. Its composition
and sensor offsets cannot both be inferred from the reference reading alone.
For each model separately:
    offset = reference_sensor - model(reference_recipe, reference_temperature)
    sensor_prediction = model(composition, point_temperature) + fixed_offset
The hybrid reference includes A(w) before estimating its sensor offset.
Offsets are fixed throughout the series, not re-estimated at later points.
The reference row is an assigned known recipe, not an inverse estimate.
Follow-up rows report composition, changes from the reference in percentage
points, and relative changes in percent (undefined for a zero reference).
All matches found are retained; no continuity or evaporation constraint is
silently used to select a branch. An effective offset can also absorb model
bias at the reference; it does not establish that the bias is purely a sensor
error or remains constant away from that reference.

By default, the search covers the full nonnegative composition domain
(total mass fraction 100%). Optional limits: --al-min 1 --al-max 3
--ipa-min 0 --ipa-max 10. The same bounds apply to BOTH models.
Multiple starting points are used, and all distinct numerical matches found
are reported. This is not a mathematical proof of completeness or uniqueness.
If no numerical match is found, the best approximation is explicitly labelled
and reported with its remaining sensor residuals.

Default numerical tolerances: 0.0001 kg/m3 and 0.0001 m/s.
These are solver tolerances, NOT sensor or model uncertainties.
Least-squares scaling: 0.1 kg/m3 and 1 m/s, also not an uncertainty estimate.
Absolute mode uses no sensor offset correction. Neither mode performs an
evaporation inversion or infers absolute mass loss from concentrations alone.
Hybrid = InkCalculator(w,T) + A(w); A(w) has no learned temperature term.

Results are printed and saved as timestamped CSV and JSON files under
results/ink_composition_inversion/. Models and source data are unchanged.
Use --repo-root to specify a different project directory.
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

import numpy as np
from scipy.optimize import least_squares


FIELD_RELATIVE = Path("laboratory_measurement_L-Com/results/calibration_field") / (
    "probe_9_10_11__Kennfeld_v2_23__korrigiert__20260928_110809/calibration_field.json"
)
TITLES = {"physics": "Ink Calculator (physics model)",
          "hybrid": "Hybrid calibration field (physics + A(w))"}


@dataclass(frozen=True)
class Sensors:
    density: float
    sound: float
    temperature: float

    def validate(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Sensor readings must be finite.")
        if self.density <= 0 or self.sound <= 0 or self.temperature <= -273.15:
            raise ValueError("Density and sound velocity must be positive; temperature must exceed -273.15 deg C.")


@dataclass(frozen=True)
class Search:
    al_min: float = 0.0
    al_max: float = 32.0
    ipa_min: float = 0.0
    ipa_max: float = 100.0
    pg_per_al: float = 2.0
    mg_per_al: float = .125
    rho_tolerance: float = 1e-4
    sound_tolerance: float = 1e-4
    max_evaluations: int = 250

    @property
    def group_factor(self):
        return 1.0 + self.pg_per_al + self.mg_per_al

    @property
    def feasible_al_max(self):
        return min(self.al_max, (100.0 - self.ipa_min) / self.group_factor)

    def validate(self):
        if not all(math.isfinite(float(value)) for value in asdict(self).values()):
            raise ValueError("Search parameters must be finite.")
        if min(self.al_min, self.ipa_min, self.pg_per_al, self.mg_per_al) < 0:
            raise ValueError("Concentrations and ratios must not be negative.")
        if self.al_max <= self.al_min or self.ipa_max <= self.ipa_min:
            raise ValueError("Upper search bounds must exceed lower bounds.")
        if self.feasible_al_max <= self.al_min or self.ipa_min >= 100:
            raise ValueError("The search bounds do not allow a feasible composition.")
        if min(self.rho_tolerance, self.sound_tolerance) <= 0 or self.max_evaluations < 1:
            raise ValueError("Tolerances and the maximum evaluation count must be positive.")

    def composition(self, unit):
        # Optimize on a unit square mapped to the physical mass-balance polygon.
        # This enforces nonnegative water without a penalty or an invalid model call.
        al = self.al_min + float(unit[0]) * (self.feasible_al_max - self.al_min)
        ipa_upper = max(self.ipa_min, min(self.ipa_max, 100.0 - self.group_factor * al))
        ipa = self.ipa_min + float(unit[1]) * (ipa_upper - self.ipa_min)
        pg, mg = self.pg_per_al * al, self.mg_per_al * al
        # Roundoff at the water-free boundary must not make sum(al,ipa,pg,mg)>100.
        if al + ipa + pg + mg > 100.0:
            ipa = max(self.ipa_min, ipa - (al + ipa + pg + mg - 100.) - 1e-12)
        water = max(0.0, 100.0 - al - ipa - pg - mg)
        return np.array([al, ipa, pg, mg, water], dtype=float)

    def unit_coordinates(self, al, ipa):
        al = float(np.clip(al, self.al_min, self.feasible_al_max))
        upper = max(self.ipa_min, min(self.ipa_max, 100.0 - self.group_factor * al))
        return np.array([(al - self.al_min) / (self.feasible_al_max - self.al_min),
                         float(np.clip((ipa - self.ipa_min) / max(upper - self.ipa_min, 1e-12), 0., 1.))])


def finite_number(text):
    try:
        value = float(text.replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Please enter a number.") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("The number must be finite.")
    return value


def ask_number(prompt, default=None, positive=False):
    while True:
        suffix = f" [{default:g}]" if default is not None else ""
        answer = input(prompt + suffix + ": ").strip()
        try:
            value = float(default) if not answer and default is not None else finite_number(answer)
        except (argparse.ArgumentTypeError, ValueError):
            print("Please enter a finite number, for example 1006.6.")
            continue
        if positive and value <= 0:
            print("The value must be greater than zero.")
            continue
        return value


def find_repo(explicit):
    candidates = [explicit] if explicit else [Path(__file__).resolve().parent, Path.cwd()]
    for candidate in candidates:
        if candidate is not None and (candidate / "ink_calculator.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Place this script next to ink_calculator.py in the project root "
                            "or specify --repo-root PATH.")


def load_calculator(repo: Path):
    path = repo / "ink_calculator.py"
    spec = importlib.util.spec_from_file_location("composition_inversion_ink_calculator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Ink Calculator: {path}")
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
    """Schema-v3 IDW, same neighbors and property-specific quality weights."""

    def __init__(self, path: Path, repo: Path):
        import numpy as np
        import pandas as pd

        self.path = path.resolve()
        self.model = json.loads(path.read_text(encoding="utf-8-sig"))
        if (self.model.get("schema") != "ink-residual-calibration-field"
                or self.model.get("schema_version") != 3):
            raise ValueError("An ink-residual-calibration-field with schema version 3 is required.")
        if not self.model.get("nodes"):
            raise ValueError("The calibration field contains no nodes.")
        self.axes = self.model["composition_axes"]
        allowed = {"Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"}
        if not self.axes or len(set(self.axes)) != len(self.axes) or set(self.axes) - allowed:
            raise ValueError("Invalid composition axes in the calibration field.")
        self.nodes = pd.DataFrame(self.model["nodes"])
        self.coordinates = self.nodes[self.axes].to_numpy(float)
        self.minimum = self.coordinates.min(axis=0)
        self.maximum = self.coordinates.max(axis=0)
        interpolation = self.model["interpolation"]
        self.scale = np.asarray(interpolation["scale"], dtype=float)
        self.power = float(interpolation.get("power", 2.0))
        requested = int(interpolation.get("neighbors", 4))
        self.count = len(self.nodes) if requested <= 0 else min(requested, len(self.nodes))
        if (self.scale.shape != (len(self.axes),) or np.any(~np.isfinite(self.scale))
                or np.any(self.scale <= 0) or not math.isfinite(self.power) or self.power <= 0
                or not np.isfinite(self.coordinates).all()):
            raise ValueError("Invalid IDW parameters or node coordinates.")
        self.properties = []
        for value_col, se_col in [("A_Rho_kg_m3", "A_Rho_SE_kg_m3"), ("A_C_m_s", "A_C_SE_m_s")]:
            values = self.nodes[value_col].to_numpy(float)
            se = pd.to_numeric(self.nodes[se_col], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite correction values: {value_col}")
            self.properties.append((values, se))
        provenance = self.model.get("calculator", {})
        inputs = {repo / "ink_calculator.py": provenance.get("sha256")}
        inputs.update({repo / "tables_parameters" / name: expected
                       for name, expected in provenance.get("table_sha256", {}).items()})
        for file, expected in inputs.items():
            if expected and expected not in equivalent_hashes(file):
                raise ValueError(f"The calibration field does not match {file.name}. "
                                 "Use the corresponding model version or rebuild the A(w) calibration.")

    def evaluate(self, al: float, ipa: float, pg: float, mg: float):
        import numpy as np

        lookup = dict(zip(("Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct"), (al, ipa, pg, mg)))
        target = np.asarray([lookup[axis] for axis in self.axes])
        distances = np.linalg.norm((self.coordinates - target) / self.scale, axis=1)
        indexes = np.argsort(distances)[:self.count]
        if distances[indexes[0]] < 1e-12:
            indexes = indexes[:1]
            geometric = np.ones(1)
        else:
            geometric = 1.0 / np.maximum(distances[indexes], 1e-12) ** self.power
        estimates, uncertainties = [], []
        for values_all, se_all in self.properties:
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


def choose_field(repo, explicit=None, interactive=True):
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            # A CLI-relative path is relative to cwd; fall back to repository.
            path = path if path.exists() else repo / path
        path = path / "calibration_field.json" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(f"Calibration field not found: {path}")
        return path.resolve()
    directory = repo / "laboratory_measurement_L-Com/results/calibration_field"
    paths = sorted(directory.glob("**/calibration_field.json")) if directory.is_dir() else []
    preferred = repo / FIELD_RELATIVE
    if preferred in paths:
        paths.remove(preferred)
        paths.insert(0, preferred)
    if not interactive:
        if preferred.is_file():
            return preferred.resolve()
        raise FileNotFoundError("Please specify --calibration-field PATH.")
    print("\nWhich hybrid calibration field should be used?")
    for number, path in enumerate(paths, 1):
        print(f"  {number}) {path.parent.name}")
    print("  M) Enter the path to another calibration field")
    while True:
        answer = input("Selection [1]: " if paths else "Path to calibration_field.json: ").strip()
        if paths and not answer:
            return paths[0].resolve()
        if paths and answer.isdigit() and 1 <= int(answer) <= len(paths):
            return paths[int(answer) - 1].resolve()
        if paths and answer.lower() == "m":
            answer = input("Path to calibration_field.json: ").strip()
        try:
            if not answer:
                raise ValueError("Please enter a path.")
            return choose_field(repo, Path(answer.strip('"')), interactive=False)
        except (OSError, ValueError) as exc:
            print(exc)


class ForwardModel:
    def __init__(self, calculator, sensors, field=None, offset=(0., 0.)):
        self.calculator, self.sensors, self.field = calculator, sensors, field
        self.offset = np.asarray(offset, dtype=float)
        if self.offset.shape != (2,) or not np.isfinite(self.offset).all():
            raise ValueError("The sensor offset must contain two finite values: density and sound velocity.")
        self.messages = set()
        self.failed_calls = 0

    def predict(self, composition):
        al, ipa, pg, mg, _ = composition
        arguments = dict(al=float(al), ipa=float(ipa), pg=float(pg), mg=float(mg),
                         temperature=self.sensors.temperature)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            physics = np.array([1000. * self.calculator.density(**arguments),
                                self.calculator.sound_velocity(**arguments)], dtype=float)
        self.messages.update(str(item.message) for item in caught)
        if not np.isfinite(physics).all():
            raise ValueError("Nonfinite model predictions.")
        correction, uncertainty, outside = self.field.evaluate(al, ipa, pg, mg) if self.field else (
            [0., 0.], [0., 0.], [])
        predicted = physics + np.asarray(correction) + self.offset
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite hybrid predictions.")
        return predicted, physics, correction, uncertainty, outside


def starting_points(search, field=None, extra=()):
    # Cover the full domain and add dense starts in the water-rich ink region.
    starts = [np.array([a, b]) for a in np.linspace(0., 1., 7)
              for b in np.linspace(0., 1., 9)]
    starts += [search.unit_coordinates(al, ipa) for al in (0., 1., 1.8, 3., 5.)
               for ipa in (0., 2., 4., 8., 15.)]
    if field:
        starts += [search.unit_coordinates(node["Al_wt_pct"], node["IPA_wt_pct"])
                   for node in field.model["nodes"]]
    starts += [search.unit_coordinates(al, ipa) for al, ipa in extra]
    unique = []
    for start in starts:
        if not any(np.linalg.norm(start - previous) < 1e-9 for previous in unique):
            unique.append(start)
    return unique


def invert(model, search, extra=(), progress=True):
    search.validate()
    target = np.array([model.sensors.density, model.sensors.sound])
    scales = np.array([.1, 1.])
    roots, best = [], None
    starts = starting_points(search, model.field, extra)
    succeeded = 0

    def residual(unit):
        predicted, *_ = model.predict(search.composition(unit))
        return (predicted - target) / scales

    for index, start in enumerate(starts):
        try:
            fit = least_squares(residual, start, bounds=(np.zeros(2), np.ones(2)),
                                x_scale="jac", max_nfev=search.max_evaluations,
                                ftol=1e-11, xtol=1e-11, gtol=1e-11)
            composition = search.composition(fit.x)
            predicted, physics, correction, uncertainty, outside = model.predict(composition)
            difference = predicted - target
            score = float(np.linalg.norm(difference / scales))
            exact = bool(abs(difference[0]) <= search.rho_tolerance
                         and abs(difference[1]) <= search.sound_tolerance)
            result = dict(composition=dict(zip(("Al", "IPA", "PG", "MG", "Water"), map(float, composition))),
                          solids_wt_pct=float(composition[0] + composition[3]),
                          density=float(predicted[0]), sound=float(predicted[1]),
                          density_difference=float(difference[0]), sound_difference=float(difference[1]),
                          physics_density=float(physics[0]), physics_sound=float(physics[1]),
                          correction_density=float(correction[0]), correction_sound=float(correction[1]),
                          correction_density_spread=float(uncertainty[0]), correction_sound_spread=float(uncertainty[1]),
                          outside_axes=outside, numerical_match=exact,
                          optimizer_converged=bool(fit.success), scaled_residual=score)
            result.update(sensor_offset_density=float(model.offset[0]), sensor_offset_sound=float(model.offset[1]),
                          base_model_density=float(predicted[0] - model.offset[0]),
                          base_model_sound=float(predicted[1] - model.offset[1]),
                          offset_corrected_sensor_density=float(target[0] - model.offset[0]),
                          offset_corrected_sensor_sound=float(target[1] - model.offset[1]))
            succeeded += 1
            if best is None or score < best["scaled_residual"]:
                best = result
            if exact:
                duplicate = next((i for i, previous in enumerate(roots)
                                  if max(abs(previous["composition"][name] - result["composition"][name])
                                         for name in ("Al", "IPA")) < 1e-4), None)
                if duplicate is None:
                    roots.append(result)
                elif score < roots[duplicate]["scaled_residual"]:
                    roots[duplicate] = result
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            model.failed_calls += 1
            model.messages.add(str(exc))
        if progress and (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(starts)} starting points checked; {len(roots)} numerical match(es).", flush=True)
    roots.sort(key=lambda row: (row["composition"]["Al"], row["composition"]["IPA"]))
    status = "multiple_matches" if len(roots) > 1 else "one_match_found" if roots else (
        "approximation_only" if best is not None else "no_valid_evaluation")
    return dict(status=status, solutions=roots, best_approximation=best if not roots else None,
                starts_tried=len(starts), starts_with_result=succeeded,
                failed_starts=model.failed_calls, model_messages=sorted(model.messages),
                search_complete_proof=False)


def print_result(name, report):
    print("\n" + TITLES[name])
    print("-" * 85)
    if "error" in report:
        print("Not calculated: " + report["error"])
        return
    status = report["status"]
    if status == "no_valid_evaluation":
        print("No valid calculation. See the result file for model messages.")
        return
    if status == "multiple_matches":
        print(f"{len(report['solutions'])} distinct numerical matches found; the composition is not unique.")
    elif status == "one_match_found":
        print("One numerical match found (not a proof of global uniqueness).")
    else:
        print("NO numerical match found. Best approximation found:")
    solutions = report["solutions"] or [report["best_approximation"]]
    print(f"{'No.':>4} {'Al [%]':>12} {'IPA [%]':>12} {'PG [%]':>12} {'MG [%]':>12} {'Water [%]':>12}")
    for index, result in enumerate(solutions, 1):
        comp = result["composition"]
        print(f"{index:>4}" + "".join(f"{comp[key]:>13.6f}" for key in ("Al", "IPA", "PG", "MG", "Water")))
        print(f"     Solids content (Al + MG): {solids_content(comp):.6f} wt-%")
        print(f"     Sensor prediction (model + offset): rho={result['density']:.6f} kg/m3; c={result['sound']:.6f} m/s")
        print(f"     Model minus sensor: {result['density_difference']:+.6g} kg/m3; "
              f"{result['sound_difference']:+.6g} m/s")
        if result["outside_axes"]:
            print("     Outside the calibration node bounds: " + ", ".join(result["outside_axes"]))
        if "change_from_reference_pp" in result:
            print("     Change from fresh ink [percentage points]:")
            print("     " + "; ".join(f"{key}: {value:+.6f}" for key, value in result["change_from_reference_pp"].items()))
            factor = result["al_concentration_factor"]
            if factor is not None:
                print(f"     Al concentration factor relative to fresh ink: {factor:.6f}")
    if report["failed_starts"]:
        print(f"     {report['failed_starts']} starting points could not be evaluated; "
              "see the result file for details.")


def solids_content(composition):
    """Return solids content (Al + MG) in weight percent."""
    return float(composition["Al"] + composition["MG"])


def reference_composition(al, ipa, search):
    """Expand a known fresh recipe; do not clip it to the inverse search bounds."""
    values = [al, ipa, search.pg_per_al * al, search.mg_per_al * al]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Fresh-ink concentrations must be finite and nonnegative.")
    total = sum(values)
    if total > 100.:
        raise ValueError("The fresh recipe exceeds 100 wt-% after adding PG and MG.")
    return dict(zip(("Al", "IPA", "PG", "MG", "Water"), values + [100. - total]))


def calibrate_reference(calculator, sensors, composition, field=None):
    """Estimate constant effective offsets as measured minus unshifted model."""
    sensors.validate()
    model = ForwardModel(calculator, sensors, field)
    values = np.array([composition[key] for key in ("Al", "IPA", "PG", "MG", "Water")])
    predicted, physics, correction, uncertainty, outside = model.predict(values)
    offset = np.array([sensors.density, sensors.sound]) - predicted
    return dict(offset_density=float(offset[0]), offset_sound=float(offset[1]),
                reference_prediction_density=float(predicted[0]), reference_prediction_sound=float(predicted[1]),
                reference_sensors=asdict(sensors), reference_composition=composition,
                reference_solids_wt_pct=solids_content(composition),
                reference_outside_axes=outside, reference_is_assigned_recipe=True,
                reference_correction_density=float(correction[0]), reference_correction_sound=float(correction[1]),
                model_messages=sorted(model.messages))


def add_reference_changes(report, reference):
    """Keep all inverse candidates and attach their changes from fresh ink."""
    candidates = report.get("solutions") or ([report["best_approximation"]]
                  if report.get("best_approximation") is not None else [])
    for result in candidates:
        composition = result["composition"]
        result["change_from_reference_pp"] = {
            key: float(composition[key] - reference[key]) for key in reference}
        result["relative_change_from_reference_pct"] = {
            key: float(100. * (composition[key] - reference[key]) / reference[key])
            if reference[key] != 0 else None for key in reference}
        result["al_concentration_factor"] = (
            float(composition["Al"] / reference["Al"]) if reference["Al"] != 0 else None)


def invert_calibrated_series(calculator, reference_sensors, reference_al, reference_ipa,
                             measurements, search, field=None, progress=True, hybrid_error=None):
    """Anchor each model once, then infer each follow-up at its own temperature.

    This reusable function takes a Sensors reference, known fresh Al/IPA
    concentrations, and a sequence of Sensors follow-ups. Each model gets its
    own offset because the hybrid reference prediction already includes A(w).
    No reference inversion, offset update, or automatic branch selection occurs.
    """
    search.validate()
    reference_sensors.validate()
    measurements = list(measurements)
    if not measurements:
        raise ValueError("A calibrated series requires at least one follow-up point.")
    for sensors in measurements:
        sensors.validate()
    reference = reference_composition(reference_al, reference_ipa, search)
    result = dict(reference=dict(label="P0", sensors=asdict(reference_sensors), composition=reference,
                                 solids_wt_pct=solids_content(reference)),
                  models={}, points=[])
    for name in ("physics", "hybrid"):
        if name == "hybrid" and field is None:
            result["models"][name] = dict(error=hybrid_error or "No hybrid calibration field is available.")
            continue
        try:
            calibration = calibrate_reference(calculator, reference_sensors, reference,
                                              field if name == "hybrid" else None)
            result["models"][name] = dict(calibration=calibration)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            result["models"][name] = dict(error=str(exc))
    previous_candidates = {name: [(reference_al, reference_ipa)] for name in result["models"]}
    for index, sensors in enumerate(measurements, 1):
        point = dict(label=f"P{index}", sensors=asdict(sensors), models={})
        for name, model_info in result["models"].items():
            if "error" in model_info:
                point["models"][name] = dict(status="error", error=model_info["error"])
                continue
            calibration = model_info["calibration"]
            offset = (calibration["offset_density"], calibration["offset_sound"])
            if progress:
                print(f"\nCalculating {point['label']}: {TITLES[name]} ...", flush=True)
            model = ForwardModel(calculator, sensors, field if name == "hybrid" else None, offset=offset)
            report = invert(model, search, extra=previous_candidates[name], progress=progress)
            add_reference_changes(report, reference)
            point["models"][name] = report
            candidates = report["solutions"] or ([report["best_approximation"]]
                         if report["best_approximation"] is not None else [])
            previous_candidates[name] = [(reference_al, reference_ipa)] + [
                (row["composition"]["Al"], row["composition"]["IPA"]) for row in candidates]
        result["points"].append(point)
    return result


def print_series(series):
    reference = series["reference"]
    print("\nP0: fresh-ink reference (assigned known recipe; not an inverse estimate)")
    print("  " + "; ".join(f"{key}: {value:.6f} wt-%" for key, value in reference["composition"].items()))
    print(f"  Solids content (Al + MG): {solids_content(reference['composition']):.6f} wt-%")
    for name, info in series["models"].items():
        if "error" in info:
            print(f"  {TITLES[name]}: {info['error']}")
            continue
        calibration = info["calibration"]
        print(f"  {TITLES[name]} fixed offsets (sensor minus model): "
              f"{calibration['offset_density']:+.6f} kg/m3; {calibration['offset_sound']:+.6f} m/s")
        if calibration["reference_outside_axes"]:
            print("    Reference outside calibration node bounds: " + ", ".join(calibration["reference_outside_axes"]))
    for point in series["points"]:
        sensors = point["sensors"]
        print(f"\n{point['label']}: rho={sensors['density']:.6f} kg/m3, "
              f"c={sensors['sound']:.6f} m/s, T={sensors['temperature']:g} deg C")
        for name, report in point["models"].items():
            print_result(name, report)


def save_series_results(repo, output, reference_sensors, search, series, field_path):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = [repo / "ink_calculator.py", *sorted((repo / "tables_parameters").glob("*.csv"))]
    if field_path is not None and field_path.is_file():
        paths.append(field_path)
    payload = dict(mode="series", search=asdict(search), series=series,
                   calibration_field=str(field_path) if field_path else None,
                   source_sha256={str(path): sha256(path) for path in paths},
                   units=dict(composition="wt-%", change="percentage points", relative_change="%",
                              density="kg/m3", sound="m/s", temperature="deg C"),
                   offset_definition="reference sensor minus unshifted model, separately for each model",
                   offsets_constant_across_points=True, numerical_tolerance_is_measurement_uncertainty=False)
    json_path = output / f"composition_series_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    components = ("Al", "IPA", "PG", "MG", "Water")
    columns = ["Point", "Model", "Status", "Solution", "Reference_Is_Assigned_Recipe",
               *[key + "_wt_pct" for key in components], "Solids_wt_pct",
               *["Delta_" + key + "_pp" for key in components],
               *["Relative_" + key + "_change_pct" for key in components],
               "Al_Concentration_Factor", "Input_Rho_kg_m3", "Input_C_m_s", "Temperature_C",
               "Sensor_Offset_Rho_kg_m3", "Sensor_Offset_C_m_s", "Corrected_Input_Rho_kg_m3", "Corrected_Input_C_m_s",
               "Base_Model_Rho_kg_m3", "Base_Model_C_m_s", "Predicted_Sensor_Rho_kg_m3", "Predicted_Sensor_C_m_s",
               "Delta_Rho_kg_m3", "Delta_C_m_s", "Outside_Axes", "Error"]
    csv_path = output / f"composition_series_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, info in series["models"].items():
            if "error" in info:
                writer.writerow(dict(Point="P0", Model=name, Status="error", Error=info["error"]))
                continue
            calibration = info["calibration"]
            reference = series["reference"]["composition"]
            row = dict(Point="P0", Model=name, Status="reference_anchor", Solution=1, Reference_Is_Assigned_Recipe=True,
                       Input_Rho_kg_m3=reference_sensors.density, Input_C_m_s=reference_sensors.sound,
                       Temperature_C=reference_sensors.temperature,
                       Sensor_Offset_Rho_kg_m3=calibration["offset_density"], Sensor_Offset_C_m_s=calibration["offset_sound"],
                       Corrected_Input_Rho_kg_m3=calibration["reference_prediction_density"],
                       Corrected_Input_C_m_s=calibration["reference_prediction_sound"],
                       Base_Model_Rho_kg_m3=calibration["reference_prediction_density"],
                       Base_Model_C_m_s=calibration["reference_prediction_sound"],
                       Predicted_Sensor_Rho_kg_m3=reference_sensors.density, Predicted_Sensor_C_m_s=reference_sensors.sound,
                       Delta_Rho_kg_m3=0., Delta_C_m_s=0., Outside_Axes=",".join(calibration["reference_outside_axes"]),
                       Al_Concentration_Factor=1. if reference["Al"] != 0 else "",
                       Solids_wt_pct=solids_content(reference))
            for key in components:
                row[key + "_wt_pct"] = reference[key]
                row["Delta_" + key + "_pp"] = 0.
                row["Relative_" + key + "_change_pct"] = 0. if reference[key] != 0 else ""
            writer.writerow(row)
        for point in series["points"]:
            sensors = point["sensors"]
            for name, report in point["models"].items():
                candidates = report.get("solutions") or ([report["best_approximation"]]
                             if report.get("best_approximation") is not None else [None])
                for index, result in enumerate(candidates, 1):
                    row = dict(Point=point["label"], Model=name, Status=report["status"], Solution=index if result else "",
                               Reference_Is_Assigned_Recipe=False, Input_Rho_kg_m3=sensors["density"],
                               Input_C_m_s=sensors["sound"], Temperature_C=sensors["temperature"], Error=report.get("error", ""))
                    if result:
                        for key in components:
                            row[key + "_wt_pct"] = result["composition"][key]
                            row["Delta_" + key + "_pp"] = result["change_from_reference_pp"][key]
                            row["Relative_" + key + "_change_pct"] = result["relative_change_from_reference_pct"][key]
                        row.update(Solids_wt_pct=solids_content(result["composition"]),
                                   Al_Concentration_Factor=result["al_concentration_factor"],
                                   Sensor_Offset_Rho_kg_m3=result["sensor_offset_density"], Sensor_Offset_C_m_s=result["sensor_offset_sound"],
                                   Corrected_Input_Rho_kg_m3=result["offset_corrected_sensor_density"],
                                   Corrected_Input_C_m_s=result["offset_corrected_sensor_sound"],
                                   Base_Model_Rho_kg_m3=result["base_model_density"], Base_Model_C_m_s=result["base_model_sound"],
                                   Predicted_Sensor_Rho_kg_m3=result["density"], Predicted_Sensor_C_m_s=result["sound"],
                                   Delta_Rho_kg_m3=result["density_difference"], Delta_C_m_s=result["sound_difference"],
                                   Outside_Axes=",".join(result["outside_axes"]))
                    writer.writerow(row)
    return json_path, csv_path


def ask_mode():
    print("\nWhat would you like to do?")
    print("  1) Absolute composition from a single sensor reading")
    print("  2) Calibrate with fresh ink and evaluate follow-up points")
    while True:
        answer = input("Selection [2]: ").strip() or "2"
        if answer in ("1", "2"):
            return "single" if answer == "1" else "series"
        print("Please enter 1 or 2.")


def ask_sensors(label, density=None, sound=None, temperature=None, default_temperature=23.):
    print("\n" + label)
    while True:
        sensors = Sensors(density if density is not None else ask_number("Density [kg/m3]", positive=True),
                          sound if sound is not None else ask_number("Sound velocity [m/s]", positive=True),
                          temperature if temperature is not None else ask_number("Temperature [deg C]", default_temperature))
        try:
            sensors.validate()
            return sensors
        except ValueError as exc:
            if any(value is not None for value in (density, sound, temperature)):
                raise
            print(exc)


def collect_series_inputs(args, search):
    print("\nFresh-ink calibration requires a known recipe; a reference reading alone is insufficient.")
    while True:
        al = args.reference_al if args.reference_al is not None else ask_number(
            "Known fresh-ink Al [wt-%]", 180. / 9922.5 * 100.)
        ipa = args.reference_ipa if args.reference_ipa is not None else ask_number(
            "Known fresh-ink IPA [wt-%]", 360. / 9922.5 * 100.)
        try:
            composition = reference_composition(al, ipa, search)
            break
        except ValueError as exc:
            if args.reference_al is not None or args.reference_ipa is not None:
                raise
            print(exc)
    print("Assigned fresh recipe: " + "; ".join(f"{key}={value:.6f} wt-%" for key, value in composition.items()))
    print(f"Solids content (Al + MG): {solids_content(composition):.6f} wt-%")
    reference = ask_sensors("P0: fresh-ink reference sensor reading", args.reference_density,
                            args.reference_sound, args.reference_temperature)
    if args.point:
        points = [Sensors(*values) for values in args.point]
        for sensors in points:
            sensors.validate()
    else:
        while True:
            answer = input("\nNumber of follow-up points [1]: ").strip() or "1"
            if answer.isdigit() and int(answer) >= 1:
                count = int(answer)
                break
            print("Please enter a positive integer.")
        points, temperature = [], reference.temperature
        for index in range(1, count + 1):
            sensors = ask_sensors(f"P{index}: follow-up sensor reading", default_temperature=temperature)
            points.append(sensors)
            temperature = sensors.temperature
    return reference, al, ipa, points


def save_results(repo, output, sensors, search, reports, field_path):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = [repo / "ink_calculator.py", *sorted((repo / "tables_parameters").glob("*.csv"))]
    if field_path is not None and field_path.is_file():
        paths.append(field_path)
    payload = dict(sensors=asdict(sensors), search=asdict(search),
                   calibration_field=str(field_path) if field_path else None,
                   source_sha256={str(path): sha256(path) for path in paths}, models=reports,
                   units=dict(composition="wt-%", density="kg/m3", sound="m/s", temperature="deg C"),
                   numerical_tolerance_is_measurement_uncertainty=False)
    json_path = output / f"composition_inversion_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["Model", "Status", "Solution", "Al_wt_pct", "IPA_wt_pct", "PG_wt_pct", "MG_wt_pct", "Water_wt_pct",
               "Solids_wt_pct",
               "Input_Rho_kg_m3", "Input_C_m_s", "Temperature_C", "Predicted_Rho_kg_m3", "Predicted_C_m_s",
               "Delta_Rho_kg_m3", "Delta_C_m_s", "Outside_Axes", "Error"]
    csv_path = output / f"composition_inversion_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, report in reports.items():
            solutions = report.get("solutions") or ([report["best_approximation"]]
                         if report.get("best_approximation") is not None else [None])
            for index, result in enumerate(solutions, 1):
                row = dict(Model=name, Status=report.get("status", "error"), Solution=index if result else "",
                           Input_Rho_kg_m3=sensors.density, Input_C_m_s=sensors.sound,
                           Temperature_C=sensors.temperature, Error=report.get("error", ""))
                if result:
                    row.update({key + "_wt_pct": value for key, value in result["composition"].items()})
                    row.update(Solids_wt_pct=solids_content(result["composition"]),
                               Predicted_Rho_kg_m3=result["density"], Predicted_C_m_s=result["sound"],
                               Delta_Rho_kg_m3=result["density_difference"], Delta_C_m_s=result["sound_difference"],
                               Outside_Axes=",".join(result["outside_axes"]))
                writer.writerow(row)
    return json_path, csv_path


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--mode", choices=("single", "series"), help="Absolute inversion or fresh-ink calibrated series")
    result.add_argument("--density", "--rho", dest="density", type=finite_number, help="Density [kg/m3]")
    result.add_argument("--sound", type=finite_number, help="Sound velocity [m/s]")
    result.add_argument("--temperature", type=finite_number, help="Temperature [deg C]")
    result.add_argument("--reference-al", type=finite_number, help="Known fresh-ink Al [wt-%%]")
    result.add_argument("--reference-ipa", type=finite_number, help="Known fresh-ink IPA [wt-%%]")
    result.add_argument("--reference-density", type=finite_number, help="Fresh-ink reference density [kg/m3]")
    result.add_argument("--reference-sound", type=finite_number, help="Fresh-ink reference sound velocity [m/s]")
    result.add_argument("--reference-temperature", type=finite_number, help="Fresh-ink reference temperature [deg C]")
    result.add_argument("--point", type=finite_number, nargs=3, action="append",
                        metavar=("DENSITY", "SOUND", "TEMPERATURE"), help="Follow-up reading; repeat for multiple points")
    result.add_argument("--calibration-field", type=Path)
    result.add_argument("--repo-root", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--pg-per-al", type=finite_number, default=2.)
    result.add_argument("--mg-per-al", type=finite_number, default=.125)
    result.add_argument("--al-min", type=finite_number, default=0.)
    result.add_argument("--al-max", type=finite_number, help="Default: 100/(1+PG/Al+MG/Al)")
    result.add_argument("--ipa-min", type=finite_number, default=0.)
    result.add_argument("--ipa-max", type=finite_number, default=100.)
    result.add_argument("--rho-tolerance", type=finite_number, default=1e-4)
    result.add_argument("--sound-tolerance", type=finite_number, default=1e-4)
    result.add_argument("--max-evaluations", type=int, default=250)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        repo = find_repo(args.repo_root)
        factor = 1 + args.pg_per_al + args.mg_per_al
        if args.pg_per_al < 0 or args.mg_per_al < 0:
            raise ValueError("PG/Al and MG/Al must not be negative.")
        search = Search(args.al_min, args.al_max if args.al_max is not None else 100. / factor,
                        args.ipa_min, args.ipa_max, args.pg_per_al, args.mg_per_al,
                        args.rho_tolerance, args.sound_tolerance, args.max_evaluations)
        search.validate()
        reference_values = (args.reference_al, args.reference_ipa, args.reference_density,
                            args.reference_sound, args.reference_temperature)
        series_requested = bool(args.point) or any(value is not None for value in reference_values)
        single_requested = any(value is not None for value in (args.density, args.sound, args.temperature))
        mode = args.mode or ("series" if series_requested else "single" if single_requested else ask_mode())
        if mode == "series" and single_requested:
            raise ValueError("Series mode uses --reference-density/--reference-sound/--reference-temperature "
                             "and --point, not the single-reading options.")
        if mode == "single" and series_requested:
            raise ValueError("Reference and --point arguments require --mode series.")
        print("\nInk composition inversion using BOTH models")
        print(f"PG = {search.pg_per_al:g} * Al; MG = {search.mg_per_al:g} * Al; Water = remainder to 100%.")
        if mode == "series":
            series_inputs = collect_series_inputs(args, search)
            interactive = any(value is None for value in reference_values) or not args.point
        else:
            sensors = ask_sensors("Single sensor reading", args.density, args.sound, args.temperature)
            interactive = any(value is None for value in (args.density, args.sound, args.temperature))
        path, field, field_error = None, None, None
        try:
            path = choose_field(repo, args.calibration_field, interactive=interactive)
            field = CalibrationField(path, repo)
            print(f"Calibration field: {path}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            field_error = str(exc)
            print("Hybrid model unavailable: " + field_error)
        calculator = load_calculator(repo)
        print(f"Search domain: Al {search.al_min:g}–{search.feasible_al_max:g} %, "
              f"IPA {search.ipa_min:g}–{min(100., search.ipa_max):g} %; "
              "additionally constrained by Water >= 0.")
        if mode == "series":
            reference, al, ipa, points = series_inputs
            series = invert_calibrated_series(calculator, reference, al, ipa, points, search,
                                              field=field, hybrid_error=field_error)
            print_series(series)
            paths = save_series_results(repo, args.output_dir or repo / "results/ink_composition_inversion",
                                         reference, search, series, path)
            print("\nResults saved:")
            for saved in paths:
                print("  " + str(saved.resolve()))
            return 0 if all(report["status"] in ("one_match_found", "multiple_matches")
                            for point in series["points"] for report in point["models"].values()) else 2
        reports = {}
        for name in ("physics", "hybrid"):
            if name == "hybrid" and field is None:
                reports[name] = dict(status="error", error=field_error)
                print_result(name, reports[name])
                continue
            print("\nCalculating " + TITLES[name] + " ...", flush=True)
            extra = [(row["composition"]["Al"], row["composition"]["IPA"])
                     for row in reports.get("physics", {}).get("solutions", [])] if name == "hybrid" else []
            model = ForwardModel(calculator, sensors, field if name == "hybrid" else None)
            reports[name] = invert(model, search, extra=extra)
            print_result(name, reports[name])
        paths = save_results(repo, args.output_dir or repo / "results/ink_composition_inversion",
                             sensors, search, reports, path)
        print("\nResults saved:")
        for saved in paths:
            print("  " + str(saved.resolve()))
        return 0 if all(report["status"] in ("one_match_found", "multiple_matches") for report in reports.values()) else 2
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
