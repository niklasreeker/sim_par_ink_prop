"""
ink_calculator.py
=====================================================================
Unified property calculator for a waterborne aluminum-pigment ink:

        encapsulated Al pigment  +  Water  +  IPA (2-propanol)  +  PG (propane-1,2-diol)

A single, reusable entry point that computes four physical properties
from the same composition (mass %) and temperature (deg C):

        1. density            [g/cm3]     (volume-contraction aware)
        2. refractive index   [nD]        (liquid matrix, Gladstone-Dale)
        3. sound velocity     [m/s]       (Wood / Urick effective medium)
        4. viscosity          [mPa.s]     (log-additive carrier + suspension)

This module merges four previously separate scripts
(ink_density.py, ink_refractive.py, ink_sound.py, ink_viscosity.py)
into one consistent interface.

Density, refractive index and viscosity use the same robust interpolation
order: shape-preserving PCHIP over concentration first, followed by a
local linear interpolation over temperature. Sparse temperature columns
are ignored locally when their surrounding concentration gap is too wide.

The SOUND model was revised (v3):
    * binary tables are stored as deviations from pure water,
      c(w, T) = c_water(T) + dc(w, T), with c_water(T) from the
      Marczak (1997) reference curve. Sparse temperature anchors only
      have to describe the slowly varying mixing deviation; the exact
      water curve carries the main temperature trend. Absolute
      calibration offsets of a data set (e.g. the 0% anchor of
      pg_sound.csv, 1498.0 vs. the true 1496.69 m/s) cancel out.
    * concentration interpolation uses the best-supported reference
      column (normally 25 C) and shape-preserving PCHIP.
    * the concentration-dependent temperature coefficient is fitted
      separately from CSV rows with actual multi-temperature values.
      Sparse 20 C columns can no longer create a fictitious low-
      concentration curve across a large internal data gap.
    * internal concentration gaps are detected and reported as model
      warnings instead of being mistaken for fully supported data.
    * additive calibration:  ink.calibrate_sound(measured_c, ...)
    * quick sanity checks:   ink.sound_calc.self_test()

---------------------------------------------------------------------
REQUIRED DATA FILES  (folder: tables_parameters)
---------------------------------------------------------------------
    ipa_density.csv     pg_density.csv      ->  Mass_Percent, Density_XXC ...      [g/cm3]
    ipa_refractive.csv  pg_refractive.csv   ->  Mass_Percent, Refractive_XXC ...   [nD]
    ipa_sound.csv       pg_sound.csv        ->  Mass_Percent, SoundVelocity_XXC .. [m/s]
    ipa_viscosity.csv   pg_viscosity.csv    ->  <mass%>, Viscosity_XXC ...         [mPa.s]

---------------------------------------------------------------------
QUICK START
---------------------------------------------------------------------
    from ink_calculator import InkCalculator

    ink = InkCalculator(tables_dir="tables_parameters")

    # all four properties at once:
    props = ink.compute(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    print(props)

    # or each property on its own (returns a plain number):
    rho  = ink.density(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    n_d  = ink.refractive_index(ipa=3.64, pg=3.64, temperature=25.0)
    c    = ink.sound_velocity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    eta  = ink.viscosity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)

Convention: water is the remainder, water = 100 - al - ipa - pg.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator


# =====================================================================
#  Shared material constants  (all overridable)
# =====================================================================
RHO_WATER = 0.998       # g/cm3  (~20-25 C)
RHO_IPA = 0.785         # g/cm3
RHO_PG = 1.036          # g/cm3
RHO_PIGMENT = 2.700     # g/cm3   intrinsic Al density, NOT powder packing density
BULK_MODULUS_ALUMINUM = 76.0e9   # Pa   (compressibility beta_Al = 1 / K_Al)


# =====================================================================
#  Shared interpolation for density, refractive index and viscosity
# =====================================================================
@dataclass(frozen=True)
class TableInterpolationResult:
    """Value and diagnostics returned by a tabulated-property lookup."""

    value: float
    temperatures_used: tuple
    concentration_outside: bool = False
    temperature_outside: bool = False
    weak_local_support: bool = False


class PchipTemperatureTable:
    """Robust interpolation of a sparse concentration/temperature CSV.

    Interpolation order is always:

        1. shape-preserving PCHIP over mass percent in each usable column,
        2. local linear interpolation over temperature.

    A temperature column is considered usable only when its largest internal
    concentration gap does not exceed ``max_local_gap_wt_pct`` and the target
    concentration lies inside its measured range. This stable, column-level
    quality decision avoids discontinuities that would otherwise occur when
    a sparse curve switches on and off across concentration. It also prevents
    a column with, for example, only 0 and 52 wt% in the operating range from
    overriding two nearby, well-supported temperature curves.

    Duplicate concentrations are combined by their median. Non-numeric and
    missing cells are ignored. Concentrations outside a column are clamped;
    temperature extrapolation uses the nearest local slope and is limited to
    ``max_temperature_extrapolation_C`` beyond the supported range.
    """

    def __init__(self, path, prefix, max_local_gap_wt_pct=20.0,
                 max_temperature_extrapolation_C=10.0,
                 minimum_value=None):
        self.path = str(path)
        self.prefix = prefix
        self.max_local_gap_wt_pct = float(max_local_gap_wt_pct)
        self.max_temperature_extrapolation_C = float(
            max_temperature_extrapolation_C)
        self.minimum_value = minimum_value

        frame = pd.read_csv(path)
        if "Mass_Percent" not in frame.columns:
            raise ValueError(f"'{path}' has no 'Mass_Percent' column.")

        mass_percent = pd.to_numeric(
            frame["Mass_Percent"], errors="coerce")
        pattern = re.compile(rf"{re.escape(prefix)}_(\d+(?:\.\d+)?)C")
        self.columns = {}

        for column_name in frame.columns:
            match = pattern.fullmatch(column_name)
            if not match:
                continue
            temperature = float(match.group(1))
            values = pd.to_numeric(frame[column_name], errors="coerce")
            valid = np.isfinite(mass_percent) & np.isfinite(values)
            if valid.sum() < 2:
                continue

            points = pd.DataFrame({
                "mass_percent": mass_percent[valid].astype(float),
                "value": values[valid].astype(float),
            })
            points = (points.groupby("mass_percent", as_index=False)["value"]
                      .median().sort_values("mass_percent"))
            x = points["mass_percent"].to_numpy(float)
            y = points["value"].to_numpy(float)
            if len(x) < 2 or np.any(np.diff(x) <= 0.0):
                continue
            if minimum_value is not None and np.any(y <= minimum_value):
                raise ValueError(
                    f"'{path}', column '{column_name}' contains values "
                    f"<= {minimum_value}.")

            self.columns[temperature] = {
                "x": x,
                "y": y,
                "interp": PchipInterpolator(x, y, extrapolate=False),
                "max_gap": float(np.max(np.diff(x))),
            }

        if not self.columns:
            raise ValueError(
                f"'{path}' has no usable '{prefix}_XXC' columns with at "
                "least two finite values.")

        self.available_temps = sorted(self.columns)
        self.well_supported_temps = [
            temperature for temperature in self.available_temps
            if (self.columns[temperature]["max_gap"]
                <= self.max_local_gap_wt_pct)
        ]
        if not self.well_supported_temps:
            self.well_supported_temps = list(self.available_temps)
        self.mp_min = min(float(column["x"][0])
                          for column in self.columns.values())
        self.mp_max = max(float(column["x"][-1])
                          for column in self.columns.values())

    @staticmethod
    def _enclosing_gap(points, value):
        points = np.asarray(points, dtype=float)
        if points.size == 0 or value < points[0] or value > points[-1]:
            return np.inf
        position = int(np.searchsorted(points, value))
        if (position < len(points)
                and np.isclose(points[position], value, atol=1e-10)):
            adjacent_gaps = []
            if position > 0:
                adjacent_gaps.append(points[position] - points[position - 1])
            if position + 1 < len(points):
                adjacent_gaps.append(points[position + 1] - points[position])
            return float(max(adjacent_gaps)) if adjacent_gaps else np.inf
        if position == 0 or position == len(points):
            return np.inf
        return float(points[position] - points[position - 1])

    def _is_locally_supported(self, temperature, mass_percent):
        if temperature not in self.well_supported_temps:
            return False
        gap = self._enclosing_gap(
            self.columns[temperature]["x"], mass_percent)
        return gap <= self.max_local_gap_wt_pct

    def locally_supported_temperatures(self, mass_percent):
        return [
            temperature for temperature in self.available_temps
            if self._is_locally_supported(temperature, mass_percent)
        ]

    def _column_value(self, temperature, mass_percent):
        column = self.columns[temperature]
        concentration = float(np.clip(
            mass_percent, column["x"][0], column["x"][-1]))
        return float(column["interp"](concentration))

    @staticmethod
    def _temperature_pair(temperatures, target_temperature):
        temperatures = sorted(temperatures)
        exact = [temperature for temperature in temperatures
                 if np.isclose(temperature, target_temperature, atol=1e-10)]
        if exact:
            return (exact[0],)
        if len(temperatures) == 1:
            return (temperatures[0],)

        lower = [temperature for temperature in temperatures
                 if temperature < target_temperature]
        upper = [temperature for temperature in temperatures
                 if temperature > target_temperature]
        if lower and upper:
            return (lower[-1], upper[0])
        if lower:
            return (lower[-2], lower[-1])
        return (upper[0], upper[1])

    def evaluate(self, mass_percent, temperature):
        mass_percent = float(mass_percent)
        temperature = float(temperature)
        if not np.isfinite(mass_percent) or not np.isfinite(temperature):
            raise ValueError("Concentration and temperature must be finite.")

        concentration_outside = not self.mp_in_range(mass_percent)
        supported = self.locally_supported_temperatures(mass_percent)
        weak_support = not supported

        if not supported:
            supported = [
                t for t in self.available_temps
                if (self.columns[t]["x"][0] <= mass_percent
                    <= self.columns[t]["x"][-1])
            ]
        if not supported:
            supported = list(self.available_temps)

        temperatures_used = self._temperature_pair(supported, temperature)
        temperature_outside = not (
            supported[0] <= temperature <= supported[-1])

        values = [self._column_value(t, mass_percent)
                  for t in temperatures_used]
        if len(temperatures_used) == 1:
            value = values[0]
        else:
            t_low, t_high = temperatures_used
            effective_temperature = float(np.clip(
                temperature,
                supported[0] - self.max_temperature_extrapolation_C,
                supported[-1] + self.max_temperature_extrapolation_C,
            ))
            fraction = ((effective_temperature - t_low)
                        / (t_high - t_low))
            value = values[0] + fraction * (values[1] - values[0])

        if self.minimum_value is not None:
            value = max(float(value), np.nextafter(
                float(self.minimum_value), np.inf))

        weak_support = weak_support or any(
            not self._is_locally_supported(t, mass_percent)
            for t in temperatures_used)
        return TableInterpolationResult(
            value=float(value),
            temperatures_used=tuple(temperatures_used),
            concentration_outside=concentration_outside,
            temperature_outside=temperature_outside,
            weak_local_support=weak_support,
        )

    def value(self, mass_percent, temperature):
        return self.evaluate(mass_percent, temperature).value

    def temp_in_range(self, temperature):
        return self.available_temps[0] <= temperature <= self.available_temps[-1]

    def mp_in_range(self, mass_percent):
        return self.mp_min <= mass_percent <= self.mp_max

    def support_warnings(self, mass_percent, temperature):
        result = self.evaluate(mass_percent, temperature)
        warnings = []
        if result.concentration_outside:
            warnings.append(
                f"concentration {mass_percent:.2f} wt% is outside the "
                f"table range {self.mp_min:g}-{self.mp_max:g} wt%")
        if result.weak_local_support:
            warnings.append(
                f"weak local concentration support near "
                f"{mass_percent:.2f} wt%")
        if result.temperature_outside:
            warnings.append(
                f"temperature {temperature:.2f} C is outside the locally "
                "supported temperature range")
        return warnings


# =====================================================================
# ====================================================================
#  1) DENSITY MODEL   (ported from ink_density.py)
# ====================================================================
# =====================================================================
class InkDensityCalculator:
    """
    Theoretical density of an ink made of Aluminum pigment, Water, IPA
    and/or PG. Uses empirical binary density tables to account for the
    volume contraction of real solvent/water mixtures.
    """

    DENSITY_ALUMINUM = RHO_PIGMENT   # g/cm3

    def __init__(self, tables_dir="tables_parameters"):
        self.tables_dir = tables_dir
        self.tables = {'IPA': None, 'PG': None}
        self._load_tables()

    def _load_tables(self):
        """Load all available density columns from both binary tables."""
        for solvent, filename in (("IPA", "ipa_density.csv"),
                                  ("PG", "pg_density.csv")):
            path = os.path.join(self.tables_dir, filename)
            try:
                self.tables[solvent] = PchipTemperatureTable(
                    path, "Density", max_local_gap_wt_pct=20.0,
                    max_temperature_extrapolation_C=10.0,
                    minimum_value=0.0,
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"Warning: {solvent} density table not loaded ({error}).")

    def get_liquid_density(self, solvent_type, mass_percent, target_temp=25):
        """
        Real density [g/cm3] of a binary liquid (Water + solvent),
        interpolated in concentration and temperature.

        :param solvent_type: 'IPA' or 'PG'
        :param mass_percent: solvent mass percent within the liquid phase
        :param target_temp:  temperature in Celsius
        """
        if solvent_type not in self.tables:
            raise ValueError(f"Unknown solvent type: {solvent_type}")
        table = self.tables[solvent_type]
        if table is None:
            raise ValueError(
                f"Missing {solvent_type} density table in '{self.tables_dir}'.")
        return table.value(mass_percent, target_temp)

    def calculate_density(self, pct_al, pct_ipa=0.0, pct_pg=0.0, target_temp=25):
        """
        Total ink density [g/cm3]. Handles single solvents directly and
        applies the pseudo-binary approximation when IPA and PG coexist.

        :param pct_al:  mass percent aluminum pigment
        :param pct_ipa: mass percent isopropanol
        :param pct_pg:  mass percent propylene glycol
        :param target_temp: temperature in Celsius
        """
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg
        if pct_water < 0:
            raise ValueError("Total mass percentage exceeds 100%. Check your inputs.")

        frac_al = pct_al / 100.0
        term_al = frac_al / self.DENSITY_ALUMINUM

        # Case 1: IPA only
        if pct_ipa > 0 and pct_pg == 0:
            pct_liquid_total = pct_ipa + pct_water
            pct_solvent_in_liquid = (pct_ipa / pct_liquid_total) * 100.0
            rho_liquid = self.get_liquid_density('IPA', pct_solvent_in_liquid, target_temp)
            return 1.0 / (term_al + ((pct_liquid_total / 100.0) / rho_liquid))

        # Case 2: PG only
        elif pct_pg > 0 and pct_ipa == 0:
            pct_liquid_total = pct_pg + pct_water
            pct_solvent_in_liquid = (pct_pg / pct_liquid_total) * 100.0
            rho_liquid = self.get_liquid_density('PG', pct_solvent_in_liquid, target_temp)
            return 1.0 / (term_al + ((pct_liquid_total / 100.0) / rho_liquid))

        # Case 3: IPA + PG (pseudo-binary approximation)
        elif pct_ipa > 0 and pct_pg > 0:
            total_solvent = pct_ipa + pct_pg
            ratio_ipa = pct_ipa / total_solvent
            ratio_pg = pct_pg / total_solvent

            mass_mix_ipa = pct_ipa + (pct_water * ratio_ipa)
            mass_mix_pg = pct_pg + (pct_water * ratio_pg)

            pct_ipa_in_mix = (pct_ipa / mass_mix_ipa) * 100.0
            pct_pg_in_mix = (pct_pg / mass_mix_pg) * 100.0

            rho_mix_ipa = self.get_liquid_density('IPA', pct_ipa_in_mix, target_temp)
            rho_mix_pg = self.get_liquid_density('PG', pct_pg_in_mix, target_temp)

            term_mix_ipa = (mass_mix_ipa / 100.0) / rho_mix_ipa
            term_mix_pg = (mass_mix_pg / 100.0) / rho_mix_pg

            return 1.0 / (term_al + term_mix_ipa + term_mix_pg)

        # Case 4: pure water (+ Al), no solvents
        else:
            rho_water = self.get_liquid_density('IPA', 0.0, target_temp)
            return 1.0 / (term_al + ((pct_water / 100.0) / rho_water))


# =====================================================================
# ====================================================================
#  2) REFRACTIVE-INDEX MODEL   (ported from ink_refractive.py)
# ====================================================================
# =====================================================================
class InkRefractiveCalculator:
    """
    Refractive index (nD) of the liquid matrix (Water + IPA and/or PG).
    Uses empirical nD tables plus the Gladstone-Dale relation with a
    pseudo-binary approximation for ternary mixtures. The aluminum
    pigment is treated as not participating in matrix refraction.
    """

    def __init__(self, tables_dir="tables_parameters", density_calculator=None):
        self.tables_dir = tables_dir
        self.tables = {'IPA': None, 'PG': None}

        # density calculator is required for the Gladstone-Dale step
        if density_calculator is None:
            self.density_calc = InkDensityCalculator(tables_dir=tables_dir)
        else:
            self.density_calc = density_calculator

        self._load_tables()

    def _load_tables(self):
        """Load all available refractive-index temperature columns."""
        for solvent, filename in (("IPA", "ipa_refractive.csv"),
                                  ("PG", "pg_refractive.csv")):
            path = os.path.join(self.tables_dir, filename)
            try:
                self.tables[solvent] = PchipTemperatureTable(
                    path, "Refractive", max_local_gap_wt_pct=20.0,
                    max_temperature_extrapolation_C=10.0,
                    minimum_value=0.0,
                )
            except (FileNotFoundError, ValueError) as error:
                print(
                    f"Warning: {solvent} refractive table not loaded "
                    f"({error}).")

    def get_liquid_refractive_index(self, solvent_type, mass_percent, target_temp=25):
        """
        Refractive index (nD) of a binary liquid (Water + solvent).

        With two or more measured temperature columns, PCHIP is evaluated
        over concentration first and the result is interpolated locally and
        linearly over temperature. With only one temperature column, a
        concentration-dependent thermo-optic coefficient is used as a
        documented fallback.
        """
        if solvent_type not in self.tables:
            raise ValueError(f"Unknown solvent type: {solvent_type}")
        table = self.tables[solvent_type]
        if table is None:
            raise ValueError(
                f"Missing {solvent_type} refractive table in "
                f"'{self.tables_dir}'.")

        if len(table.available_temps) >= 2:
            return table.value(mass_percent, target_temp)

        reference_temperature = table.available_temps[0]
        reference_value = table.value(mass_percent, reference_temperature)
        dn_dt_water = -0.00010
        dn_dt_solvent = {"IPA": -0.00040, "PG": -0.00038}[solvent_type]
        fraction = float(np.clip(mass_percent, 0.0, 100.0)) / 100.0
        dn_dt_mix = dn_dt_water + fraction * (
            dn_dt_solvent - dn_dt_water)
        return reference_value + dn_dt_mix * (
            target_temp - reference_temperature)

    def calculate_refractive_index(self, pct_al=0.0, pct_ipa=0.0, pct_pg=0.0, target_temp=25):
        """
        Refractive index (nD) of the liquid matrix. Handles single
        solvents directly and applies the Gladstone-Dale pseudo-binary
        approximation when IPA and PG coexist.
        """
        # Determine the actual water fraction including the aluminum mass.
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg
        if pct_water < 0:
            raise ValueError("Total mass percentage exceeds 100%. Check your inputs.")

        # Total mass of the liquid phase.
        pct_liquid_total = pct_ipa + pct_pg + pct_water
        if pct_liquid_total <= 0:
            return 1.0  # Fallback

        # Solvent concentrations within the liquid phase.
        ipa_in_liq = (pct_ipa / pct_liquid_total) * 100.0
        pg_in_liq = (pct_pg / pct_liquid_total) * 100.0
        water_in_liq = (pct_water / pct_liquid_total) * 100.0

        # Case 1: pure water
        if ipa_in_liq == 0 and pg_in_liq == 0:
            return self.get_liquid_refractive_index('IPA', 0.0, target_temp)

        # Case 2: IPA only
        elif ipa_in_liq > 0 and pg_in_liq == 0:
            return self.get_liquid_refractive_index('IPA', ipa_in_liq, target_temp)

        # Case 3: PG only
        elif pg_in_liq > 0 and ipa_in_liq == 0:
            return self.get_liquid_refractive_index('PG', pg_in_liq, target_temp)

        # Case 4: IPA + PG (Gladstone-Dale pseudo-binary)
        elif ipa_in_liq > 0 and pg_in_liq > 0:
            total_solvent_in_liq = ipa_in_liq + pg_in_liq
            ratio_ipa = ipa_in_liq / total_solvent_in_liq
            ratio_pg = pg_in_liq / total_solvent_in_liq

            mass_mix_ipa = ipa_in_liq + (water_in_liq * ratio_ipa)
            mass_mix_pg = pg_in_liq + (water_in_liq * ratio_pg)

            pct_ipa_in_mix = (ipa_in_liq / mass_mix_ipa) * 100.0
            pct_pg_in_mix = (pg_in_liq / mass_mix_pg) * 100.0

            n_1 = self.get_liquid_refractive_index('IPA', pct_ipa_in_mix, target_temp)
            rho_1 = self.density_calc.get_liquid_density('IPA', pct_ipa_in_mix, target_temp)

            n_2 = self.get_liquid_refractive_index('PG', pct_pg_in_mix, target_temp)
            rho_2 = self.density_calc.get_liquid_density('PG', pct_pg_in_mix, target_temp)

            # specific refraction (Gladstone-Dale): R = (n - 1) / rho
            r_1 = (n_1 - 1.0) / rho_1
            r_2 = (n_2 - 1.0) / rho_2

            # mix specific refractions by mass fraction in the total liquid
            w_mix_1 = mass_mix_ipa / 100.0
            w_mix_2 = mass_mix_pg / 100.0
            r_total = (w_mix_1 * r_1) + (w_mix_2 * r_2)

            # convert back using the real ternary-liquid density (Al = 0%)
            rho_liquid_total = self.density_calc.calculate_density(
                pct_al=0.0, pct_ipa=ipa_in_liq, pct_pg=pg_in_liq, target_temp=target_temp)

            return (r_total * rho_liquid_total) + 1.0


# =====================================================================
# ====================================================================
#  3) SOUND-VELOCITY MODEL   (ported from ink_sound.py)
# ====================================================================
# =====================================================================
# ---------------------------------------------------------------------
#  Reference curve: speed of sound in pure water at atmospheric pressure
#  Anchor values from Marczak (1997), J. Acoust. Soc. Am. 102(5),
#  2776-2779 (accuracy of the underlying equation: < 0.05 m/s).
# ---------------------------------------------------------------------
_WATER_SOUND_T = np.array(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
_WATER_SOUND_C = np.array(
    [1402.39, 1426.15, 1447.27, 1465.93, 1482.34, 1496.69,
     1509.13, 1519.81, 1528.86, 1536.43, 1542.57])
_WATER_SOUND_INTERP = PchipInterpolator(
    _WATER_SOUND_T, _WATER_SOUND_C, extrapolate=True)


def water_sound_velocity(temperature_C):
    """Speed of sound in pure water [m/s] (Marczak 1997 anchors, PCHIP).
    Intended range 0..50 C; extrapolated smoothly outside."""
    return float(_WATER_SOUND_INTERP(temperature_C))


class _PropertyTable:
    """Interpolation of sparse concentration/temperature property tables.

    Density tables use shape-preserving concentration curves followed by
    linear temperature interpolation. Sound tables (``reference`` supplied)
    use the more robust v3 construction

        c(w, T) = c_water(T) + dc_ref(w) + (T - T_ref) * k_T(w)

    where ``dc_ref`` is taken from the best-supported reference column
    (normally 25 C) and ``k_T`` is estimated only from rows that contain at
    least two actual temperature measurements. Large internal concentration
    gaps therefore no longer masquerade as fully supported data.
    """

    MAX_LOCAL_GAP_WT_PCT = 12.0
    MAX_ABS_DEVIATION_SLOPE = 5.0  # m/s/K; safety bound for sparse source data
    MAX_LOCAL_SLOPE_M_S_PER_WT_PCT = 12.0
    MIN_SPIKE_RESIDUAL_M_S = 8.0

    @classmethod
    def _remove_isolated_spikes(cls, mass_percent, values):
        """Remove sharp one-point reversals caused by conflicting source rows.

        The merged PG table contains near-duplicate concentrations from
        different sources (for example around 31 and 80 wt%). A point is
        removed only if it forms a strong local direction reversal and lies
        far from the straight line between its neighbours.
        """
        x = np.asarray(mass_percent, dtype=float)
        y = np.asarray(values, dtype=float)
        keep = list(range(len(x)))
        changed = True
        while changed and len(keep) >= 4:
            changed = False
            for position in range(1, len(keep) - 1):
                left, center, right = keep[position - 1:position + 2]
                slope_left = (y[center] - y[left]) / (x[center] - x[left])
                slope_right = (y[right] - y[center]) / (x[right] - x[center])
                if slope_left * slope_right >= 0:
                    continue
                predicted = y[left] + (y[right] - y[left]) * (
                    (x[center] - x[left]) / (x[right] - x[left])
                )
                residual = abs(y[center] - predicted)
                if (
                    max(abs(slope_left), abs(slope_right))
                    > cls.MAX_LOCAL_SLOPE_M_S_PER_WT_PCT
                    and residual > cls.MIN_SPIKE_RESIDUAL_M_S
                ):
                    del keep[position]
                    changed = True
                    break
        return x[keep], y[keep]

    def __init__(self, path, prefix, reference=None):
        self.path = path
        self.prefix = prefix
        self.reference = reference
        raw = pd.read_csv(path)
        if "Mass_Percent" not in raw.columns:
            raise ValueError(f"'{path}' has no 'Mass_Percent' column.")
        raw["Mass_Percent"] = pd.to_numeric(
            raw["Mass_Percent"], errors="coerce")
        for column_name in raw.columns[1:]:
            raw[column_name] = pd.to_numeric(
                raw[column_name], errors="coerce")
        raw = raw[np.isfinite(raw["Mass_Percent"])].copy()
        if raw.empty:
            raise ValueError(f"'{path}' has no finite concentrations.")

        # Median aggregation makes repeated source concentrations harmless.
        df = (raw.groupby("Mass_Percent", as_index=False).median(numeric_only=True)
              .sort_values("Mass_Percent").reset_index(drop=True))
        self.mp_full = df["Mass_Percent"].to_numpy(dtype=float)
        self.df = df
        self.columns = {}
        self.column_names = {}
        pattern = re.compile(rf"{prefix}_(\d+(?:\.\d+)?)C")

        for col in df.columns:
            match = pattern.fullmatch(col)
            if not match:
                continue
            temperature = float(match.group(1))
            y_full = df[col].to_numpy(dtype=float)
            valid = np.isfinite(y_full)
            mp_valid = self.mp_full[valid]
            y_valid = y_full[valid]
            if len(mp_valid) < 2:
                continue
            if self.reference is not None and len(mp_valid) >= 4:
                mp_valid, y_valid = self._remove_isolated_spikes(mp_valid, y_valid)

            interp = PchipInterpolator(mp_valid, y_valid, extrapolate=False)
            derivative = interp.derivative()
            offset = 0.0
            if self.reference is not None:
                if mp_valid[0] <= 0.0 <= mp_valid[-1]:
                    offset = float(interp(0.0))
                else:
                    offset = float(self.reference(temperature))

            self.columns[temperature] = {
                "interp": interp,
                "mp": mp_valid,
                "y": y_valid,
                "mp_min": float(mp_valid[0]),
                "mp_max": float(mp_valid[-1]),
                "y_lo": float(y_valid[0]),
                "y_hi": float(y_valid[-1]),
                "d_lo": float(derivative(mp_valid[0])),
                "d_hi": float(derivative(mp_valid[-1])),
                "offset": offset,
            }
            self.column_names[temperature] = col

        if not self.columns:
            raise ValueError(f"'{path}' has no valid '{prefix}_XXC' columns.")

        self.available_temps = sorted(self.columns)
        self.mp_min = min(column["mp_min"] for column in self.columns.values())
        self.mp_max = max(column["mp_max"] for column in self.columns.values())
        self.ref_temp = 25.0 if 25.0 in self.columns else min(
            self.columns,
            key=lambda temperature: (
                -(len(self.columns[temperature]["mp"])),
                abs(temperature - 25.0),
            ),
        )

        self.temp_slope_mp = np.array([], dtype=float)
        self.temp_slope_values = np.array([], dtype=float)
        self.temp_slope_interp = None
        if self.reference is not None:
            self._build_deviation_temperature_slope()

    @staticmethod
    def _enclosing_gap(points, value):
        """Width of the measured concentration interval surrounding value."""
        points = np.asarray(points, dtype=float)
        if points.size == 0 or value < points[0] or value > points[-1]:
            return np.inf
        position = int(np.searchsorted(points, value))
        if position < len(points) and np.isclose(points[position], value, atol=1e-10):
            return 0.0
        if position == 0 or position == len(points):
            return np.inf
        return float(points[position] - points[position - 1])

    def _col_value(self, temperature, mass_percent):
        """One concentration curve, with bounded linear edge continuation."""
        column = self.columns[temperature]
        if mass_percent < column["mp_min"]:
            value = column["y_lo"] + column["d_lo"] * (
                mass_percent - column["mp_min"]
            )
        elif mass_percent > column["mp_max"]:
            value = column["y_hi"] + column["d_hi"] * (
                mass_percent - column["mp_max"]
            )
        else:
            value = float(column["interp"](mass_percent))
        return value - column["offset"]

    def _row_deviation_slope(self, row_index):
        """Local d(dc)/dT at T_ref from actual values in one CSV row."""
        samples = []
        for temperature in self.available_temps:
            raw = self.df.at[row_index, self.column_names[temperature]]
            if pd.notna(raw):
                samples.append(
                    (temperature, float(raw) - self.columns[temperature]["offset"])
                )
        if len(samples) < 2:
            return None

        samples.sort()
        temperatures = np.array([item[0] for item in samples], dtype=float)
        deviations = np.array([item[1] for item in samples], dtype=float)
        reference_index = np.flatnonzero(np.isclose(temperatures, self.ref_temp))

        if reference_index.size:
            index = int(reference_index[0])
            if 0 < index < len(samples) - 1:
                slope = (deviations[index + 1] - deviations[index - 1]) / (
                    temperatures[index + 1] - temperatures[index - 1]
                )
            elif index < len(samples) - 1:
                slope = (deviations[index + 1] - deviations[index]) / (
                    temperatures[index + 1] - temperatures[index]
                )
            else:
                slope = (deviations[index] - deviations[index - 1]) / (
                    temperatures[index] - temperatures[index - 1]
                )
        else:
            position = int(np.searchsorted(temperatures, self.ref_temp))
            if 0 < position < len(samples):
                lo, hi = position - 1, position
            elif position == 0:
                lo, hi = 0, 1
            else:
                lo, hi = len(samples) - 2, len(samples) - 1
            slope = (deviations[hi] - deviations[lo]) / (
                temperatures[hi] - temperatures[lo]
            )
        return float(
            np.clip(
                slope,
                -self.MAX_ABS_DEVIATION_SLOPE,
                self.MAX_ABS_DEVIATION_SLOPE,
            )
        )

    def _build_deviation_temperature_slope(self):
        """Build k_T(w) only from rows with real multi-temperature data."""
        concentrations = []
        slopes = []
        for row_index, mass_percent in enumerate(self.mp_full):
            slope = self._row_deviation_slope(row_index)
            if slope is not None and np.isfinite(slope):
                concentrations.append(float(mass_percent))
                slopes.append(slope)

        if not concentrations:
            return
        table = pd.DataFrame({"mp": concentrations, "slope": slopes})
        table = table.groupby("mp", as_index=False)["slope"].median().sort_values("mp")
        self.temp_slope_mp = table["mp"].to_numpy(float)
        self.temp_slope_values = table["slope"].to_numpy(float)
        if len(table) >= 2:
            self.temp_slope_interp = PchipInterpolator(
                self.temp_slope_mp,
                self.temp_slope_values,
                extrapolate=False,
            )

    def _temperature_slope(self, mass_percent):
        if self.temp_slope_interp is None:
            return 0.0
        concentration = float(
            np.clip(mass_percent, self.temp_slope_mp[0], self.temp_slope_mp[-1])
        )
        return float(self.temp_slope_interp(concentration))

    def temps_covering(self, mass_percent):
        """Temperatures with locally supported concentration data."""
        return [
            temperature
            for temperature in self.available_temps
            if self._enclosing_gap(self.columns[temperature]["mp"], mass_percent)
            <= self.MAX_LOCAL_GAP_WT_PCT
        ]

    def covering_temp_range(self, mass_percent):
        temperatures = self.temps_covering(mass_percent)
        if not temperatures:
            return self.ref_temp, self.ref_temp
        return temperatures[0], temperatures[-1]

    def support_warnings(self, mass_percent, temperature):
        """Human-readable diagnostics for weakly supported sound predictions."""
        if self.reference is None:
            return []
        warnings_out = []
        reference_gap = self._enclosing_gap(
            self.columns[self.ref_temp]["mp"], mass_percent
        )
        if reference_gap > self.MAX_LOCAL_GAP_WT_PCT:
            warnings_out.append(
                f"reference concentration curve has an internal gap of "
                f"{reference_gap:.1f} wt% near {mass_percent:.1f} wt%"
            )
        slope_gap = self._enclosing_gap(self.temp_slope_mp, mass_percent)
        if slope_gap > self.MAX_LOCAL_GAP_WT_PCT:
            warnings_out.append(
                f"temperature-slope curve has an internal gap of "
                f"{slope_gap:.1f} wt% near {mass_percent:.1f} wt%"
            )
        t_low, t_high = self.covering_temp_range(mass_percent)
        if not (t_low <= temperature <= t_high):
            warnings_out.append(
                f"temperature {temperature:.1f} C is outside locally supported "
                f"anchors {t_low:g}-{t_high:g} C at {mass_percent:.1f} wt%"
            )
        return warnings_out

    def value(self, mass_percent, temperature):
        if self.reference is None:
            values = np.array(
                [self._col_value(t, mass_percent) for t in self.available_temps],
                dtype=float,
            )
            if len(self.available_temps) == 1:
                return float(values[0])
            temperature_interpolator = interp1d(
                self.available_temps,
                values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return float(temperature_interpolator(temperature))

        base = float(self.reference(temperature))
        concentration_deviation = self._col_value(self.ref_temp, mass_percent)
        temperature_correction = (
            temperature - self.ref_temp
        ) * self._temperature_slope(mass_percent)
        return base + concentration_deviation + temperature_correction

    def temp_in_range(self, temperature):
        return self.available_temps[0] <= temperature <= self.available_temps[-1]

    def mp_in_range(self, mass_percent):
        return self.mp_min <= mass_percent <= self.mp_max


@dataclass
class SoundResult:
    """Result container for the sound-velocity model."""
    sound_velocity: float          # m/s
    density: float                 # g/cm3 (from the Wood/Urick mixing)
    temperature: float             # C
    composition: dict              # {'Al':.., 'IPA':.., 'PG':.., 'Water':..}
    vol_fraction_al: float = 0.0   # phi_Al (-)
    calibration_offset: float = 0.0  # additive offset [m/s] included in sound_velocity
    warnings: list = field(default_factory=list)


class InkSoundCalculator:
    """
    Sound velocity of an Al-pigment / Water / IPA / PG ink.

    Sound velocity cannot be mixed linearly. Instead density (rho) and
    adiabatic compressibility (beta) are mixed over volume fractions and
    converted back via Newton-Laplace:  c = 1 / sqrt(rho * beta).
    The solid pigment is added with the Wood/Urick effective-medium
    equation; beta_Al comes from the bulk modulus K_Al.

    v3: the binary sound tables are evaluated as deviations from the
    Marczak pure-water reference curve (see _PropertyTable), which
    removes absolute offsets of the data sets' own water anchors. The
    concentration curve and its temperature coefficient are modelled
    separately so internal gaps in sparse columns cannot dominate the
    operating range.
    An additive calibration offset [m/s] can be set via calibrate().
    """

    DENSITY_ALUMINUM = RHO_PIGMENT          # g/cm3
    BULK_MODULUS_ALUMINUM = BULK_MODULUS_ALUMINUM  # Pa

    def __init__(self, tables_dir="tables_parameters"):
        self.tables_dir = tables_dir
        self.density = {"IPA": None, "PG": None}
        self.sound = {"IPA": None, "PG": None}
        self.calibration_offset = 0.0        # additive, m/s
        self._load("IPA", "ipa_density.csv", "ipa_sound.csv")
        self._load("PG", "pg_density.csv", "pg_sound.csv")

    def _load(self, solvent, density_file, sound_file):
        dpath = os.path.join(self.tables_dir, density_file)
        spath = os.path.join(self.tables_dir, sound_file)
        try:
            self.density[solvent] = PchipTemperatureTable(
                dpath, "Density", max_local_gap_wt_pct=20.0,
                max_temperature_extrapolation_C=10.0,
                minimum_value=0.0,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Note: {solvent} density table not loaded ({e}).")
        try:
            self.sound[solvent] = _PropertyTable(
                spath, "SoundVelocity", reference=water_sound_velocity)
        except (FileNotFoundError, ValueError) as e:
            print(f"Note: {solvent} sound table not loaded ({e}).")

    def _binary_phase(self, solvent, pct_solvent, pct_water, temperature, warnings):
        """One binary liquid sub-phase -> (volume_cm3, rho_SI, beta_SI)."""
        mass_liquid = pct_solvent + pct_water          # g, on a 100-g basis
        if mass_liquid <= 0:
            return None

        pct_in_liquid = (pct_solvent / mass_liquid) * 100.0

        dtab = self.density[solvent]
        stab = self.sound[solvent]
        if dtab is None or stab is None:
            raise ValueError(
                f"Missing {solvent} table(s) in '{self.tables_dir}'. "
                f"Cannot evaluate a mixture containing {solvent}.")

        if not dtab.mp_in_range(pct_in_liquid) or not stab.mp_in_range(pct_in_liquid):
            warnings.append(
                f"{solvent} concentration {pct_in_liquid:.1f}% is outside the "
                f"tabulated range -> extrapolated.")

        for issue in stab.support_warnings(pct_in_liquid, temperature):
            warnings.append(f"{solvent} sound: {issue}.")
        for issue in dtab.support_warnings(pct_in_liquid, temperature):
            warnings.append(f"{solvent} density: {issue}.")

        rho_L = dtab.value(pct_in_liquid, temperature)     # g/cm3
        c_L = stab.value(pct_in_liquid, temperature)       # m/s

        rho_si = rho_L * 1000.0                            # kg/m3
        beta = 1.0 / (rho_si * c_L ** 2)                   # 1/Pa
        vol = mass_liquid / rho_L                          # cm3 (100-g basis)
        return (vol, rho_si, beta)

    def calculate(self, pct_al, pct_ipa=0.0, pct_pg=0.0, temperature=25.0):
        """Compute sound velocity (and Wood-model density) of the ink -> SoundResult."""
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg
        if pct_al < 0 or pct_ipa < 0 or pct_pg < 0:
            raise ValueError("Mass percentages must be non-negative.")
        if pct_water < -1e-9:
            raise ValueError(
                f"Al + IPA + PG = {pct_al + pct_ipa + pct_pg:.3f}% exceeds 100%.")
        pct_water = max(pct_water, 0.0)

        warnings = []
        beta_al = 1.0 / self.BULK_MODULUS_ALUMINUM
        rho_al_si = self.DENSITY_ALUMINUM * 1000.0

        # phase list: (volume_cm3, rho_SI, beta_SI); start with aluminum
        phases = []
        if pct_al > 0:
            phases.append((pct_al / self.DENSITY_ALUMINUM, rho_al_si, beta_al))

        # assemble the liquid phase(s)
        if pct_ipa > 0 and pct_pg == 0:                         # IPA only
            phases.append(self._binary_phase("IPA", pct_ipa, pct_water,
                                             temperature, warnings))
        elif pct_pg > 0 and pct_ipa == 0:                       # PG only
            phases.append(self._binary_phase("PG", pct_pg, pct_water,
                                             temperature, warnings))
        elif pct_ipa > 0 and pct_pg > 0:                        # IPA + PG (pseudo-binary)
            total_solvent = pct_ipa + pct_pg
            water_ipa = pct_water * (pct_ipa / total_solvent)
            water_pg = pct_water * (pct_pg / total_solvent)
            phases.append(self._binary_phase("IPA", pct_ipa, water_ipa,
                                             temperature, warnings))
            phases.append(self._binary_phase("PG", pct_pg, water_pg,
                                             temperature, warnings))
        else:                                                   # water (+ Al) only
            phases.append(self._binary_phase("IPA", 0.0, pct_water,
                                             temperature, warnings))

        phases = [p for p in phases if p is not None]
        if not phases:
            raise ValueError("Empty mixture: nothing to compute.")

        # Wood / Urick mixing over volume fractions
        V_total = sum(v for v, _, _ in phases)
        rho_mix_si = sum((v / V_total) * rho for v, rho, _ in phases)
        beta_mix = sum((v / V_total) * beta for v, _, beta in phases)
        c_mix = float(1.0 / np.sqrt(rho_mix_si * beta_mix)) + self.calibration_offset

        vol_al = (pct_al / self.DENSITY_ALUMINUM) if pct_al > 0 else 0.0
        phi_al = vol_al / V_total

        if pct_al > 5.0:
            warnings.append(
                "Al > 5 % : Wood/Urick is a dilute-suspension model; "
                "accuracy may degrade at high pigment loading.")

        return SoundResult(
            sound_velocity=c_mix,
            density=rho_mix_si / 1000.0,
            temperature=temperature,
            composition={"Al": pct_al, "IPA": pct_ipa, "PG": pct_pg, "Water": pct_water},
            vol_fraction_al=phi_al,
            calibration_offset=self.calibration_offset,
            warnings=warnings,
        )

    def sound_velocity(self, pct_al, pct_ipa=0.0, pct_pg=0.0, temperature=25.0):
        """Return only the sound velocity in m/s."""
        return self.calculate(pct_al, pct_ipa, pct_pg, temperature).sound_velocity

    def calibrate(self, measured_velocity, pct_al, pct_ipa=0.0, pct_pg=0.0,
                  temperature=25.0):
        """
        Set the additive calibration offset [m/s] so the model reproduces
        a measured sound velocity exactly at the given state point.
        Returns the offset (measured - model).
        """
        self.calibration_offset = 0.0
        base = self.calculate(pct_al, pct_ipa, pct_pg, temperature).sound_velocity
        self.calibration_offset = measured_velocity - base
        return self.calibration_offset

    def self_test(self, verbose=True):
        """
        Quick sanity checks of the sound model. Returns a list of issue
        strings (empty list = all checks passed). Checks:
          1. solvent-free limit reproduces the pure-water reference,
          2. continuity of c over temperature (no interpolation jumps),
          3. continuity of c over concentration.
        Runs with the calibration offset temporarily set to zero.
        """
        issues = []
        saved_offset = self.calibration_offset
        self.calibration_offset = 0.0
        try:
            # 1) water limit vs. Marczak reference
            for T in (15.0, 20.0, 22.0, 25.0, 30.0, 35.0):
                c = self.sound_velocity(0.0, 0.0, 0.0, T)
                ref = water_sound_velocity(T)
                if abs(c - ref) > 0.2:
                    issues.append(
                        f"water limit at {T:g} C: model {c:.2f} vs. "
                        f"reference {ref:.2f} m/s (|d| > 0.2).")

            # 2) continuity over temperature (max ~6 m/s per K allowed)
            for (al, ipa, pg) in ((1.82, 3.64, 3.64), (0.0, 25.0, 0.0),
                                  (0.0, 0.0, 35.0), (0.0, 8.0, 0.0)):
                prev = None
                for T in np.arange(16.0, 32.001, 0.25):
                    c = self.sound_velocity(al, ipa, pg, float(T))
                    if prev is not None and abs(c - prev) > 1.5:
                        issues.append(
                            f"T-scan jump at Al={al} IPA={ipa} PG={pg}, "
                            f"T={T:.2f} C: dc={c - prev:+.1f} m/s per 0.25 K.")
                        break
                    prev = c

            # 3) continuity over concentration (incl. column boundaries)
            for T in (20.0, 23.0):
                for solvent in ("IPA", "PG"):
                    prev = None
                    for w in np.arange(0.0, 60.001, 0.5):
                        ipa = float(w) if solvent == "IPA" else 0.0
                        pg = float(w) if solvent == "PG" else 0.0
                        c = self.sound_velocity(0.0, ipa, pg, T)
                        if prev is not None and abs(c - prev) > 8.0:
                            issues.append(
                                f"{solvent} concentration scan jump at "
                                f"{w:.1f}%, T={T:g} C: dc={c - prev:+.1f} "
                                f"m/s per 0.5%.")
                            break
                        prev = c
        finally:
            self.calibration_offset = saved_offset

        if verbose:
            if issues:
                print(f"Sound self-test: {len(issues)} issue(s) found:")
                for msg in issues:
                    print(f"  ! {msg}")
            else:
                print("Sound self-test: all checks passed "
                      "(water limit, T- and concentration continuity).")
        return issues


# =====================================================================
# ====================================================================
#  4) VISCOSITY MODEL   (ported from ink_viscosity.py)
# ====================================================================
# =====================================================================
def _load_viscosity_table(path):
    """Load a positive viscosity table for PCHIP/linear interpolation."""
    return PchipTemperatureTable(
        path, "Viscosity", max_local_gap_wt_pct=20.0,
        max_temperature_extrapolation_C=10.0,
        minimum_value=0.0,
    )


def _interp_viscosity_table(table, mass_percent, temperature_C):
    """PCHIP over concentration, then local linear interpolation over T."""
    result = table.evaluate(mass_percent, temperature_C)
    concentration_issue = (
        result.concentration_outside or result.weak_local_support)
    return result.value, concentration_issue, result.temperature_outside


class InkViscosityModel:
    """
    Estimates the viscosity [mPa.s] of the ink in two steps:

      1) Liquid carrier (Water + IPA + PG): log-additive overlay of the
         measured binary data,  eta_carrier = eta(W+IPA) * eta(W+PG) / eta(W).
      2) Solid pigment: a relative-viscosity factor from the suspension
         model (Einstein / Batchelor / Krieger-Dougherty) using the
         pigment volume fraction phi.

    Physically motivated ESTIMATE, not a substitute for measurement.
    Use .calibrate(...) to anchor the model to a measured value.
    """

    def __init__(self, tables_dir="tables_parameters",
                 rho_water=RHO_WATER, rho_ipa=RHO_IPA, rho_pg=RHO_PG,
                 rho_pigment=RHO_PIGMENT,
                 suspension_model="batchelor",
                 einstein_coeff=2.5,     # c1 (1st order)
                 batchelor_coeff=7.2,    # c2 (2nd order)
                 intrinsic_viscosity=2.5, phi_max=0.63,  # Krieger-Dougherty only
                 calibration_factor=1.0):
        self.ipa = _load_viscosity_table(os.path.join(tables_dir, "ipa_viscosity.csv"))
        self.pg = _load_viscosity_table(os.path.join(tables_dir, "pg_viscosity.csv"))
        self.rho_water = rho_water
        self.rho_ipa = rho_ipa
        self.rho_pg = rho_pg
        self.rho_pigment = rho_pigment
        model = suspension_model.lower()
        if model not in ("batchelor", "einstein", "krieger-dougherty"):
            raise ValueError("suspension_model must be 'batchelor', 'einstein' or "
                             "'krieger-dougherty'.")
        self.suspension_model = model
        self.einstein_coeff = einstein_coeff
        self.batchelor_coeff = batchelor_coeff
        self.intrinsic_viscosity = intrinsic_viscosity
        self.phi_max = phi_max
        self.calibration_factor = calibration_factor

    def water_viscosity(self, temperature_C):
        """Water viscosity from the 0% row of the IPA table [mPa.s]."""
        eta, _, t_oob = _interp_viscosity_table(self.ipa, 0.0, temperature_C)
        return eta, t_oob

    def _suspension_factor(self, phi):
        """Relative viscosity increase from the particles -> (factor, warnings)."""
        warnings = []
        c1 = self.einstein_coeff
        if self.suspension_model == "einstein":
            factor = 1.0 + c1 * phi
        elif self.suspension_model == "batchelor":
            c2 = self.batchelor_coeff
            factor = 1.0 + c1 * phi + c2 * phi ** 2
        else:  # krieger-dougherty
            if phi >= self.phi_max:
                warnings.append(f"phi={phi:.3f} >= phi_max={self.phi_max} "
                                f"-- Krieger-Dougherty no longer valid.")
                return float("inf"), warnings
            factor = (1.0 - phi / self.phi_max) ** (-self.intrinsic_viscosity * self.phi_max)

        if phi > 0.10:
            warnings.append(f"Volume fraction phi={phi:.3f} > 0.10 -- polynomial "
                            f"approximations (Einstein/Batchelor) are valid only for "
                            f"dilute suspensions.")
        return factor, warnings

    def _model_label(self):
        if self.suspension_model == "einstein":
            return f"Einstein (1 + {self.einstein_coeff}*phi)"
        if self.suspension_model == "batchelor":
            return (f"Batchelor 2nd order "
                    f"(1 + {self.einstein_coeff}*phi + {self.batchelor_coeff}*phi^2)")
        return f"Krieger-Dougherty ([eta]={self.intrinsic_viscosity}, phi_max={self.phi_max})"

    def estimate(self, water, ipa, pg, aluminum, temperature_C, verbose=False):
        """Estimate the ink viscosity. All fractions in mass percent."""
        warnings = []
        total = water + ipa + pg + aluminum
        if abs(total - 100.0) > 0.5:
            warnings.append(f"Sum of fractions = {total:.2f} % (not 100). "
                            f"Ratios are used instead.")

        m_w, m_i, m_p, m_al = water, ipa, pg, aluminum
        carrier_mass = m_w + m_i + m_p
        if carrier_mass <= 0:
            raise ValueError("Liquid carrier (Water+IPA+PG) is 0 -- not computable.")

        # --- Step 1: table lookups + carrier (Arrhenius-like) ---
        ipa_bin = 100.0 * m_i / (m_i + m_w) if (m_i + m_w) > 0 else 0.0
        pg_bin = 100.0 * m_p / (m_p + m_w) if (m_p + m_w) > 0 else 0.0

        eta_w, t_oob_w = self.water_viscosity(temperature_C)
        if m_i > 0.0:
            eta_wi, mp_oob_i, t_oob_i = _interp_viscosity_table(
                self.ipa, ipa_bin, temperature_C)
        else:
            eta_wi, mp_oob_i, t_oob_i = eta_w, False, t_oob_w
        if m_p > 0.0:
            eta_wp, mp_oob_p, t_oob_p = _interp_viscosity_table(
                self.pg, pg_bin, temperature_C)
        else:
            eta_wp, mp_oob_p, t_oob_p = eta_w, False, t_oob_w

        if t_oob_w or t_oob_i or t_oob_p:
            warnings.append(
                f"Temperature {temperature_C} C is outside the locally "
                "supported viscosity range -- bounded local linear "
                "extrapolation was used.")
        if mp_oob_i:
            warnings.append(
                f"IPA pseudo-fraction {ipa_bin:.2f} % is outside or weakly "
                "supported by the IPA viscosity table.")
        if mp_oob_p:
            warnings.append(
                f"PG pseudo-fraction {pg_bin:.2f} % is outside or weakly "
                "supported by the PG viscosity table.")

        eta_carrier = eta_wi * eta_wp / eta_w

        # --- carrier density (inverse mixing rule) ---
        w_w, w_i, w_p = m_w / carrier_mass, m_i / carrier_mass, m_p / carrier_mass
        inv_rho = w_w / self.rho_water + w_i / self.rho_ipa + w_p / self.rho_pg
        rho_carrier = 1.0 / inv_rho

        # --- Step 2: pigment volume fraction + suspension factor ---
        v_al = m_al / self.rho_pigment
        v_carrier = carrier_mass / rho_carrier
        phi = v_al / (v_al + v_carrier) if (v_al + v_carrier) > 0 else 0.0

        susp_factor, susp_warn = self._suspension_factor(phi)
        warnings.extend(susp_warn)

        if w_w < 0.5:
            warnings.append("Water fraction in the carrier < 50 % -- estimate less "
                            "reliable (model designed for water-rich mixtures).")

        eta_ink = eta_carrier * susp_factor * self.calibration_factor

        result = {
            "viscosity_mPas": eta_ink,
            "temperature_C": temperature_C,
            "suspension_model": self._model_label(),
            "eta_water": eta_w,
            "eta_water_ipa_binary": eta_wi,
            "eta_water_pg_binary": eta_wp,
            "eta_carrier": eta_carrier,
            "ipa_pseudo_pct": ipa_bin,
            "pg_pseudo_pct": pg_bin,
            "rho_carrier": rho_carrier,
            "phi_pigment_vol": phi,
            "suspension_factor": susp_factor,
            "calibration_factor": self.calibration_factor,
            "warnings": warnings,
        }

        if verbose:
            self._print_report(water, ipa, pg, aluminum, result)
        return result

    def calibrate(self, measured_viscosity, water, ipa, pg, aluminum, temperature_C):
        """Set calibration_factor so the model matches a measured value exactly."""
        self.calibration_factor = 1.0
        base = self.estimate(water, ipa, pg, aluminum, temperature_C, verbose=False)
        self.calibration_factor = measured_viscosity / base["viscosity_mPas"]
        return self.calibration_factor

    @staticmethod
    def _print_report(water, ipa, pg, aluminum, r):
        print("=" * 60)
        print("Ink viscosity (estimate)")
        print("-" * 60)
        print(f"  Composition [mass %]: Water {water}, IPA {ipa}, PG {pg}, Al {aluminum}")
        print(f"  Temperature:          {r['temperature_C']} C")
        print(f"  Suspension model:     {r['suspension_model']}")
        print("-" * 60)
        print(f"  eta water:            {r['eta_water']:.3f} mPa*s")
        print(f"  pseudo-binary IPA:    {r['ipa_pseudo_pct']:.2f} % "
              f"-> {r['eta_water_ipa_binary']:.3f} mPa*s")
        print(f"  pseudo-binary PG:     {r['pg_pseudo_pct']:.2f} % "
              f"-> {r['eta_water_pg_binary']:.3f} mPa*s")
        print(f"  eta carrier (liquid): {r['eta_carrier']:.3f} mPa*s")
        print(f"  pigment vol fraction: {r['phi_pigment_vol']*100:.2f} vol-%")
        print(f"  suspension factor:    x{r['suspension_factor']:.4f}")
        if r["calibration_factor"] != 1.0:
            print(f"  calibration factor:   x{r['calibration_factor']:.4f}")
        print("-" * 60)
        print(f"  => eta ink:           {r['viscosity_mPas']:.3f} mPa*s")
        if r["warnings"]:
            print("-" * 60)
            for w in r["warnings"]:
                print(f"  ! {w}")
        print("=" * 60)


# =====================================================================
# ====================================================================
#  UNIFIED FACADE
# ====================================================================
# =====================================================================
@dataclass
class InkProperties:
    """Combined result of all four property models."""
    composition: dict          # mass % {'Al','IPA','PG','Water'}
    temperature: float         # C
    density: float             # g/cm3
    refractive_index: float    # nD
    sound_velocity: float      # m/s
    viscosity: float           # mPa.s
    vol_fraction_al: float     # phi_Al (-)
    details: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def __str__(self):
        c = self.composition
        lines = [
            "=" * 56,
            "  INK PROPERTIES",
            "-" * 56,
            "  Composition (mass %):",
            f"     Aluminum : {c['Al']:7.3f} %",
            f"     IPA      : {c['IPA']:7.3f} %",
            f"     PG       : {c['PG']:7.3f} %",
            f"     Water    : {c['Water']:7.3f} %",
            f"  Temperature : {self.temperature:7.2f} C",
            "  - - - - - - - - - - - - - - - - - - - - - - - - - - - - ",
            f"  Density            : {self.density:9.5f}  g/cm3",
            f"  Refractive index   : {self.refractive_index:9.5f}  nD",
            f"  Sound velocity     : {self.sound_velocity:9.2f}  m/s",
            f"  Viscosity          : {self.viscosity:9.3f}  mPa.s",
            f"  Al volume fraction : {self.vol_fraction_al * 100:9.3f}  %",
            "=" * 56,
        ]
        for w in self.warnings:
            lines.append(f"  [!] {w}")
        return "\n".join(lines)


class InkCalculator:
    """
    One reusable entry point for all ink properties.

    The aluminum / IPA / PG mass percentages are given explicitly and
    water is taken as the remainder (water = 100 - al - ipa - pg), unless
    an explicit ``water`` value is supplied. ``rho_pigment`` is the
    intrinsic/effective density of one pigment particle, not the loose or
    tapped packing density of a powder bed. It is applied consistently to
    density, sound and viscosity calculations.

        ink = InkCalculator(tables_dir="tables_parameters")
        print(ink.compute(al=1.82, ipa=3.64, pg=3.64, temperature=25.0))
    """

    def __init__(self, tables_dir="tables_parameters",
                 suspension_model="batchelor",
                 einstein_coeff=2.5, batchelor_coeff=7.2,
                 intrinsic_viscosity=2.5, phi_max=0.63,
                 rho_pigment=RHO_PIGMENT):
        self.tables_dir = tables_dir
        if rho_pigment <= 0.0:
            raise ValueError("rho_pigment must be positive.")

        # one shared density calculator (reused by the optical model)
        self.density_calc = InkDensityCalculator(tables_dir=tables_dir)
        self.density_calc.DENSITY_ALUMINUM = float(rho_pigment)
        self.refractive_calc = InkRefractiveCalculator(
            tables_dir=tables_dir, density_calculator=self.density_calc)
        self.sound_calc = InkSoundCalculator(tables_dir=tables_dir)
        self.sound_calc.DENSITY_ALUMINUM = float(rho_pigment)
        self.viscosity_model = InkViscosityModel(
            tables_dir=tables_dir,
            suspension_model=suspension_model,
            einstein_coeff=einstein_coeff,
            batchelor_coeff=batchelor_coeff,
            intrinsic_viscosity=intrinsic_viscosity,
            phi_max=phi_max,
            rho_pigment=rho_pigment)

    # ---- helpers -----------------------------------------------------
    @staticmethod
    def _validate_state(al, ipa, pg, temperature):
        values = {"Al": al, "IPA": ipa, "PG": pg,
                  "temperature": temperature}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Composition and temperature must be finite.")
        if min(al, ipa, pg) < 0.0:
            raise ValueError("Mass percentages must be non-negative.")
        if al + ipa + pg > 100.0 + 1e-9:
            raise ValueError(
                f"Al + IPA + PG = {al + ipa + pg:.3f}% exceeds 100%.")

    @classmethod
    def _water_remainder(cls, al, ipa, pg, water, temperature=25.0):
        cls._validate_state(al, ipa, pg, temperature)
        if water is None:
            water = 100.0 - al - ipa - pg
        if not np.isfinite(water) or water < -1e-9:
            raise ValueError("Water mass percentage must be finite and non-negative.")
        if abs(al + ipa + pg + water - 100.0) > 1e-6:
            raise ValueError(
                "Explicit Al + IPA + PG + water must sum to 100 mass %."
            )
        return max(water, 0.0)

    # ---- individual properties --------------------------------------
    def density(self, al=0.0, ipa=0.0, pg=0.0, temperature=25.0):
        """Ink density [g/cm3]."""
        self._validate_state(al, ipa, pg, temperature)
        return self.density_calc.calculate_density(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)

    def refractive_index(self, al=0.0, ipa=0.0, pg=0.0, temperature=25.0):
        """Refractive index (nD) of the liquid matrix (pigment excluded)."""
        self._validate_state(al, ipa, pg, temperature)
        return self.refractive_calc.calculate_refractive_index(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)

    def sound_velocity(self, al=0.0, ipa=0.0, pg=0.0, temperature=25.0):
        """Sound velocity [m/s]."""
        self._validate_state(al, ipa, pg, temperature)
        return self.sound_calc.sound_velocity(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, temperature=temperature)

    def viscosity(self, al=0.0, ipa=0.0, pg=0.0, temperature=25.0, water=None):
        """Ink viscosity [mPa.s]."""
        water = self._water_remainder(
            al, ipa, pg, water, temperature=temperature)
        return self.viscosity_model.estimate(
            water=water, ipa=ipa, pg=pg, aluminum=al,
            temperature_C=temperature, verbose=False)["viscosity_mPas"]

    # ---- everything at once -----------------------------------------
    def compute(self, al=0.0, ipa=0.0, pg=0.0, temperature=25.0, water=None):
        """
        Compute all four properties at once and return an InkProperties
        object (printable). Water is the remainder unless given explicitly.
        """
        water = self._water_remainder(
            al, ipa, pg, water, temperature=temperature)
        warnings = []
        details = {}

        # --- density ---
        rho = self.density_calc.calculate_density(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)

        # --- refractive index (matrix only) ---
        n_d = self.refractive_calc.calculate_refractive_index(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)

        # --- sound velocity ---
        sound = self.sound_calc.calculate(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, temperature=temperature)
        warnings.extend(sound.warnings)
        details["density_wood_model"] = sound.density

        # --- viscosity ---
        visc = self.viscosity_model.estimate(
            water=water, ipa=ipa, pg=pg, aluminum=al,
            temperature_C=temperature, verbose=False)
        warnings.extend(visc["warnings"])
        details["viscosity"] = visc

        return InkProperties(
            composition={"Al": al, "IPA": ipa, "PG": pg, "Water": water},
            temperature=temperature,
            density=rho,
            refractive_index=n_d,
            sound_velocity=sound.sound_velocity,
            viscosity=visc["viscosity_mPas"],
            vol_fraction_al=sound.vol_fraction_al,
            details=details,
            warnings=warnings,
        )

    # ---- calibration passthrough ------------------------------------
    def calibrate_viscosity(self, measured_viscosity, al, ipa, pg,
                            temperature=25.0, water=None):
        """Anchor the viscosity model to a measured value (sets its calibration factor)."""
        water = self._water_remainder(
            al, ipa, pg, water, temperature=temperature)
        return self.viscosity_model.calibrate(
            measured_viscosity, water=water, ipa=ipa, pg=pg,
            aluminum=al, temperature_C=temperature)

    def calibrate_sound(self, measured_velocity, al, ipa=0.0, pg=0.0,
                        temperature=25.0):
        """
        Anchor the sound model to a measured value (sets its additive
        offset in m/s). Returns the offset (measured - model).
        """
        return self.sound_calc.calibrate(
            measured_velocity, pct_al=al, pct_ipa=ipa, pct_pg=pg,
            temperature=temperature)

    # ---- pigment-paste mode -----------------------------------------
    @staticmethod
    def paste_composition(paste, ipa=0.0, pg=0.0,
                          solids_fraction=0.20,
                          ipa_fraction=0.40, pg_fraction=0.40,
                          rho_particle=2.20):
        """
        Convert a pigment-PASTE dosage [mass %] into effective model
        inputs. Defaults match the internal composition of ECOLEAF
        SL 120:  20 % encapsulated Al pigment + 40 % IPA + 40 % PG.
        (Consistent with the SDS: PG is not hazardous, hence absent
        from SDS section 3 but present in its DNEL/PNEC tables;
        IPA 30-50 %; total VOC 81 % ~ IPA + PG.)

        Mapping:
          * the encapsulated pigment (metal + shell) is one particle
            phase with density rho_particle (estimate ~2.2 g/cm3;
            determinable exactly from a measured paste density via
            1/rho_paste = solids/rho_p + w_IPA/rho_IPA + w_PG/rho_PG).
            The exact value has little effect at phi < 1 vol-%.
          * carrier IPA and PG go into the liquid phase.
          * any unassigned mass fraction of the paste is treated as
            water (for SL 120 the fractions sum to 1, so none).

        For an older PG-free paste (e.g. the Anton Paar trial inks) use
        ipa_fraction=0.80, pg_fraction=0.0.

        The standard ink (1 part paste + 10 parts water) is
        paste = 100/11 = 9.0909 %  ->  1.818 % Al, 3.636 % IPA,
        3.636 % PG, 90.909 % water.

        Returns a dict: {'al', 'ipa', 'pg', 'rho_particle'}.
        """
        if min(solids_fraction, ipa_fraction, pg_fraction) < 0:
            raise ValueError("Paste fractions must be non-negative.")
        if solids_fraction + ipa_fraction + pg_fraction > 1.0 + 1e-9:
            raise ValueError("Paste fractions must not exceed 1 in total.")
        if rho_particle <= 0:
            raise ValueError("rho_particle must be positive.")

        return {"al": paste * solids_fraction,
                "ipa": ipa + paste * ipa_fraction,
                "pg": pg + paste * pg_fraction,
                "rho_particle": rho_particle}

    def compute_from_paste(self, paste, ipa=0.0, pg=0.0, temperature=25.0,
                           solids_fraction=0.20,
                           ipa_fraction=0.40, pg_fraction=0.40,
                           rho_particle=2.20):
        """
        Compute all four properties for an ink specified via pigment-
        PASTE dosage (see paste_composition). The effective particle
        density (encapsulated pigment) is applied consistently to the
        density, sound and viscosity models for this call. The particle
        bulk modulus is left at K_Al; at phi < 1 vol-% the resulting
        error in c is < 0.5 m/s.

        Example (standard ink, 1 part SL 120 + 10 parts water):
            props = ink.compute_from_paste(paste=100.0 / 11.0)
        """
        comp = self.paste_composition(
            paste, ipa=ipa, pg=pg, solids_fraction=solids_fraction,
            ipa_fraction=ipa_fraction, pg_fraction=pg_fraction,
            rho_particle=rho_particle)

        saved = (self.density_calc.DENSITY_ALUMINUM,
                 self.sound_calc.DENSITY_ALUMINUM,
                 self.viscosity_model.rho_pigment)
        self.density_calc.DENSITY_ALUMINUM = comp["rho_particle"]
        self.sound_calc.DENSITY_ALUMINUM = comp["rho_particle"]
        self.viscosity_model.rho_pigment = comp["rho_particle"]
        try:
            props = self.compute(al=comp["al"], ipa=comp["ipa"],
                                 pg=comp["pg"], temperature=temperature)
        finally:
            (self.density_calc.DENSITY_ALUMINUM,
             self.sound_calc.DENSITY_ALUMINUM,
             self.viscosity_model.rho_pigment) = saved

        props.details["paste"] = {
            "paste_pct": paste,
            "solids_fraction": solids_fraction,
            "ipa_fraction": ipa_fraction,
            "pg_fraction": pg_fraction,
            "effective_particle_pct": comp["al"],
            "ipa_from_paste_pct": paste * ipa_fraction,
            "pg_from_paste_pct": paste * pg_fraction,
            "rho_particle": comp["rho_particle"],
        }
        return props


# =====================================================================
#  Example / direct use
# =====================================================================
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    tables = os.path.join(here, "tables_parameters")

    ink = InkCalculator(tables_dir=tables)

    # # original ink: 1.82% Al, 3.64% IPA, 3.64% PG, rest water, 25 C
    # props = ink.compute(al=1.82, ipa=3.64, pg=3.64, temperature=25)
    # print(props)

    # individual numbers, if you only need one property:
    #   rho = ink.density(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    #   n_d = ink.refractive_index(ipa=3.64, pg=3.64, temperature=25.0)
    #   c   = ink.sound_velocity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    #   eta = ink.viscosity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)

    # anchor the viscosity estimate to a measured value:
    #   ink.calibrate_viscosity(1.9, al=1.82, ipa=3.64, pg=3.64, temperature=25.0)

    # anchor the sound model to a measured value (additive offset, m/s):
    #   ink.calibrate_sound(1542.1, al=1.82, ipa=3.64, pg=3.64, temperature=25.0)

    # quick sanity check of the sound model (water limit + continuity):
    ink.sound_calc.self_test(verbose=True)
    print()

    props = ink.compute(al=1.8128, ipa=3.6256, pg=3.6256, temperature=23.66)
    print(props)