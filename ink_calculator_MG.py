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
into one consistent interface. The underlying physics, formulas and
interpolation methods of each original model are preserved unchanged,
so results are identical to running the four scripts individually.

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

    # with dissolved methyl gallate (dilute-solute corrections, mass %):
    props = ink.compute(al=1.81, ipa=3.63, pg=3.63, mg=0.23, temperature=25.0)

    # or each property on its own (returns a plain number):
    rho  = ink.density(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    n_d  = ink.refractive_index(ipa=3.64, pg=3.64, temperature=25.0)
    c    = ink.sound_velocity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    eta  = ink.viscosity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)

Convention: water is the remainder, water = 100 - al - ipa - pg.
"""

import os
import re
import csv
import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# =====================================================================
#  Shared material constants  (all overridable)
# =====================================================================
RHO_WATER = 0.998       # g/cm3  (~20-25 C)
RHO_IPA = 0.785         # g/cm3
RHO_PG = 1.036          # g/cm3
RHO_PIGMENT = 2.700     # g/cm3   bulk aluminum; encapsulated may be lower
BULK_MODULUS_ALUMINUM = 76.0e9   # Pa   (compressibility beta_Al = 1 / K_Al)

# --- Methyl gallate (MG, methyl 3,4,5-trihydroxybenzoate, CAS 99-24-1) ---
# MG is a DISSOLVED solute (not a suspended solid, not a miscible cosolvent),
# therefore its solution behaviour is described by apparent molar quantities,
# NOT by the crystal density. See MethylGallateModel for sources.
M_MG = 184.15           # g/mol   molar mass (user / PubChem CID 7428)
RHO_MG_SOLID = 1.526    # g/cm3   crystal density (database) -- REFERENCE ONLY
V_PHI_MG = 115.9        # cm3/mol apparent molar volume in water (group additivity)
R_MOL_MG = 41.77        # cm3/mol molar refraction, Na-D (Eisenlohr/Vogel increments)
K_PHI_S_MG = -25.0e-15  # m3/(mol Pa) apparent molar isentropic compression (estimate)
B_JD_MG = 0.45e-3       # m3/mol  Jones-Dole B coefficient (estimate)


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
        self.interpolators = {'IPA': {}, 'PG': {}}
        self._load_tables()

    def _load_tables(self):
        """Load CSV files and build the concentration interpolators."""
        ipa_path = os.path.join(self.tables_dir, "ipa_density.csv")
        pg_path = os.path.join(self.tables_dir, "pg_density.csv")

        # IPA table (e.g. 20C and 30C) -> linear in concentration
        if os.path.exists(ipa_path):
            df_ipa = pd.read_csv(ipa_path)
            self.interpolators['IPA']['20C'] = interp1d(
                df_ipa['Mass_Percent'], df_ipa['Density_20C'], kind='linear')
            self.interpolators['IPA']['30C'] = interp1d(
                df_ipa['Mass_Percent'], df_ipa['Density_30C'], kind='linear')
        else:
            print(f"Warning: '{ipa_path}' not found. IPA density calculations will fail.")

        # PG table (e.g. 20C, 25C, 30C) -> cubic in concentration
        if os.path.exists(pg_path):
            df_pg = pd.read_csv(pg_path)
            self.interpolators['PG']['20C'] = interp1d(
                df_pg['Mass_Percent'], df_pg['Density_20C'], kind='cubic')
            self.interpolators['PG']['25C'] = interp1d(
                df_pg['Mass_Percent'], df_pg['Density_25C'], kind='cubic')
            self.interpolators['PG']['30C'] = interp1d(
                df_pg['Mass_Percent'], df_pg['Density_30C'], kind='cubic')
        else:
            print(f"Warning: '{pg_path}' not found. PG density calculations will fail.")

    def get_liquid_density(self, solvent_type, mass_percent, target_temp=25):
        """
        Real density [g/cm3] of a binary liquid (Water + solvent),
        interpolated in concentration and temperature.

        :param solvent_type: 'IPA' or 'PG'
        :param mass_percent: solvent mass percent within the liquid phase
        :param target_temp:  temperature in Celsius (20 .. 30)
        """
        if not (20 <= target_temp <= 30):
            raise ValueError("Target temperature must be between 20 C and 30 C.")

        funcs = self.interpolators[solvent_type]

        if solvent_type == 'IPA':
            dens_20 = float(funcs['20C'](mass_percent))
            dens_30 = float(funcs['30C'](mass_percent))
            slope = (dens_30 - dens_20) / (30 - 20)
            return dens_20 + slope * (target_temp - 20)

        elif solvent_type == 'PG':
            dens_20 = float(funcs['20C'](mass_percent))
            dens_25 = float(funcs['25C'](mass_percent))
            dens_30 = float(funcs['30C'](mass_percent))

            if target_temp == 20:
                return dens_20
            if target_temp == 25:
                return dens_25
            if target_temp == 30:
                return dens_30

            if 20 < target_temp < 25:
                slope = (dens_25 - dens_20) / (25 - 20)
                return dens_20 + slope * (target_temp - 20)
            else:  # 25 < target_temp < 30
                slope = (dens_30 - dens_25) / (30 - 25)
                return dens_25 + slope * (target_temp - 25)

        else:
            raise ValueError(f"Unknown solvent type: {solvent_type}")

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
        self.interpolators = {'IPA': {}, 'PG': {}}

        # density calculator is required for the Gladstone-Dale step
        if density_calculator is None:
            self.density_calc = InkDensityCalculator(tables_dir=tables_dir)
        else:
            self.density_calc = density_calculator

        self._load_tables()

    def _load_tables(self):
        """Load refractive-index CSV files and build interpolators."""
        ipa_path = os.path.join(self.tables_dir, "ipa_refractive.csv")
        pg_path = os.path.join(self.tables_dir, "pg_refractive.csv")

        # IPA: single empirical anchor (25C)
        if os.path.exists(ipa_path):
            df_ipa = pd.read_csv(ipa_path)
            self.interpolators['IPA']['25C'] = interp1d(
                df_ipa['Mass_Percent'], df_ipa['Refractive_25C'], kind='cubic')
        else:
            print(f"Warning: '{ipa_path}' not found. IPA optical calculations will fail.")

        # PG: two empirical anchors (22C and 25C)
        if os.path.exists(pg_path):
            df_pg = pd.read_csv(pg_path)
            self.interpolators['PG']['22C'] = interp1d(
                df_pg['Mass_Percent'], df_pg['Refractive_22C'], kind='cubic')
            self.interpolators['PG']['25C'] = interp1d(
                df_pg['Mass_Percent'], df_pg['Refractive_25C'], kind='cubic')
        else:
            print(f"Warning: '{pg_path}' not found. PG optical calculations will fail.")

    def get_liquid_refractive_index(self, solvent_type, mass_percent, target_temp=25):
        """
        Refractive index (nD) of a binary liquid (Water + solvent) at any
        temperature. IPA is corrected with a thermo-optic coefficient
        (dn/dT); PG is interpolated/extrapolated from its two anchors.
        """
        funcs = self.interpolators[solvent_type]

        if solvent_type == 'IPA':
            n_25 = float(funcs['25C'](mass_percent))
            if target_temp == 25:
                return n_25
            # thermo-optic coefficients (per degC): water ~ -1e-4, IPA ~ -4e-4
            dn_dt_water = -0.00010
            dn_dt_ipa = -0.00040
            dn_dt_mix = dn_dt_water + (mass_percent / 100.0) * (dn_dt_ipa - dn_dt_water)
            delta_t = target_temp - 25.0
            return n_25 + (dn_dt_mix * delta_t)

        elif solvent_type == 'PG':
            temps = [22, 25]
            n_vals = [
                float(funcs['22C'](mass_percent)),
                float(funcs['25C'](mass_percent)),
            ]
            temp_interpolator = interp1d(
                temps, n_vals, kind='linear', fill_value='extrapolate')
            return float(temp_interpolator(target_temp))

        else:
            raise ValueError(f"Unknown solvent type: {solvent_type}")

    def calculate_refractive_index(self, pct_al=0.0, pct_ipa=0.0, pct_pg=0.0, target_temp=25):
        """
        Refractive index (nD) of the liquid matrix. Handles single
        solvents directly and applies the Gladstone-Dale pseudo-binary
        approximation when IPA and PG coexist.
        """
        # NEU: Wir berechnen die wahre Wassermenge unter Einbezug von Aluminium
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg
        if pct_water < 0:
            raise ValueError("Total mass percentage exceeds 100%. Check your inputs.")

        # Masse der reinen flüssigen Phase
        pct_liquid_total = pct_ipa + pct_pg + pct_water
        if pct_liquid_total <= 0:
            return 1.0  # Fallback

        # Wahre Konzentrationen der Lösungsmittel IN der Flüssigphase
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
class _PropertyTable:
    """
    Loads a CSV with a 'Mass_Percent' column and one or more
    '<prefix>_<temp>C' columns and provides value(mass_percent, temp)
    by interpolating over concentration, then over temperature.
    """

    def __init__(self, path, prefix):
        self.path = path
        self.prefix = prefix
        df = pd.read_csv(path)

        if "Mass_Percent" not in df.columns:
            raise ValueError(f"'{path}' has no 'Mass_Percent' column.")

        mp_full = df["Mass_Percent"].to_numpy(dtype=float)
        order = np.argsort(mp_full)
        mp_full = mp_full[order]

        self.mp_min, self.mp_max = float(mp_full.min()), float(mp_full.max())

        self.temps = {}
        pattern = re.compile(rf"{prefix}_(\d+(?:\.\d+)?)C")
        for col in df.columns:
            m = pattern.fullmatch(col)
            if m:
                t = float(m.group(1))
                y_full = df[col].to_numpy(dtype=float)[order]

                # NEU: Filtere leere Zellen (NaN) für diese spezifische Spalte heraus
                valid_mask = ~np.isnan(y_full)
                mp_valid = mp_full[valid_mask]
                y_valid = y_full[valid_mask]

                # Überspringe die Spalte komplett, falls sie gar keine Daten enthält
                if len(mp_valid) == 0:
                    continue

                # Bestimme die Interpolationsmethode anhand der GÜLTIGEN Datenpunkte
                kind = "cubic" if len(mp_valid) >= 4 else "linear"

                self.temps[t] = interp1d(
                    mp_valid, y_valid, kind=kind, bounds_error=False, fill_value="extrapolate")

        if not self.temps:
            raise ValueError(f"'{path}' has no valid '{prefix}_XXC' columns.")
        self.available_temps = sorted(self.temps)

    def value(self, mass_percent, temperature):
        """Interpolated property at the given mass percent and temperature."""
        vals = np.array(
            [float(self.temps[t](mass_percent)) for t in self.available_temps])
        if temperature in self.temps:
            return float(self.temps[temperature](mass_percent))
        if len(self.available_temps) == 1:
            return float(vals[0])
        temp_interp = interp1d(
            self.available_temps, vals, kind="linear",
            bounds_error=False, fill_value="extrapolate")
        return float(temp_interp(temperature))

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
    warnings: list = field(default_factory=list)


class InkSoundCalculator:
    """
    Sound velocity of an Al-pigment / Water / IPA / PG ink.

    Sound velocity cannot be mixed linearly. Instead density (rho) and
    adiabatic compressibility (beta) are mixed over volume fractions and
    converted back via Newton-Laplace:  c = 1 / sqrt(rho * beta).
    The solid pigment is added with the Wood/Urick effective-medium
    equation; beta_Al comes from the bulk modulus K_Al.
    """

    DENSITY_ALUMINUM = RHO_PIGMENT          # g/cm3
    BULK_MODULUS_ALUMINUM = BULK_MODULUS_ALUMINUM  # Pa

    def __init__(self, tables_dir="tables_parameters"):
        self.tables_dir = tables_dir
        self.density = {"IPA": None, "PG": None}
        self.sound = {"IPA": None, "PG": None}
        self._load("IPA", "ipa_density.csv", "ipa_sound.csv")
        self._load("PG", "pg_density.csv", "pg_sound.csv")

    def _load(self, solvent, density_file, sound_file):
        dpath = os.path.join(self.tables_dir, density_file)
        spath = os.path.join(self.tables_dir, sound_file)
        try:
            self.density[solvent] = _PropertyTable(dpath, "Density")
        except (FileNotFoundError, ValueError) as e:
            print(f"Note: {solvent} density table not loaded ({e}).")
        try:
            self.sound[solvent] = _PropertyTable(spath, "SoundVelocity")
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
        if not dtab.temp_in_range(temperature) or not stab.temp_in_range(temperature):
            warnings.append(
                f"Temperature {temperature:.1f} C is outside the tabulated "
                f"range -> extrapolated.")

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
        c_mix = 1.0 / np.sqrt(rho_mix_si * beta_mix)

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
            warnings=warnings,
        )

    def sound_velocity(self, pct_al, pct_ipa=0.0, pct_pg=0.0, temperature=25.0):
        """Return only the sound velocity in m/s."""
        return self.calculate(pct_al, pct_ipa, pct_pg, temperature).sound_velocity


# =====================================================================
# ====================================================================
#  4) VISCOSITY MODEL   (ported from ink_viscosity.py)
# ====================================================================
# =====================================================================
def _load_viscosity_table(path):
    """Read a viscosity CSV and return a dict of support points (stdlib only)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        temps = [float(h.replace("Viscosity_", "").rstrip("Cc")) for h in header[1:]]
        mass_percent, eta = [], []
        for row in reader:
            if not row:
                continue
            mass_percent.append(float(row[0]))
            eta.append([float(v) for v in row[1:]])
    order = sorted(range(len(mass_percent)), key=lambda i: mass_percent[i])
    mass_percent = [mass_percent[i] for i in order]
    eta = [eta[i] for i in order]
    return {"temps": temps, "mass_percent": mass_percent, "eta": eta, "path": path}


def _interp_1d(xs, ys, x):
    """Linear interpolation. Returns (value, out_of_range); clamps at edges."""
    if x <= xs[0]:
        return ys[0], (x < xs[0] - 1e-9)
    if x >= xs[-1]:
        return ys[-1], (x > xs[-1] + 1e-9)
    i = bisect.bisect_right(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0), False


def _interp_viscosity_table(table, mass_percent, temperature_C):
    """Bilinear interpolation in (mass percent, temperature)."""
    temps = table["temps"]
    mps = table["mass_percent"]
    grid = table["eta"]
    eta_vs_T, mp_oob = [], False
    for j in range(len(temps)):
        col = [grid[i][j] for i in range(len(mps))]
        val, oob = _interp_1d(mps, col, mass_percent)
        eta_vs_T.append(val)
        mp_oob = mp_oob or oob
    eta, t_oob = _interp_1d(temps, eta_vs_T, temperature_C)
    return eta, mp_oob, t_oob


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
        eta_wi, mp_oob_i, t_oob_i = _interp_viscosity_table(self.ipa, ipa_bin, temperature_C)
        eta_wp, mp_oob_p, t_oob_p = _interp_viscosity_table(self.pg, pg_bin, temperature_C)

        if t_oob_w or t_oob_i or t_oob_p:
            warnings.append(f"Temperature {temperature_C} C outside the table range "
                            f"-- clamped to the edge.")
        if mp_oob_i:
            warnings.append(f"IPA pseudo-fraction {ipa_bin:.2f} % outside the IPA table.")
        if mp_oob_p:
            warnings.append(f"PG pseudo-fraction {pg_bin:.2f} % outside the PG table.")

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
#  5) METHYL GALLATE (MG) SOLUTE MODEL -- dilute-solution corrections
# ====================================================================
# =====================================================================
class MethylGallateModel:
    """
    First-order solute corrections for methyl gallate (MG) dissolved in
    the aqueous carrier of the ink.

    WHY A SEPARATE MODEL
    --------------------
    MG cannot be treated like IPA/PG (no full-range binary tables exist,
    MG is only sparingly soluble in water, ~20-25 g/L) and not like the
    pigment (it is dissolved, not suspended). In the dilute regime
    (w_MG << 1, design point ~0.23 mass %) every property change is
    strictly linear in concentration, so each property needs exactly one
    well-defined increment, taken from apparent molar quantities:

        density           apparent molar volume V_phi
        refractive index  molar refraction R  +  Lorentz-Lorenz
        sound velocity    apparent molar isentropic compression K_phi_s
        viscosity         Jones-Dole:  eta = eta0 * (1 + B*C)

    IMPORTANT: the crystal density (RHO_MG_SOLID = 1.526 g/cm3) is kept
    for reference only. Dissolved MG occupies V_phi = 115.9 cm3/mol
    (effective 1.589 g/cm3) because the three phenolic OH groups
    electrostrict their hydration water. Using the crystal density
    would underestimate the density increment and (worse) miss the
    hydration stiffening that dominates the sound-velocity increment.

    PARAMETER SOURCES / TRACEABILITY
    --------------------------------
    V_PHI_MG   = V_phi(gallic acid, aq, ~100 cm3/mol, literature)
                 + CH2 group increment (+15.9 cm3/mol).
    R_MOL_MG   = Eisenlohr (Z. Phys. Chem. 75 (1911) 585) / Vogel
                 (J. Chem. Soc. 1948, 1833) atomic refractions for
                 C8H8O5: 8*C + 8*H + 3*O(OH) + O(C=O) + O(ester)
                 + 3*C=C(aromatic) = 41.77 cm3/mol.
    K_PHI_S_MG = group estimate for a trihydroxy aromatic ester,
                 band -15e-15 .. -35e-15 m3/(mol Pa)
                 (cf. Hoiland in Hinz (ed.), Thermodynamic Data for
                 Biochemistry and Biotechnology, Springer 1986;
                 Galema & Hoiland, J. Phys. Chem. 95 (1991) 5321).
                 >>> largest model uncertainty: replace with the value
                 from ONE differential measurement water vs water+MG
                 (see fit_k_phi_s_from_measurement).
    B_JD_MG    = gallic-acid analogy + CH2, band 0.35e-3 .. 0.55e-3
                 m3/mol (Jones-Dole; A*sqrt(C) term = 0, nonelectrolyte).

    Validity: dilute solutions (w_MG < ~1 %), 20..30 C. All parameters
    are per-instance and overridable, e.g.
        MethylGallateModel(k_phi_s=-18e-15)
    """

    SOLUBILITY_WATER_G_PER_L = 22.0   # approx. water solubility at 25 C

    def __init__(self,
                 molar_mass=M_MG,              # g/mol
                 rho_solid=RHO_MG_SOLID,       # g/cm3 (reference only)
                 v_phi=V_PHI_MG,               # cm3/mol
                 molar_refraction=R_MOL_MG,    # cm3/mol
                 k_phi_s=K_PHI_S_MG,           # m3/(mol Pa)
                 b_jones_dole=B_JD_MG):        # m3/mol
        self.M = molar_mass
        self.rho_solid = rho_solid
        self.v_phi = v_phi
        self.R_mol = molar_refraction
        self.k_phi_s = k_phi_s
        self.B = b_jones_dole

    # -- specific volume of the DISSOLVED solute [cm3/g] ---------------
    @property
    def v_specific(self):
        return self.v_phi / self.M

    # ------------------------------------------------------------------
    def validity_warnings(self, w_mg, w_mg_liquid, rho_liquid):
        """Dilute-limit and solubility checks (fractions 0..1)."""
        warnings = []
        conc_g_per_l = w_mg_liquid * rho_liquid * 1000.0
        if conc_g_per_l > self.SOLUBILITY_WATER_G_PER_L:
            warnings.append(
                f"MG in the liquid phase = {conc_g_per_l:.1f} g/L exceeds the "
                f"approximate water solubility (~{self.SOLUBILITY_WATER_G_PER_L:.0f} g/L) "
                f"-- undissolved MG violates the solute model.")
        if w_mg > 0.01:
            warnings.append(
                f"w_MG = {w_mg * 100:.2f} % -- the dilute-solution corrections are "
                f"first order in concentration; accuracy degrades above ~1 %.")
        return warnings

    # ------------------------------------------------------------------
    def corrected_density(self, rho_base, w_mg, rho_water):
        """
        Density with MG [g/cm3] from the volume balance 1/rho = sum(w_i*v_i).

        rho_base  : density of the SAME liquid/ink with MG replaced by water
        w_mg      : MG mass fraction (0..1) referred to rho_base's basis
        rho_water : pure-water density at the same temperature [g/cm3]

        MG replaces water  ->  d(1/rho) = w_mg * (v_MG - v_water).
        """
        d_inv_rho = w_mg * (self.v_specific - 1.0 / rho_water)
        return 1.0 / (1.0 / rho_base + d_inv_rho)

    def corrected_refractive_index(self, n_base, w_mg_liquid, rho_liquid):
        """
        Matrix refractive index with MG via Lorentz-Lorenz.

        n_base      : nD of the liquid matrix without MG
        w_mg_liquid : MG mass fraction WITHIN the liquid phase (0..1)
        rho_liquid  : density of the liquid phase incl. MG [g/cm3]

        LL(solute) = R_mol / V_phi; the LL function mixes linearly in
        volume fraction (exact in the dilute limit).
        """
        ll_base = (n_base ** 2 - 1.0) / (n_base ** 2 + 2.0)
        ll_mg = self.R_mol / self.v_phi
        phi_mg = w_mg_liquid * rho_liquid * self.v_specific   # volume fraction
        ll_new = ll_base + phi_mg * (ll_mg - ll_base)
        return float(np.sqrt((1.0 + 2.0 * ll_new) / (1.0 - ll_new)))

    def corrected_sound_velocity(self, c_base, rho_base_si, rho_new_si, w_mg):
        """
        Sound velocity with MG [m/s] from the isentropic-compression
        balance (per kg of ink, SI units):

            kappa_new * V_new = kappa_base * V_base + n_MG * K_phi_s

        with V = 1/rho, V_base = (1 - w_mg)/rho_base (base-fluid share)
        and kappa_base from Newton-Laplace. The negative K_phi_s adds the
        hydration stiffening that a solid-phase (Wood/Urick) treatment
        of MG would miss entirely.
        """
        kappa_base = 1.0 / (rho_base_si * c_base ** 2)
        n_mg = w_mg / (self.M / 1000.0)          # mol per kg of ink
        v_new = 1.0 / rho_new_si                 # m3/kg
        v_base = (1.0 - w_mg) / rho_base_si      # m3/kg
        kappa_new = (kappa_base * v_base + n_mg * self.k_phi_s) / v_new
        return float(1.0 / np.sqrt(rho_new_si * kappa_new))

    def viscosity_factor(self, w_mg_liquid, rho_liquid):
        """
        Jones-Dole multiplier (1 + B*C) for the carrier viscosity.
        C = molarity of MG in the liquid phase [mol/m3];
        rho_liquid in g/cm3. A*sqrt(C) term omitted (nonelectrolyte).
        """
        c_molar = (w_mg_liquid * rho_liquid * 1000.0) / (self.M / 1000.0)
        return 1.0 + self.B * c_molar

    # ------------------------------------------------------------------
    def fit_k_phi_s_from_measurement(self, c_measured, c_base,
                                     rho_base_si, rho_new_si, w_mg):
        """
        Invert the compression balance: obtain K_phi_s from ONE measured
        differential sound velocity (e.g. water vs water + 0.23 % MG) and
        store it on the instance. Returns the fitted value [m3/(mol Pa)].
        """
        kappa_base = 1.0 / (rho_base_si * c_base ** 2)
        kappa_meas = 1.0 / (rho_new_si * c_measured ** 2)
        n_mg = w_mg / (self.M / 1000.0)
        v_new = 1.0 / rho_new_si
        v_base = (1.0 - w_mg) / rho_base_si
        self.k_phi_s = (kappa_meas * v_new - kappa_base * v_base) / n_mg
        return self.k_phi_s


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
            f"     MG       : {c.get('MG', 0.0):7.3f} %",
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
    an explicit ``water`` value is supplied.

        ink = InkCalculator(tables_dir="tables_parameters")
        print(ink.compute(al=1.82, ipa=3.64, pg=3.64, temperature=25.0))
    """

    def __init__(self, tables_dir="tables_parameters",
                 suspension_model="batchelor",
                 einstein_coeff=2.5, batchelor_coeff=7.2,
                 intrinsic_viscosity=2.5, phi_max=0.63,
                 rho_pigment=RHO_PIGMENT,
                 mg_model=None):
        self.tables_dir = tables_dir

        # dilute-solute model for methyl gallate (parameters overridable)
        self.mg_model = mg_model if mg_model is not None else MethylGallateModel()

        # one shared density calculator (reused by the optical model)
        self.density_calc = InkDensityCalculator(tables_dir=tables_dir)
        self.refractive_calc = InkRefractiveCalculator(
            tables_dir=tables_dir, density_calculator=self.density_calc)
        self.sound_calc = InkSoundCalculator(tables_dir=tables_dir)
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
    def _water_remainder(al, ipa, pg, water, mg=0.0):
        if water is None:
            water = 100.0 - al - ipa - pg - mg
        if water < -1e-9:
            raise ValueError(
                f"Al + IPA + PG + MG = {al + ipa + pg + mg:.3f}% exceeds 100% "
                f"(water would be negative).")
        return max(water, 0.0)

    def _mg_context(self, al, ipa, pg, mg, temperature):
        """
        Common quantities for the MG solute corrections.

        The BASE state for every correction is the same ink with the MG
        mass replaced by water (the tabulated models compute exactly
        that, because they take water as the remainder of al/ipa/pg).
        """
        w_mg = mg / 100.0
        liquid_mass = 100.0 - al                 # liquid matrix incl. MG, mass-%
        w_mg_liq = (mg / liquid_mass) if liquid_mass > 0 else 0.0

        # pure-water density at T from the 0 % row of the IPA table
        rho_water = self.density_calc.get_liquid_density('IPA', 0.0, temperature)

        # liquid matrix (no pigment), MG replaced by water -> then corrected
        ipa_liq = ipa / liquid_mass * 100.0 if liquid_mass > 0 else 0.0
        pg_liq = pg / liquid_mass * 100.0 if liquid_mass > 0 else 0.0
        rho_liq_base = self.density_calc.calculate_density(
            pct_al=0.0, pct_ipa=ipa_liq, pct_pg=pg_liq, target_temp=temperature)
        rho_liq = self.mg_model.corrected_density(rho_liq_base, w_mg_liq, rho_water)

        return {"w_mg": w_mg, "w_mg_liq": w_mg_liq,
                "rho_water": rho_water, "rho_liq": rho_liq}

    # ---- individual properties --------------------------------------
    def density(self, al=0.0, ipa=0.0, pg=0.0, mg=0.0, temperature=25.0):
        """Ink density [g/cm3]. mg = mass % dissolved methyl gallate."""
        rho = self.density_calc.calculate_density(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)
        if mg > 0:
            ctx = self._mg_context(al, ipa, pg, mg, temperature)
            rho = self.mg_model.corrected_density(rho, ctx["w_mg"], ctx["rho_water"])
        return rho

    def refractive_index(self, al=0.0, ipa=0.0, pg=0.0, mg=0.0, temperature=25.0):
        """Refractive index (nD) of the liquid matrix (pigment excluded)."""
        n_d = self.refractive_calc.calculate_refractive_index(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)
        if mg > 0:
            ctx = self._mg_context(al, ipa, pg, mg, temperature)
            n_d = self.mg_model.corrected_refractive_index(
                n_d, ctx["w_mg_liq"], ctx["rho_liq"])
        return n_d

    def sound_velocity(self, al=0.0, ipa=0.0, pg=0.0, mg=0.0, temperature=25.0):
        """Sound velocity [m/s]."""
        res = self.sound_calc.calculate(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, temperature=temperature)
        c = res.sound_velocity
        if mg > 0:
            ctx = self._mg_context(al, ipa, pg, mg, temperature)
            rho_base_si = res.density * 1000.0
            rho_new_si = self.mg_model.corrected_density(
                res.density, ctx["w_mg"], ctx["rho_water"]) * 1000.0
            c = self.mg_model.corrected_sound_velocity(
                c, rho_base_si, rho_new_si, ctx["w_mg"])
        return c

    def viscosity(self, al=0.0, ipa=0.0, pg=0.0, mg=0.0, temperature=25.0, water=None):
        """Ink viscosity [mPa.s]."""
        water = self._water_remainder(al, ipa, pg, water, mg=mg)
        # base carrier: MG mass counted as water (dilute-solvent view) ...
        eta = self.viscosity_model.estimate(
            water=water + mg, ipa=ipa, pg=pg, aluminum=al,
            temperature_C=temperature, verbose=False)["viscosity_mPas"]
        # ... then the Jones-Dole solute factor on top
        if mg > 0:
            ctx = self._mg_context(al, ipa, pg, mg, temperature)
            eta *= self.mg_model.viscosity_factor(ctx["w_mg_liq"], ctx["rho_liq"])
        return eta

    # ---- everything at once -----------------------------------------
    def compute(self, al=0.0, ipa=0.0, pg=0.0, mg=0.0, temperature=25.0, water=None):
        """
        Compute all four properties at once and return an InkProperties
        object (printable). Water is the remainder unless given explicitly.
        mg = mass % of dissolved methyl gallate (dilute-solute model).
        """
        water = self._water_remainder(al, ipa, pg, water, mg=mg)
        warnings = []
        details = {}

        # --- base ink: identical recipe with the MG mass counted as water ---
        # (the tabulated models take water as the remainder of al/ipa/pg,
        #  which is exactly this base state)

        # --- density ---
        rho_base = self.density_calc.calculate_density(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)
        rho = rho_base

        # --- refractive index (matrix only) ---
        n_base = self.refractive_calc.calculate_refractive_index(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, target_temp=temperature)
        n_d = n_base

        # --- sound velocity ---
        sound = self.sound_calc.calculate(
            pct_al=al, pct_ipa=ipa, pct_pg=pg, temperature=temperature)
        warnings.extend(sound.warnings)
        details["density_wood_model"] = sound.density
        c_sound = sound.sound_velocity

        # --- viscosity (base carrier incl. MG mass as water) ---
        visc = self.viscosity_model.estimate(
            water=water + mg, ipa=ipa, pg=pg, aluminum=al,
            temperature_C=temperature, verbose=False)
        warnings.extend(visc["warnings"])
        details["viscosity"] = visc
        eta = visc["viscosity_mPas"]

        # --- methyl gallate solute corrections ---
        if mg > 0:
            ctx = self._mg_context(al, ipa, pg, mg, temperature)
            warnings.extend(self.mg_model.validity_warnings(
                ctx["w_mg"], ctx["w_mg_liq"], ctx["rho_liq"]))

            rho = self.mg_model.corrected_density(
                rho_base, ctx["w_mg"], ctx["rho_water"])
            n_d = self.mg_model.corrected_refractive_index(
                n_base, ctx["w_mg_liq"], ctx["rho_liq"])

            rho_base_wood_si = sound.density * 1000.0
            rho_new_wood_si = self.mg_model.corrected_density(
                sound.density, ctx["w_mg"], ctx["rho_water"]) * 1000.0
            c_sound = self.mg_model.corrected_sound_velocity(
                sound.sound_velocity, rho_base_wood_si, rho_new_wood_si, ctx["w_mg"])

            jd_factor = self.mg_model.viscosity_factor(
                ctx["w_mg_liq"], ctx["rho_liq"])
            eta = eta * jd_factor

            details["methyl_gallate"] = {
                "w_mg_liquid": ctx["w_mg_liq"],
                "conc_g_per_L_liquid": ctx["w_mg_liq"] * ctx["rho_liq"] * 1000.0,
                "delta_density_gcm3": rho - rho_base,
                "delta_refractive_index": n_d - n_base,
                "delta_sound_velocity_ms": c_sound - sound.sound_velocity,
                "jones_dole_factor": jd_factor,
                "v_phi_cm3mol": self.mg_model.v_phi,
                "k_phi_s_m3molPa": self.mg_model.k_phi_s,
            }

        return InkProperties(
            composition={"Al": al, "IPA": ipa, "PG": pg, "MG": mg, "Water": water},
            temperature=temperature,
            density=rho,
            refractive_index=n_d,
            sound_velocity=c_sound,
            viscosity=eta,
            vol_fraction_al=sound.vol_fraction_al,
            details=details,
            warnings=warnings,
        )

    # ---- calibration passthrough ------------------------------------
    def calibrate_viscosity(self, measured_viscosity, al, ipa, pg,
                            temperature=25.0, water=None):
        """Anchor the viscosity model to a measured value (sets its calibration factor)."""
        water = self._water_remainder(al, ipa, pg, water)
        return self.viscosity_model.calibrate(
            measured_viscosity, water=water, ipa=ipa, pg=pg,
            aluminum=al, temperature_C=temperature)


# =====================================================================
#  Example / direct use
# =====================================================================
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    tables = os.path.join(here, "tables_parameters")

    ink = InkCalculator(tables_dir=tables)

    # original ink: 1.82% Al, 3.64% IPA, 3.64% PG, rest water, 25 C
    props_old = ink.compute(al=160/88, ipa=320/88, pg=320/88, temperature=25)
    print(props_old)

    # new ink with methyl gallate:
    # 1.81% Al, 3.63% IPA, 3.63% PG, 0.23% MG, rest water (90.70%), 25 C
    props_new = ink.compute(al=160/88.2, ipa=320/88.2, pg=320/88.2, mg=20/88.2, temperature=25)
    print(props_new)

    print("Recipe change  (new - old):")
    print(f"  Delta density          : {props_new.density - props_old.density:+.5f} g/cm3")
    print(f"  Delta refractive index : "
          f"{props_new.refractive_index - props_old.refractive_index:+.5f}")
    print(f"  Delta sound velocity   : "
          f"{props_new.sound_velocity - props_old.sound_velocity:+.2f} m/s")
    print(f"  Delta viscosity        : "
          f"{props_new.viscosity - props_old.viscosity:+.4f} mPa.s")
    if "methyl_gallate" in props_new.details:
        d = props_new.details["methyl_gallate"]
        print(f"  (pure MG contribution: d_rho {d['delta_density_gcm3']:+.5f}, "
              f"d_n {d['delta_refractive_index']:+.5f}, "
              f"d_c {d['delta_sound_velocity_ms']:+.2f} m/s, "
              f"Jones-Dole x{d['jones_dole_factor']:.4f})")

    # individual numbers, if you only need one property:
    #   rho = ink.density(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    #   n_d = ink.refractive_index(ipa=3.64, pg=3.64, temperature=25.0)
    #   c   = ink.sound_velocity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)
    #   eta = ink.viscosity(al=1.82, ipa=3.64, pg=3.64, temperature=25.0)

    # anchor the viscosity estimate to a measured value:
    #   ink.calibrate_viscosity(1.9, al=1.82, ipa=3.64, pg=3.64, temperature=25.0)