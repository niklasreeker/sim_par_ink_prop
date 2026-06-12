"""
ink_sound.py
=====================================================================
Reusable tool to compute the SOUND VELOCITY (and density) of a
waterborne aluminum-pigment ink made of:

        encapsulated Al pigment  +  Water  +  IPA  +  PG

You enter the composition (mass %) and the temperature, and the tool
returns the sound velocity in m/s. The ink is assumed to be highly
aqueous (typically > 80 % water, < 3 % Al), which is exactly the
regime where the model below is valid.

---------------------------------------------------------------------
PHYSICAL METHOD (identical to the approach developed earlier)
---------------------------------------------------------------------
Sound velocity cannot be mixed linearly. Instead we mix DENSITY (rho)
and adiabatic COMPRESSIBILITY (beta) and convert back via Newton-Laplace:

        c = 1 / sqrt(rho * beta)            beta = 1 / (rho * c^2)

Step 1 - Liquid matrix (Water + solvent), from MEASURED binary data:
        rho_L : interpolated from *_density.csv  (real, contracted density)
        c_L   : interpolated from *_sound.csv
        beta_L = 1 / (rho_L * c_L^2)

Step 2 - Add the solid pigment via the WOOD / URICK equation
        (effective-medium, long-wavelength limit), using volume fractions:

        rho_mix  = sum_i ( phi_i * rho_i )
        beta_mix = sum_i ( phi_i * beta_i )
        c_mix    = 1 / sqrt(rho_mix * beta_mix)

        with phi_i = (m_i / rho_i) / sum_j (m_j / rho_j)

For aluminum, beta is taken from the BULK modulus K_Al (NOT from the
6320 m/s longitudinal solid-bar velocity):

        beta_Al = 1 / K_Al ,  K_Al ~ 76 GPa

Ternary solvent (IPA + PG present together) is handled with the same
pseudo-binary split as the density model: water is distributed between
the two solvents by mass ratio, forming two virtual binary liquids;
Wood then mixes three phases (Al, IPA-mix, PG-mix).

---------------------------------------------------------------------
REQUIRED DATA FILES (folder: tables_parameters)
---------------------------------------------------------------------
    ipa_density.csv , pg_density.csv
        column 1 : Mass_Percent
        columns  : Density_20C, Density_25C, ...        [g/cm3]
    ipa_sound.csv   , pg_sound.csv
        column 1 : Mass_Percent
        columns  : SoundVelocity_20C, SoundVelocity_25C, ...   [m/s]

Mass-percent rows may be unevenly spaced; they are interpolated
(cubic if >= 4 points, otherwise linear). Temperature is interpolated
linearly between whatever "_XXC" columns are present.
=====================================================================
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# =====================================================================
#  Helper: one interpolatable property table (density OR sound velocity)
# =====================================================================
class _PropertyTable:
    """
    Loads a CSV with a 'Mass_Percent' column and one or more
    '<prefix>_<temp>C' columns, and provides value(mass_percent, temp)
    by interpolating over concentration and then over temperature.
    """

    def __init__(self, path, prefix):
        self.path = path
        self.prefix = prefix
        df = pd.read_csv(path)

        if "Mass_Percent" not in df.columns:
            raise ValueError(f"'{path}' has no 'Mass_Percent' column.")

        mp = df["Mass_Percent"].to_numpy(dtype=float)
        order = np.argsort(mp)              # sort unevenly spaced support points
        mp = mp[order]
        self.mp_min, self.mp_max = float(mp.min()), float(mp.max())

        kind = "cubic" if len(mp) >= 4 else "linear"

        # build one concentration-interpolator per available temperature
        self.temps = {}
        pattern = re.compile(rf"{prefix}_(\d+(?:\.\d+)?)C")
        for col in df.columns:
            m = pattern.fullmatch(col)
            if m:
                t = float(m.group(1))
                y = df[col].to_numpy(dtype=float)[order]
                self.temps[t] = interp1d(
                    mp, y, kind=kind, bounds_error=False, fill_value="extrapolate"
                )

        if not self.temps:
            raise ValueError(
                f"'{path}' has no '{prefix}_XXC' columns (e.g. {prefix}_25C)."
            )
        self.available_temps = sorted(self.temps)

    def value(self, mass_percent, temperature):
        """Interpolated property at the given mass percent and temperature."""
        # evaluate at each available temperature for this concentration
        vals = np.array(
            [float(self.temps[t](mass_percent)) for t in self.available_temps]
        )
        if temperature in self.temps:
            return float(self.temps[temperature](mass_percent))
        if len(self.available_temps) == 1:
            return float(vals[0])
        temp_interp = interp1d(
            self.available_temps, vals, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )
        return float(temp_interp(temperature))

    # -- range checks (for friendly warnings) --
    def temp_in_range(self, temperature):
        return self.available_temps[0] <= temperature <= self.available_temps[-1]

    def mp_in_range(self, mass_percent):
        return self.mp_min <= mass_percent <= self.mp_max


# =====================================================================
#  Result container
# =====================================================================
@dataclass
class InkResult:
    sound_velocity: float          # m/s
    density: float                 # g/cm3
    temperature: float             # C
    composition: dict              # {'Al':.., 'IPA':.., 'PG':.., 'Water':..}
    vol_fraction_al: float = 0.0   # phi_Al (-)
    warnings: list = field(default_factory=list)

    def __str__(self):
        c = self.composition
        lines = [
            "----------------------------------------------------",
            f"  Composition (mass %):",
            f"     Aluminum : {c['Al']:6.3f} %",
            f"     IPA      : {c['IPA']:6.3f} %",
            f"     PG       : {c['PG']:6.3f} %",
            f"     Water    : {c['Water']:6.3f} %",
            f"  Temperature : {self.temperature:6.2f} C",
            "  - - - - - - - - - - - - - - - - - - - - - - - - - -",
            f"  Density            : {self.density:8.4f} g/cm3",
            f"  Sound velocity     : {self.sound_velocity:8.2f} m/s",
            f"  Al volume fraction : {self.vol_fraction_al*100:7.3f} %",
            "----------------------------------------------------",
        ]
        for w in self.warnings:
            lines.append(f"  [!] {w}")
        return "\n".join(lines)


# =====================================================================
#  Main calculator
# =====================================================================
class InkSoundCalculator:
    """Sound velocity of an Al-pigment / Water / IPA / PG ink."""

    # Aluminum properties
    DENSITY_ALUMINUM = 2.700        # g/cm3
    BULK_MODULUS_ALUMINUM = 76.0e9  # Pa  (compressibility beta_Al = 1/K_Al)

    def __init__(self, tables_dir="tables_parameters"):
        self.tables_dir = tables_dir
        # property tables are loaded lazily / tolerantly:
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

    # -----------------------------------------------------------------
    #  One binary liquid sub-phase  ->  (volume_cm3, rho_SI, beta_SI)
    # -----------------------------------------------------------------
    def _binary_phase(self, solvent, pct_solvent, pct_water, temperature, warnings):
        mass_liquid = pct_solvent + pct_water          # g, on a 100-g basis
        if mass_liquid <= 0:
            return None

        # solvent concentration WITHIN the liquid phase (Al excluded)
        pct_in_liquid = (pct_solvent / mass_liquid) * 100.0

        dtab = self.density[solvent]
        stab = self.sound[solvent]
        if dtab is None or stab is None:
            raise ValueError(
                f"Missing {solvent} table(s) in '{self.tables_dir}'. "
                f"Cannot evaluate a mixture containing {solvent}."
            )

        # range warnings (extrapolation still works, but flag it)
        if not dtab.mp_in_range(pct_in_liquid) or not stab.mp_in_range(pct_in_liquid):
            warnings.append(
                f"{solvent} concentration {pct_in_liquid:.1f}% is outside the "
                f"tabulated range -> extrapolated."
            )
        if not dtab.temp_in_range(temperature) or not stab.temp_in_range(temperature):
            warnings.append(
                f"Temperature {temperature:.1f} C is outside the tabulated "
                f"range -> extrapolated."
            )

        rho_L = dtab.value(pct_in_liquid, temperature)     # g/cm3
        c_L = stab.value(pct_in_liquid, temperature)       # m/s

        rho_si = rho_L * 1000.0                            # kg/m3
        beta = 1.0 / (rho_si * c_L ** 2)                   # 1/Pa
        vol = mass_liquid / rho_L                          # cm3 (100-g basis)
        return (vol, rho_si, beta)

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------
    def calculate(self, pct_al, pct_ipa=0.0, pct_pg=0.0, temperature=25.0):
        """
        Compute sound velocity and density of the ink.

        Parameters
        ----------
        pct_al, pct_ipa, pct_pg : float
            Mass percentages of aluminum, IPA and PG. Water is the
            remainder (100 - Al - IPA - PG).
        temperature : float
            Temperature in degrees Celsius.

        Returns
        -------
        InkResult
        """
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg
        if pct_al < 0 or pct_ipa < 0 or pct_pg < 0:
            raise ValueError("Mass percentages must be non-negative.")
        if pct_water < -1e-9:
            raise ValueError(
                f"Al + IPA + PG = {pct_al + pct_ipa + pct_pg:.3f}% exceeds 100%."
            )
        pct_water = max(pct_water, 0.0)

        warnings = []
        beta_al = 1.0 / self.BULK_MODULUS_ALUMINUM
        rho_al_si = self.DENSITY_ALUMINUM * 1000.0

        # phase list: (volume_cm3, rho_SI, beta_SI); start with aluminum
        phases = []
        if pct_al > 0:
            phases.append((pct_al / self.DENSITY_ALUMINUM, rho_al_si, beta_al))

        # --- assemble the liquid phase(s) ---
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

        # --- Wood / Urick mixing over volume fractions ---
        V_total = sum(v for v, _, _ in phases)
        rho_mix_si = sum((v / V_total) * rho for v, rho, _ in phases)
        beta_mix = sum((v / V_total) * beta for v, _, beta in phases)
        c_mix = 1.0 / np.sqrt(rho_mix_si * beta_mix)

        vol_al = (pct_al / self.DENSITY_ALUMINUM) if pct_al > 0 else 0.0
        phi_al = vol_al / V_total

        # gentle validity hint for the Wood model
        if pct_al > 5.0:
            warnings.append(
                "Al > 5 % : Wood/Urick is a dilute-suspension model; "
                "accuracy may degrade at high pigment loading."
            )

        return InkResult(
            sound_velocity=c_mix,
            density=rho_mix_si / 1000.0,
            temperature=temperature,
            composition={"Al": pct_al, "IPA": pct_ipa,
                         "PG": pct_pg, "Water": pct_water},
            vol_fraction_al=phi_al,
            warnings=warnings,
        )

    # convenience wrapper returning just the number
    def sound_velocity(self, pct_al, pct_ipa=0.0, pct_pg=0.0, temperature=25.0):
        """Return only the sound velocity in m/s."""
        return self.calculate(pct_al, pct_ipa, pct_pg, temperature).sound_velocity


# =====================================================================
#  Interactive command-line use
# =====================================================================
def _ask_float(prompt, default=0.0):
    raw = input(prompt).strip().replace(",", ".")
    if raw == "":
        return default
    return float(raw)


def run_interactive():
    print("=" * 54)
    print("  Ink sound-velocity calculator")
    print("  (Al pigment + Water + IPA + PG)")
    print("=" * 54)
    print("  Enter mass percentages. Press Enter to accept the default.\n")

    calc = InkSoundCalculator(tables_dir="../tables_parameters")

    while True:
        try:
            al = _ask_float("  Aluminum  [mass %] (default 0)  : ", 0.0)
            ipa = _ask_float("  IPA       [mass %] (default 0)  : ", 0.0)
            pg = _ask_float("  PG        [mass %] (default 0)  : ", 0.0)
            temp = _ask_float("  Temperature [C]   (default 25) : ", 25.0)

            result = calc.calculate(al, ipa, pg, temp)
            print()
            print(result)
        except ValueError as e:
            print(f"  Input error: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  Error: {e}")

        again = input("\n  Another calculation? [y/N]: ").strip().lower()
        if again not in ("y", "yes", "j", "ja"):
            print("  Done.")
            break


def example():
    """Programmatic usage example (no interaction)."""
    calc = InkSoundCalculator(tables_dir="../tables_parameters")
    # original ink: 1.82% Al, 3.64% IPA, 3.64% PG, rest water, 25 C
    result = calc.calculate(pct_al=1.82, pct_ipa=3.64, pct_pg=3.64, temperature=25.0)
    print(result)
    # or just the number:
    c = calc.sound_velocity(1.82, 3.64, 3.64, 25.0)
    print(f"\n  -> sound velocity = {c:.2f} m/s")


if __name__ == "__main__":
    # Use the interactive prompt by default; fall back to the example
    # if no interactive input is available (e.g. when piped/automated).
    import sys
    if sys.stdin and sys.stdin.isatty():
        run_interactive()
    else:
        example()