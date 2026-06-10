import pandas as pd
from scipy.interpolate import interp1d
import os


class InkDensityCalculator:
    """
    A calculator to determine the theoretical density of ink mixtures
    consisting of Aluminum pigments, Water, IPA, and/or PG.
    It uses empirical density tables to account for volume contraction.
    """

    # Constants
    DENSITY_ALUMINUM = 2.700  # g/cm³

    def __init__(self, tables_dir="tables_parameters"):
        """
        Initializes the calculator and loads the density tables into memory.

        :param tables_dir: Directory where the CSV files are stored.
        """
        self.tables_dir = tables_dir
        self.interpolators = {
            'IPA': {},
            'PG': {}
        }
        self._load_tables()

    def _load_tables(self):
        """Internal method to load CSV files and setup interpolation functions."""
        ipa_path = os.path.join(self.tables_dir, "ipa_density.csv")
        pg_path = os.path.join(self.tables_dir, "pg_density.csv")

        # Load IPA table (Contains 20C and 30C)
        if os.path.exists(ipa_path):
            df_ipa = pd.read_csv(ipa_path)
            self.interpolators['IPA']['20C'] = interp1d(df_ipa['Mass_Percent'], df_ipa['Density_20C'], kind='linear')
            self.interpolators['IPA']['30C'] = interp1d(df_ipa['Mass_Percent'], df_ipa['Density_30C'], kind='linear')
        else:
            print(f"Warning: '{ipa_path}' not found. IPA calculations will fail.")

        # Load PG table (Contains 20C, 25C, 30C)
        if os.path.exists(pg_path):
            df_pg = pd.read_csv(pg_path)
            self.interpolators['PG']['20C'] = interp1d(df_pg['Mass_Percent'], df_pg['Density_20C'], kind='cubic')
            self.interpolators['PG']['25C'] = interp1d(df_pg['Mass_Percent'], df_pg['Density_25C'], kind='cubic')
            self.interpolators['PG']['30C'] = interp1d(df_pg['Mass_Percent'], df_pg['Density_30C'], kind='cubic')
        else:
            print(f"Warning: '{pg_path}' not found. PG calculations will fail.")

    def get_liquid_density(self, solvent_type, mass_percent, target_temp=25):
        """
        Interpolates the real density of a binary liquid mixture (Water + Solvent).

        :param solvent_type: 'IPA' or 'PG'
        :param mass_percent: The mass percentage of the solvent in the liquid phase
        :param target_temp: The target temperature in Celsius (between 20 and 30)
        :return: Interpolated real density in g/cm³
        """
        if not (20 <= target_temp <= 30):
            raise ValueError("Target temperature must be between 20°C and 30°C.")

        funcs = self.interpolators[solvent_type]

        if solvent_type == 'IPA':
            dens_20 = float(funcs['20C'](mass_percent))
            dens_30 = float(funcs['30C'](mass_percent))
            # Linear interpolation for temperature
            slope = (dens_30 - dens_20) / (30 - 20)
            return dens_20 + slope * (target_temp - 20)

        elif solvent_type == 'PG':
            dens_20 = float(funcs['20C'](mass_percent))
            dens_25 = float(funcs['25C'](mass_percent))
            dens_30 = float(funcs['30C'](mass_percent))

            if target_temp == 20: return dens_20
            if target_temp == 25: return dens_25
            if target_temp == 30: return dens_30

            # Interpolate between the closest temperature brackets
            if 20 < target_temp < 25:
                slope = (dens_25 - dens_20) / (25 - 20)
                return dens_20 + slope * (target_temp - 20)
            else:  # 25 < target_temp < 30
                slope = (dens_30 - dens_25) / (30 - 25)
                return dens_25 + slope * (target_temp - 25)

    def calculate_density(self, pct_al, pct_ipa=0.0, pct_pg=0.0, target_temp=25):
        """
        Calculates the total density of the ink.
        Automatically handles single solvents or applies the pseudo-binary
        approximation if both IPA and PG are present.

        :param pct_al: Mass percent of Aluminum pigment
        :param pct_ipa: Mass percent of Isopropanol
        :param pct_pg: Mass percent of Propylene Glycol
        :param target_temp: Temperature in Celsius
        :return: Total density in g/cm³
        """
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg

        if pct_water < 0:
            raise ValueError("Total mass percentage exceeds 100%. Check your inputs.")

        frac_al = pct_al / 100.0
        term_al = frac_al / self.DENSITY_ALUMINUM

        # Case 1: Only IPA is present
        if pct_ipa > 0 and pct_pg == 0:
            pct_liquid_total = pct_ipa + pct_water
            pct_solvent_in_liquid = (pct_ipa / pct_liquid_total) * 100.0
            rho_liquid = self.get_liquid_density('IPA', pct_solvent_in_liquid, target_temp)

            return 1.0 / (term_al + ((pct_liquid_total / 100.0) / rho_liquid))

        # Case 2: Only PG is present
        elif pct_pg > 0 and pct_ipa == 0:
            pct_liquid_total = pct_pg + pct_water
            pct_solvent_in_liquid = (pct_pg / pct_liquid_total) * 100.0
            rho_liquid = self.get_liquid_density('PG', pct_solvent_in_liquid, target_temp)

            return 1.0 / (term_al + ((pct_liquid_total / 100.0) / rho_liquid))

        # Case 3: Both IPA and PG are present (Pseudo-Binary Approximation)
        elif pct_ipa > 0 and pct_pg > 0:
            # 1. Proportional split of water based on solvent mass ratio
            total_solvent = pct_ipa + pct_pg
            ratio_ipa = pct_ipa / total_solvent
            ratio_pg = pct_pg / total_solvent

            # 2. Form virtual sub-mixtures
            mass_mix_ipa = pct_ipa + (pct_water * ratio_ipa)
            mass_mix_pg = pct_pg + (pct_water * ratio_pg)

            # 3. Calculate concentration inside each virtual mixture
            pct_ipa_in_mix = (pct_ipa / mass_mix_ipa) * 100.0
            pct_pg_in_mix = (pct_pg / mass_mix_pg) * 100.0

            # 4. Get real density for both virtual mixtures
            rho_mix_ipa = self.get_liquid_density('IPA', pct_ipa_in_mix, target_temp)
            rho_mix_pg = self.get_liquid_density('PG', pct_pg_in_mix, target_temp)

            # 5. Calculate total density using the 3-part formula
            term_mix_ipa = (mass_mix_ipa / 100.0) / rho_mix_ipa
            term_mix_pg = (mass_mix_pg / 100.0) / rho_mix_pg

            return 1.0 / (term_al + term_mix_ipa + term_mix_pg)

        # Case 4: Pure water + Aluminum (No solvents)
        else:
            # Density of pure water varies slightly, using approx 0.997 at 25C
            # For exactness, we fetch 0% IPA table value
            rho_water = self.get_liquid_density('IPA', 0.0, target_temp)
            return 1.0 / (term_al + ((pct_water / 100.0) / rho_water))


# =====================================================================
# Example Usage
# =====================================================================
if __name__ == "__main__":

    # Initialize the calculator (loads CSVs into memory once)
    calculator = InkDensityCalculator(tables_dir="tables_parameters")

    print("--- Single Solvent Tests (Your 4 Data Points at 25°C) ---")
    points = [
        (0.5, 22.0),
        (2.2, 22.0),
        (0.5, 10.0),
        (2.2, 10.0)
    ]

    for al, ipa in points:
        density = calculator.calculate_density(pct_al=al, pct_ipa=ipa, target_temp=25)
        print(f"Al: {al}%, IPA: {ipa}% -> {density:.4f} g/cm³")

    print("\n--- Complex Ternary Solvent Test (Pseudo-Binary Approximation) ---")
    # Example: 2.2% Al, 4% IPA, 4% PG (Rest is water)
    complex_density = calculator.calculate_density(pct_al=2.2, pct_ipa=4.0, pct_pg=4.0, target_temp=25)
    print(f"Al: 2.2%, IPA: 4.0%, PG: 4.0% -> {complex_density:.4f} g/cm³")