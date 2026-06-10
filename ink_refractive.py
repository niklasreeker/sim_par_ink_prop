import pandas as pd
from scipy.interpolate import interp1d
import os

# Import the density calculator from your existing module
from ink_density import InkDensityCalculator


class InkRefractiveCalculator:
    """
    A calculator to determine the theoretical refractive index of the liquid matrix
    in ink mixtures (Water, IPA, and/or PG).
    It uses empirical refractive index tables and the Gladstone-Dale relation
    combined with a pseudo-binary approximation for ternary mixtures.
    """

    def __init__(self, tables_dir="tables_parameters", density_calculator=None):
        """
        Initializes the refractive index calculator and loads the CSV tables.

        :param tables_dir: Directory where the CSV files are stored.
        :param density_calculator: An existing instance of InkDensityCalculator.
                                   If None, a new instance will be created.
        """
        self.tables_dir = tables_dir
        self.interpolators = {
            'IPA': {},
            'PG': {}
        }

        # Link to the density calculator (needed for Gladstone-Dale equation)
        if density_calculator is None:
            self.density_calc = InkDensityCalculator(tables_dir=tables_dir)
        else:
            self.density_calc = density_calculator

        self._load_tables()

    def _load_tables(self):
        """Internal method to load CSV files and setup interpolation functions."""
        ipa_path = os.path.join(self.tables_dir, "ipa_refractive.csv")
        pg_path = os.path.join(self.tables_dir, "pg_refractive.csv")

        # Load IPA refractive table (Contains strictly empirical 25C data)
        if os.path.exists(ipa_path):
            df_ipa = pd.read_csv(ipa_path)
            self.interpolators['IPA']['25C'] = interp1d(df_ipa['Mass_Percent'], df_ipa['Refractive_25C'], kind='cubic')
        else:
            print(f"Warning: '{ipa_path}' not found. IPA optical calculations will fail.")

        # Load PG refractive table (Contains strictly empirical 22C and 25C data)
        if os.path.exists(pg_path):
            df_pg = pd.read_csv(pg_path)
            self.interpolators['PG']['22C'] = interp1d(df_pg['Mass_Percent'], df_pg['Refractive_22C'], kind='cubic')
            self.interpolators['PG']['25C'] = interp1d(df_pg['Mass_Percent'], df_pg['Refractive_25C'], kind='cubic')
        else:
            print(f"Warning: '{pg_path}' not found. PG optical calculations will fail.")

    def get_liquid_refractive_index(self, solvent_type, mass_percent, target_temp=25):
        """
        Interpolates/Extrapolates the real refractive index of a binary liquid mixture
        (Water + Solvent) for ANY given temperature.

        :param solvent_type: 'IPA' or 'PG'
        :param mass_percent: The mass percentage of the solvent in the liquid phase
        :param target_temp: The target temperature in Celsius
        :return: Interpolated/Extrapolated refractive index (nD)
        """
        funcs = self.interpolators[solvent_type]

        if solvent_type == 'IPA':
            # We only have one empirical anchor for IPA (25°C)
            n_25 = float(funcs['25C'](mass_percent))

            if target_temp == 25:
                return n_25
            else:
                # Estimate the thermo-optic coefficient (dn/dT)
                # Pure water: approx -0.0001 per °C
                # Pure IPA: approx -0.0004 per °C (Derived from Herráez & Belda, 2006)
                dn_dt_water = -0.00010
                dn_dt_ipa = -0.00040

                # Linear mixing of the temperature coefficient based on mass fraction
                dn_dt_mix = dn_dt_water + (mass_percent / 100.0) * (dn_dt_ipa - dn_dt_water)

                # Apply temperature correction
                delta_t = target_temp - 25.0
                return n_25 + (dn_dt_mix * delta_t)

        elif solvent_type == 'PG':
            # We have two empirical anchors for PG (22°C and 25°C) -> Math extrapolation is possible
            temps = [22, 25]
            n_vals = [
                float(funcs['22C'](mass_percent)),
                float(funcs['25C'](mass_percent))
            ]

            # Dynamically interpolate or extrapolate along the temperature axis
            temp_interpolator = interp1d(temps, n_vals, kind='linear', fill_value='extrapolate')
            return float(temp_interpolator(target_temp))

        else:
            raise ValueError(f"Unknown solvent type: {solvent_type}")

    def calculate_refractive_index(self, pct_ipa=0.0, pct_pg=0.0, target_temp=25):
        """
        Calculates the total refractive index of the liquid matrix.
        Automatically handles single solvents or applies the pseudo-binary
        Gladstone-Dale approximation if both IPA and PG are present.

        Note: The Aluminium pigment does not participate in light refraction of the matrix.

        :param pct_ipa: Mass percent of Isopropanol in the liquid
        :param pct_pg: Mass percent of Propylene Glycol in the liquid
        :param target_temp: Temperature in Celsius
        :return: Total refractive index (nD)
        """
        pct_water = 100.0 - pct_ipa - pct_pg

        if pct_water < 0:
            raise ValueError("Total solvent mass percentage exceeds 100%. Check your inputs.")

        # Case 1: Pure Water
        if pct_ipa == 0 and pct_pg == 0:
            return self.get_liquid_refractive_index('IPA', 0.0, target_temp)

        # Case 2: Only IPA is present
        elif pct_ipa > 0 and pct_pg == 0:
            return self.get_liquid_refractive_index('IPA', pct_ipa, target_temp)

        # Case 3: Only PG is present
        elif pct_pg > 0 and pct_ipa == 0:
            return self.get_liquid_refractive_index('PG', pct_pg, target_temp)

        # Case 4: Both IPA and PG are present (Gladstone-Dale Approximation)
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

            # 4. Get empirical refractive index (n) and density (rho) for both virtual mixtures
            n_1 = self.get_liquid_refractive_index('IPA', pct_ipa_in_mix, target_temp)
            rho_1 = self.density_calc.get_liquid_density('IPA', pct_ipa_in_mix, target_temp)

            n_2 = self.get_liquid_refractive_index('PG', pct_pg_in_mix, target_temp)
            rho_2 = self.density_calc.get_liquid_density('PG', pct_pg_in_mix, target_temp)

            # 5. Calculate specific refraction (R) for both virtual mixtures (Gladstone-Dale)
            r_1 = (n_1 - 1.0) / rho_1
            r_2 = (n_2 - 1.0) / rho_2

            # 6. Mix specific refractions based on their mass fraction in the total liquid
            w_mix_1 = mass_mix_ipa / 100.0
            w_mix_2 = mass_mix_pg / 100.0
            r_total = (w_mix_1 * r_1) + (w_mix_2 * r_2)

            # 7. Final Step: Convert total specific refraction back to refractive index
            # We get the real density of the ternary liquid mixture (Aluminium = 0%)
            rho_liquid_total = self.density_calc.calculate_density(pct_al=0.0, pct_ipa=pct_ipa, pct_pg=pct_pg,
                                                                   target_temp=target_temp)

            n_total = (r_total * rho_liquid_total) + 1.0

            return n_total


# =====================================================================
# Example Usage
# =====================================================================
if __name__ == "__main__":

    print("Initializing optical calculator...")
    optical_calc = InkRefractiveCalculator()

    print("\n--- IPA Temperature Extrapolation Test (Using dn/dT) ---")
    test_temps = [20, 25, 30]
    for t in test_temps:
        n_d = optical_calc.calculate_refractive_index(pct_ipa=50.0, pct_pg=0.0, target_temp=t)
        print(f"50% IPA / 50% Water at {t}°C -> nD = {n_d:.4f}")

    print("\n--- Ternary Mixture Test (Gladstone-Dale Approach) at 25°C ---")
    # Example: 10% IPA, 5% PG, 85% Water
    complex_n_d = optical_calc.calculate_refractive_index(pct_ipa=10.0, pct_pg=5.0, target_temp=25)
    print(f"Liquid: 10.0% IPA | 5.0% PG | 85.0% Water -> nD = {complex_n_d:.4f}")