"""
Erzeugt zwei CSV-Dateien mit experimentellen Viskositaetsdaten:
  - ipa_viscosity.csv : 2-Propanol (IPA) + Wasser
  - pg_viscosity.csv  : Propylenglykol (Propan-1,2-diol, PG) + Wasser

Quellen / Spaltenlogik der Stammtabellen:
  IPA:  x1 = Molenbruch 2-Propanol,  T = 293.15 ... 333.15 K (Schrittweite 5 K)
  PG :  x1 = Molenbruch WASSER (!),  Komponente 2 = Propan-1,2-diol,
        T = 298.15 ... 338.15 K (Schrittweite 10 K)

Die Stammtabellen geben die Zusammensetzung als Molenbruch an. Dieses Skript
rechnet den Molenbruch der jeweils interessierenden Komponente (IPA bzw. PG)
in Massenprozent um. Die Spalte heisst "Mass_Percent", die Viskositaetsspalten
"Viscosity_<T>C" (T in Grad Celsius), Werte in mPa*s.
"""

import os
import csv

# --- Molare Massen [g/mol] ---
M_WATER = 18.01528
M_IPA = 60.096   # 2-Propanol,        C3H8O
M_PG = 76.095    # Propan-1,2-diol,   C3H8O2

OUTPUT_DIR = "../tables_parameters"
MASS_PERCENT_DECIMALS = 4


def mole_fraction_to_mass_percent(x_solute, M_solute, M_solvent=M_WATER):
    """Molenbruch des betrachteten Stoffs -> Massenprozent (binaeres Gemisch mit Wasser)."""
    m_solute = x_solute * M_solute
    m_solvent = (1.0 - x_solute) * M_solvent
    return 100.0 * m_solute / (m_solute + m_solvent)


# === Daten 1: 2-Propanol (1) + Wasser (2) ===
# x1 = Molenbruch 2-Propanol
ipa_temps_C = [20, 25, 30, 35, 40, 45, 50, 55, 60]   # 293.15 ... 333.15 K
ipa_x1 = [
    0.00000, 0.01000, 0.02000, 0.05000, 0.07069, 0.10000, 0.20003, 0.30003,
    0.40003, 0.50004, 0.60005, 0.70013, 0.79990, 0.89994, 1.00000,
]
ipa_eta = [
    [1.002, 0.890, 0.797, 0.719, 0.653, 0.596, 0.547, 0.504, 0.466],
    [1.170, 1.029, 0.912, 0.815, 0.733, 0.664, 0.604, 0.554, 0.510],
    [1.369, 1.188, 1.040, 0.923, 0.823, 0.739, 0.668, 0.609, 0.556],
    [2.062, 1.725, 1.467, 1.263, 1.098, 0.966, 0.856, 0.767, 0.691],
    [2.544, 2.089, 1.735, 1.480, 1.270, 1.103, 0.969, 0.862, 0.774],
    [3.054, 2.472, 2.041, 1.717, 1.458, 1.261, 1.101, 0.973, 0.868],
    [3.741, 3.040, 2.504, 2.109, 1.789, 1.539, 1.337, 1.175, 1.037],
    [3.726, 3.068, 2.551, 2.153, 1.835, 1.580, 1.374, 1.207, 1.066],
    [3.481, 2.894, 2.425, 2.062, 1.765, 1.525, 1.328, 1.168, 1.033],
    [3.180, 2.667, 2.253, 1.926, 1.657, 1.437, 1.256, 1.107, 0.981],
    [2.888, 2.441, 2.076, 1.787, 1.544, 1.345, 1.179, 1.041, 0.926],
    [2.655, 2.256, 1.934, 1.669, 1.449, 1.265, 1.113, 0.985, 0.877],
    [2.484, 2.127, 1.824, 1.585, 1.379, 1.208, 1.064, 0.944, 0.840],
    [2.395, 2.054, 1.772, 1.538, 1.341, 1.176, 1.035, 0.919, 0.818],
    [2.414, 2.070, 1.785, 1.546, 1.347, 1.176, 1.033, 0.914, 0.811],
]

# === Daten 2: Wasser (1) + Propan-1,2-diol / PG (2) ===
# x1 = Molenbruch WASSER  ->  Molenbruch PG = 1 - x1
pg_temps_C = [25, 35, 45, 55, 65]   # 298.15 ... 338.15 K
pg_x1_water = [
    0.0440, 0.0987, 0.1484, 0.2498, 0.3494, 0.4494, 0.5029, 0.5499, 0.5597,
    0.6501, 0.7499, 0.8500, 0.9003, 0.9115, 0.9154, 0.9270, 0.9367, 0.9504,
    0.9551, 0.9600, 0.9696, 0.9797, 0.9902, 0.9923, 0.9948,
]
pg_eta = [
    [41.249, 23.027, 12.150, 9.223, 6.720],
    [38.410, 21.431, 11.313, 8.612, 6.302],
    [35.744, 19.930, 10.517, 8.046, 5.915],
    [30.202, 16.803, 8.875, 6.905, 5.133],
    [24.829, 13.774, 7.318, 5.842, 4.398],
    [19.708, 10.900, 5.893, 4.844, 3.698],
    [17.139, 9.465, 5.202, 4.335, 3.337],
    [14.997, 8.274, 4.639, 3.901, 3.027],
    [14.565, 8.034, 4.528, 3.811, 2.961],
    [10.839, 5.979, 3.576, 3.002, 2.378],
    [7.287, 4.049, 2.675, 2.147, 1.750],
    [4.325, 2.473, 1.864, 1.360, 1.156],
    [3.049, 1.810, 1.462, 1.012, 0.882],
    [2.782, 1.673, 1.370, 0.942, 0.825],
    [2.692, 1.626, 1.338, 0.917, 0.806],
    [2.425, 1.491, 1.242, 0.848, 0.749],
    [2.208, 1.379, 1.160, 0.793, 0.702],
    [1.907, 1.227, 1.043, 0.720, 0.639],
    [1.806, 1.177, 1.003, 0.695, 0.618],
    [1.701, 1.124, 0.960, 0.672, 0.597],
    [1.501, 1.023, 0.876, 0.627, 0.556],
    [1.294, 0.919, 0.784, 0.582, 0.515],
    [1.083, 0.815, 0.688, 0.539, 0.474],
    [1.041, 0.794, 0.668, 0.531, 0.466],
    [0.992, 0.769, 0.645, 0.521, 0.457],
]


def build_rows(x_list, eta_rows, M_solute, x_is_water):
    """Baut (Massenprozent, [Viskositaeten])-Zeilen und sortiert aufsteigend nach Massenprozent."""
    rows = []
    for x, etas in zip(x_list, eta_rows):
        x_solute = (1.0 - x) if x_is_water else x
        mp = mole_fraction_to_mass_percent(x_solute, M_solute)
        rows.append((round(mp, MASS_PERCENT_DECIMALS), etas))
    rows.sort(key=lambda r: r[0])
    return rows


def write_csv(path, temps_C, rows):
    header = ["Mass_Percent"] + [f"Viscosity_{t}C" for t in temps_C]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for mass_percent, etas in rows:
            writer.writerow([mass_percent] + list(etas))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ipa_rows = build_rows(ipa_x1, ipa_eta, M_IPA, x_is_water=False)
    write_csv(os.path.join(OUTPUT_DIR, "ipa_viscosity.csv"), ipa_temps_C, ipa_rows)

    pg_rows = build_rows(pg_x1_water, pg_eta, M_PG, x_is_water=True)
    write_csv(os.path.join(OUTPUT_DIR, "pg_viscosity.csv"), pg_temps_C, pg_rows)

    print(f"Fertig. CSV-Dateien liegen in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  - ipa_viscosity.csv ({len(ipa_rows)} Zeilen)")
    print(f"  - pg_viscosity.csv  ({len(pg_rows)} Zeilen)")


if __name__ == "__main__":
    main()