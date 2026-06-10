"""
Erzeugt die Datei  tables_parameters/ipa_sound.csv  von Grund auf.

Spalten:
    Mass_Percent, SoundVelocity_18C, SoundVelocity_20C,
    SoundVelocity_22C, SoundVelocity_25C

- SoundVelocity_25C: vollstaendige Tabelle (Messwerte bei 298,15 K / 25 C)
  ueber den gesamten Bereich 0 bis 100 Massenprozent.
- SoundVelocity_18C / 20C / 22C: nur fuer 0 bis 9,69 Massenprozent,
  darueber leer (lineare Naeherung gilt nur fuer verduennte Loesungen).

Berechnung 18/20/22 C:
    SoundVelocity = u_Wasser(Literatur) + Steigung * Mass_Percent
  - Startwert bei 0 % = Literaturwert reines Wasser nach Marczak (1997), ITS-90
  - Steigung aus den in Massenprozent umgerechneten Geradengleichungen

Ausfuehren im Projekt-Wurzelverzeichnis:  python create_ipa_sound_csv.py
"""

import csv
import os

# ---------------------------------------------------------------------
# Messdaten bei 25 C (298,15 K): Massenanteil 2-Propanol und u in m/s
# ---------------------------------------------------------------------
mass_fraction = [
    0.0000, 0.0333, 0.0635, 0.0936, 0.0969, 0.1220, 0.1487, 0.1760, 0.2008,
    0.2250, 0.2479, 0.2701, 0.3136, 0.3513, 0.3902, 0.4216, 0.4349, 0.4545,
    0.4990, 0.5524, 0.6105, 0.6712, 0.7237, 0.7690, 0.8089, 0.8443, 0.8759,
    0.9043, 0.9257, 0.9455, 0.9603, 0.9747, 0.9868, 0.9995, 1.0000,
]
speed_25C = [
    1496.69, 1525.88, 1552.31, 1577.40, 1579.81, 1597.93, 1612.69, 1620.85,
    1620.00, 1611.80, 1599.19, 1584.51, 1552.76, 1523.00, 1496.35, 1471.80,
    1464.34, 1452.24, 1423.68, 1391.08, 1357.24, 1322.77, 1294.40, 1270.58,
    1249.86, 1231.50, 1214.83, 1199.47, 1187.33, 1175.58, 1166.32, 1156.64,
    1147.98, 1137.50, 1137.07,
]

# ---------------------------------------------------------------------
# Parameter fuer 18 / 20 / 22 C
# ---------------------------------------------------------------------
U_WASSER = {18: 1476.07, 20: 1482.38, 22: 1488.36}   # Marczak (1997) [m/s]
STEIGUNG = {18: 9.93, 20: 9.28, 22: 8.98}            # aus Geradengleichungen
GRENZE = 9.69                                         # obere Gueltigkeit [Massen-%]

# ---------------------------------------------------------------------
# CSV schreiben
# ---------------------------------------------------------------------
ORDNER = "tables_parameters"
DATEI = os.path.join(ORDNER, "ipa_sound.csv")


def velocity(T, mass_percent):
    return round(U_WASSER[T] + STEIGUNG[T] * mass_percent, 2)


def main():
    os.makedirs(ORDNER, exist_ok=True)

    with open(DATEI, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mass_Percent",
            "SoundVelocity_18C",
            "SoundVelocity_20C",
            "SoundVelocity_22C",
            "SoundVelocity_25C",
        ])

        for w, u25 in zip(mass_fraction, speed_25C):
            mass_percent = round(w * 100, 2)
            if mass_percent <= GRENZE:
                u18 = velocity(18, mass_percent)
                u20 = velocity(20, mass_percent)
                u22 = velocity(22, mass_percent)
            else:
                u18 = u20 = u22 = ""      # ausserhalb Gueltigkeitsbereich
            writer.writerow([mass_percent, u18, u20, u22, u25])

    print(f"Datei erstellt: {DATEI}")


if __name__ == "__main__":
    main()