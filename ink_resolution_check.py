"""
ink_resolution_final.py
=====================================================================
Bewertung der Sensorauflösung.
Zeigt den Signalhub sauber im Plot an, vermeidet wissenschaftliche
Notation und markiert den Toleranzbereich von 1.72% bis 1.92% Al.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importiere den Calculator (muss im selben Ordner liegen)
from ink_calculator import InkCalculator

# =====================================================================
# 1. Konfiguration & Konstanten
# =====================================================================
AL_IDEAL = 1.82
AL_LOWER = 1.72  # -0.1% Toleranz
AL_UPPER = 1.92  # +0.1% Toleranz
PG_BASE = 3.64
TEMP_BASE = 25.0

IPA_0_EVAP = 3.64  # Idealzustand
IPA_50_EVAP = 1.82
IPA_100_EVAP = 0.0

# Hardware-Auflösungen
RES_DENS_BIEGE = 0.0001  # g/cm³
RES_DENS_CORIOLIS = 0.0005  # g/cm³
RES_RI = 0.0002  # nD
RES_SOUND = 0.01  # m/s
RES_VISC_PCT = 0.01  # 1% vom Messwert

ink = InkCalculator(tables_dir="tables_parameters")
al_range = np.linspace(1.0, 3.0, 100)

sns.set_theme(style="whitegrid")


# =====================================================================
# Hilfsfunktion für saubere Dezimalzahlen
# =====================================================================
def format_decimal(val):
    """Verhindert wissenschaftliche Notation (z.B. 4e-05) und gibt saubere Dezimalzahlen aus."""
    s = f"{val:.6f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if s == "":
        s = "0"
    return s


# =====================================================================
# Hilfsfunktion zum Plotten
# =====================================================================
def plot_measurement(ax, property_name, ylabel, unit, title, resolutions):
    y_0, y_50, y_100 = [], [], []

    # 1. Kurven berechnen
    for al in al_range:
        y_0.append(getattr(ink.compute(al=al, ipa=IPA_0_EVAP, pg=PG_BASE, temperature=TEMP_BASE), property_name))
        y_50.append(getattr(ink.compute(al=al, ipa=IPA_50_EVAP, pg=PG_BASE, temperature=TEMP_BASE), property_name))
        y_100.append(getattr(ink.compute(al=al, ipa=IPA_100_EVAP, pg=PG_BASE, temperature=TEMP_BASE), property_name))

    y_0, y_50, y_100 = np.array(y_0), np.array(y_50), np.array(y_100)

    # 2. Exakte Werte für die Toleranzgrenzen berechnen (auf der blauen Ideal-Kurve)
    val_lower = getattr(ink.compute(al=AL_LOWER, ipa=IPA_0_EVAP, pg=PG_BASE, temperature=TEMP_BASE), property_name)
    val_upper = getattr(ink.compute(al=AL_UPPER, ipa=IPA_0_EVAP, pg=PG_BASE, temperature=TEMP_BASE), property_name)

    # Signalhub berechnen
    signalhub = abs(val_upper - val_lower)

    # 3. In den Plot zeichnen

    # Markierung des Bereiches (Fläche + feine Begrenzungslinien)
    ax.axvspan(AL_LOWER, AL_UPPER, color='gray', alpha=0.15, label='Toleranzbereich (±0.1% Al)')
    ax.axvline(AL_LOWER, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(AL_UPPER, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(AL_IDEAL, color='black', linestyle='--', linewidth=1.5, label='Idealzustand (1.82% Al)')

    # Sensor-Bänder zeichnen
    for res_value, res_name, res_color, res_alpha in resolutions:
        # Viskosität wird prozentual berechnet, der Rest absolut
        lower_bound = y_0 * (1.0 - res_value) if property_name == 'viscosity' else y_0 - res_value
        upper_bound = y_0 * (1.0 + res_value) if property_name == 'viscosity' else y_0 + res_value
        ax.fill_between(al_range, lower_bound, upper_bound, color=res_color, alpha=res_alpha, label=res_name)

    # Hauptlinien
    ax.plot(al_range, y_0, color='tab:blue', linewidth=2, label='IPA 0% verdampft')
    ax.plot(al_range, y_50, color='tab:orange', linewidth=2, linestyle='-.', label='IPA 50% verdampft')
    ax.plot(al_range, y_100, color='tab:red', linewidth=2, linestyle=':', label='IPA 100% verdampft')

    # 4. Formatierung & Text
    ax.set_xlabel('Aluminiumgehalt in Tinte [%]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Textbox für den Signalhub (Angepasster Text)
    textstr = f"Signalhub von 1,72% bis 1,92% Al:\nΔ = {format_decimal(signalhub)} {unit}"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Legende automatisch platzieren
    ax.legend(loc='best', fontsize=9)
    ax.set_xlim(1.0, 3.0)


# =====================================================================
# Hauptausführung
# =====================================================================
def run():
    print("Berechne finale Graphen...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Dichte
    plot_measurement(
        ax=axs[0, 0], property_name='density',
        ylabel='Dichte [g/cm³]', unit='g/cm³',
        title='Dichtemessung',
        resolutions=[
            (RES_DENS_CORIOLIS, f'Coriolis (±{RES_DENS_CORIOLIS})', 'cyan', 0.4),
            (RES_DENS_BIEGE, f'Biegeschwinger (±{RES_DENS_BIEGE})', 'blue', 0.5)
        ]
    )

    # 2. Brechungsindex
    plot_measurement(
        ax=axs[0, 1], property_name='refractive_index',
        ylabel='Brechungsindex [nD]', unit='nD',
        title='Brechungsindex Messung',
        resolutions=[(RES_RI, f'Refraktometer (±{RES_RI})', 'purple', 0.3)]
    )

    # 3. Schallgeschwindigkeit
    plot_measurement(
        ax=axs[1, 0], property_name='sound_velocity',
        ylabel='Schallgeschwindigkeit [m/s]', unit='m/s',
        title='Akustische Messung',
        resolutions=[(RES_SOUND, f'Ultraschall (±{RES_SOUND})', 'green', 0.4)]
    )

    # 4. Viskosität
    plot_measurement(
        ax=axs[1, 1], property_name='viscosity',
        ylabel='Viskosität [mPa.s]', unit='mPa.s',
        title='Viskositätsmessung',
        resolutions=[(RES_VISC_PCT, f'Viskosimeter (±{RES_VISC_PCT * 100}%)', 'red', 0.2)]
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()