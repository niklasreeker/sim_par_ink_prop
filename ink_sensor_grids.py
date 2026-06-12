"""
ink_sensor_grids.py
=====================================================================
Netzdiagramme (Sensor-Fusion) zur Entkopplung von Aluminium und IPA.
X-Achse: Dichte (Hauptsensor)
Y-Achsen: Brechungsindex, Schallgeschwindigkeit, Viskosität (Kandidaten für Zweitsensor)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Importiere deinen Calculator aus der lokalen Datei ink_calculator.py
from ink_calculator import InkCalculator

# =====================================================================
# 1. Konfiguration
# =====================================================================
AL_BASE = 1.82
IPA_BASE = 3.64
PG_BASE = 3.64
TEMP_BASE = 25.0

# Initialisiere den Calculator
ink = InkCalculator(tables_dir="tables_parameters")

# Berechne den Idealzustand
base = ink.compute(al=AL_BASE, ipa=IPA_BASE, pg=PG_BASE, temperature=TEMP_BASE)

# Style für die Plots festlegen
sns.set_theme(style="whitegrid")


def plot_sensor_fusion_grids():
    print("Berechne Netzdiagramme...")

    # Raster definieren: Al: 1.82 +- 1% -> ~0.82% bis 2.82%
    al_levels = [0.82, 1.32, 1.82, 2.32, 2.82]
    # IPA: 0% bis 5%
    ipa_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Hochauflösende Arrays für flüssige Linien
    al_fine = np.linspace(0.82, 2.82, 30)
    ipa_fine = np.linspace(0.0, 5.0, 30)

    # Figur mit 3 Subplots nebeneinander erstellen
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # -----------------------------------------------------------------
    # Schritt 1: Linien für konstantes Aluminium (IPA variiert) -> BLAU
    # -----------------------------------------------------------------
    for al in al_levels:
        rho_v, nd_v, c_v, eta_v = [], [], [], []
        for ipa in ipa_fine:
            p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
            rho_v.append(p.density)
            nd_v.append(p.refractive_index)
            c_v.append(p.sound_velocity)
            eta_v.append(p.viscosity)

        lw = 2.5 if al == 1.82 else 1.0
        alpha = 0.8 if al == 1.82 else 0.5

        axs[0].plot(rho_v, nd_v, color='blue', linewidth=lw, alpha=alpha)
        axs[1].plot(rho_v, c_v, color='blue', linewidth=lw, alpha=alpha)
        axs[2].plot(rho_v, eta_v, color='blue', linewidth=lw, alpha=alpha)

        # Beschriftung am Startpunkt der Linie
        axs[0].text(rho_v[0], nd_v[0], f' {al}% Al', color='blue', fontsize=9, va='center')
        axs[1].text(rho_v[0], c_v[0], f' {al}% Al', color='blue', fontsize=9, va='center')
        axs[2].text(rho_v[0], eta_v[0], f' {al}% Al', color='blue', fontsize=9, va='center')

    # -----------------------------------------------------------------
    # Schritt 2: Linien für konstantes IPA (Aluminium variiert) -> ORANGE
    # -----------------------------------------------------------------
    for ipa in ipa_levels:
        rho_v, nd_v, c_v, eta_v = [], [], [], []
        for al in al_fine:
            p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
            rho_v.append(p.density)
            nd_v.append(p.refractive_index)
            c_v.append(p.sound_velocity)
            eta_v.append(p.viscosity)

        lw = 2.5 if ipa == 3.0 or ipa == 4.0 else 1.0  # Hervorhebung der Nähe zum Ideal
        alpha = 0.8 if ipa == 3.0 or ipa == 4.0 else 0.5

        axs[0].plot(rho_v, nd_v, color='darkorange', linewidth=lw, alpha=alpha)
        axs[1].plot(rho_v, c_v, color='darkorange', linewidth=lw, alpha=alpha)
        axs[2].plot(rho_v, eta_v, color='darkorange', linewidth=lw, alpha=alpha)

        # Beschriftung am Endpunkt der Linie
        axs[0].text(rho_v[-1], nd_v[-1], f' {ipa}% IPA', color='darkorange', fontsize=9, ha='right', va='bottom')
        axs[1].text(rho_v[-1], c_v[-1], f' {ipa}% IPA', color='darkorange', fontsize=9, ha='right', va='bottom')
        axs[2].text(rho_v[-1], eta_v[-1], f' {ipa}% IPA', color='darkorange', fontsize=9, ha='right', va='bottom')

    # -----------------------------------------------------------------
    # Schritt 3: Idealzustand markieren & Formatierung
    # -----------------------------------------------------------------
    for i, ax in enumerate(axs):
        y_ideal = [base.refractive_index, base.sound_velocity, base.viscosity][i]
        ax.plot(base.density, y_ideal, 'r*', markersize=12, zorder=5)

        ax.set_xlabel('Dichte [g/cm³] (Hauptsensor)', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)

    axs[0].set_ylabel('Brechungsindex [nD]', fontsize=11)
    axs[0].set_title('Option A: Dichte + Optik', fontsize=13, pad=10)

    axs[1].set_ylabel('Schallgeschwindigkeit [m/s]', fontsize=11)
    axs[1].set_title('Option B: Dichte + Akustik', fontsize=13, pad=10)

    axs[2].set_ylabel('Viskosität [mPa.s]', fontsize=11)
    axs[2].set_title('Option C: Dichte + Viskosität', fontsize=13, pad=10)

    # Legende global erstellen
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='darkorange', lw=2),
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12)]
    fig.legend(custom_lines, ['Konstantes Aluminium (IPA variiert)', 'Konstantes IPA (Aluminium variiert)',
                              'Idealzustand (1.82% Al, 3.64% IPA)'],
               loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    # Platz für die Legende machen
    plt.subplots_adjust(top=0.88)

    print("Fertig! Diagramme werden angezeigt.")
    plt.show()


if __name__ == "__main__":
    plot_sensor_fusion_grids()