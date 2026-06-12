"""
ink_single_grids.py
=====================================================================
Netzdiagramme als 4 einzelne Plots.
Gibt zusätzlich die berechneten Werte der Arbeitspunkte in der Konsole aus.
System: Wasser + IPA + Aluminium (PG = 0.0%)
Bereich: IPA (10% - 22%), Aluminium (0.5% - 2.2%)
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
PG_BASE = 0.0
TEMP_BASE = 25.0

ink = InkCalculator(tables_dir="tables_parameters")

# Bereiche definieren
MIN_AL, MAX_AL = 0.5, 2.2
MIN_IPA, MAX_IPA = 10.0, 22.0

# Gitter-Auflösung
NUM_LINES = 12
al_levels = np.linspace(MIN_AL, MAX_AL, NUM_LINES)
ipa_levels = np.linspace(MIN_IPA, MAX_IPA, NUM_LINES)

# Hohe Auflösung für die Zeichnung der Linien (flüssige Kurven)
al_fine = np.linspace(MIN_AL, MAX_AL, 50)
ipa_fine = np.linspace(MIN_IPA, MAX_IPA, 50)

# Eckpunkte (Arbeitspunkte)
CORNERS = [
    (MIN_AL, MIN_IPA),
    (MAX_AL, MIN_IPA),
    (MIN_AL, MAX_IPA),
    (MAX_AL, MAX_IPA)
]

# Standard Python Style
sns.set_theme(style="whitegrid")


# =====================================================================
# 2. Konsolenausgabe der Arbeitspunkte
# =====================================================================
def print_operating_points():
    print("=" * 60)
    print(f"BERECHNETE WERTE DER ARBEITSPUNKTE (Tinte bei {TEMP_BASE}°C)")
    print("=" * 60)

    for al, ipa in CORNERS:
        p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
        print(f"Arbeitspunkt: {al:.1f}% Al, {ipa:.1f}% IPA")
        print(f"  Dichte:                {p.density:.4f} g/cm³")
        print(f"  Schallgeschwindigkeit: {p.sound_velocity:.2f} m/s")
        print(f"  Brechungsindex:        {p.refractive_index:.4f} nD")
        print(f"  Viskosität:            {p.viscosity:.3f} mPa.s")
        print("-" * 60)


# =====================================================================
# 3. Plot-Funktion für einzelne Diagramme
# =====================================================================
def plot_single_grid(x_attr, y_attr, xlabel, ylabel, title):
    """Erstellt ein separates Fenster (Figure) für das übergebene Netzdiagramm."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # 1. Konstantes Aluminium (Blau) -> IPA variiert entlang der Linie
    for al in al_levels:
        x_vals, y_vals = [], []
        for ipa in ipa_fine:
            p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
            x_vals.append(getattr(p, x_attr))
            y_vals.append(getattr(p, y_attr))
        ax.plot(x_vals, y_vals, color='blue', alpha=0.6, linewidth=1.5)

    # 2. Konstantes IPA (Rot) -> Aluminium variiert entlang der Linie
    for ipa in ipa_levels:
        x_vals, y_vals = [], []
        for al in al_fine:
            p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
            x_vals.append(getattr(p, x_attr))
            y_vals.append(getattr(p, y_attr))
        ax.plot(x_vals, y_vals, color='red', alpha=0.6, linewidth=1.5)

    # 3. Arbeitspunkte (Ecken) markieren
    for al, ipa in CORNERS:
        p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
        x_val = getattr(p, x_attr)
        y_val = getattr(p, y_attr)

        # Punkt einzeichnen
        ax.plot(x_val, y_val, marker='o', color='black', markersize=6, zorder=5)

        # Text hinzufügen (mit weißem Hintergrund)
        label_text = f'{al}% Al\n{ipa}% IPA'
        ax.annotate(label_text, (x_val, y_val),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='black',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    zorder=6)

    # 4. Achsen und Titel formatieren
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, color='black', fontweight='bold', pad=15)

    # 5. Eigene Legende pro Plot
    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8)
    ]
    ax.legend(custom_lines,
              ['Konstantes IPA (Rot)', 'Konstantes Aluminium (Blau)', 'Arbeitspunkte'],
              loc='best', fontsize=10)

    plt.tight_layout()


# =====================================================================
# 4. Hauptausführung
# =====================================================================
def run():
    # 1. Werte in der Konsole ausgeben
    print_operating_points()

    haupt_titel = f"Tinte (Al + IPA + Wasser) bei {TEMP_BASE}°C"

    # 2. Die vier Diagramme generieren (jedes öffnet ein eigenes Fenster)
    print("Erstelle Diagramm 1: Dichte vs. Schallgeschwindigkeit...")
    plot_single_grid(
        x_attr='density', y_attr='sound_velocity',
        xlabel='Dichte [g/cm³]', ylabel='Schallgeschwindigkeit [m/s]',
        title=f'{haupt_titel}\n(Dichte vs. Schallgeschwindigkeit)'
    )

    print("Erstelle Diagramm 2: Schallgeschwindigkeit vs. Brechungsindex...")
    plot_single_grid(
        x_attr='sound_velocity', y_attr='refractive_index',
        xlabel='Schallgeschwindigkeit [m/s]', ylabel='Brechungsindex [nD]',
        title=f'{haupt_titel}\n(Schallgeschwindigkeit vs. Brechungsindex)'
    )

    print("Erstelle Diagramm 3: Dichte vs. Viskosität...")
    plot_single_grid(
        x_attr='density', y_attr='viscosity',
        xlabel='Dichte [g/cm³]', ylabel='Viskosität [mPa.s]',
        title=f'{haupt_titel}\n(Dichte vs. Viskosität)'
    )

    print("Erstelle Diagramm 4: Dichte vs. Brechungsindex...")
    plot_single_grid(
        x_attr='density', y_attr='refractive_index',
        xlabel='Dichte [g/cm³]', ylabel='Brechungsindex [nD]',
        title=f'{haupt_titel}\n(Dichte vs. Brechungsindex)'
    )

    print("Alle Diagramme erstellt. Zeige Fenster an...")
    plt.show()


if __name__ == "__main__":
    run()