"""
ink_monitoring_grids.py
=====================================================================
Netzdiagramme zur Überwachung der Tinte (Kreuzempfindlichkeit).
System: Wasser + IPA + PG + Aluminium
Basis: PG konstant bei 3.64%

Bereich:
 - Aluminium: 1.0% bis 2.64% (Ideal: 1.82%)
 - IPA:       0.0% bis 5.0%  (Ideal: 3.64%)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Importiere deinen (korrigierten) Calculator
from ink_calculator import InkCalculator

# =====================================================================
# 1. Konfiguration
# =====================================================================
AL_BASE = 1.82
IPA_BASE = 3.64
PG_BASE = 3.64
TEMP_BASE = 25.0

ink = InkCalculator(tables_dir="../tables_parameters")

# Bereiche definieren
MIN_AL, MAX_AL = 1.0, 2.64
MIN_IPA, MAX_IPA = 0.0, 5.0

# Gitterlinien erzeugen und die Idealwerte explizit hinzufügen
al_levels = np.linspace(MIN_AL, MAX_AL, 9)
if AL_BASE not in al_levels:
    al_levels = np.sort(np.unique(np.append(al_levels, AL_BASE)))

ipa_levels = np.linspace(MIN_IPA, MAX_IPA, 11)
if IPA_BASE not in ipa_levels:
    ipa_levels = np.sort(np.unique(np.append(ipa_levels, IPA_BASE)))

# Hohe Auflösung für weiche Kurvenzeichnung
al_fine = np.linspace(MIN_AL, MAX_AL, 50)
ipa_fine = np.linspace(MIN_IPA, MAX_IPA, 50)

# Eckpunkte für die Beschriftung
CORNERS = [
    (MIN_AL, MIN_IPA),
    (MAX_AL, MIN_IPA),
    (MIN_AL, MAX_IPA),
    (MAX_AL, MAX_IPA)
]

sns.set_theme(style="whitegrid")


# =====================================================================
# 2. Konsolenausgabe
# =====================================================================
def print_operating_points():
    print("=" * 60)
    print(f"BERECHNETE WERTE FÜR DIE MASCHINENSTEUERUNG ({TEMP_BASE}°C)")
    print(f"Konstanter PG-Gehalt: {PG_BASE}%")
    print("=" * 60)

    # Idealzustand ausgeben
    p_ideal = ink.compute(al=AL_BASE, ipa=IPA_BASE, pg=PG_BASE, temperature=TEMP_BASE)
    print(">>> IDEALER ARBEITSPUNKT <<<")
    print(f"  {AL_BASE}% Al, {IPA_BASE}% IPA")
    print(f"  Dichte:                {p_ideal.density:.4f} g/cm³")
    print(f"  Brechungsindex:        {p_ideal.refractive_index:.4f} nD")
    print(f"  Schallgeschwindigkeit: {p_ideal.sound_velocity:.2f} m/s")
    print(f"  Viskosität:            {p_ideal.viscosity:.3f} mPa.s")
    print("-" * 60)

    print("GRENZWERTE (ECKEN DES DIAGRAMMS):")
    for al, ipa in CORNERS:
        p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
        print(f"  [{al:.2f}% Al, {ipa:.2f}% IPA] -> Dichte: {p.density:.4f}, nD: {p.refractive_index:.4f}")
    print("=" * 60)


# =====================================================================
# 3. Plot-Funktion
# =====================================================================
def plot_single_grid(x_attr, y_attr, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Konstantes Aluminium (Blau) -> IPA variiert
    for al in al_levels:
        x_vals, y_vals = [], []
        for ipa in ipa_fine:
            p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
            x_vals.append(getattr(p, x_attr))
            y_vals.append(getattr(p, y_attr))

        # Ideal-Linie dicker zeichnen
        is_ideal = np.isclose(al, AL_BASE, atol=0.001)
        lw = 2.5 if is_ideal else 1.0
        alpha = 0.9 if is_ideal else 0.5
        ax.plot(x_vals, y_vals, color='blue', alpha=alpha, linewidth=lw)

    # 2. Konstantes IPA (Rot) -> Aluminium variiert
    for ipa in ipa_levels:
        x_vals, y_vals = [], []
        for al in al_fine:
            p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
            x_vals.append(getattr(p, x_attr))
            y_vals.append(getattr(p, y_attr))

        # Ideal-Linie dicker zeichnen
        is_ideal = np.isclose(ipa, IPA_BASE, atol=0.001)
        lw = 2.5 if is_ideal else 1.0
        alpha = 0.9 if is_ideal else 0.5
        ax.plot(x_vals, y_vals, color='red', alpha=alpha, linewidth=lw)

    # 3. Arbeitspunkt (Stern) markieren
    p_ideal = ink.compute(al=AL_BASE, ipa=IPA_BASE, pg=PG_BASE, temperature=TEMP_BASE)
    ax.plot(getattr(p_ideal, x_attr), getattr(p_ideal, y_attr),
            marker='*', color='gold', markeredgecolor='black', markersize=18, zorder=10)

    ax.annotate('Idealzustand',
                (getattr(p_ideal, x_attr), getattr(p_ideal, y_attr)),
                xytext=(15, -15), textcoords='offset points',
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="gold", ec="black", alpha=0.9), zorder=10)

    # 4. Ecken (Grenzen) markieren
    for al, ipa in CORNERS:
        p = ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)
        x_val = getattr(p, x_attr)
        y_val = getattr(p, y_attr)

        ax.plot(x_val, y_val, marker='o', color='black', markersize=5, zorder=5)
        ax.annotate(f'{al}% Al\n{ipa}% IPA', (x_val, y_val),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7), zorder=6)

    # 5. Formatierung
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, color='black', fontweight='bold', pad=15)

    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], marker='*', color='gold', markeredgecolor='black', markersize=12, linestyle='None')
    ]
    ax.legend(custom_lines,
              ['Konstantes IPA (Rot)', 'Konstantes Aluminium (Blau)', 'Idealzustand (1.82% Al, 3.64% IPA)'],
              loc='best', fontsize=10)

    plt.tight_layout()


# =====================================================================
# 4. Hauptausführung
# =====================================================================
def run():
    print_operating_points()

    haupt_titel = f"Tinte ({PG_BASE}% PG) bei {TEMP_BASE}°C"

    print("Erstelle Diagramm 1: Dichte vs. Brechungsindex...")
    plot_single_grid(
        x_attr='density', y_attr='refractive_index',
        xlabel='Dichte [g/cm³]', ylabel='Brechungsindex [nD]',
        title=f'{haupt_titel}\n(Dichte vs. Brechungsindex)'
    )

    print("Erstelle Diagramm 2: Dichte vs. Schallgeschwindigkeit...")
    plot_single_grid(
        x_attr='density', y_attr='sound_velocity',
        xlabel='Dichte [g/cm³]', ylabel='Schallgeschwindigkeit [m/s]',
        title=f'{haupt_titel}\n(Dichte vs. Schallgeschwindigkeit)'
    )

    print("Erstelle Diagramm 3: Dichte vs. Viskosität...")
    plot_single_grid(
        x_attr='density', y_attr='viscosity',
        xlabel='Dichte [g/cm³]', ylabel='Viskosität [mPa.s]',
        title=f'{haupt_titel}\n(Dichte vs. Viskosität)'
    )

    print("Alle Diagramme erstellt. Zeige Fenster an...")
    plt.show()


if __name__ == "__main__":
    run()