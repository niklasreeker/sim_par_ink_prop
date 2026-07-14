"""
plot_signal_vs_uncertainties.py
=====================================================================
Balkendiagramm je Messgroesse (Dichte, Brechungsindex, Schall-
geschwindigkeit, Viskositaet) bei 25 C mit vier Balken:

    1. Signalhub Aluminium (1,72 -> 1,92 Massen-% Al)
    2. Aufloesung des Messgeraets
    3. Temperaturfehler (Sensorgenauigkeit +-0,1 C, ohne Methylgallat)
    4. Einfluss Methylgallat (0,23 Massen-%, Differenz mit/ohne MG)

Werte: eigene Berechnungen (ink_calculator) + Praesentation
"Berechnung Messgroessen" (Folien "Validierung der Aufloesung" /
"Abhaengigkeit von der Temperatur").

Nutzung:  python plot_signal_vs_uncertainties.py
Erzeugt:  signal_vs_uncertainties_25C.png (300 dpi)
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
#  Konfiguration
# ---------------------------------------------------------------
LOG_SCALE = False        # True: logarithmische y-Achse (bei stark
                         # unterschiedlichen Groessenordnungen sinnvoll)
OUTPUT_FILE = "signal_vs_uncertainties_25C.png"

# ---------------------------------------------------------------
#  Daten (25 C)
#  Reihenfolge der Balken:
#  [Signalhub Al, Aufloesung, Temp.-Fehler +-0,1C, Einfluss MG]
# ---------------------------------------------------------------
DATA = {
    "Dichtemessung": {
        "values": [0.001279, 0.00005, 0.000033, 0.00084],
        "unit": "g/cm³",
        "fmt": "{:.6f}",
    },
    "Brechungsindex": {
        "values": [0.000015, 0.0001, 0.000015, 0.00037],
        "unit": "nD",
        "fmt": "{:.6f}",
    },
    "Schallgeschwindigkeit": {
        "values": [0.323173, 0.01, 0.150518, 0.91],
        "unit": "m/s",
        "fmt": "{:.3f}",
    },
    "Viskosität": {
        "values": [0.003496, 0.013, 0.001933, 0.0061],
        "unit": "mPa·s",
        "fmt": "{:.4f}",
    },
}

BAR_LABELS = [
    "Signalhub\n(1,72–1,92 % Al)",
    "Auflösung\n(Messgerät)",
    "Temp.-Fehler\n(±0,1 °C)",
    "Einfluss MG\n(0,23 %)",
]
BAR_COLORS = ["#2e9e4f", "#3a7bd5", "#d94a3d", "#e8a13a"]

# ---------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------
def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(
        "Vergleich: Nutzsignal vs. Messunsicherheiten und MG-Einfluss (25 °C)",
        fontsize=14, fontweight="bold")

    for ax, (title, d) in zip(axes.flat, DATA.items()):
        values = d["values"]
        x = np.arange(len(values))
        bars = ax.bar(x, values, color=BAR_COLORS,
                      edgecolor="black", linewidth=0.6, zorder=3)

        # Wertbeschriftung ueber jedem Balken
        for bar, val in zip(bars, values):
            ax.annotate(d["fmt"].format(val),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

        # Referenzlinie: Signalhub (Balken 1) zum direkten Vergleich
        ax.axhline(values[0], color=BAR_COLORS[0], linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=2)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(f"Messwertänderung [{d['unit']}]")
        ax.set_xticks(x)
        ax.set_xticklabels(BAR_LABELS, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3, zorder=0)

        if LOG_SCALE:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, max(values) * 1.18)


    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Gespeichert: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()