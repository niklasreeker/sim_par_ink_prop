"""
plot_refractive_index.py
=====================================================================
Reproduziert die beiden Diagramme aus der Präsentationsfolie
"Refractive Index of ternary Ink solvent (Water/PG/IPA)" -- aber mit
den *berechneten* Brechungsindizes aus ink_calculator.py statt den
gemessenen Kalibrierwerten.

Berechnet nD (25 C) fuer drei Loesungsmittelsysteme in Wasser, jeweils
ueber den gesamten Bereich von 0 % bis 100 % Loesungsmittelanteil:

    * IPA (100)      -> reines Isopropanol in Wasser
    * PG/IPA (5:3)   -> Propylenglykol : Isopropanol = 5 : 3
    * PG/IPA (5:2)   -> Propylenglykol : Isopropanol = 5 : 2

Erzeugt werden:
    * refractive_index_charts.png  -> Zwei-Panel-Abbildung wie die Folie
                                      (links Zoom 0-15 %, rechts 0-100 %)
    * refractive_index_full.png    -> nur das grosse Diagramm (0-100 %)
    * refractive_index_zoom.png    -> nur der Zoom (0-15 %)
    * refractive_index_values.csv  -> alle berechneten nD-Werte

Voraussetzung: ink_calculator.py und der Ordner tables_parameters/
liegen neben diesem Skript (oder werden automatisch gefunden).
=====================================================================
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")           # dateibasiertes Rendern (kein Display noetig)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter, MultipleLocator


# =====================================================================
#  KONFIGURATION
# =====================================================================
TEMPERATURE = 25.0              # Messtemperatur [deg C]

# Loesungsmittelsysteme: (Anzeigename, IPA-Anteil, PG-Anteil, Farbe)
# Die Anteile sind Massenverhaeltnisse innerhalb des Loesungsmittels.
SYSTEMS = [
    ("PG/IPA (5:3)", 3.0 / 8.0, 5.0 / 8.0, "#4472C4"),   # blau
    ("PG/IPA (5:2)", 2.0 / 7.0, 5.0 / 7.0, "#ED7D31"),   # orange
    ("IPA (100)",    1.0,       0.0,       "#A5A5A5"),   # grau
]

# Loesungsmittelanteile (Massen-%) an denen "gemessen" (= gerechnet) wird.
CONTENTS_PCT = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12.5, 15, 20, 25, 30,
     35, 40, 45, 50, 60, 70, 80, 90, 100], dtype=float
)

# Deutsche Zahlendarstellung (Komma als Dezimaltrennzeichen) wie im Original.
DECIMAL_COMMA = True

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
#  ink_calculator importieren + Tabellenordner finden
# =====================================================================
def locate_and_import():
    """Findet ink_calculator.py und den Ordner tables_parameters."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        from ink_calculator import InkCalculator
    except ImportError as exc:
        raise SystemExit(
            "Konnte 'ink_calculator.py' nicht importieren. Lege dieses Skript "
            "in denselben Ordner wie ink_calculator.py.\n  Details: %s" % exc
        )

    # tables_parameters neben dem Skript oder im aktuellen Arbeitsordner suchen
    candidates = [
        os.path.join(here, "../tables_parameters"),
        os.path.join(os.getcwd(), "../tables_parameters"),
    ]
    tables_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if tables_dir is None:
        raise SystemExit(
            "Ordner 'tables_parameters' nicht gefunden. Er muss neben diesem "
            "Skript liegen und die CSV-Tabellen (ipa_refractive.csv, "
            "pg_refractive.csv, ipa_density.csv, pg_density.csv, ...) enthalten."
        )
    return InkCalculator, tables_dir


# =====================================================================
#  Hilfsfunktionen
# =====================================================================
def de(value, decimals):
    """Zahl formatieren, optional mit Komma als Dezimaltrennzeichen."""
    s = f"{value:.{decimals}f}"
    return s.replace(".", ",") if DECIMAL_COMMA else s


def compute_series(ink, ipa_frac, pg_frac):
    """
    Berechnet nD ueber alle CONTENTS_PCT fuer ein Loesungsmittelsystem.
    Punkte ausserhalb des tabellierten Bereichs werden uebersprungen.

    Rueckgabe: (x_fraction, nD, skipped_contents)
    """
    xs, ys, skipped = [], [], []
    for content in CONTENTS_PCT:
        ipa = content * ipa_frac
        pg = content * pg_frac
        try:
            n_d = ink.refractive_index(al=0.0, ipa=ipa, pg=pg,
                                       temperature=TEMPERATURE)
            xs.append(content / 100.0)      # als Bruch 0..1 fuer die x-Achse
            ys.append(n_d)
        except Exception:                   # z.B. ausserhalb der Interpolation
            skipped.append(content)
    return np.array(xs), np.array(ys), skipped


def fit_trend(x_frac, y):
    """
    Polynom-Trendlinie 3. Grades (wie Excel), x = Loesungsmittelbruch 0..1.
    Rueckgabe: (coeffs_hoch->niedrig, r2, x_dense, y_dense)
    """
    degree = min(3, len(x_frac) - 1)
    coeffs = np.polyfit(x_frac, y, degree)
    poly = np.poly1d(coeffs)

    # Bestimmtheitsmass R^2
    y_hat = poly(x_frac)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    x_dense = np.linspace(x_frac.min(), x_frac.max(), 400)
    return coeffs, r2, x_dense, poly(x_dense)


def poly_label(coeffs, r2):
    """Baut den zweizeiligen Excel-artigen Trendlinien-Text."""
    # coeffs sind hoch -> niedrig; auf Grad 3 auffuellen
    c = list(coeffs)
    while len(c) < 4:
        c.insert(0, 0.0)
    a, b, cc, d = c[-4], c[-3], c[-2], c[-1]

    def term(coef, suffix, first=False):
        sign = "" if (first and coef >= 0) else (" + " if coef >= 0 else " - ")
        if first and coef < 0:
            sign = "-"
        return f"{sign}{de(abs(coef), 4)}{suffix}"

    eq = ("y = " + term(a, "x\u00b3", first=True) + term(b, "x\u00b2")
          + term(cc, "x") + term(d, ""))
    return eq + "\n" + f"R\u00b2 = {de(r2, 4)}"


# =====================================================================
#  Achsen-Styling im Excel-Look
# =====================================================================
def style_axes(ax, y_decimals):
    ax.grid(True, color="#D9D9D9", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#BFBFBF")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#595959", labelsize=9, length=0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _pos: de(v, y_decimals)))
    ax.set_xlabel("content of solvent(mixture)", fontsize=9, color="#595959")
    ax.set_ylabel("refractive index nD", fontsize=9, color="#595959")


def draw_panel(ax, series, zoom):
    """Zeichnet einen Chart-Bereich (zoom=True -> Ausschnitt 0-15 %)."""
    for name, color, xs, ys, trend in series:
        cx, r2, xd, yd = trend
        ax.plot(xd, yd, linestyle=":", color=color, linewidth=1.4, zorder=2)
        ax.scatter(xs, ys, s=22, color=color, edgecolors="white",
                   linewidths=0.4, zorder=3, label=name)

    if zoom:
        ax.set_xlim(-0.01, 0.15)
        ax.set_ylim(1.330, 1.350)
        ax.set_xticks(np.arange(-0.01, 0.151, 0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.001))
        style_axes(ax, y_decimals=3)
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.32, 1.42)
        ax.set_xticks(np.arange(0.0, 1.01, 0.10))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        style_axes(ax, y_decimals=2)
        ax.set_title("nD of different ink solvent systems",
                     fontsize=12, color="#595959", pad=12)
        # Trendlinien-Gleichungen wie im Original platzieren
        anchors = {"PG/IPA (5:2)": (0.40, 0.90),
                   "PG/IPA (5:3)": (0.58, 0.66),
                   "IPA (100)":    (0.60, 0.40)}
        for name, color, xs, ys, trend in series:
            if name in anchors:
                fx, fy = anchors[name]
                ax.text(fx, fy, poly_label(trend[0], trend[1]),
                        transform=ax.transAxes, color=color, fontsize=8,
                        ha="left", va="center")


# =====================================================================
#  Hauptprogramm
# =====================================================================
def main():
    InkCalculator, tables_dir = locate_and_import()
    print(f"Tabellen: {tables_dir}")
    print(f"Temperatur: {TEMPERATURE} C\n")

    ink = InkCalculator(tables_dir=tables_dir)

    # --- alle Systeme berechnen ---
    series = []          # (name, color, xs, ys, trend)
    csv_rows = []        # fuer Export
    for name, ipa_frac, pg_frac, color in SYSTEMS:
        xs, ys, skipped = compute_series(ink, ipa_frac, pg_frac)
        if len(xs) < 2:
            print(f"[!] {name}: zu wenige gueltige Punkte -- uebersprungen.")
            continue
        trend = fit_trend(xs, ys)
        series.append((name, color, xs, ys, trend))

        print(f"{name}: {len(xs)} Punkte berechnet "
              f"(0-{xs.max()*100:.0f} % Loesungsmittel), "
              f"nD-Bereich {ys.min():.5f}..{ys.max():.5f}")
        if skipped:
            print(f"    uebersprungen (ausserhalb Tabellenbereich): "
                  f"{', '.join(f'{s:g}%' for s in skipped)}")

        for xf, n in zip(xs, ys):
            csv_rows.append((name, xf * 100.0, n))

    if not series:
        raise SystemExit("Keine berechenbaren Datenreihen -- Abbruch.")

    # --- CSV-Export ---
    csv_path = os.path.join(OUT_DIR, "../refractive_index_values.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("System,Content_solvent_percent,Refractive_index_nD\n")
        for name, content, n in csv_rows:
            f.write(f"{name},{content:g},{n:.5f}\n")
    print(f"\nWerte gespeichert: {csv_path}")

    # --- kombinierte Zwei-Panel-Abbildung (wie die Folie) ---
    fig, (ax_zoom, ax_full) = plt.subplots(
        1, 2, figsize=(16, 6.2),
        gridspec_kw={"width_ratios": [1, 1.55], "wspace": 0.22})
    fig.patch.set_facecolor("white")
    draw_panel(ax_zoom, series, zoom=True)
    draw_panel(ax_full, series, zoom=False)

    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.16, top=0.90, left=0.05, right=0.98)

    combined = os.path.join(OUT_DIR, "refractive_index_charts.png")
    fig.savefig(combined, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Abbildung gespeichert: {combined}")

    # --- Einzeldiagramme ---
    for zoom, fname, size in [(False, "refractive_index_full.png", (9, 6)),
                              (True, "refractive_index_zoom.png", (7, 6))]:
        f1, a1 = plt.subplots(figsize=size)
        f1.patch.set_facecolor("white")
        draw_panel(a1, series, zoom=zoom)
        h, l = a1.get_legend_handles_labels()
        a1.legend(h, l, loc="lower center", ncol=len(series),
                  frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.22))
        path = os.path.join(OUT_DIR, fname)
        f1.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(f1)
        print(f"Abbildung gespeichert: {path}")

    plt.close(fig)
    print("\nFertig.")


if __name__ == "__main__":
    main()