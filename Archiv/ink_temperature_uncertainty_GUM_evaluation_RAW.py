"""
ink_temperature_uncertainty_GUM_evaluation.py
=====================================================================
Erweiterung von 'ink_resolution_final.py':
Einfluss der Temperatursensor-Genauigkeit auf die effektive Auflösung.
=====================================================================

Hintergrund
-----------
Jede Messgröße P (Dichte, Brechungsindex, Schallgeschwindigkeit,
Viskosität) ist temperaturabhängig. Die Unsicherheit des
Temperatursensors (ΔT = ±0,1 °C bzw. ±0,01 °C) pflanzt sich über die
Temperaturempfindlichkeit dP/dT auf die Messgröße fort:

        u_T(P) = |dP/dT| · ΔT

Diese temperaturbedingte Unsicherheit wird mit der reinen
Hardware-Auflösung des Sensors quadratisch kombiniert
(unabhängige, unkorrelierte Fehlerquellen, vgl. GUM):

        u_ges = sqrt( u_sensor² + u_T² )

Bewertung: u_ges wird mit dem Signalhub ΔS über den Toleranzbereich
(1,72 % … 1,92 % Al) verglichen. Daraus folgt die Zahl der noch
trennbaren Stufen im Toleranzband:

        N = ΔS / (2 · u_ges)

Die Temperaturempfindlichkeit dP/dT wird numerisch über einen
zentralen Differenzenquotienten direkt aus dem InkCalculator-Modell
gewonnen – es müssen also keine Formeln von Hand hinterlegt werden.
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
AL_LOWER = 1.72   # -0.1 % Toleranz
AL_UPPER = 1.92   # +0.1 % Toleranz
PG_BASE = 3.64
TEMP_BASE = 25.0

IPA_0_EVAP = 3.64   # Idealzustand
IPA_50_EVAP = 1.82
IPA_100_EVAP = 0.0

# Hardware-Auflösungen
RES_DENS_BIEGE = 0.0001     # g/cm³
RES_RI = 0.0002             # nD
RES_SOUND = 0.01            # m/s
RES_VISC_PCT = 0.01         # 1 % vom Messwert (relativ!)

# --- NEU: Temperatursensor-Genauigkeiten [°C] -----------------------
# 0.1 °C  = Standardsensor
# 0.01 °C = nach genauerer Kalibrierung
TEMP_ACCURACIES = [0.1, 0.01]

# Schrittweite für die numerische Ableitung dP/dT [°C].
# Klein genug für lokale Linearität, groß genug gegen Tabellenrauschen.
DT_DERIV = 0.5

ink = InkCalculator(tables_dir="tables_parameters")
al_range = np.linspace(1.0, 3.0, 100)

sns.set_theme(style="whitegrid")


# =====================================================================
# 2. Hilfsfunktionen
# =====================================================================
def format_decimal(val):
    """Verhindert wissenschaftliche Notation (z.B. 4e-05) und gibt
    saubere Dezimalzahlen aus."""
    s = f"{val:.6f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if s in ("", "-"):
        s = "0"
    return s


def prop_at(property_name, al, temperature, ipa=IPA_0_EVAP):
    """Eine Messgröße bei gegebenem Al-Gehalt, Temperatur und IPA-Zustand."""
    return getattr(
        ink.compute(al=al, ipa=ipa, pg=PG_BASE, temperature=temperature),
        property_name,
    )


def dP_dT(property_name, al, temperature, ipa=IPA_0_EVAP, h=DT_DERIV):
    """Numerische Temperaturempfindlichkeit dP/dT über zentralen
    Differenzenquotienten (Einheit: [P]/°C)."""
    p_plus = prop_at(property_name, al, temperature + h, ipa)
    p_minus = prop_at(property_name, al, temperature - h, ipa)
    return (p_plus - p_minus) / (2.0 * h)


def sensor_uncertainty(res_value, is_relative, y_values):
    """Sensorauflösung als ABSOLUTE Unsicherheit zurückgeben.
    Bei relativen Sensoren (Viskosität: 1 % vom Messwert) wird mit dem
    lokalen Messwert multipliziert, damit eine quadratische Kombination
    mit u_T (ebenfalls absolut) möglich ist."""
    if is_relative:
        return res_value * np.abs(y_values)
    return np.full_like(np.asarray(y_values, dtype=float), res_value)


def verdict(steps):
    """Heuristische Eignungsbewertung anhand der trennbaren Stufen N."""
    if steps >= 5:
        return "sehr gut geeignet"
    if steps >= 2:
        return "geeignet"
    if steps >= 1:
        return "grenzwertig"
    return "ungeeignet"


# =====================================================================
# 3. Analyse + Plot je Messgröße
# =====================================================================
def analyze_and_plot(ax, property_name, ylabel, unit, title,
                     res_value, res_label, is_relative=False):
    # ---- Kurven (Ideal + Verdampfungszustände) bei Basistemperatur ----
    y_0 = np.array([prop_at(property_name, al, TEMP_BASE, IPA_0_EVAP) for al in al_range])
    y_50 = np.array([prop_at(property_name, al, TEMP_BASE, IPA_50_EVAP) for al in al_range])
    y_100 = np.array([prop_at(property_name, al, TEMP_BASE, IPA_100_EVAP) for al in al_range])

    # ---- Temperaturempfindlichkeit entlang der Ideal-Kurve ----
    dpdt = np.array([dP_dT(property_name, al, TEMP_BASE) for al in al_range])

    # ---- Sensor-Unsicherheit (absolut, ggf. relativ umgerechnet) ----
    u_sensor = sensor_uncertainty(res_value, is_relative, y_0)

    # ---- Signalhub über den Toleranzbereich (auf der Ideal-Kurve) ----
    val_lower = prop_at(property_name, AL_LOWER, TEMP_BASE)
    val_upper = prop_at(property_name, AL_UPPER, TEMP_BASE)
    signalhub = abs(val_upper - val_lower)

    # ---- Markierung des Toleranzbereichs ----
    ax.axvspan(AL_LOWER, AL_UPPER, color='gray', alpha=0.15, label='Toleranzbereich (±0,1 % Al)')
    ax.axvline(AL_LOWER, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(AL_UPPER, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(AL_IDEAL, color='black', linestyle='--', linewidth=1.5, label='Idealzustand (1,82 % Al)')

    # ---- Kombinierte Unsicherheitsbänder (breit -> schmal zeichnen) ----
    # Die Bänder sind verschachtelt: u_sensor <= u_ges(0,01°C) <= u_ges(0,1°C)
    band_styles = {0.1: ('#d62728', 0.16), 0.01: ('#ff7f0e', 0.30)}
    for dT in sorted(TEMP_ACCURACIES, reverse=True):
        u_T = np.abs(dpdt) * dT
        u_tot = np.sqrt(u_sensor**2 + u_T**2)
        color, alpha = band_styles.get(dT, ('purple', 0.2))
        ax.fill_between(al_range, y_0 - u_tot, y_0 + u_tot,
                        color=color, alpha=alpha,
                        label=f'± gesamt (Sensor ⊕ Temp ±{format_decimal(dT)} °C)')

    # ---- Reine Sensorauflösung (schmalstes Band, oben) ----
    ax.fill_between(al_range, y_0 - u_sensor, y_0 + u_sensor,
                    color='tab:blue', alpha=0.45, label=res_label)

    # ---- Hauptkurven ----
    ax.plot(al_range, y_0, color='tab:blue', linewidth=2, label='IPA 0 % verdampft (ideal)')
    ax.plot(al_range, y_50, color='tab:orange', linewidth=1.2, linestyle='-.', alpha=0.7, label='IPA 50 % verdampft')
    ax.plot(al_range, y_100, color='tab:red', linewidth=1.2, linestyle=':', alpha=0.7, label='IPA 100 % verdampft')

    # ---- Kennzahlen am Idealpunkt (1,82 % Al) ----
    dpdt_ideal = dP_dT(property_name, AL_IDEAL, TEMP_BASE)
    y_ideal = prop_at(property_name, AL_IDEAL, TEMP_BASE)
    u_sensor_ideal = res_value * abs(y_ideal) if is_relative else res_value
    steps_sensor = signalhub / (2.0 * u_sensor_ideal) if u_sensor_ideal > 0 else np.inf

    per_dT = {}
    for dT in TEMP_ACCURACIES:
        u_T_ideal = abs(dpdt_ideal) * dT
        u_tot_ideal = np.sqrt(u_sensor_ideal**2 + u_T_ideal**2)
        steps = signalhub / (2.0 * u_tot_ideal) if u_tot_ideal > 0 else np.inf
        per_dT[dT] = dict(u_T=u_T_ideal, u_tot=u_tot_ideal, steps=steps)

    # ---- Textbox (kompakt) ----
    lines = [
        f"Signalhub Δ = {format_decimal(signalhub)} {unit}",
        f"dP/dT ≈ {format_decimal(dpdt_ideal)} {unit}/°C",
        f"Sensor: ±{format_decimal(u_sensor_ideal)} {unit}  (N≈{steps_sensor:.1f})",
    ]
    for dT in TEMP_ACCURACIES:
        r = per_dT[dT]
        lines.append(
            f"+Temp ±{format_decimal(dT)} °C → ges. ±{format_decimal(r['u_tot'])} {unit}"
            f"  (N≈{r['steps']:.1f}, {verdict(r['steps'])})"
        )
    textstr = "\n".join(lines)
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', bbox=props)

    # ---- Formatierung ----
    ax.set_xlabel('Aluminiumgehalt in Tinte [%]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=7.5)
    ax.set_xlim(1.0, 3.0)

    return {
        'signalhub': signalhub,
        'unit': unit,
        'dpdt_ideal': dpdt_ideal,
        'u_sensor': u_sensor_ideal,
        'steps_sensor': steps_sensor,
        'per_dT': per_dT,
    }


# =====================================================================
# 4. Konsolen-Zusammenfassung
# =====================================================================
def print_summary(collected):
    print("\n" + "=" * 92)
    print("ZUSAMMENFASSUNG  –  Bewertung am Idealpunkt (1,82 % Al, 25 °C, IPA 0 %)")
    print("=" * 92)
    for title, d in collected:
        unit = d['unit']
        print(f"\n▶ {title}")
        print(f"   Signalhub (1,72 → 1,92 % Al):   Δ = {format_decimal(d['signalhub'])} {unit}")
        print(f"   Temp.-Empfindlichkeit dP/dT:    {format_decimal(d['dpdt_ideal'])} {unit}/°C")
        print(f"   Sensorauflösung:                ± {format_decimal(d['u_sensor'])} {unit}"
              f"   ->  N = {d['steps_sensor']:.1f}  ({verdict(d['steps_sensor'])})")
        for dT in TEMP_ACCURACIES:
            r = d['per_dT'][dT]
            print(f"   + Temp.-Sensor ±{format_decimal(dT):<5} °C:     "
                  f"u_T = ±{format_decimal(r['u_T'])} {unit}  |  "
                  f"u_ges = ±{format_decimal(r['u_tot'])} {unit}  |  "
                  f"N = {r['steps']:.1f}  ({verdict(r['steps'])})")
    print("\n" + "=" * 92)
    print("N = Signalhub / (2 · u_ges) = Anzahl trennbarer Stufen im Toleranzband.")
    print("Heuristik:  N >= 5 sehr gut | 2-5 geeignet | 1-2 grenzwertig | < 1 ungeeignet")
    print("=" * 92 + "\n")


# =====================================================================
# 5. Vergleichs-Balkendiagramm (Eignung je Messgröße)
# =====================================================================
def plot_summary(collected):
    labels = [t for t, _ in collected]
    x = np.arange(len(labels))
    width = 0.25

    def clamp(vals):
        return [max(v, 1e-2) for v in vals]  # für log-Achse

    n_sensor = clamp([d['steps_sensor'] for _, d in collected])
    n_001 = clamp([d['per_dT'][0.01]['steps'] for _, d in collected]) if 0.01 in TEMP_ACCURACIES else None
    n_01 = clamp([d['per_dT'][0.1]['steps'] for _, d in collected]) if 0.1 in TEMP_ACCURACIES else None

    fig, ax = plt.subplots(figsize=(11, 6))

    groups = [('nur Sensor (ideal)', n_sensor, '#1f77b4', -width)]
    if n_001 is not None:
        groups.append(('+ Temp ±0,01 °C', n_001, '#ff7f0e', 0.0))
    if n_01 is not None:
        groups.append(('+ Temp ±0,1 °C', n_01, '#d62728', width))

    for lbl, vals, col, pos in groups:
        bars = ax.bar(x + pos, vals, width, label=lbl, color=col)
        ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)

    ax.axhline(2, color='green', linestyle=':', linewidth=1.2)
    ax.axhline(1, color='red', linestyle='--', linewidth=1.2)
    ax.text(len(labels) - 0.4, 2.05, 'N=2 (geeignet)', color='green', fontsize=8, ha='right')
    ax.text(len(labels) - 0.4, 1.02, 'N=1 (Grenze)', color='red', fontsize=8, ha='right')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_ylabel('Trennbare Stufen N im Toleranzband  (log-Skala)')
    ax.set_title('Eignung der Messgrößen – Einfluss der Temperatursensor-Genauigkeit',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', which='both', alpha=0.4)
    fig.tight_layout()
    return fig


# =====================================================================
# 6. Hauptausführung
# =====================================================================
def run():
    print("Analysiere Temperatureinfluss auf die Sensorauflösung ...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    cfg = [
        (axs[0, 0], 'density', 'Dichte [g/cm³]', 'g/cm³',
         'Dichtemessung (Biegeschwinger)',
         RES_DENS_BIEGE, f'Biegeschwinger (±{RES_DENS_BIEGE} g/cm³)', False),

        (axs[0, 1], 'refractive_index', 'Brechungsindex [nD]', 'nD',
         'Brechungsindex (Refraktometer)',
         RES_RI, f'Refraktometer (±{RES_RI} nD)', False),

        (axs[1, 0], 'sound_velocity', 'Schallgeschwindigkeit [m/s]', 'm/s',
         'Akustische Messung (Ultraschall)',
         RES_SOUND, f'Ultraschall (±{RES_SOUND} m/s)', False),

        (axs[1, 1], 'viscosity', 'Viskosität [mPa·s]', 'mPa·s',
         'Viskositätsmessung',
         RES_VISC_PCT, f'Viskosimeter (±{RES_VISC_PCT * 100:.0f} %)', True),
    ]

    collected = []
    for (ax, prop, ylabel, unit, title, res, reslabel, isrel) in cfg:
        data = analyze_and_plot(ax, prop, ylabel, unit, title, res, reslabel, isrel)
        collected.append((title, data))

    fig.suptitle('Einfluss der Temperatursensor-Genauigkeit auf die effektive Auflösung',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    print_summary(collected)
    plot_summary(collected)

    plt.show()


if __name__ == "__main__":
    run()