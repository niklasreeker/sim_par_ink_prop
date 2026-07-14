import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ==========================================
# PFAD-ANPASSUNG FÜR IMPORT
# ==========================================
# Fügt den übergeordneten Ordner (Projektordner) zum Python-Pfad hinzu,
# da das Skript im Ordner "plots" liegt und "ink_calculator.py" eine Ebene darüber.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ink_calculator import InkCalculator

# ==========================================
# HILFSFUNKTIONEN
# ==========================================
def calculate_evaporation(w_before, w_after):
    """
    Berechnet die kumulierte verdampfte Masse (bereinigt um Pipetten-Entnahmen).
    """
    cum_evap = [0.0]  # Bei t=0 ist noch nichts verdampft
    current_evap = 0.0

    for i in range(len(w_before) - 1):
        # Verdunstete Menge im Zeitfenster
        evaporated_step = w_after[i] - w_before[i + 1]
        current_evap += evaporated_step
        cum_evap.append(current_evap)

    return cum_evap

def analyze_binary_experiment(ink_calc, time, ri, temp):
    """
    Berechnet die Zusammensetzung NUR anhand des Brechungsindex (für Versuch 1).
    """
    pct_ipa_list = []
    pct_water_list = []
    pct_pg_list = [0.0] * len(time)  # PG ist immer 0

    for i in range(len(time)):
        n_meas = ri[i]
        T = temp[i]

        def objective(pct_ipa):
            n_calc = ink_calc.refractive_index(al=0.0, ipa=pct_ipa, pg=0.0, temperature=T)
            return abs(n_calc - n_meas)

        res = minimize_scalar(objective, bounds=(0, 100), method='bounded')

        pct_ipa_list.append(res.x)
        pct_water_list.append(100.0 - res.x)

    return time, pct_ipa_list, pct_water_list, pct_pg_list


def analyze_ternary_experiment(ink_calc, time, ri, w_before, w_after, temp, m_pg_initial):
    """
    Berechnet die Zusammensetzung durch Brechungsindex und Massen-Tracking (für Versuch 2 & 3).
    """
    m_pg = m_pg_initial
    pct_ipa_list = []
    pct_water_list = []
    pct_pg_list = []

    for i in range(len(time)):
        w_bef = w_before[i]
        n_meas = ri[i]
        T = temp[i]

        def objective(m_ipa):
            pct_ipa = (m_ipa / w_bef) * 100.0
            pct_pg = (m_pg / w_bef) * 100.0

            if pct_ipa < 0 or (pct_ipa + pct_pg) >= 100:
                return 999.0

            n_calc = ink_calc.refractive_index(al=0.0, ipa=pct_ipa, pg=pct_pg, temperature=T)
            return abs(n_calc - n_meas)

        max_ipa_possible = max(0.0, w_bef - m_pg)
        res = minimize_scalar(objective, bounds=(0, max_ipa_possible), method='bounded')

        m_ipa_current = res.x
        m_water_current = w_bef - m_pg - m_ipa_current

        pct_ipa_list.append((m_ipa_current / w_bef) * 100.0)
        pct_water_list.append((m_water_current / w_bef) * 100.0)
        pct_pg_list.append((m_pg / w_bef) * 100.0)

        if i < len(time) - 1:
            drop_mass = w_before[i] - w_after[i]
            frac_pg = m_pg / w_before[i]
            m_pg -= (drop_mass * frac_pg)

    return time, pct_ipa_list, pct_water_list, pct_pg_list


# ==========================================
# DATEN VORBEREITEN
# ==========================================

# Versuch 1: IPA + Wasser (Raumtemperatur)
t1_ri = [0, 11, 30, 50, 70, 91, 120, 190, 210, 232, 253, 280, 301, 330]
ri1 = [1.3538, 1.3572, 1.3482, 1.3461, 1.3440, 1.3436, 1.3431, 1.3423, 1.3420, 1.3418, 1.3414, 1.3411, 1.3407, 1.3402]
temp1 = [22.8, 22.7, 22.5, 22.3, 22.1, 22.1, 22.1, 22.1, 22.2, 22.2, 22.4, 22.5, 22.5, 22.7]

t1_w = [0, 11, 30, 50, 70]
w1_before = [16.05, 14.70, 13.24, 12.25, 11.37]
w1_after = [15.37, 14.00, 12.70, 11.80, 10.90]

# Versuch 2: IPA + Wasser + PG (Raumtemperatur)
t2 = [0, 20, 35, 102, 120, 144, 160, 192, 212, 238]
ri2 = [1.3745, 1.3744, 1.3737, 1.3712, 1.3708, 1.3703, 1.3700, 1.3697, 1.3697, 1.3697]
# Gewichte mit +44.65g Offset ab Minute 192 wegen Gefäßwechsel bereinigt
w2_before = [20.40, 18.60, 17.50, 15.17, 14.15, 13.28, 12.48, -32.70 + 44.65, -33.42 + 44.65, -34.05 + 44.65]
w2_after = [19.55, 18.10, 17.05, 14.50, 13.67, 12.65, 12.00, -33.45 + 44.65, -34.04 + 44.65, -34.91 + 44.65]
temp2 = [22.1, 22.0, 22.0, 22.2, 22.1, 22.3, 22.2, 22.4, 22.5, 22.6]
pg2_initial = 5.05

# Versuch 3: IPA + Wasser + PG (60°C -> 80°C)
t3 = [0, 5, 10, 20, 30, 40, 50, 60, 70]
ri3 = [1.3775, 1.3773, 1.3768, 1.3768, 1.3764, 1.3768, 1.3779, 1.3819, 1.3851]
w3_before = [20.32, 18.93, 17.97, 15.87, 14.20, 12.63, 11.52, 9.23, 8.02]
w3_after = [19.60, 18.37, 17.28, 15.35, 13.52, 12.20, 10.28, 8.78, 8.02]  # Letzter Wert als Platzhalter
temp3 = [22.2, 22.3, 22.4, 22.4, 22.5, 22.5, 22.6, 22.6, 22.9]
pg3_initial = 5.22


# ==========================================
# DATEN BERECHNEN
# ==========================================
print("Berechne physikalische Verdunstung...")
evap1 = calculate_evaporation(w1_before, w1_after)
evap2 = calculate_evaporation(w2_before, w2_after)
evap3 = calculate_evaporation(w3_before, w3_after)

print("Initialisiere InkCalculator...")
# Passe den Ordner für tables_parameters an, falls dieser auch im Hauptordner liegt
tables_path = os.path.join(parent_dir, "tables_parameters")
ink = InkCalculator(tables_dir=tables_path)

print("Berechne Versuch 1 (direkt über Brechungsindex)...")
t1_res, pct_ipa1, pct_water1, pct_pg1 = analyze_binary_experiment(ink, t1_ri, ri1, temp1)

print("Berechne Versuch 2 (ternäres System)...")
t2_res, pct_ipa2, pct_water2, pct_pg2 = analyze_ternary_experiment(ink, t2, ri2, w2_before, w2_after, temp2, pg2_initial)

print("Berechne Versuch 3 (ternäres System)...")
t3_res, pct_ipa3, pct_water3, pct_pg3 = analyze_ternary_experiment(ink, t3, ri3, w3_before, w3_after, temp3, pg3_initial)


# ==========================================
# PLOTTING FUNKTIONEN
# ==========================================
def plot_experiment(ax, t_ri, ri, t_w, evap, title, annotate_80c=False, transfer_time=None):
    # Linke Achse: Brechungsindex (Blau)
    color_ri = 'tab:blue'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Zeit (Minuten)', fontsize=10)
    ax.set_ylabel('Brechungsindex (nD)', color=color_ri, fontsize=11)
    line1, = ax.plot(t_ri, ri, marker='o', color=color_ri, linewidth=2, label='Brechungsindex')
    ax.tick_params(axis='y', labelcolor=color_ri)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Rechte Achse: Kumulierte Verdunstung (Rot)
    ax2 = ax.twinx()
    color_w = 'tab:red'
    ax2.set_ylabel('Kumulierte verdampfte Masse (g)', color=color_w, fontsize=11)
    line2, = ax2.plot(t_w, evap, marker='s', color=color_w, linewidth=2, linestyle='-', label='Verdampfte Masse')
    ax2.set_ylim(0, max(evap) * 1.1)
    ax2.tick_params(axis='y', labelcolor=color_w)

    # Markierung für das Umfüllen
    if transfer_time is not None:
        ax.axvline(x=transfer_time, color='purple', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(transfer_time + 2, 0.45, 'Umfüllen in kleineres Gefäß', color='purple', fontsize=10, transform=ax.get_xaxis_transform())

    # Markierung für 80°C Erwärmung
    if annotate_80c:
        ax.axvline(x=50, color='gray', linestyle=':', linewidth=2)
        ax.text(52, min(ri), 'Platte auf 80°C erwärmt', color='gray', fontsize=10, verticalalignment='bottom')

    # Gemeinsame Legende
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

def plot_composition(ax, t, ipa_pct, water_pct, pg_pct, title, annotate_80c=False, transfer_time=None):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Zeit (Minuten)', fontsize=10)
    ax.set_ylabel('Massenanteil (%)', fontsize=11)

    # IPA und Wasser plotten
    ax.plot(t, ipa_pct, marker='o', color='tab:blue', linewidth=2, label='Isopropanol (IPA)')
    ax.plot(t, water_pct, marker='s', color='tab:cyan', linewidth=2, linestyle='--', label='Wasser')

    # PG nur plotten, wenn es in der Mischung ist
    if max(pg_pct) > 0.1:
        ax.plot(t, pg_pct, marker='^', color='tab:orange', linewidth=2, linestyle='-.', label='Propylenglykol (PG)')

    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')

    # Markierung für das Umfüllen
    if transfer_time is not None:
        ax.axvline(x=transfer_time, color='purple', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(transfer_time + 2, 45, 'Umfüllen in kleineres Gefäß', color='purple', fontsize=10)

    # Markierung für 80°C Erwärmung
    if annotate_80c:
        ax.axvline(x=50, color='gray', linestyle=':', linewidth=2)
        ax.text(52, 5, 'Platte auf 80°C erwärmt', color='gray', fontsize=10)


# ==========================================
# ABBILDUNGEN ERSTELLEN
# ==========================================
print("Erstelle Diagramme...")

# --- Abbildung 1: Verdunstung und Brechungsindex ---
fig1, axs1 = plt.subplots(3, 1, figsize=(10, 15))
fig1.suptitle('Physikalische Verdunstung & Brechungsindex der Mischungen', fontsize=16, fontweight='bold')

plot_experiment(axs1[0], t1_ri, ri1, t1_w, evap1, 'Versuch 1: IPA + Wasser (Raumtemperatur)', transfer_time=70)
plot_experiment(axs1[1], t2, ri2, t2, evap2, 'Versuch 2: IPA + Wasser + PG (Raumtemperatur)', transfer_time=160)
plot_experiment(axs1[2], t3, ri3, t3, evap3, 'Versuch 3: IPA + Wasser + PG (60°C -> 80°C)', annotate_80c=True)

fig1.tight_layout()
fig1.subplots_adjust(top=0.93)


# --- Abbildung 2: Prozentuale Zusammensetzung ---
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 15))
fig2.suptitle('Prozentuale Zusammensetzung der Restmischung über die Zeit', fontsize=16, fontweight='bold')

plot_composition(axs2[0], t1_res, pct_ipa1, pct_water1, pct_pg1, 'Versuch 1: IPA + Wasser (Kompletter Verlauf)', transfer_time=70)
plot_composition(axs2[1], t2_res, pct_ipa2, pct_water2, pct_pg2, 'Versuch 2: IPA + Wasser + PG (Raumtemperatur)', transfer_time=160)
plot_composition(axs2[2], t3_res, pct_ipa3, pct_water3, pct_pg3, 'Versuch 3: IPA + Wasser + PG (60°C -> 80°C)', annotate_80c=True)

fig2.tight_layout()
fig2.subplots_adjust(top=0.93)


# Beide Fenster gleichzeitig anzeigen
plt.show()