"""
ink_simulation.py
=====================================================================
Simulation der Sensoreignung für die Al-Tinte.
Szenario A: Relative Sensitivität auf Aluminium-Schwankungen
Szenario B: Relative Sensitivität auf IPA-Verdampfung
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importiere deinen Calculator aus der lokalen Datei ink_calculator.py
from ink_calculator import InkCalculator

# =====================================================================
# 1. Konfiguration & Basiswerte (Der Idealzustand)
# =====================================================================
AL_BASE = 1.82
IPA_BASE = 3.64
PG_BASE = 3.64
TEMP_BASE = 25.0

# Initialisiere den Calculator (Pfade zum Ordner ggf. anpassen)
ink = InkCalculator(tables_dir="tables_parameters")

# Berechne die Basiswerte als Referenz (Dies entspricht 0% Signaländerung)
base_props = ink.compute(al=AL_BASE, ipa=IPA_BASE, pg=PG_BASE, temperature=TEMP_BASE)
BASE_RHO = base_props.density
BASE_ND = base_props.refractive_index
BASE_C = base_props.sound_velocity
BASE_ETA = base_props.viscosity

# Style für die Plots festlegen
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def run_simulation():
    print("Starte Simulation...")
    print(
        f"Basiswerte (Idealzustand): Dichte={BASE_RHO:.4f}, nD={BASE_ND:.4f}, Schall={BASE_C:.1f}, Viskosität={BASE_ETA:.3f}")

    plot_scenario_a_al_sensitivity()
    plot_scenario_b_ipa_evaporation()

    print("Simulation abgeschlossen. Diagramme werden angezeigt.")
    plt.show()


# =====================================================================
# 2. Szenario A: Relative Sensitivität (Nur Aluminium schwankt)
# =====================================================================
def plot_scenario_a_al_sensitivity():
    # Al-Gehalt variieren: von 1.5% bis 2.2%
    al_range = np.linspace(1.5, 2.2, 50)

    delta_rho = []
    delta_nd = []
    delta_c = []
    delta_eta = []

    for al in al_range:
        props = ink.compute(al=al, ipa=IPA_BASE, pg=PG_BASE, temperature=TEMP_BASE)

        # Prozentuale Abweichung vom Idealzustand
        delta_rho.append(((props.density / BASE_RHO) - 1.0) * 100.0)
        delta_nd.append(((props.refractive_index / BASE_ND) - 1.0) * 100.0)
        delta_c.append(((props.sound_velocity / BASE_C) - 1.0) * 100.0)
        delta_eta.append(((props.viscosity / BASE_ETA) - 1.0) * 100.0)

    plt.figure()
    plt.plot(al_range, delta_rho, label='Dichte', linewidth=2, color='tab:blue')
    plt.plot(al_range, delta_c, label='Schallgeschwindigkeit', linewidth=2, color='tab:orange')
    plt.plot(al_range, delta_eta, label='Viskosität', linewidth=2, color='tab:red')
    plt.plot(al_range, delta_nd, label='Brechungsindex (nD)', linewidth=2, linestyle='--', color='tab:green')

    plt.axvline(AL_BASE, color='gray', linestyle=':', label='Idealzustand (1.82%)')
    plt.axvspan(AL_BASE - 0.1, AL_BASE + 0.1, color='gray', alpha=0.2, label='Ziel-Toleranz (±0.1%)')

    plt.title('Szenario A: Signaländerung durch Aluminium-Schwankung', fontsize=14, pad=15)
    plt.xlabel('Aluminiumgehalt in der Tinte [%]', fontsize=12)
    plt.ylabel('Signaländerung relativ zum Idealzustand [%]', fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()


# =====================================================================
# 3. Szenario B: Relative Sensitivität bei IPA Verdampfung
# =====================================================================
def plot_scenario_b_ipa_evaporation():
    # IPA-Gehalt variieren: von 3.64% (Ideal) runter auf 0% (komplett verdampft)
    # Wir berechnen 50 Zwischenschritte
    ipa_range = np.linspace(IPA_BASE, 0.0, 50)

    delta_rho = []
    delta_nd = []
    delta_c = []
    delta_eta = []

    for ipa in ipa_range:
        # Al, PG und Temp bleiben konstant auf Idealwert, nur IPA sinkt.
        # Der Rest wird durch den Calculator automatisch als Wasser aufgefüllt.
        props = ink.compute(al=AL_BASE, ipa=ipa, pg=PG_BASE, temperature=TEMP_BASE)

        # Prozentuale Abweichung vom Idealzustand
        delta_rho.append(((props.density / BASE_RHO) - 1.0) * 100.0)
        delta_nd.append(((props.refractive_index / BASE_ND) - 1.0) * 100.0)
        delta_c.append(((props.sound_velocity / BASE_C) - 1.0) * 100.0)
        delta_eta.append(((props.viscosity / BASE_ETA) - 1.0) * 100.0)

    plt.figure()
    plt.plot(ipa_range, delta_rho, label='Dichte', linewidth=2, color='tab:blue')
    plt.plot(ipa_range, delta_c, label='Schallgeschwindigkeit', linewidth=2, color='tab:orange')
    plt.plot(ipa_range, delta_eta, label='Viskosität', linewidth=2, color='tab:red')
    plt.plot(ipa_range, delta_nd, label='Brechungsindex (nD)', linewidth=2, color='tab:green')

    # X-Achse invertieren, damit man von links (voll) nach rechts (leer) liest
    plt.xlim(IPA_BASE, 0.0)

    # Vertikale Markierungen für den Kontext
    plt.axvline(IPA_BASE, color='gray', linestyle=':', label='Start / Idealzustand (3.64% IPA)')
    plt.axvline(0.0, color='black', linestyle=':', label='Vollständig verdampft (0% IPA)')

    plt.title('Szenario B: Signaländerung durch IPA-Verdampfung', fontsize=14, pad=15)
    plt.xlabel('Verbleibender IPA-Gehalt in der Tinte [%] ➔ (Verdampfung)', fontsize=12)
    plt.ylabel('Signaländerung relativ zum Idealzustand [%]', fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()


if __name__ == "__main__":
    run_simulation()