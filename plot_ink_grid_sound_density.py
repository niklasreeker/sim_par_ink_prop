"""
ink_sound_and_plot.py
=====================================================================
Berechnet Schallgeschwindigkeit UND Dichte von Tinten aus
Aluminium-Pigment + (IPA / PG) + Wasser und stellt das Ergebnis als
Netzdiagramm (Dichte vs. Schallgeschwindigkeit) dar.

Physik
-------
Schallgeschwindigkeit laesst sich NICHT linear mischen. Gemischt
werden Dichte (rho) und adiabatische Kompressibilitaet (beta);
zurueckgerechnet wird ueber Newton-Laplace:

        c = 1 / sqrt(rho * beta)          beta = 1 / (rho * c^2)

Vorgehen je Arbeitspunkt:
  1) Fluessigmatrix (Wasser + Loesemittel): gemessene Binaerdaten.
       - rho_L : aus InkDensityCalculator (Volumenkontraktion enthalten)
       - c_L   : aus *_sound.csv (interpoliert)
       - beta_L = 1 / (rho_L * c_L^2)
  2) Aluminium ueber die Wood/Urick-Gleichung einrechnen (Volumenanteile):
       - rho_mix = sum(phi_i * rho_i)
       - beta_mix = sum(phi_i * beta_i)
       - beta_Al  = 1 / K_Al   (Kompressionsmodul, NICHT 6320 m/s!)
       - c_mix    = 1 / sqrt(rho_mix * beta_mix)

Die Dichte-Logik wird 1:1 aus deinem ink_density.py uebernommen,
damit beide Skripte konsistent bleiben (rho_mix == calculate_density).
=====================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # entfernen, falls interaktiv gewuenscht
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d

from ink_density import InkDensityCalculator


# ---------------------------------------------------------------------
#  Schallgeschwindigkeits-Rechner
# ---------------------------------------------------------------------
class InkSoundVelocityCalculator:
    """
    Berechnet die Schallgeschwindigkeit der Tinte nach Wood/Urick.
    Nutzt einen InkDensityCalculator fuer die (realen) Fluessigdichten.
    """

    # Aluminium-Eigenschaften
    DENSITY_ALUMINUM = 2.700          # g/cm³
    BULK_MODULUS_ALUMINUM = 76.0e9    # Pa  (Kompressionsmodul K_Al)

    def __init__(self, density_calculator, tables_dir="tables_parameters"):
        self.tables_dir = tables_dir
        self.density_calc = density_calculator
        # sound_tables[solvent] = {temp_float: interp1d(mass_percent -> c)}
        self.sound_tables = {'IPA': {}, 'PG': {}}
        self._load_tables()

    # -- CSV laden -----------------------------------------------------
    def _load_one(self, path, key):
        if not os.path.exists(path):
            print(f"Warning: '{path}' not found. {key} sound calc will fail.")
            return
        df = pd.read_csv(path)
        mp = df['Mass_Percent'].to_numpy(dtype=float)
        order = np.argsort(mp)           # Stuetzstellen sortieren (unregelmaessig)
        mp = mp[order]
        kind = 'cubic' if len(mp) >= 4 else 'linear'
        temps = {}
        for col in df.columns:
            m = re.match(r'SoundVelocity_(\d+(?:\.\d+)?)C', col)
            if not m:
                continue
            t = float(m.group(1))
            y = df[col].to_numpy(dtype=float)[order]
            # bounds_error=False + extrapolate: kein Crash bei kleinen
            # Randueberschreitungen; Werte sollten aber im Datenbereich liegen.
            temps[t] = interp1d(mp, y, kind=kind,
                                bounds_error=False, fill_value="extrapolate")
        self.sound_tables[key] = temps

    def _load_tables(self):
        self._load_one(os.path.join(self.tables_dir, "ipa_sound.csv"), 'IPA')
        self._load_one(os.path.join(self.tables_dir, "pg_sound.csv"), 'PG')

    # -- c der binaeren Fluessigkeit (Loesemittel + Wasser) ------------
    def get_liquid_sound_velocity(self, solvent_type, mass_percent, target_temp=25):
        temps = self.sound_tables[solvent_type]
        if not temps:
            raise ValueError(f"Keine Schalldaten fuer {solvent_type} geladen.")
        if target_temp in temps:
            return float(temps[target_temp](mass_percent))
        # ueber Temperatur interpolieren (beliebige vorhandene Temperaturen)
        avail = sorted(temps.keys())
        c_vals = np.array([float(temps[t](mass_percent)) for t in avail])
        if len(avail) == 1:
            return float(c_vals[0])
        t_interp = interp1d(avail, c_vals, kind='linear',
                            bounds_error=False, fill_value="extrapolate")
        return float(t_interp(target_temp))

    # -- eine binaere Fluessigphase als (Volumen, rho_SI, beta_SI) -----
    def _binary_phase(self, solvent, pct_solvent, pct_water, target_temp):
        mass_liquid = pct_solvent + pct_water           # g (100-g-Basis)
        if mass_liquid <= 0:
            return None
        pct_in_liquid = (pct_solvent / mass_liquid) * 100.0
        rho_L = self.density_calc.get_liquid_density(solvent, pct_in_liquid, target_temp)  # g/cm³
        c_L = self.get_liquid_sound_velocity(solvent, pct_in_liquid, target_temp)          # m/s
        rho_si = rho_L * 1000.0                          # kg/m³
        beta = 1.0 / (rho_si * c_L**2)                   # Pa⁻¹
        vol = mass_liquid / rho_L                        # cm³ (100-g-Basis)
        return (vol, rho_si, beta)

    # -- Hauptmethode --------------------------------------------------
    def calculate(self, pct_al, pct_ipa=0.0, pct_pg=0.0, target_temp=25):
        """
        Gibt (c_mix [m/s], rho_mix [g/cm³]) der Tinte zurueck.
        Struktur analog zu InkDensityCalculator.calculate_density().
        """
        pct_water = 100.0 - pct_al - pct_ipa - pct_pg
        if pct_water < 0:
            raise ValueError("Massenanteile > 100 %. Eingaben pruefen.")

        beta_al = 1.0 / self.BULK_MODULUS_ALUMINUM
        rho_al_si = self.DENSITY_ALUMINUM * 1000.0

        # Liste von Phasen: (Volumen cm³, rho_SI kg/m³, beta_SI Pa⁻¹)
        phases = [(pct_al / self.DENSITY_ALUMINUM, rho_al_si, beta_al)]  # Aluminium

        if pct_ipa > 0 and pct_pg == 0:                       # nur IPA
            phases.append(self._binary_phase('IPA', pct_ipa, pct_water, target_temp))

        elif pct_pg > 0 and pct_ipa == 0:                     # nur PG
            phases.append(self._binary_phase('PG', pct_pg, pct_water, target_temp))

        elif pct_ipa > 0 and pct_pg > 0:                      # IPA + PG (Pseudo-binaer)
            total = pct_ipa + pct_pg
            water_ipa = pct_water * (pct_ipa / total)
            water_pg = pct_water * (pct_pg / total)
            phases.append(self._binary_phase('IPA', pct_ipa, water_ipa, target_temp))
            phases.append(self._binary_phase('PG', pct_pg, water_pg, target_temp))

        else:                                                 # nur Wasser + Al
            phases.append(self._binary_phase('IPA', 0.0, pct_water, target_temp))

        phases = [p for p in phases if p is not None]

        # Wood/Urick-Mischung ueber Volumenanteile
        V = sum(v for v, _, _ in phases)
        rho_mix_si = sum((v / V) * rho for v, rho, _ in phases)
        beta_mix = sum((v / V) * beta for v, _, beta in phases)
        c_mix = 1.0 / np.sqrt(rho_mix_si * beta_mix)

        return c_mix, rho_mix_si / 1000.0  # c in m/s, rho in g/cm³


# ---------------------------------------------------------------------
#  Auswertung
# ---------------------------------------------------------------------
def main():
    TEMP = 25.0
    dens = InkDensityCalculator(tables_dir="tables_parameters")
    snd = InkSoundVelocityCalculator(dens, tables_dir="tables_parameters")

    # --- Arbeitspunkte (pct_al, pct_ipa) ---
    # Hinweis: In deiner Anfrage stand "2,2% Al, 22% IPA" zweimal; der
    # vierte Punkt ist hier als 2,2% Al / 10% IPA angenommen (Tippfehler?).
    work_points = [
        (0.5, 22.0),
        (2.2, 22.0),
        (0.5, 10.0),
        (2.2, 10.0),
    ]

    print(f"=== Arbeitspunkte (Al + IPA + Wasser) bei {TEMP:.0f} °C ===")
    print(f"{'Al [%]':>7} {'IPA [%]':>8} {'rho [g/cm³]':>12} {'c [m/s]':>10}")
    wp_results = []
    for al, ipa in work_points:
        c, rho = snd.calculate(pct_al=al, pct_ipa=ipa, target_temp=TEMP)
        wp_results.append((al, ipa, rho, c))
        print(f"{al:7.2f} {ipa:8.1f} {rho:12.4f} {c:10.2f}")

    # --- Gitter fuer das Netzdiagramm (Arbeitspunkte + Punkte dazwischen) ---
    al_grid = np.linspace(0.5, 2.2, 6)     # Aluminium-Achse
    ipa_grid = np.linspace(10.0, 22.0, 7)  # IPA-Achse

    RHO = np.zeros((len(al_grid), len(ipa_grid)))
    C = np.zeros_like(RHO)
    for i, al in enumerate(al_grid):
        for j, ipa in enumerate(ipa_grid):
            c, rho = snd.calculate(pct_al=al, pct_ipa=ipa, target_temp=TEMP)
            RHO[i, j] = rho
            C[i, j] = c

    # --- Netzdiagramm: x = Dichte, y = Schallgeschwindigkeit ---
    fig, ax = plt.subplots(figsize=(9, 6.5))

    al_colors = cm.viridis(np.linspace(0.15, 0.9, len(al_grid)))
    ipa_colors = cm.autumn(np.linspace(0.0, 0.7, len(ipa_grid)))

    # Linien konstanten Al-Gehalts (IPA variiert)
    for i in range(len(al_grid)):
        ax.plot(RHO[i, :], C[i, :], '-', color=al_colors[i], lw=1.6, zorder=2)
    # Linien konstanten IPA-Gehalts (Al variiert)
    for j in range(len(ipa_grid)):
        ax.plot(RHO[:, j], C[:, j], '-', color='0.55', lw=0.9, zorder=1)

    # Gitterknoten
    ax.scatter(RHO.ravel(), C.ravel(), s=14, color='0.35', zorder=3)

    # Arbeitspunkte hervorheben
    for al, ipa, rho, c in wp_results:
        ax.scatter([rho], [c], s=110, facecolor='crimson',
                   edgecolor='black', lw=1.2, zorder=5)
        ax.annotate(f"{al:.1f}% Al / {ipa:.0f}% IPA",
                    (rho, c), textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5, fontweight='bold')

    # Beschriftung der Netz-Raender (Richtungssinn)
    ax.annotate("steigender Al-Gehalt", (RHO[-1, 0], C[-1, 0]),
                textcoords="offset points", xytext=(-10, -16),
                fontsize=8, color='0.3', ha='right')
    ax.annotate("steigender IPA-Gehalt", (RHO[0, -1], C[0, -1]),
                textcoords="offset points", xytext=(6, -2),
                fontsize=8, color=al_colors[0])

    ax.set_xlabel("Dichte  $\\rho$  [g/cm³]")
    ax.set_ylabel("Schallgeschwindigkeit  $c$  [m/s]")
    ax.set_title(f"Tinte (Al + IPA + Wasser) bei {TEMP:.0f} °C — Netzdiagramm")
    ax.grid(True, ls=':', alpha=0.4)

    # Legende
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=al_colors[len(al_colors)//2], lw=1.6,
               label='konst. Al-Linie (IPA variiert)'),
        Line2D([0], [0], color='0.55', lw=0.9,
               label='konst. IPA-Linie (Al variiert)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
               markeredgecolor='black', markersize=9, label='Arbeitspunkt'),
    ]
    ax.legend(handles=legend_elems, loc='best', fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    out = "ink_sound_density_net.png"
    fig.savefig(out, dpi=160)
    print(f"\nNetzdiagramm gespeichert: {out}")


if __name__ == "__main__":
    main()