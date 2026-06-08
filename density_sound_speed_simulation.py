#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erweitertes Modell fuer c(Al, IPA) ternaerer Tinten (Al / 2-Propanol / Wasser), 25 C.

Idee (gegenueber dem reinen Wood-Modell):
  Jedes Pigmentteilchen bindet eine Fluessigkeits-INTERPHASE (skaliert mit der
  Pigmentmasse ~ spez. Oberflaeche). Diese Interphase
    (a) entzieht der mobilen BULK-Phase bevorzugt Wasser
        -> der Bulk wird IPA-reicher und verschiebt sich entlang der
           NICHTLINEAREN Traegerkurve c0(w) (mit Maximum!).
           Da 10 % und 22 % auf verschiedenen Aesten des Maximums liegen,
           erzeugt dieselbe Verschiebung GEGENLAEUFIGE Aenderungen von c.
    (b) ist selbst eine eigene, steifere/dichtere Phase (gebundenes Wasser,
        Elektrostriktion) und geht so in die Wood-Mischung ein.

Drei-Phasen-Wood: Al + Interphase + Bulk.
Gefittet werden 3 Parameter an die 4 Messwerte:
    beta_W  : gebundene Wassermasse pro g Al   [g/g]   (Mechanismus a)
    f_kappa : Kompressibilitaet der Interphase / reines Wasser  (<1, Mechanismus b)
    eps_rho : Dichte-Erhoehung der Interphase   (Elektrostriktion)

WICHTIG: Die Traegerkurve c0(w) ist die dominierende ANNAHME. Sie sollte durch
eine eigene feine c(IPA)-Messung bei 0,5 % Al ersetzt werden (Experiment 5).
"""

import numpy as np
from scipy.optimize import least_squares

# ====================================================================
# Reinstoffe (25 C)
# ====================================================================
RHO_W, C_W = 997.05, 1496.7
RHO_AL     = 2700.0
KAPPA_AL   = 1.0 / 76.0e9
KAPPA_WPUR = 1.0 / (RHO_W * C_W**2)        # Kompressibilitaet reines Wasser

def kappa(rho, c):
    return 1.0 / (rho * c * c)

# ====================================================================
# Traegerkurve c0(w), rho0(w) der MOBILEN Bulk-Phase (0 % Al)
# w = IPA-Massenanteil im Bulk. Maximum bei ~15 % (vom Anwender beobachtet:
# Maximum zwischen 10 % und 22 %). >>> Hier eigene Messwerte eintragen! <<<
# ====================================================================
_W   = np.array([0.00, 0.05, 0.08, 0.10, 0.13, 0.16, 0.19, 0.22, 0.30, 0.50, 1.00])
_C0  = np.array([1496.7,1545.,1568.,1582.,1612.,1636.,1632.,1618.,1598.,1505.,1140.])
_RHO0= np.array([997.05,989.,984.,981.5,977.,972.,967.,961.,949.,905.,781.3])

def carrier(w):
    c   = np.interp(w, _W, _C0)
    rho = np.interp(w, _W, _RHO0)
    return rho, c

# ====================================================================
# Erweitertes 3-Phasen-Modell
# ====================================================================
def model_c(al_pct, ipa_pct, beta_W, f_kappa, eps_rho):
    """Liefert (rho_eff [kg/m3], c [m/s]) fuer eine Rezeptur in Massen-%."""
    m_al  = float(al_pct)
    m_ipa = float(ipa_pct)
    m_w   = 100.0 - m_al - m_ipa

    # (a) gebundenes Wasser -> Bulk wird IPA-reicher
    dm_w = beta_W * m_al
    dm_w = min(dm_w, 0.95 * m_w)               # nicht mehr binden als vorhanden
    m_w_bulk  = m_w - dm_w
    m_ipa_bulk = m_ipa                          # IPA bleibt (diagnostiziert)
    w_bulk = m_ipa_bulk / (m_ipa_bulk + m_w_bulk)
    rho_bulk, c_bulk = carrier(w_bulk)          # entlang nichtlinearer Kurve!
    kap_bulk = kappa(rho_bulk, c_bulk)

    # (b) Interphase (gebundenes, dichteres/steiferes Wasser)
    rho_int = RHO_W * (1.0 + eps_rho)
    kap_int = f_kappa * KAPPA_WPUR
    m_int   = dm_w

    # Volumina (in cm3 fuer 100 g Ansatz; Dichten in g/cm3)
    V_al   = m_al        / (RHO_AL  / 1000.0)
    V_int  = m_int       / (rho_int / 1000.0)
    V_bulk = (m_ipa_bulk + m_w_bulk) / (rho_bulk / 1000.0)
    V_tot  = V_al + V_int + V_bulk
    phi_al, phi_int, phi_bulk = V_al/V_tot, V_int/V_tot, V_bulk/V_tot

    # Effektive Dichte (SI) und Wood-Kompressibilitaet
    rho_eff = 100.0 / V_tot * 1000.0            # g/cm3 -> kg/m3
    kap_eff = phi_al*KAPPA_AL + phi_int*kap_int + phi_bulk*kap_bulk
    c_eff   = 1.0 / np.sqrt(rho_eff * kap_eff)
    return rho_eff, c_eff

# ====================================================================
# Messwerte (Al%, IPA%) -> c_gemessen
# ====================================================================
data = [
    ("AP1", 0.5, 22.0, 1616.0),
    ("AP2", 2.2, 22.0, 1595.0),
    ("AP3", 0.5, 10.0, 1580.0),
    ("AP4", 2.2, 10.0, 1608.0),
]

def residuals(p):
    bW, fk, er = p
    return [model_c(al, ipa, bW, fk, er)[1] - c_meas
            for _, al, ipa, c_meas in data]

# ====================================================================
# Fit
# ====================================================================
p0     = [1.0, 0.70, 0.02]
bounds = ([0.0, 0.20, 0.0], [10.0, 1.00, 0.15])
sol = least_squares(residuals, p0, bounds=bounds)
bW, fk, er = sol.x

print("="*64)
print("GEFITTETE PARAMETER")
print(f"  beta_W  (gebundenes Wasser pro g Al) = {bW:6.3f} g/g")
print(f"  f_kappa (Interphasen-Kompressibilitaet/Wasser) = {fk:6.3f}")
print(f"  eps_rho (Dichte-Erhoehung Interphase) = {er*100:5.2f} %")
print("="*64)
print(f"{'AP':4} {'Al%':>5} {'IPA%':>6} {'c_mess':>8} {'c_modell':>9} "
      f"{'Diff':>6} {'w_bulk':>8} {'rho':>7}")
for name, al, ipa, cm in data:
    rho, cmod = model_c(al, ipa, bW, fk, er)
    # Bulk-Verschiebung zur Anzeige
    m_w = 100-al-ipa; dmw = min(bW*al, 0.95*m_w)
    wb = ipa/(ipa + (m_w-dmw))
    print(f"{name:4} {al:5.1f} {ipa:6.1f} {cm:8.1f} {cmod:9.1f} "
          f"{cmod-cm:+6.1f} {wb*100:7.2f}% {rho:7.1f}")
rms = np.sqrt(np.mean(np.array(residuals(sol.x))**2))
print(f"\nRMS-Abweichung Modell vs. Messung: {rms:.2f} m/s")

# Plausibilitaetscheck: implizierte gebundene Wasserschicht
a_s = 8.0  # angenommene spez. Oberflaeche Al-Flake [m2/g]
k_areal = bW / a_s                # g Wasser pro m2
shell_nm = k_areal / 1.0 * 1e6 / 1e9 * 1e9  # grobe Dicke [nm] (rho_w~1 g/cm3)
shell_nm = (k_areal*1e-3) / (RHO_W) * 1e9   # m -> nm
print(f"\nPlausibilitaet (Annahme a_s = {a_s:.0f} m2/g):")
print(f"  arealer Wasseranteil  = {k_areal:.3f} g/m2")
print(f"  entspr. Schichtdicke  ~ {shell_nm:.0f} nm "
      f"(>> Monolage ~0.3 nm -> 'effektive' Verschiebung, nicht wortwoertlich)")

# ====================================================================
# Netzdiagramm aus dem erweiterten Modell (zeigt die Verdrehung)
# ====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.colormaps["viridis"]

ipa_lines = np.array([10, 12, 14, 16, 18, 20, 22])
al_lines  = np.array([0.5, 0.9, 1.35, 1.8, 2.2])
al_fine  = np.linspace(0.5, 2.2, 40)
ipa_fine = np.linspace(10, 22, 40)

for k, ipa in enumerate(ipa_lines):
    pts = [model_c(a, ipa, bW, fk, er) for a in al_fine]
    rho = [p[0] for p in pts]; c = [p[1] for p in pts]
    col = cmap(k/(len(ipa_lines)-1))
    ax.plot(rho, c, "-", color=col, lw=2, zorder=2)
    ax.annotate(f"{ipa:.0f}% IPA", (rho[0], c[0]),
                textcoords="offset points", xytext=(-6,6), ha="right",
                fontsize=8, color=col, fontweight="bold")

for al in al_lines:
    pts = [model_c(al, ipa, bW, fk, er) for ipa in ipa_fine]
    rho = [p[0] for p in pts]; c = [p[1] for p in pts]
    ax.plot(rho, c, "--", color="0.45", lw=1.0, zorder=1)
    ax.annotate(f"{al:.1f}% Al", (rho[-1], c[-1]),
                textcoords="offset points", xytext=(8,0), ha="left",
                va="center", fontsize=8, color="0.35")

# Messwerte (Modell-rho, gemessenes c)
for name, al, ipa, cm in data:
    rho, _ = model_c(al, ipa, bW, fk, er)
    ax.plot(rho, cm, "o", ms=11, mfc="#d6336c", mec="white", mew=1.5, zorder=5)
    ax.annotate(f"{name}: {cm:.0f} m/s", (rho, cm),
                textcoords="offset points", xytext=(8,-14), fontsize=8,
                color="#d6336c")

ax.set_xlabel("Dichte  $\\rho$  [kg/m$^3$]", fontsize=12)
ax.set_ylabel("Schallgeschwindigkeit  $c$  [m/s]", fontsize=12)
ax.set_title("Erweitertes Interphasen-Modell (25 C) - verdrehtes Netz\n"
             "Punkte = Messwerte, Linien = Modell", fontsize=12)
ax.grid(True, ls=":", alpha=0.5)
ax.margins(0.12)
plt.tight_layout()
plt.savefig("tinte_interphasen_netz.png", dpi=150)
print("\nGespeichert: tinte_interphasen_netz.png")