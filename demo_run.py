# -*- coding: utf-8 -*-
"""Simulator (Ground Truth) + UKF-Demo mit synthetischen, verrauschten Anlagendaten."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ukf_aluminium import (ORDER, IDX, N_COMP, W_RECIPE, M_OFFSET,
                           process_step, measurement, masses_to_w, UKF)

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Szenario
# ----------------------------------------------------------------------------
dt      = 5.0                      # Abtastzeit [s]
T_end   = 8*3600                   # 8 Stunden
steps   = int(T_end/dt)
t       = np.arange(steps)*dt

M_INK0  = 100.0                    # Anfangs-Tintenmasse im Kreislauf [kg]
m0      = W_RECIPE * M_INK0        # Anfangs-Komponentenmassen
kp_true = 8.0                      # wahres k_p [mg/Umdrehung]  (Startmotiv)

# Eingangsprofile ----------------------------------------------------------
n_donor = np.full(steps, 3.0)      # Donor-Drehzahl [1/s] (Grundlast)
n_donor += 0.15*np.sin(2*np.pi*t/1800)          # langsame Schwankung
# Druckpausen (Anlage steht zwischendurch)
for a,b in [(1.2,1.5),(4.0,4.3)]:
    n_donor[int(a*3600/dt):int(b*3600/dt)] = 0.0

# Motivwechsel: k_p springt bei t = 5 h  (anderes Klebermuster)
kp_profile = np.full(steps, kp_true)
kp_profile[int(5*3600/dt):] = 1.35*kp_true

# Temperatur der Tinte am Sensor [degC]: leichte Drift + Rauschen
T_true = 25.0 + 0.8*np.sin(2*np.pi*t/(3*3600)) + 0.05*rng.standard_normal(steps).cumsum()*0.0
T_true = 25.0 + 0.8*np.sin(2*np.pi*t/(3*3600))

# Dosierereignisse: SL120 (Konzentrat) und Wasser, als kurze Pulse ----------
mdot_SL = np.zeros(steps)
mdot_W  = np.zeros(steps)
# SL120: alle 90 min ein 60s-Puls von 0.02 kg/s  -> haelt nv-Fraktion nach
for k in range(steps):
    tt = t[k]
    if (tt % (90*60)) < 60:
        mdot_SL[k] = 0.020
# Wasser: alle 30 min ein 60s-Puls, gleicht Verdampfung aus
for k in range(steps):
    tt = t[k]
    if (tt % (30*60)) < 60:
        mdot_W[k] = 0.060

def make_u(k):
    return dict(n_Donor=n_donor[k], mdot_SL=mdot_SL[k], mdot_W=mdot_W[k],
                P_Heiz=0.0, n_Vent=0.0, dz=0.0, T=T_true[k])

# ----------------------------------------------------------------------------
# Ground-Truth-Simulation (wahre Trajektorie)
# ----------------------------------------------------------------------------
X_true = np.zeros((steps, N_COMP+1))
x = np.concatenate([m0, [kp_profile[0]]])
for k in range(steps):
    x[N_COMP] = kp_profile[k]                       # wahres k_p vorgeben
    X_true[k] = x
    x = process_step(x, make_u(k), dt)

w_al_true = X_true[:, IDX["Al"]] / X_true[:, :N_COMP].sum(axis=1)
w_ipa_true = X_true[:, IDX["IPA"]] / X_true[:, :N_COMP].sum(axis=1)

# ----------------------------------------------------------------------------
# Synthetische Messungen mit realistischem Rauschen
# ----------------------------------------------------------------------------
# Rauschpegel (1-sigma):
SIG_WAAGE = 0.15      # kg  -> stark verrauscht (Zirkulation, Ruehrwerk, Pumpen)
SIG_RHO   = 0.00006   # g/cm^3 (Aufloesung 0.00005 + etwas mehr)
SIG_C     = 0.02      # m/s

# Messkanaele: Waage jede Sekunde-Ebene (hier jeder Schritt), rho/c alle 30 s
meas = np.full((steps, 3), np.nan)
mask_hist = np.zeros((steps,3), dtype=bool)
for k in range(steps):
    y = measurement(X_true[k], make_u(k))
    # Waage: immer, aber stark verrauscht + leicht korreliert (AR(1))
    meas[k,0] = y[0] + SIG_WAAGE*rng.standard_normal()
    mask_hist[k,0] = True
    if k % int(30/dt) == 0:                          # rho & c alle 30 s
        meas[k,1] = y[1] + SIG_RHO*rng.standard_normal()
        meas[k,2] = y[2] + SIG_C*rng.standard_normal()
        mask_hist[k,1] = mask_hist[k,2] = True

# ----------------------------------------------------------------------------
# UKF-Initialisierung
# ----------------------------------------------------------------------------
# Startschaetzung: leicht daneben (5% Fehler auf Massen, k_p grob geraten)
x0 = np.concatenate([m0*np.array([1.03,0.97,1.05,1.0,0.9]), [5.0]])
P0 = np.diag(np.concatenate([(0.05*m0)**2, [ 5.0**2 ]]))

# Prozessrauschen Q: klein auf Massen (Modell recht gut), groesser auf k_p (Random Walk)
q_m = (np.array([2e-4,2e-4,2e-4,5e-4,5e-5]))**2      # pro Schritt
q_kp = (0.02)**2                                    # [mg/rev]^2 pro Schritt: folgt Motivwechsel
Q = np.diag(np.concatenate([q_m, [q_kp]]))

# Messrauschen R
R = np.diag([SIG_WAAGE**2, SIG_RHO**2, SIG_C**2])

ukf = UKF(x0, P0, Q, R, alpha=0.5, beta=2.0, kappa=0.0)

# ----------------------------------------------------------------------------
# Filterlauf
# ----------------------------------------------------------------------------
est_wal   = np.zeros(steps)
est_wal_s = np.zeros(steps)
est_wipa  = np.zeros(steps)
est_kp    = np.zeros(steps)
for k in range(steps):
    u = make_u(k)
    ukf.predict(u, dt)
    m = mask_hist[k]
    if m.any():
        ukf.update(meas[k], u, mask=m)
    est_wal[k]   = ukf.w_al()
    est_wal_s[k] = ukf.w_al_std()
    est_wipa[k]  = ukf.x[IDX["IPA"]]/ukf.x[:N_COMP].sum()
    est_kp[k]    = ukf.x[N_COMP]

# ----------------------------------------------------------------------------
# Fehlermetriken
# ----------------------------------------------------------------------------
# nach Einschwingen (ab 30 min)
i0 = int(30*60/dt)
err = (est_wal - w_al_true)[i0:]
rmse = np.sqrt(np.mean(err**2))
mae  = np.mean(np.abs(err))
inband = np.mean(np.abs(err) < 0.001)*100     # Anteil im +-0.1%-Pkt Band
print(f"RMSE(w_Al)      = {rmse*100:.4f} %-Pkt")
print(f"MAE (w_Al)      = {mae*100:.4f} %-Pkt")
print(f"Anteil |Fehler|<0.1%-Pkt = {inband:.1f} %")
print(f"k_p wahr (Start/Ende) = {kp_true:.2f} / {1.35*kp_true:.2f} mg/rev")
print(f"k_p Schaetzung Ende    = {est_kp[-1]:.2f} mg/rev")

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
th = t/3600.0
fig, ax = plt.subplots(3,1, figsize=(11,11), sharex=True)

# (1) w_Al
ax[0].axhspan(1.82-0.1, 1.82+0.1, color="#7A5EA8", alpha=0.12, label="Zielband ±0,1 %-Pkt")
ax[0].plot(th, w_al_true*100, color="#222", lw=1.6, label="wahr")
ax[0].plot(th, est_wal*100, color="#D85A30", lw=1.2, label="UKF-Schätzung")
ax[0].fill_between(th, (est_wal-2*est_wal_s)*100, (est_wal+2*est_wal_s)*100,
                   color="#D85A30", alpha=0.20, label="±2σ")
ax[0].axvline(5.0, color="#3B7BB8", ls="--", lw=1, alpha=0.7)
ax[0].text(5.05, ax[0].get_ylim()[1], " Motivwechsel", color="#3B7BB8", va="top", fontsize=9)
ax[0].set_ylabel("w_Al  [%]")
ax[0].set_title("Aluminiumkonzentration: UKF-Schätzung vs. Wahrheit")
ax[0].legend(loc="upper right", fontsize=9); ax[0].grid(alpha=0.3)

# (2) IPA-Anteil (Depletion)
ax[1].plot(th, w_ipa_true*100, color="#222", lw=1.6, label="wahr")
ax[1].plot(th, est_wipa*100, color="#3B7BB8", lw=1.2, label="UKF-Schätzung")
ax[1].set_ylabel("w_IPA  [%]")
ax[1].set_title("IPA-Anteil (Verdampfung / Depletion)")
ax[1].legend(loc="upper right", fontsize=9); ax[1].grid(alpha=0.3)

# (3) k_p online geschaetzt
ax[2].plot(th, kp_profile, color="#222", lw=1.6, label="wahr")
ax[2].plot(th, est_kp, color="#2E8B57", lw=1.2, label="UKF-Schätzung")
ax[2].axvline(5.0, color="#3B7BB8", ls="--", lw=1, alpha=0.7)
ax[2].set_ylabel("k_p  [mg/Umdr.]")
ax[2].set_xlabel("Zeit  [h]")
ax[2].set_title("Online-Schätzung des Austrags pro Umdrehung k_p (unbekanntes Druckmotiv)")
ax[2].legend(loc="upper right", fontsize=9); ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ukf_demo.png", dpi=130, bbox_inches="tight")
print("Plot gespeichert: ukf_demo.png")
