# -*- coding: utf-8 -*-
"""Simulator (Ground Truth) + UKF-Demo. Messmodell = validierter ink_calculator.
   Zustand ohne Additiv: x = [m_Al, m_PG, m_IPA, m_W, k_p]."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ukf_aluminium import (IDX, NC, W_RECIPE, process_step, measurement, UKF, T_MIN, T_MAX)

rng = np.random.default_rng(7)

# ---- Szenario --------------------------------------------------------------
dt    = 5.0
T_end = 8*3600
steps = int(T_end/dt)
t     = np.arange(steps)*dt

M_INK0  = 100.0
m0      = W_RECIPE * M_INK0
kp_true = 8.0                               # [mg/Umdrehung]  Startmotiv

n_donor = np.full(steps, 3.0) + 0.15*np.sin(2*np.pi*t/1800)
for a,b in [(1.2,1.5),(4.0,4.3)]:           # Druckpausen
    n_donor[int(a*3600/dt):int(b*3600/dt)] = 0.0

kp_profile = np.full(steps, kp_true)
kp_profile[int(5*3600/dt):] = 1.35*kp_true  # Motivwechsel bei 5 h

# Tintentemperatur am Sensor: im gueltigen Tabellenbereich (24..26 degC)
T_true = 25.0 + 0.8*np.sin(2*np.pi*t/(3*3600))

mdot_SL = np.where((t % (90*60)) < 60, 0.020, 0.0)   # SL120-Pulse alle 90 min
mdot_W  = np.where((t % (30*60)) < 60, 0.060, 0.0)   # Wasser-Pulse alle 30 min

def make_u(k):
    return dict(n_Donor=n_donor[k], mdot_SL=mdot_SL[k], mdot_W=mdot_W[k], T=T_true[k])

# ---- Ground Truth ----------------------------------------------------------
X_true = np.zeros((steps, NC+1))
x = np.concatenate([m0, [kp_profile[0]]])
for k in range(steps):
    x[NC] = kp_profile[k]
    X_true[k] = x
    x = process_step(x, make_u(k), dt)
w_al_true  = X_true[:, IDX["Al"]]  / X_true[:, :NC].sum(axis=1)
w_ipa_true = X_true[:, IDX["IPA"]] / X_true[:, :NC].sum(axis=1)

# ---- Synthetische Messungen (reales h(x) + Rauschen) -----------------------
SIG_WAAGE = 0.15      # kg  (stark verrauscht: Zirkulation, Ruehrwerk, Pumpen)
SIG_RHO   = 0.00006   # g/cm^3
SIG_C     = 0.02      # m/s

meas = np.full((steps,3), np.nan); avail = np.zeros((steps,3), bool)
for k in range(steps):
    y = measurement(X_true[k], make_u(k))
    meas[k,0] = y[0] + SIG_WAAGE*rng.standard_normal(); avail[k,0] = True
    if k % int(30/dt) == 0:                  # rho & c alle 30 s
        meas[k,1] = y[1] + SIG_RHO*rng.standard_normal()
        meas[k,2] = y[2] + SIG_C*rng.standard_normal()
        avail[k,1] = avail[k,2] = True

# ---- UKF -------------------------------------------------------------------
x0 = np.concatenate([m0*np.array([1.03,0.97,1.05,1.0]), [5.0]])   # bewusst falsch
P0 = np.diag(np.concatenate([(0.05*m0)**2, [5.0**2]]))
q_m  = (np.array([2e-4,2e-4,2e-4,5e-4]))**2
Q = np.diag(np.concatenate([q_m, [0.02**2]]))
R = np.diag([SIG_WAAGE**2, SIG_RHO**2, SIG_C**2])
ukf = UKF(x0, P0, Q, R, alpha=0.5, beta=2.0, kappa=0.0)

est_wal=np.zeros(steps); est_wal_s=np.zeros(steps); est_wipa=np.zeros(steps); est_kp=np.zeros(steps)
for k in range(steps):
    u = make_u(k)
    ukf.predict(u, dt)
    if avail[k].any():
        ukf.update(meas[k], u, mask=avail[k])
    est_wal[k]=ukf.w_al(); est_wal_s[k]=ukf.w_al_std()
    est_wipa[k]=ukf.x[IDX["IPA"]]/ukf.x[:NC].sum(); est_kp[k]=ukf.x[NC]

# ---- Metriken --------------------------------------------------------------
i0 = int(30*60/dt)
err = (est_wal - w_al_true)[i0:]
print(f"RMSE(w_Al)   = {np.sqrt(np.mean(err**2))*100:.4f} %-Pkt")
print(f"MAE (w_Al)   = {np.mean(np.abs(err))*100:.4f} %-Pkt")
print(f"im ±0,1-Band = {np.mean(np.abs(err)<0.001)*100:.1f} %")
print(f"k_p Ende: wahr {1.35*kp_true:.2f} / geschätzt {est_kp[-1]:.2f} mg/Umdr.")

# ---- Plots -----------------------------------------------------------------
th = t/3600.0
fig, ax = plt.subplots(3,1, figsize=(11,11), sharex=True)
ax[0].axhspan(1.82-0.1,1.82+0.1, color="#7A5EA8", alpha=0.12, label="Zielband ±0,1 %-Pkt")
ax[0].plot(th, w_al_true*100, color="#222", lw=1.6, label="wahr")
ax[0].plot(th, est_wal*100, color="#D85A30", lw=1.2, label="UKF-Schätzung")
ax[0].fill_between(th, (est_wal-2*est_wal_s)*100, (est_wal+2*est_wal_s)*100, color="#D85A30", alpha=0.20, label="±2σ")
ax[0].axvline(5.0, color="#3B7BB8", ls="--", lw=1, alpha=0.7)
ax[0].text(5.05, ax[0].get_ylim()[1]," Motivwechsel", color="#3B7BB8", va="top", fontsize=9)
ax[0].set_ylabel("w_Al  [%]"); ax[0].set_title("Aluminiumkonzentration: UKF vs. Wahrheit (Messmodell = ink_calculator)")
ax[0].legend(loc="upper right", fontsize=9); ax[0].grid(alpha=0.3)

ax[1].plot(th, w_ipa_true*100, color="#222", lw=1.6, label="wahr")
ax[1].plot(th, est_wipa*100, color="#3B7BB8", lw=1.2, label="UKF-Schätzung")
ax[1].set_ylabel("w_IPA  [%]"); ax[1].set_title("IPA-Anteil (Verdampfung / Depletion)")
ax[1].legend(loc="upper right", fontsize=9); ax[1].grid(alpha=0.3)

ax[2].plot(th, kp_profile, color="#222", lw=1.6, label="wahr")
ax[2].plot(th, est_kp, color="#2E8B57", lw=1.2, label="UKF-Schätzung")
ax[2].axvline(5.0, color="#3B7BB8", ls="--", lw=1, alpha=0.7)
ax[2].set_ylabel("k_p  [mg/Umdr.]"); ax[2].set_xlabel("Zeit  [h]")
ax[2].set_title("Online-Schätzung k_p (unbekanntes Druckmotiv)")
ax[2].legend(loc="upper right", fontsize=9); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("ukf_demo.png", dpi=130, bbox_inches="tight")
print("Plot gespeichert: ukf_demo.png")
