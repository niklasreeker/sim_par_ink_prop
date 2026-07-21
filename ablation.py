# -*- coding: utf-8 -*-
"""Ablationsstudie: Wert des Schallsensors. Filter laeuft mit
   (A) Waage+Dichte+Schall  vs.  (B) Waage+Dichte.
   Vergleich der w_Al-Unsicherheit und des Fehlers ueber die Zeit."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ukf_aluminium import IDX, N_COMP, W_RECIPE, process_step, measurement, UKF

rng = np.random.default_rng(7)
dt=5.0; T_end=8*3600; steps=int(T_end/dt); t=np.arange(steps)*dt
M0=100.0; m0=W_RECIPE*M0

n_donor=np.full(steps,3.0)+0.15*np.sin(2*np.pi*t/1800)
for a,b in [(1.2,1.5),(4.0,4.3)]: n_donor[int(a*3600/dt):int(b*3600/dt)]=0.0
kp_profile=np.full(steps,8.0); kp_profile[int(5*3600/dt):]=1.35*8.0
T_true=25.0+0.8*np.sin(2*np.pi*t/(3*3600))
mdot_SL=np.where((t%(90*60))<60,0.020,0.0)
mdot_W =np.where((t%(30*60))<60,0.060,0.0)
def make_u(k): return dict(n_Donor=n_donor[k],mdot_SL=mdot_SL[k],mdot_W=mdot_W[k],
                           P_Heiz=0,n_Vent=0,dz=0,T=T_true[k])

# Ground truth
X=np.zeros((steps,N_COMP+1)); x=np.concatenate([m0,[kp_profile[0]]])
for k in range(steps):
    x[N_COMP]=kp_profile[k]; X[k]=x; x=process_step(x,make_u(k),dt)
w_al_true=X[:,IDX["Al"]]/X[:,:N_COMP].sum(1)

SIG=dict(waage=0.15,rho=0.00006,c=0.02)
meas=np.zeros((steps,3)); avail=np.zeros((steps,3),bool)
for k in range(steps):
    y=measurement(X[k],make_u(k))
    meas[k,0]=y[0]+SIG["waage"]*rng.standard_normal(); avail[k,0]=True
    if k%int(30/dt)==0:
        meas[k,1]=y[1]+SIG["rho"]*rng.standard_normal()
        meas[k,2]=y[2]+SIG["c"]*rng.standard_normal(); avail[k,1]=avail[k,2]=True

def run(use_sound):
    x0=np.concatenate([m0*np.array([1.03,0.97,1.05,1.0,0.9]),[5.0]])
    P0=np.diag(np.concatenate([(0.05*m0)**2,[5.0**2]]))
    Q=np.diag(np.concatenate([(np.array([2e-4,2e-4,2e-4,5e-4,5e-5]))**2,[0.02**2]]))
    R=np.diag([SIG["waage"]**2,SIG["rho"]**2,SIG["c"]**2])
    f=UKF(x0,P0,Q,R,alpha=0.5,beta=2.0,kappa=0.0)
    wal=np.zeros(steps); s=np.zeros(steps)
    for k in range(steps):
        u=make_u(k); f.predict(u,dt)
        m=avail[k].copy()
        if not use_sound: m[2]=False       # Schallkanal deaktivieren
        if m.any(): f.update(meas[k],u,mask=m)
        wal[k]=f.w_al(); s[k]=f.w_al_std()
    return wal,s

walA,sA=run(True)      # Dichte + Schall
walB,sB=run(False)     # nur Dichte

th=t/3600
fig,ax=plt.subplots(2,1,figsize=(11,8),sharex=True)
ax[0].plot(th,sA*100,color="#D85A30",lw=1.5,label="Waage + Dichte + Schall")
ax[0].plot(th,sB*100,color="#3B7BB8",lw=1.5,label="Waage + Dichte (ohne Schall)")
ax[0].axhline(0.05,color="#888",ls=":",lw=1,label="halbes Zielband (0,05 %-Pkt)")
ax[0].set_ylabel("Unsicherheit σ(w_Al) [%-Pkt]")
ax[0].set_title("Wert des Schallsensors: Unsicherheit der Al-Schätzung")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

ax[1].plot(th,np.abs(walA-w_al_true)*100,color="#D85A30",lw=1.0,label="|Fehler| mit Schall")
ax[1].plot(th,np.abs(walB-w_al_true)*100,color="#3B7BB8",lw=1.0,label="|Fehler| ohne Schall")
ax[1].axhline(0.1,color="#7A5EA8",ls="--",lw=1,label="Zielgrenze 0,1 %-Pkt")
ax[1].set_ylabel("|Fehler w_Al| [%-Pkt]"); ax[1].set_xlabel("Zeit [h]")
ax[1].set_title("Tatsächlicher Schätzfehler im Vergleich")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("ukf_ablation.png",dpi=130,bbox_inches="tight")

i0=int(30*60/dt)
print(f"RMSE mit Schall : {np.sqrt(np.mean((walA-w_al_true)[i0:]**2))*100:.4f} %-Pkt")
print(f"RMSE ohne Schall: {np.sqrt(np.mean((walB-w_al_true)[i0:]**2))*100:.4f} %-Pkt")
print(f"mittlere σ mit Schall : {np.mean(sA[i0:])*100:.4f} %-Pkt")
print(f"mittlere σ ohne Schall: {np.mean(sB[i0:])*100:.4f} %-Pkt")
print("Plot gespeichert: ukf_ablation.png")
