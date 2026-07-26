# -*- coding: utf-8 -*-
"""
================================================================================
Datengetriebene Zustandsschaetzung der Aluminiumkonzentration in einer Drucktinte
UKF-Kernmodul  --  Messmodell = validierter ink_calculator (reale Labortabellen)
================================================================================

Aenderungen ggü. der ersten Version:
  * Additiv entfernt  ->  Zustand x = [m_Al, m_PG, m_IPA, m_W, k_p]  (4 Massen + k_p)
  * Messmodell h(x): synthetische Kennfelder ERSETZT durch ink_calculator.InkCalculator
    (Dichte volumenkontraktions-korrigiert, Schall via Wood/Urick; gegen Messungen validiert)
  * Temperatur-Absicherung: Sigma-Punkte auf gueltigen Tabellenbereich [20,30] degC geklemmt
  * k_p in mg/Umdrehung (numerische Skalierung des UKF)

Einheiten: Masse [kg], Zeit [s], Dichte [g/cm^3], Schall [m/s], Temperatur [degC]
"""

import numpy as np
from ink_calculator import InkCalculator

# =============================================================================
# 1. KOMPONENTEN, REZEPTUR, PROZESSPARAMETER
# =============================================================================
ORDER = ["Al", "PG", "IPA", "W"]                 # feste Reihenfolge (ohne Additiv)
IDX = {k: i for i, k in enumerate(ORDER)}
NC = len(ORDER)

# Rezeptur-Massenanteile der frischen Tinte (ohne Add): Al 1.82, PG 3.64, IPA 3.64, Rest W
W_RECIPE = np.array([0.0182, 0.0364, 0.0364, 1.0 - 0.0182 - 0.0364 - 0.0364])

# SL120-Konzentrat = wasserfreie Fraktion (Al, PG, IPA), auf 1 normiert
_nv = np.array([0.0182, 0.0364, 0.0364, 0.0])
S_SL = _nv / _nv.sum()                            # [Al, PG, IPA, W=0]

# Austragsverhaeltnis beim Druck (nur Al, PG), Summe = 1.  ANNAHME: Tank-Verhaeltnis.
# -> experimentell zu bestimmen (Bilanzversuche, alpha).
_sol = np.array([0.0182, 0.0364])
ALPHA = np.zeros(NC)
ALPHA[[IDX["Al"], IDX["PG"]]] = _sol / _sol.sum()

# Verdampfungskoeffizienten (EXPERIMENTELL; hier illustrativ)
# ṁ_ev,i = k_i0 * w_i  (+ k_iT * w_i * g(u), Trockneranteil hier deaktiviert)
K_IPA0 = 0.0139       # IPA-Luftverdunstung [kg/s]  -> IPA-Zeitkonstante ~2 h
K_W0   = 0.0016       # Wasser-Luftverdunstung [kg/s]

M_OFFSET = 6.0        # Hold-up ausserhalb des Tanks [kg]

# Gueltiger Temperaturbereich der Labortabellen (Dichtemodell ist hart begrenzt)
T_MIN, T_MAX = 20.0, 30.0

# =============================================================================
# 2. MESSMODELL h(x)  --  validierter ink_calculator
# =============================================================================
# Eine gemeinsame Instanz (Tabellen werden einmalig geladen).
INK = InkCalculator(tables_dir="tables_parameters")

def _clip_T(T):
    """Temperatur auf gueltigen Tabellenbereich klemmen (schuetzt Sigma-Punkte
       vor der ValueError-Grenze des Dichtemodells)."""
    return min(max(T, T_MIN), T_MAX)

def masses_to_w(m):
    tot = m.sum()
    return m / tot if tot > 0 else np.zeros_like(m)

def measurement(x, u):
    """h(x) -> [y_Waage, y_rho, y_c] mit realem Kennfeld."""
    m = x[:NC]
    tot = m.sum()
    al  = 100.0 * m[IDX["Al"]]  / tot
    pg  = 100.0 * m[IDX["PG"]]  / tot
    ipa = 100.0 * m[IDX["IPA"]] / tot
    T = _clip_T(u["T"])
    y_waage = tot - M_OFFSET
    y_rho   = INK.density(al=al, ipa=ipa, pg=pg, temperature=T)
    y_c     = INK.sound_velocity(al=al, ipa=ipa, pg=pg, temperature=T)
    return np.array([y_waage, y_rho, y_c])

# =============================================================================
# 3. PROZESSMODELL f(x, u)  --  Massenbilanz + Verdampfung (Euler)
# =============================================================================
def process_step(x, u, dt):
    """Ein Euler-Schritt. x = [m_Al, m_PG, m_IPA, m_W, k_p], k_p in mg/Umdrehung."""
    m = x[:NC].copy()
    kp = x[NC]                                     # [mg/Umdrehung]
    w = masses_to_w(m)

    mdot_print = kp * 1e-6 * u["n_Donor"]          # ṁ_print [kg/s]
    dm = np.zeros(NC)
    dm[IDX["Al"]]  = S_SL[IDX["Al"]]  * u["mdot_SL"] - ALPHA[IDX["Al"]] * mdot_print
    dm[IDX["PG"]]  = S_SL[IDX["PG"]]  * u["mdot_SL"] - ALPHA[IDX["PG"]] * mdot_print
    dm[IDX["IPA"]] = S_SL[IDX["IPA"]] * u["mdot_SL"] - K_IPA0 * w[IDX["IPA"]]
    dm[IDX["W"]]   = u["mdot_W"]                     - K_W0   * w[IDX["W"]]

    m_new = np.maximum(m + dt * dm, 1e-9)          # Nichtnegativitaet
    kp_new = max(kp, 1e-9)                          # k_p >= 0 (Random Walk via Q)
    return np.concatenate([m_new, [kp_new]])

# =============================================================================
# 4. UKF  (from scratch, nur NumPy)
# =============================================================================
class UKF:
    def __init__(self, x0, P0, Q, R, alpha=0.5, beta=2.0, kappa=0.0):
        self.n = len(x0)
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.Q = Q; self.R = R
        lam = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + lam)
        self.Wm = np.full(2*self.n+1, 1.0/(2*(self.n+lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam/(self.n+lam)
        self.Wc[0] = lam/(self.n+lam) + (1 - alpha**2 + beta)

    def _sigma_points(self):
        P = self.P + 1e-15*np.eye(self.n)
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(P)
            S = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-15)))
        pts = np.zeros((2*self.n+1, self.n))
        pts[0] = self.x
        for i in range(self.n):
            pts[i+1]        = self.x + self.gamma * S[:, i]
            pts[self.n+i+1] = self.x - self.gamma * S[:, i]
        return pts

    def predict(self, u, dt):
        pts = self._sigma_points()
        prop = np.array([process_step(p, u, dt) for p in pts])
        self.x = self.Wm @ prop
        P = self.Q.copy()
        for i in range(2*self.n+1):
            d = prop[i] - self.x
            P += self.Wc[i] * np.outer(d, d)
        self.P = P

    def update(self, y, u, mask=None):
        pts = self._sigma_points()
        Y = np.array([measurement(p, u) for p in pts])
        if mask is None:
            mask = np.ones(Y.shape[1], dtype=bool)
        Y = Y[:, mask]; y = np.asarray(y)[mask]; R = self.R[np.ix_(mask, mask)]
        y_hat = self.Wm @ Y
        S = R.copy(); C = np.zeros((self.n, Y.shape[1]))
        for i in range(2*self.n+1):
            dy = Y[i] - y_hat; dx = pts[i] - self.x
            S += self.Wc[i] * np.outer(dy, dy)
            C += self.Wc[i] * np.outer(dx, dy)
        K = C @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_hat)
        self.P = self.P - K @ S @ K.T
        self.x[:NC] = np.maximum(self.x[:NC], 1e-9)
        self.x[NC]  = max(self.x[NC], 1e-9)

    def w_al(self):
        m = self.x[:NC]; return m[IDX["Al"]] / m.sum()

    def w_al_std(self):
        """1-sigma Unsicherheit von w_Al via Delta-Methode aus P."""
        m = self.x[:NC]; tot = m.sum()
        J = np.zeros(self.n)
        for i in range(NC):
            J[i] = ((1.0 if i == IDX["Al"] else 0.0)*tot - m[IDX["Al"]]) / tot**2
        return np.sqrt(max(J @ self.P @ J, 0.0))

if __name__ == "__main__":
    print("Kernmodul geladen (4 Komponenten, ink_calculator als h(x)).")
    print(f"  Sum(W_RECIPE)={W_RECIPE.sum():.4f}  Sum(S_SL)={S_SL.sum():.4f}  Sum(ALPHA)={ALPHA.sum():.4f}")
    x = np.concatenate([W_RECIPE*100.0, [8.0]])
    u = dict(n_Donor=3.0, mdot_SL=0.0, mdot_W=0.0, T=25.0)
    print("  h(x) @Rezeptur:", np.round(measurement(x, u), 5))
