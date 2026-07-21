# -*- coding: utf-8 -*-
"""
================================================================================
Datengetriebene Zustandsschaetzung der Aluminiumkonzentration in einer Drucktinte
UKF-Referenzimplementierung (Unscented Kalman Filter), NumPy-only, transparent.
================================================================================

Struktur:
  1. Physikalische Parameter & Rezeptur
  2. Kennfelder (Messmodell): Dichte (physikbasiert) + Schall (kalibriert-linear)
     -> HIER die realen Labor-Fits einsetzen
  3. Prozessmodell f(x,u): Massenbilanz + Verdampfung (Euler-Schritt)
  4. Messmodell h(x,u): [Waage, Dichte, Schall]
  5. UKF (from scratch)
  6. Simulator (Ground Truth) + Demo mit synthetischen, verrauschten Daten

Zustand:  x = [m_Al, m_PG, m_IPA, m_W, m_Add, k_p]     (5 Massen + 1 augm. Parameter)
Einheiten: Masse [kg], Zeit [s], Dichte [g/cm^3], Schall [m/s], Temperatur [degC]
"""

import numpy as np

# =============================================================================
# 1. PHYSIKALISCHE PARAMETER & REZEPTUR
# =============================================================================
ORDER = ["Al", "PG", "IPA", "W", "Add"]          # feste Reihenfolge der Komponenten
IDX = {k: i for i, k in enumerate(ORDER)}
N_COMP = len(ORDER)

# Komponentendichten [g/cm^3] (~20-25 degC)
RHO = np.array([2.70, 1.036, 0.786, 0.998, 1.40])

# Rezeptur-Massenanteile der frischen Tinte  (= definierter Anfangszustand / Anker)
W_RECIPE = np.array([0.01814, 0.03628, 0.03628, 0.90703, 0.00227])

# SL120-Konzentrat = wasserfreie Fraktion (Al, PG, IPA, Add), auf 1 normiert
_nv = np.array([0.01814, 0.03628, 0.03628, 0.0, 0.00227])
S_SL = _nv / _nv.sum()      # s_i des Konzentrats: [Al, PG, IPA, W=0, Add]

# Austragsverhaeltnis beim Druck (Al, PG, Add), Summe = 1
# ANNAHME hier: Austrag im Tank-Verhaeltnis der nv-Fraktion (alpha ~ s).
# -> Diese alpha experimentell bestimmen (Bilanzversuche)!
_nvsolid = np.array([0.01814, 0.03628, 0.00227])   # Al, PG, Add
ALPHA = np.zeros(N_COMP)
ALPHA[[IDX["Al"], IDX["PG"], IDX["Add"]]] = _nvsolid / _nvsolid.sum()

# Verdampfungskoeffizienten  (EXPERIMENTELL zu bestimmen; hier illustrativ)
# ṁ_ev,i = k_i0 * w_i  +  k_iT * w_i * g(u)
K_IPA0 = 0.0139     # Luft-Verdunstung IPA  [kg/s] -> IPA-Zeitkonstante ~2 h
K_W0   = 0.0016     # Luft-Verdunstung Wasser [kg/s]
K_IPAT = 0.0        # Trockneranteil (in dieser Demo aus, da Verdampfung exp. bekannt)
K_WT   = 0.0
def g_dryer(u):     # Trockner-Intensitaet (hier 0, Platzhalter fuer P_Heiz/n_Vent/dz)
    return 0.0

# Hold-up ausserhalb des Tanks [kg] (Leitungen, Wanne, Rollen)
M_OFFSET = 6.0

# =============================================================================
# 2. KENNFELDER (MESSMODELL)  ---  hier reale Labor-Fits einsetzen
# =============================================================================
def density_map(w, T):
    """Dichte [g/cm^3] aus Zusammensetzung (ideale Volumenmischung) + T-Korrektur.
       Reproduziert den gemessenen Al-Signalhub (~1.3 mg/cm^3 / 0.2%-Pkt) auf 2%."""
    inv = np.sum(w / RHO)
    rho = 1.0 / inv
    # lineare Temperaturkorrektur (dominiert durch Wasser), Referenz 25 degC
    rho += -0.00033 * (T - 25.0)     # ~ -0.33 mg/cm^3 pro K
    return rho

# Schall-Kennfeld: linear um Betriebspunkt, kalibriert an den gemessenen Zahlen.
# Sensitivitaeten [ (m/s) pro Massenanteil-Einheit ]:
C0 = 1500.0                          # Basis-Schallgeschwindigkeit [m/s] bei Rezeptur, 25 degC
DC_DW = np.array([
    161.5,    # Al   -> 0.323 m/s ueber 0.2%-Pkt (gemessen)
    420.0,    # PG   -> Glykole heben c stark
    380.0,    # IPA  -> Alkohol hebt c im verduennten Bereich stark  (IPA-Beobachtbarkeit!)
    0.0,      # W    (Referenzkomponente)
    50.0,     # Add  (klein)
])
DC_DT = 1.51                         # [m/s / K]  (gemessen: 0.151 m/s / 0.1 K)

def sound_map(w, T):
    """Schallgeschwindigkeit [m/s] aus Zusammensetzung + Temperatur (linear)."""
    return C0 + DC_DW @ (w - W_RECIPE) + DC_DT * (T - 25.0)

# =============================================================================
# 3. PROZESSMODELL  f(x, u)
# =============================================================================
# Eingang u (dict):  n_Donor, mdot_SL, mdot_W, P_Heiz, n_Vent, dz, T
def masses_to_w(m):
    tot = m.sum()
    return m / tot if tot > 0 else np.zeros_like(m)

def process_step(x, u, dt):
    """Ein Euler-Schritt des Zustands x = [m_Al,m_PG,m_IPA,m_W,m_Add, k_p].
       WICHTIG: k_p wird in mg/Umdrehung gefuehrt (O(1..10)), damit der UKF
       numerisch nicht an der Skalen-Diskrepanz zu den Massen (kg) scheitert."""
    m = x[:N_COMP].copy()
    kp = x[N_COMP]                                      # [mg/Umdrehung]
    w = masses_to_w(m)

    mdot_print = kp * 1e-6 * u["n_Donor"]               # ṁ_print = k_p * n_Donor  [kg/s]
    gu = g_dryer(u)
    mdot_ev_IPA = K_IPA0 * w[IDX["IPA"]] + K_IPAT * w[IDX["IPA"]] * gu
    mdot_ev_W   = K_W0   * w[IDX["W"]]   + K_WT   * w[IDX["W"]]   * gu

    dm = np.zeros(N_COMP)
    dm[IDX["Al"]]  = S_SL[IDX["Al"]]  * u["mdot_SL"] - ALPHA[IDX["Al"]]  * mdot_print
    dm[IDX["PG"]]  = S_SL[IDX["PG"]]  * u["mdot_SL"] - ALPHA[IDX["PG"]]  * mdot_print
    dm[IDX["Add"]] = S_SL[IDX["Add"]] * u["mdot_SL"] - ALPHA[IDX["Add"]] * mdot_print
    dm[IDX["IPA"]] = S_SL[IDX["IPA"]] * u["mdot_SL"] - mdot_ev_IPA
    dm[IDX["W"]]   = u["mdot_W"]                      - mdot_ev_W

    m_new = m + dt * dm
    m_new = np.maximum(m_new, 1e-9)                    # Nichtnegativitaet (Nebenbed.)
    kp_new = max(kp, 1e-12)                            # k_p >= 0 (Random Walk via Q)
    return np.concatenate([m_new, [kp_new]])

# =============================================================================
# 4. MESSMODELL  h(x, u)   ->  [y_Waage, y_rho, y_c]
# =============================================================================
def measurement(x, u):
    m = x[:N_COMP]
    w = masses_to_w(m)
    y_waage = m.sum() - M_OFFSET
    y_rho   = density_map(w, u["T"])
    y_c     = sound_map(w, u["T"])
    return np.array([y_waage, y_rho, y_c])

# =============================================================================
# 5. UKF  (from scratch)
# =============================================================================
class UKF:
    def __init__(self, x0, P0, Q, R, alpha=1e-2, beta=2.0, kappa=0.0):
        self.n = len(x0)
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.Q = Q; self.R = R
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        lam = alpha**2 * (self.n + kappa) - self.n
        self.lam = lam
        self.gamma = np.sqrt(self.n + lam)
        # Gewichte
        self.Wm = np.full(2*self.n+1, 1.0/(2*(self.n+lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam/(self.n+lam)
        self.Wc[0] = lam/(self.n+lam) + (1 - alpha**2 + beta)

    def _sigma_points(self):
        P = self.P + 1e-15*np.eye(self.n)             # Jitter fuer Cholesky-Stabilitaet
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # Notfall: Eigenwert-Reparatur
            vals, vecs = np.linalg.eigh(P)
            vals = np.maximum(vals, 1e-12)
            S = vecs @ np.diag(np.sqrt(vals))
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
        self._sigma_cache = prop                       # fuer Korrektur wiederverwenden

    def update(self, y, u, mask=None):
        """mask: bool-Array, welche Messkanaele aktuell verfuegbar sind."""
        pts = self._sigma_points()
        Y = np.array([measurement(p, u) for p in pts])
        if mask is None:
            mask = np.ones(Y.shape[1], dtype=bool)
        Y = Y[:, mask]
        y = np.asarray(y)[mask]
        R = self.R[np.ix_(mask, mask)]

        y_hat = self.Wm @ Y
        S = R.copy()
        C = np.zeros((self.n, Y.shape[1]))
        for i in range(2*self.n+1):
            dy = Y[i] - y_hat
            dx = pts[i] - self.x
            S += self.Wc[i] * np.outer(dy, dy)
            C += self.Wc[i] * np.outer(dx, dy)
        K = C @ np.linalg.inv(S)
        self.x = self.x + K @ (y - y_hat)
        self.P = self.P - K @ S @ K.T
        # Nichtnegativitaet nach Update (Projektion)
        self.x[:N_COMP] = np.maximum(self.x[:N_COMP], 1e-9)
        self.x[N_COMP]  = max(self.x[N_COMP], 1e-12)

    def w_al(self):
        m = self.x[:N_COMP]
        return m[IDX["Al"]] / m.sum()

    def w_al_std(self):
        """1-sigma Unsicherheit von w_Al via lineare Fehlerfortpflanzung aus P."""
        m = self.x[:N_COMP]; tot = m.sum()
        J = np.zeros(self.n)
        for i in range(N_COMP):
            J[i] = ((1.0 if i == IDX["Al"] else 0.0)*tot - m[IDX["Al"]]) / tot**2
        var = J @ self.P @ J
        return np.sqrt(max(var, 0.0))

print("Modul geladen. Parameter konsistent:",
      f"Sum(W_RECIPE)={W_RECIPE.sum():.4f}, Sum(ALPHA)={ALPHA.sum():.4f}, Sum(S_SL)={S_SL.sum():.4f}")
