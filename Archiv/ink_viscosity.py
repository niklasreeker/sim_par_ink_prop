"""
ink_viscosity.py
================
Schaetzt die Viskositaet einer waessrigen Tinte aus

    gekapseltes Aluminiumpigment + Wasser + IPA (2-Propanol) + PG (Propan-1,2-diol)

bei vorgegebener Zusammensetzung (in Massenprozent) und Temperatur (in Grad C).

Grundlage sind die beiden Stuetztabellen im Ordner "tables_parameters":
    ipa_viscosity.csv  -> Viskositaet binaerer (Wasser + IPA)-Gemische
    pg_viscosity.csv   -> Viskositaet binaerer (Wasser + PG)-Gemische

------------------------------------------------------------------------------
Modell (zwei Schritte)
------------------------------------------------------------------------------
1) Fluessiger Traeger (Wasser + IPA + PG)
   Logarithmische (Arrhenius-artige) Ueberlagerung der gemessenen Binaerdaten:

       ln eta_traeger = ln eta(W+IPA) + ln eta(W+PG) - ln eta(W)
   d.h.
       eta_traeger = eta(W+IPA) * eta(W+PG) / eta(W)

   mit w_IPA* = m_IPA/(m_IPA+m_Wasser), w_PG* = m_PG/(m_PG+m_Wasser) als
   "Pseudo-Binaer"-Massenanteilen (genau das, was in den Tabellen steht).
   Hinweis: Dies ist NICHT das klassische Arrhenius-Gesetz aus Reinstoffwerten
   (ln eta = sum x_i ln eta_i), sondern eine log-additive Ueberlagerung der
   gemessenen Binaergemische -> bildet die Nichtidealitaet realistischer ab.

2) Feststoffanteil (Aluminiumpigment)
   Suspendierte Partikel erhoehen die Viskositaet um den Faktor eta_r:

     - "einstein"          : eta_r = 1 + c1*phi
     - "batchelor"         : eta_r = 1 + c1*phi + c2*phi**2   (zweite Ordnung)
     - "krieger-dougherty" : eta_r = (1 - phi/phi_max)^(-[eta]*phi_max)

   phi ist der VOLUMENanteil des Pigments (aus Massenanteilen + Dichten).

   Koeffizienten zweiter Ordnung in der Literatur:
       c1 = 2.5  (Einstein, Hartkugeln, verduennt)
       c2 = 6.2  (Batchelor 1977, brownsche Hartkugeln)
       c2 = 7.6  (Batchelor & Green 1972, reine Dehnstroemung)
   Default hier: c1 = 2.5, c2 = 7.2 (frei einstellbar).

------------------------------------------------------------------------------
Grenzen des Modells
------------------------------------------------------------------------------
- Physikalisch motivierte SCHAETZUNG, kein Ersatz fuer eine Messung.
- Aluminiumpigmente sind oft Plaettchen -> [eta] bzw. c1/c2 koennen groesser sein.
- Reale Pigmenttinten sind haeufig nicht-newtonsch (scherverduennend).
- Ueber 'calibration_factor' an eigene Messwerte anpassbar (.calibrate(...)).
"""

import os
import csv
import bisect


# =====================================================================
# Stoffdaten (Standardwerte, alle ueberschreibbar)
# =====================================================================
RHO_WATER = 0.998     # g/cm^3  (~20-25 C)
RHO_IPA = 0.785       # g/cm^3
RHO_PG = 1.036        # g/cm^3
RHO_PIGMENT = 2.70    # g/cm^3  bulk-Aluminium; gekapselt evtl. niedriger -> anpassen


# =====================================================================
# Tabellen laden und interpolieren (nur Standardbibliothek)
# =====================================================================
def load_table(path):
    """Liest eine Viskositaets-CSV und gibt ein Dict mit Stuetzstellen zurueck."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        temps = [float(h.replace("Viscosity_", "").rstrip("Cc")) for h in header[1:]]
        mass_percent, eta = [], []
        for row in reader:
            if not row:
                continue
            mass_percent.append(float(row[0]))
            eta.append([float(v) for v in row[1:]])
    order = sorted(range(len(mass_percent)), key=lambda i: mass_percent[i])
    mass_percent = [mass_percent[i] for i in order]
    eta = [eta[i] for i in order]
    return {"temps": temps, "mass_percent": mass_percent, "eta": eta, "path": path}


def _interp_1d(xs, ys, x):
    """Lineare Interpolation. Gibt (wert, out_of_range) zurueck; klemmt am Rand."""
    if x <= xs[0]:
        return ys[0], (x < xs[0] - 1e-9)
    if x >= xs[-1]:
        return ys[-1], (x > xs[-1] + 1e-9)
    i = bisect.bisect_right(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0), False


def interp_table(table, mass_percent, temperature_C):
    """Bilineare Interpolation in (Massenprozent, Temperatur)."""
    temps = table["temps"]
    mps = table["mass_percent"]
    grid = table["eta"]
    eta_vs_T, mp_oob = [], False
    for j in range(len(temps)):
        col = [grid[i][j] for i in range(len(mps))]
        val, oob = _interp_1d(mps, col, mass_percent)
        eta_vs_T.append(val)
        mp_oob = mp_oob or oob
    eta, t_oob = _interp_1d(temps, eta_vs_T, temperature_C)
    return eta, mp_oob, t_oob


# =====================================================================
# Hauptmodell
# =====================================================================
class InkViscosityModel:
    def __init__(self, tables_dir="tables_parameters",
                 rho_water=RHO_WATER, rho_ipa=RHO_IPA, rho_pg=RHO_PG,
                 rho_pigment=RHO_PIGMENT,
                 suspension_model="batchelor",
                 einstein_coeff=2.5,     # c1 (1. Ordnung)
                 batchelor_coeff=7.2,    # c2 (2. Ordnung)
                 intrinsic_viscosity=2.5, phi_max=0.63,  # nur fuer Krieger-Dougherty
                 calibration_factor=1.0):
        self.ipa = load_table(os.path.join(tables_dir, "ipa_viscosity.csv"))
        self.pg = load_table(os.path.join(tables_dir, "pg_viscosity.csv"))
        self.rho_water = rho_water
        self.rho_ipa = rho_ipa
        self.rho_pg = rho_pg
        self.rho_pigment = rho_pigment
        model = suspension_model.lower()
        if model not in ("batchelor", "einstein", "krieger-dougherty"):
            raise ValueError("suspension_model muss 'batchelor', 'einstein' oder "
                             "'krieger-dougherty' sein.")
        self.suspension_model = model
        self.einstein_coeff = einstein_coeff
        self.batchelor_coeff = batchelor_coeff
        self.intrinsic_viscosity = intrinsic_viscosity
        self.phi_max = phi_max
        self.calibration_factor = calibration_factor

    def water_viscosity(self, temperature_C):
        """Wasser-Viskositaet aus der 0-%-Zeile der IPA-Tabelle (mPa*s)."""
        eta, _, t_oob = interp_table(self.ipa, 0.0, temperature_C)
        return eta, t_oob

    def _suspension_factor(self, phi):
        """Relative Viskositaetserhoehung durch die Partikel; gibt (faktor, warnings)."""
        warnings = []
        c1 = self.einstein_coeff
        if self.suspension_model == "einstein":
            factor = 1.0 + c1 * phi
        elif self.suspension_model == "batchelor":
            c2 = self.batchelor_coeff
            factor = 1.0 + c1 * phi + c2 * phi ** 2
        else:  # krieger-dougherty
            if phi >= self.phi_max:
                warnings.append(f"phi={phi:.3f} >= phi_max={self.phi_max} "
                                f"-- Krieger-Dougherty nicht mehr gueltig.")
                return float("inf"), warnings
            factor = (1.0 - phi / self.phi_max) ** (-self.intrinsic_viscosity * self.phi_max)

        if phi > 0.10:
            warnings.append(f"Volumenanteil phi={phi:.3f} > 0.10 -- Polynom-Naeherungen "
                            f"(Einstein/Batchelor) gelten nur fuer verduennte Suspensionen.")
        return factor, warnings

    def _model_label(self):
        if self.suspension_model == "einstein":
            return f"Einstein (1 + {self.einstein_coeff}*phi)"
        if self.suspension_model == "batchelor":
            return (f"Batchelor 2. Ordnung "
                    f"(1 + {self.einstein_coeff}*phi + {self.batchelor_coeff}*phi^2)")
        return f"Krieger-Dougherty ([eta]={self.intrinsic_viscosity}, phi_max={self.phi_max})"

    def estimate(self, water, ipa, pg, aluminum, temperature_C, verbose=True):
        """Schaetzt die Tintenviskositaet. Anteile in Massenprozent."""
        warnings = []
        total = water + ipa + pg + aluminum
        if abs(total - 100.0) > 0.5:
            warnings.append(f"Summe der Anteile = {total:.2f} % (nicht 100). "
                            f"Es werden die Verhaeltnisse verwendet.")

        m_w, m_i, m_p, m_al = water, ipa, pg, aluminum
        carrier_mass = m_w + m_i + m_p
        if carrier_mass <= 0:
            raise ValueError("Fluessiger Traeger (Wasser+IPA+PG) ist 0 -- nicht berechenbar.")

        # --- Schritt 1: Tabellen-Lookups + Traeger (Arrhenius-artig) ---
        ipa_bin = 100.0 * m_i / (m_i + m_w) if (m_i + m_w) > 0 else 0.0
        pg_bin = 100.0 * m_p / (m_p + m_w) if (m_p + m_w) > 0 else 0.0

        eta_w, t_oob_w = self.water_viscosity(temperature_C)
        eta_wi, mp_oob_i, t_oob_i = interp_table(self.ipa, ipa_bin, temperature_C)
        eta_wp, mp_oob_p, t_oob_p = interp_table(self.pg, pg_bin, temperature_C)

        if t_oob_w or t_oob_i or t_oob_p:
            warnings.append(f"Temperatur {temperature_C} C ausserhalb des Tabellenbereichs "
                            f"-- es wird auf den Rand geklemmt.")
        if mp_oob_i:
            warnings.append(f"IPA-Pseudoanteil {ipa_bin:.2f} % ausserhalb der IPA-Tabelle.")
        if mp_oob_p:
            warnings.append(f"PG-Pseudoanteil {pg_bin:.2f} % ausserhalb der PG-Tabelle.")

        eta_carrier = eta_wi * eta_wp / eta_w

        # --- Dichte des Traegers (inverse Mischungsregel) ---
        w_w, w_i, w_p = m_w / carrier_mass, m_i / carrier_mass, m_p / carrier_mass
        inv_rho = w_w / self.rho_water + w_i / self.rho_ipa + w_p / self.rho_pg
        rho_carrier = 1.0 / inv_rho

        # --- Schritt 2: Pigment-Volumenanteil + Suspensionsfaktor ---
        v_al = m_al / self.rho_pigment
        v_carrier = carrier_mass / rho_carrier
        phi = v_al / (v_al + v_carrier) if (v_al + v_carrier) > 0 else 0.0

        susp_factor, susp_warn = self._suspension_factor(phi)
        warnings.extend(susp_warn)

        if w_w < 0.5:
            warnings.append("Wasseranteil im Traeger < 50 % -- Schaetzung weniger "
                            "zuverlaessig (Modell fuer wasserreiche Gemische ausgelegt).")

        eta_ink = eta_carrier * susp_factor * self.calibration_factor

        result = {
            "viscosity_mPas": eta_ink,
            "temperature_C": temperature_C,
            "suspension_model": self._model_label(),
            "eta_water": eta_w,
            "eta_water_ipa_binary": eta_wi,
            "eta_water_pg_binary": eta_wp,
            "eta_carrier": eta_carrier,
            "ipa_pseudo_pct": ipa_bin,
            "pg_pseudo_pct": pg_bin,
            "rho_carrier": rho_carrier,
            "phi_pigment_vol": phi,
            "suspension_factor": susp_factor,
            "calibration_factor": self.calibration_factor,
            "warnings": warnings,
        }

        if verbose:
            self._print_report(water, ipa, pg, aluminum, result)
        return result

    def calibrate(self, measured_viscosity, water, ipa, pg, aluminum, temperature_C):
        """Setzt calibration_factor so, dass das Modell einen Messwert exakt trifft."""
        self.calibration_factor = 1.0
        base = self.estimate(water, ipa, pg, aluminum, temperature_C, verbose=False)
        self.calibration_factor = measured_viscosity / base["viscosity_mPas"]
        return self.calibration_factor

    @staticmethod
    def _print_report(water, ipa, pg, aluminum, r):
        print("=" * 60)
        print("Tinten-Viskositaet (Schaetzung)")
        print("-" * 60)
        print(f"  Zusammensetzung [Massen-%]: Wasser {water}, IPA {ipa}, "
              f"PG {pg}, Al {aluminum}")
        print(f"  Temperatur:                 {r['temperature_C']} C")
        print(f"  Suspensionsmodell:          {r['suspension_model']}")
        print("-" * 60)
        print(f"  eta Wasser:                 {r['eta_water']:.3f} mPa*s")
        print(f"  Pseudo-Binaer IPA:          {r['ipa_pseudo_pct']:.2f} % "
              f"-> {r['eta_water_ipa_binary']:.3f} mPa*s")
        print(f"  Pseudo-Binaer PG:           {r['pg_pseudo_pct']:.2f} % "
              f"-> {r['eta_water_pg_binary']:.3f} mPa*s")
        print(f"  eta Traeger (fluessig):     {r['eta_carrier']:.3f} mPa*s")
        print(f"  Pigment-Volumenanteil phi:  {r['phi_pigment_vol']*100:.2f} vol-%")
        print(f"  Suspensionsfaktor:          x{r['suspension_factor']:.4f}")
        if r["calibration_factor"] != 1.0:
            print(f"  Kalibrierfaktor:            x{r['calibration_factor']:.4f}")
        print("-" * 60)
        print(f"  => eta Tinte:               {r['viscosity_mPas']:.3f} mPa*s")
        if r["warnings"]:
            print("-" * 60)
            for w in r["warnings"]:
                print(f"  ! {w}")
        print("=" * 60)


# =====================================================================
# Beispiel / direkte Nutzung
# =====================================================================
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    tables = os.path.join(here, "tables_parameters")

    # Default: Batchelor 2. Ordnung mit c1 = 2.5, c2 = 7.2
    model = InkViscosityModel(tables_dir=tables,
                              suspension_model="batchelor",
                              einstein_coeff=2.5,
                              batchelor_coeff=7.2)

    model.estimate(water=85.0, ipa=5.0, pg=7.0, aluminum=3.0, temperature_C=25.0)

    # Alternative Modelle (einfach umschalten):
    #   InkViscosityModel(..., suspension_model="einstein", einstein_coeff=2.5)
    #   InkViscosityModel(..., suspension_model="krieger-dougherty",
    #                     intrinsic_viscosity=2.5, phi_max=0.63)

    # Eigener Code:
    #   res = model.estimate(88, 4, 6, 2, temperature_C=30, verbose=False)
    #   print(res["viscosity_mPas"])

    # An Messwert anpassen:
    #   model.calibrate(1.9, water=85, ipa=5, pg=7, aluminum=3, temperature_C=25)