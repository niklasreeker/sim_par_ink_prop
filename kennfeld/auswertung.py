#!/usr/bin/env python3
"""
Auswertung der Kennfeldaufnahme Pico 3000 / L-Com 5500
Masterarbeit Reeker - Versuchsstand

Projektstruktur:

    kennfeld/
    +-- auswertung.py          <- dieses Skript
    +-- messdaten/             <- hier alle CSV-Dateien der SPS ablegen
    |   +-- Kennfeld_v2.csv
    |   +-- ...
    +-- ergebnisse/            <- wird automatisch erzeugt

Ziel: Aus den Einwaagen und den Sensorwerten ein Kalibriermodell
bestimmen, das die Vorhersagerichtung

    Zusammensetzung + Temperatur  ->  Dichte, Schallgeschwindigkeit

abbildet. Das ist genau die Richtung, die spaeter als Messmodell h(x)
im UKF gebraucht wird.

Aufruf:  python3 auswertung.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS = True
except ImportError:
    PLOTS = False


# =====================================================================
# KONFIGURATION
# =====================================================================

BASIS      = Path(__file__).resolve().parent
DATENORDER = BASIS / "messdaten"
AUSGABE    = BASIS / "ergebnisse"

# --- Zusammensetzung des Konzentrats SL120 (Massenanteile) ---
# Herstellerangabe. Stellt sich spaeter ein abweichender Al-Gehalt
# heraus, wird NUR diese Zahl korrigiert und die gesamte Kampagne
# rechnet sich neu - deshalb werden in der CSV Einwaagen und keine
# fertigen Prozentwerte gespeichert.
SL120_AL  = 0.20
SL120_IPA = 0.40
SL120_PG  = 0.40

# --- Bezugstemperatur der Regression [degC] ---
# Sinnvoll in die Mitte des Messbereichs legen.
T_REF = 24.0

# --- Qualitaetsfilter ---
NUR_GUELTIGE     = True    # nur Saetze mit Gueltig == 1
MAX_MASSENVERLUST = 0.5    # [%] max. Verdunstungsverlust, 0 = Pruefung aus
MIN_ABSTAND_S     = 100    # [s] Mindestabstand zweier Saetze, s. Hinweis unten

# --- Werkskonstanten aus dem Factory Adjustment Protocol ---
# Seriennummer 85786356, L-Com 5500 HAS
WERK = dict(
    DA=1.398529e-03, DA1=-2.9704e-04, DA2=-1.1928e-08,
    DB=2112.762,     DB1=-3.66e-05,
    SA=6.357216,     SA1=1.22e-05,
    SB=2.764707,     SB1=7.51927e-05, SB2=1.085813e-07,
)

# --- Modellterme ---
# Wasser ist die Restkomponente und steckt implizit im Achsenabschnitt.
# Terme, die im Datensatz konstant null sind, werden automatisch
# verworfen (sonst waere die Designmatrix singulaer).
TERME = ["w_Al", "w_IPA", "w_PG", "w_MG", "dT"]


# =====================================================================
# EINLESEN
# =====================================================================

def lade_messdaten(ordner: Path) -> pd.DataFrame:
    """Liest alle CSV-Dateien des Ordners und haengt sie aneinander.

    Die DataLog-Dateien der S7-1500 haben zwei Eigenheiten:
      - sie enden mit einer Zeile '//END'
      - die Zahlen stehen in wissenschaftlicher Notation mit
        fuehrenden Leerzeichen
    Beides wird hier abgefangen.
    """
    dateien = sorted(ordner.glob("*.csv"))
    if not dateien:
        sys.exit(f"Keine CSV-Dateien in {ordner} gefunden.")

    teile = []
    for f in dateien:
        df = pd.read_csv(f, comment="/", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df["Quelldatei"] = f.name
        teile.append(df)
        print(f"  {f.name}: {len(df)} Datensaetze")

    df = pd.concat(teile, ignore_index=True)

    # Zeitstempel zusammensetzen. ACHTUNG: Die Spalte heisst 'UTC Time'.
    # Steht die CPU-Uhr falsch, sind diese Werte wertlos - dann in TIA
    # unter Online & Diagnose die Uhrzeit synchronisieren.
    if {"Date", "UTC Time"}.issubset(df.columns):
        df["Zeit"] = pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " +
            df["UTC Time"].astype(str).str.strip(),
            errors="coerce", utc=True)

    return df


# =====================================================================
# ZUSAMMENSETZUNG
# =====================================================================

def berechne_zusammensetzung(df: pd.DataFrame) -> pd.DataFrame:
    """Rechnet die Einwaagen in Massenanteile [Gew.-%] um."""
    df = df.copy()

    m_ges = (df["m_SL120"] + df["m_Wasser"] + df["m_IPA"]
             + df["m_PG"] + df["m_MG"])
    df["m_ges"] = m_ges

    ungueltig = m_ges <= 0
    if ungueltig.any():
        print(f"  WARNUNG: {ungueltig.sum()} Satz/Saetze mit Gesamtmasse 0 "
              f"- vermutlich Einwaagen nicht eingetragen.")

    df["w_Al"]  = 100.0 * SL120_AL * df["m_SL120"] / m_ges
    df["w_IPA"] = 100.0 * (SL120_IPA * df["m_SL120"] + df["m_IPA"]) / m_ges
    df["w_PG"]  = 100.0 * (SL120_PG * df["m_SL120"] + df["m_PG"]) / m_ges
    df["w_MG"]  = 100.0 * df["m_MG"] / m_ges
    df["w_H2O"] = 100.0 * df["m_Wasser"] / m_ges

    # Verdunstungsverlust
    hat_massen = (df["m_vorher"] > 0) & (df["m_nachher"] > 0)
    df["Verlust_pct"] = np.where(
        hat_massen,
        100.0 * (df["m_vorher"] - df["m_nachher"]) / df["m_vorher"].replace(0, np.nan),
        np.nan)

    df["dT"] = df["T_M"] - T_REF
    return df


# =====================================================================
# QUALITAETSFILTER
# =====================================================================

def filtere(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    maske = pd.Series(True, index=df.index)

    if NUR_GUELTIGE and "Gueltig" in df.columns:
        raus = df["Gueltig"] != 1
        if raus.any():
            print(f"  {raus.sum()} Satz/Saetze mit Gueltig=0 verworfen")
        maske &= ~raus

    if MAX_MASSENVERLUST > 0 and "Verlust_pct" in df.columns:
        raus = df["Verlust_pct"] > MAX_MASSENVERLUST
        if raus.any():
            print(f"  {raus.sum()} Satz/Saetze mit Massenverlust > "
                  f"{MAX_MASSENVERLUST}% verworfen")
        maske &= ~raus.fillna(False)

    raus = df["m_ges"] <= 0
    maske &= ~raus

    df = df[maske].copy()
    print(f"  {len(df)} von {n0} Saetzen verbleiben")

    # --- Hinweis auf ueberlappende Mittelungsfenster ---
    # Das Fenster umfasst 100 Messwerte bei 1 s Aktualisierungsrate,
    # also 100 s. Zwei Saetze im Abstand von 8 s teilen 92 ihrer 100
    # Werte - das sind KEINE unabhaengigen Messungen.
    if "Zeit" in df.columns and df["Zeit"].notna().sum() > 1:
        d = df.sort_values("Zeit")["Zeit"].diff().dt.total_seconds()
        eng = (d < MIN_ABSTAND_S) & d.notna()
        if eng.any():
            print(f"  WARNUNG: {eng.sum()} Satz/Saetze liegen weniger als "
                  f"{MIN_ABSTAND_S} s auseinander. Die Mittelungsfenster "
                  f"ueberlappen, die Punkte sind nicht unabhaengig.")

    return df


# =====================================================================
# KONTROLLE GEGEN DIE WERKSKONSTANTEN
# =====================================================================

def rho_aus_periode(tau, T, k=WERK):
    """Dichte aus Schwingungsperiode [us] und Temperatur [degC]."""
    return (k["DA"] * (1 + k["DA1"] * T + k["DA2"] * T**2) * tau**2
            - k["DB"] * (1 + k["DB1"] * T))


def c_aus_laufzeit(t, T, k=WERK):
    """Schallgeschwindigkeit aus Laufzeit [us] und Temperatur [degC]."""
    return (1000.0 * k["SA"] * (1 + k["SA1"] * T)
            / (t - k["SB"] * (1 + k["SB1"] * T + k["SB2"] * T**2)))


def pruefe_werkskonstanten(df: pd.DataFrame) -> pd.DataFrame:
    """Rechnet Dichte und Schall aus den Rohgroessen nach.

    Das ist die unabhaengige Kontrolle der geraeteinternen Rechnung.
    Systematische Abweichungen deuten auf einen anderen Temperaturfuehler
    (L-D Temp statt der zusammengefassten Temperature) oder auf eine
    abweichende Formelvariante hin.
    """
    df = df.copy()
    if "Per_M" in df.columns and (df["Per_M"] > 0).any():
        df["Rho_nachgerechnet"] = rho_aus_periode(df["Per_M"], df["T_M"])
        df["Rho_diff"] = df["Rho_nachgerechnet"] - df["Rho_M"]
    if "RunT_M" in df.columns and (df["RunT_M"] > 0).any():
        df["C_nachgerechnet"] = c_aus_laufzeit(df["RunT_M"], df["T_M"])
        df["C_diff"] = df["C_nachgerechnet"] - df["C_M"]
    return df


# =====================================================================
# REGRESSION
# =====================================================================

def aktive_terme(df: pd.DataFrame, terme: list[str]) -> list[str]:
    """Verwirft Terme, die im Datensatz keine Variation haben."""
    aktiv = []
    for t in terme:
        if t not in df.columns:
            continue
        if df[t].std(ddof=0) < 1e-12:
            print(f"  Term '{t}' ist konstant und wird verworfen")
            continue
        aktiv.append(t)
    return aktiv


def konditionierung(X: np.ndarray, namen: list[str]) -> dict:
    """Konditionszahl und VIF der Designmatrix.

    Das ist der wichtigste Diagnosewert der ganzen Auswertung. Kommen
    Al, IPA und PG ausschliesslich als SL120 in die Probe, sind ihre
    Anteile streng proportional - dann ist das Gleichungssystem
    singulaer und Aluminium ist NICHT von den Loesemitteln trennbar,
    egal wie gut der Sensor misst.
    """
    Xz = X[:, 1:]                      # ohne Achsenabschnitt
    if Xz.shape[1] == 0 or Xz.shape[0] <= Xz.shape[1]:
        return {"kond": float("nan"), "vif": {}}

    Xs = (Xz - Xz.mean(0)) / np.where(Xz.std(0) > 0, Xz.std(0), 1)
    kond = float(np.linalg.cond(Xs))

    vif = {}
    for i, name in enumerate(namen[1:]):
        rest = np.delete(Xs, i, axis=1)
        if rest.shape[1] == 0:
            vif[name] = 1.0
            continue
        A = np.column_stack([np.ones(len(rest)), rest])
        beta, *_ = np.linalg.lstsq(A, Xs[:, i], rcond=None)
        resid = Xs[:, i] - A @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((Xs[:, i] - Xs[:, i].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif[name] = float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return {"kond": kond, "vif": vif}


def fitte(df: pd.DataFrame, ziel: str, terme: list[str]) -> dict | None:
    """Lineare Regression ziel ~ 1 + terme, mit LOO-Kreuzvalidierung."""
    y = df[ziel].to_numpy(float)
    X = np.column_stack([np.ones(len(df))] +
                        [df[t].to_numpy(float) for t in terme])
    namen = ["const"] + terme
    n, p = X.shape

    if n < p + 1:
        print(f"\n  {ziel}: {n} Punkte fuer {p} Parameter - zu wenig. "
              f"Mindestens {p + 1} Punkte noetig, sinnvoll sind {2 * p + 5}.")
        return None

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - p
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * s2)
        # Leave-one-out ueber die Hat-Matrix, ohne n-fachen Refit
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        loo = resid / (1 - np.clip(h, 0, 1 - 1e-9))
        rmse_loo = float(np.sqrt(np.mean(loo ** 2)))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
        rmse_loo = float("nan")

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")

    return {
        "ziel": ziel, "n": n, "terme": namen,
        "koeff": dict(zip(namen, beta.tolist())),
        "stdfehler": dict(zip(namen, np.asarray(se).tolist())),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "rmse_loo": rmse_loo,
        "r2": r2,
        "resid": resid,
        "kond": konditionierung(X, namen),
    }


def bericht(fit: dict) -> None:
    print(f"\n--- {fit['ziel']} ---")
    print(f"  Punkte: {fit['n']}   R2 = {fit['r2']:.6f}")
    print(f"  {'Term':<8}{'Koeffizient':>16}{'Std.fehler':>14}")
    for t in fit["terme"]:
        print(f"  {t:<8}{fit['koeff'][t]:>16.6g}{fit['stdfehler'][t]:>14.3g}")
    print(f"  RMSE (Fit)  = {fit['rmse']:.5g}")
    print(f"  RMSE (LOO)  = {fit['rmse_loo']:.5g}   <- Startwert fuer R_diag")

    k = fit["kond"]
    if not np.isnan(k["kond"]):
        print(f"  Konditionszahl = {k['kond']:.1f}")
        if k["kond"] > 30:
            print("    ACHTUNG: > 30 bedeutet starke Kollinearitaet.")
            print("    Die Komponenten sind kaum voneinander trennbar.")
        for name, v in k["vif"].items():
            if v > 10:
                print(f"    VIF({name}) = {v:.1f}  - dieser Term ist "
                      f"weitgehend durch die anderen bestimmt")


# =====================================================================
# VORHERSAGE - die eigentliche Anwendungsrichtung
# =====================================================================

def vorhersage(koeff: dict, w_Al=0.0, w_IPA=0.0, w_PG=0.0, w_MG=0.0,
               T=T_REF) -> float:
    """Zusammensetzung [Gew.-%] + Temperatur [degC] -> Messgroesse."""
    werte = {"w_Al": w_Al, "w_IPA": w_IPA, "w_PG": w_PG,
             "w_MG": w_MG, "dT": T - T_REF}
    y = koeff.get("const", 0.0)
    for term, b in koeff.items():
        if term != "const":
            y += b * werte.get(term, 0.0)
    return y


def rezeptur_zu_anteilen(m_SL120, m_Wasser, m_IPA=0.0, m_PG=0.0, m_MG=0.0):
    """Einwaagen [g] -> Massenanteile [Gew.-%]. Fuer die Versuchsplanung."""
    m = m_SL120 + m_Wasser + m_IPA + m_PG + m_MG
    return {
        "w_Al":  100 * SL120_AL * m_SL120 / m,
        "w_IPA": 100 * (SL120_IPA * m_SL120 + m_IPA) / m,
        "w_PG":  100 * (SL120_PG * m_SL120 + m_PG) / m,
        "w_MG":  100 * m_MG / m,
        "w_H2O": 100 * m_Wasser / m,
    }


# =====================================================================
# GRAFIKEN
# =====================================================================

def plots(df: pd.DataFrame, fits: dict, ordner: Path) -> None:
    if not PLOTS:
        print("\nmatplotlib nicht verfuegbar - keine Grafiken erzeugt.")
        return

    for ziel, fit in fits.items():
        if fit is None:
            continue
        terme = [t for t in fit["terme"] if t != "const"]
        n = len(terme) + 1
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), squeeze=False)
        axes = axes[0]

        axes[0].scatter(df[ziel], df[ziel] + fit["resid"] * 0, s=18)
        axes[0].scatter(df[ziel], df[ziel] - fit["resid"], s=18)
        lim = [df[ziel].min(), df[ziel].max()]
        axes[0].plot(lim, lim, "k--", lw=0.8)
        axes[0].set_xlabel("gemessen")
        axes[0].set_ylabel("Modell")
        axes[0].set_title(f"{ziel}: Modell vs. Messung")

        for ax, t in zip(axes[1:], terme):
            ax.scatter(df[t], fit["resid"], s=18)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel(t)
            ax.set_ylabel("Residuum")
            ax.set_title(f"Residuen ueber {t}")

        fig.tight_layout()
        pfad = ordner / f"residuen_{ziel}.png"
        fig.savefig(pfad, dpi=140)
        plt.close(fig)
        print(f"  {pfad.name}")


# =====================================================================
# HAUPTPROGRAMM
# =====================================================================

def main() -> None:
    AUSGABE.mkdir(exist_ok=True)
    if not DATENORDER.exists():
        DATENORDER.mkdir()
        sys.exit(f"Ordner {DATENORDER} wurde angelegt. "
                 f"Bitte CSV-Dateien hineinlegen und erneut starten.")

    print("=" * 66)
    print("EINLESEN")
    print("=" * 66)
    df = lade_messdaten(DATENORDER)

    print("\n" + "=" * 66)
    print("ZUSAMMENSETZUNG UND FILTER")
    print("=" * 66)
    df = berechne_zusammensetzung(df)
    df = filtere(df)
    df = pruefe_werkskonstanten(df)

    if len(df) == 0:
        sys.exit("Keine verwertbaren Datensaetze uebrig.")

    print("\n" + "=" * 66)
    print("KONTROLLE GEGEN DIE WERKSKONSTANTEN")
    print("=" * 66)
    for spalte, einheit in (("Rho_diff", "kg/m3"), ("C_diff", "m/s")):
        if spalte in df.columns:
            print(f"  {spalte:<10} Mittel {df[spalte].mean():+.4f} {einheit}, "
                  f"Streuung {df[spalte].std(ddof=0):.4f}")

    print("\n" + "=" * 66)
    print("UEBERSICHT DER MESSPUNKTE")
    print("=" * 66)
    spalten = ["w_Al", "w_IPA", "w_PG", "w_MG", "T_M",
               "Rho_M", "Rho_S", "C_M", "C_S"]
    print(df[[s for s in spalten if s in df.columns]]
          .describe().T.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 66)
    print("KALIBRIERMODELL")
    print("=" * 66)
    terme = aktive_terme(df, TERME)
    if not terme:
        sys.exit("\nKeine variierenden Terme - alle Proben haben dieselbe "
                 "Zusammensetzung und Temperatur. Es gibt nichts zu fitten.")
    print(f"  Verwendete Terme: {', '.join(terme)}")

    fits = {ziel: fitte(df, ziel, terme) for ziel in ("Rho_M", "C_M")}
    for fit in fits.values():
        if fit:
            bericht(fit)

    if any(f is None for f in fits.values()):
        print("\nZu wenige Messpunkte fuer eine belastbare Kalibrierung.")
        print("Das Skript laeuft trotzdem durch, damit die Kette getestet "
              "werden kann.")

    print("\n" + "=" * 66)
    print("AUSGABE")
    print("=" * 66)

    df.to_csv(AUSGABE / "messpunkte_aufbereitet.csv", index=False)
    print("  messpunkte_aufbereitet.csv")

    ergebnis = {
        "T_ref": T_REF,
        "SL120": {"Al": SL120_AL, "IPA": SL120_IPA, "PG": SL120_PG},
        "modelle": {
            ziel: {
                "n": f["n"], "koeff": f["koeff"], "stdfehler": f["stdfehler"],
                "rmse": f["rmse"], "rmse_loo": f["rmse_loo"], "r2": f["r2"],
                "konditionszahl": f["kond"]["kond"], "vif": f["kond"]["vif"],
            }
            for ziel, f in fits.items() if f
        },
    }
    (AUSGABE / "koeffizienten.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8")
    print("  koeffizienten.json")

    plots(df, fits, AUSGABE)

    # --- Beispielvorhersage ---
    if fits["Rho_M"] and fits["C_M"]:
        print("\n" + "=" * 66)
        print("BEISPIELVORHERSAGE")
        print("=" * 66)
        anteile = rezeptur_zu_anteilen(m_SL120=9.1, m_Wasser=90.9)
        print("  Rezeptur 9.1 g SL120 + 90.9 g Wasser:")
        for k, v in anteile.items():
            print(f"    {k:<7}{v:7.3f} %")
        args = {k: v for k, v in anteile.items() if k != "w_H2O"}
        rho = vorhersage(fits["Rho_M"]["koeff"], T=25.0, **args)
        c = vorhersage(fits["C_M"]["koeff"], T=25.0, **args)
        print(f"  -> Dichte bei 25 degC:  {rho:9.3f} kg/m3 "
              f"(+/- {fits['Rho_M']['rmse_loo']:.3f})")
        print(f"  -> Schall bei 25 degC:  {c:9.3f} m/s   "
              f"(+/- {fits['C_M']['rmse_loo']:.3f})")


if __name__ == "__main__":
    main()