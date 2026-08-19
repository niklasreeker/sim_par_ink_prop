#!/usr/bin/env python3
"""
Visualisierung der Kennfeld-Messdaten
Masterarbeit Reeker - Versuchsstand

Erzeugt drei Grafiken in ergebnisse/:

  1_zeitreihe.png    Zusammensetzung je Messpunkt sowie Temperatur,
                     Dichte und Schallgeschwindigkeit ueber die Zeit,
                     jeweils mit Streuung
  2_streuung.png     Messqualitaet je Punkt - Standardabweichung und
                     Spannweite der drei Kanaele
  3_kalibrierung.png Dichte und Schall ueber dem Al-Gehalt, eingefaerbt
                     nach Temperatur. Erst aussagekraeftig, sobald
                     Proben mit unterschiedlicher Zusammensetzung
                     vermessen wurden.

WICHTIG zum Verstaendnis: Eine Zeile der CSV ist EIN Messpunkt, keine
Einzelmessung. Rho_M ist bereits der Mittelwert ueber 100 Sensorwerte,
Rho_S deren Standardabweichung und Rho_Sp die Spannweite. Die
Fehlerbalken zeigen also die Streuung INNERHALB eines Messpunkts, nicht
die zwischen den Punkten.

Aufruf:  python3 visualisierung.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASIS      = Path(__file__).resolve().parent
DATENORDER = BASIS / "messdaten"
AUSGABE    = BASIS / "ergebnisse"

SL120_AL, SL120_IPA, SL120_PG = 0.20, 0.40, 0.40

FARBEN = {
    "Al":  "#8c4a2f",
    "IPA": "#3d6b8c",
    "PG":  "#5c8c5c",
    "MG":  "#8c7a3d",
    "rho": "#1f4e79",
    "c":   "#a33b3b",
    "T":   "#4a7c59",
}


# ---------------------------------------------------------------------

def lade() -> pd.DataFrame:
    dateien = sorted(DATENORDER.glob("*.csv"))
    if not dateien:
        sys.exit(f"Keine CSV-Dateien in {DATENORDER}")

    teile = []
    for f in dateien:
        d = pd.read_csv(f, comment="/", skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        teile.append(d)
    df = pd.concat(teile, ignore_index=True)

    # Zusammensetzung
    m = (df["m_SL120"] + df["m_Wasser"] + df["m_IPA"]
         + df["m_PG"] + df["m_MG"]).replace(0, np.nan)
    df["w_Al"]  = 100 * SL120_AL * df["m_SL120"] / m
    df["w_IPA"] = 100 * (SL120_IPA * df["m_SL120"] + df["m_IPA"]) / m
    df["w_PG"]  = 100 * (SL120_PG * df["m_SL120"] + df["m_PG"]) / m
    df["w_MG"]  = 100 * df["m_MG"] / m
    df["w_H2O"] = 100 * df["m_Wasser"] / m
    df[["w_Al", "w_IPA", "w_PG", "w_MG", "w_H2O"]] = \
        df[["w_Al", "w_IPA", "w_PG", "w_MG", "w_H2O"]].fillna(0.0)

    # Zeitachse. Steht die CPU-Uhr falsch - erkennbar an einem
    # unplausiblen Jahr - wird auf den Messpunktindex ausgewichen.
    df["x"] = range(1, len(df) + 1)
    df["x_label"] = "Messpunkt"
    if {"Date", "UTC Time"}.issubset(df.columns):
        t = pd.to_datetime(df["Date"].astype(str).str.strip() + " "
                           + df["UTC Time"].astype(str).str.strip(),
                           errors="coerce")
        if t.notna().all() and t.dt.year.between(2024, 2100).all():
            df["x"] = t
            df["x_label"] = "Zeit (UTC)"
        elif t.notna().all():
            print(f"  Hinweis: CPU-Uhr steht auf {t.dt.year.iloc[0]}. "
                  f"Zeitachse durch Messpunktindex ersetzt.")
    return df


# ---------------------------------------------------------------------

def kanal(ax, df, mittel, std, span, farbe, label, einheit):
    """Ein Messkanal mit Streuungsband und Fehlerbalken."""
    x, y = df["x"], df[mittel]

    if span in df.columns:
        ax.fill_between(x, y - df[span] / 2, y + df[span] / 2,
                        color=farbe, alpha=0.18, lw=0,
                        label="Spannweite im Fenster")
    if std in df.columns:
        ax.errorbar(x, y, yerr=df[std], fmt="o-", color=farbe,
                    ms=5, lw=1.4, capsize=3, elinewidth=1.2,
                    label="Mittelwert $\\pm$ Standardabw.")
    else:
        ax.plot(x, y, "o-", color=farbe, ms=5, lw=1.4)

    ax.set_ylabel(f"{label}\n[{einheit}]")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)


def grafik_zeitreihe(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1, 1, 1]})

    # --- Zusammensetzung ---
    ax = axes[0]
    unten = np.zeros(len(df))
    for komp, farbe in (("Al", FARBEN["Al"]), ("IPA", FARBEN["IPA"]),
                        ("PG", FARBEN["PG"]), ("MG", FARBEN["MG"])):
        werte = df[f"w_{komp}"].to_numpy()
        if werte.max() <= 0:
            continue
        ax.bar(df["x"], werte, bottom=unten, color=farbe, label=komp,
               width=0.6 if df["x_label"] == "Messpunkt" else None)
        unten += werte

    if unten.max() <= 0:
        ax.text(0.5, 0.5, "Keine Aktivkomponenten - reines Wasser",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#666")
    ax.set_ylabel("Aktivkomponenten\n[Gew.-%]")
    ax.set_title("Zusammensetzung je Messpunkt (Wasser als Rest, nicht dargestellt)",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    if unten.max() > 0:
        ax.legend(fontsize=8, ncol=4, loc="upper left")

    kanal(axes[1], df, "T_M", "T_S", None,
          FARBEN["T"], "Temperatur", "°C")
    kanal(axes[2], df, "Rho_M", "Rho_S", "Rho_Sp",
          FARBEN["rho"], "Dichte", "kg/m³")
    kanal(axes[3], df, "C_M", "C_S", "C_Sp",
          FARBEN["c"], "Schallgeschw.", "m/s")

    axes[-1].set_xlabel(df["x_label"].iloc[0])
    if df["x_label"].iloc[0] == "Zeit (UTC)":
        fig.autofmt_xdate()

    fig.suptitle("Kennfeldaufnahme Pico 3000 / L-Com 5500 – Übersicht",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(AUSGABE / "1_zeitreihe.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------

def grafik_streuung(df: pd.DataFrame) -> None:
    kanaele = [("Dichte", "Rho_S", "Rho_Sp", "kg/m³", FARBEN["rho"]),
               ("Schallgeschw.", "C_S", "C_Sp", "m/s", FARBEN["c"]),
               ("Temperatur", "T_S", None, "°C", FARBEN["T"])]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    breite = 0.38
    idx = np.arange(len(df))

    for ax, (name, s, sp, einheit, farbe) in zip(axes, kanaele):
        ax.bar(idx - breite / 2, df[s], breite, color=farbe,
               label="Standardabw.")
        if sp and sp in df.columns:
            ax.bar(idx + breite / 2, df[sp], breite, color=farbe,
                   alpha=0.45, label="Spannweite")
        ax.set_title(f"{name} [{einheit}]", fontsize=10)
        ax.set_xlabel("Messpunkt")
        ax.set_xticks(idx)
        ax.set_xticklabels(df["Nr"].astype(int) if "Nr" in df.columns
                           else idx + 1, fontsize=8)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
        ax.legend(fontsize=7)

    fig.suptitle("Messqualität je Punkt – Streuung innerhalb des "
                 "Mittelungsfensters (100 Werte)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(AUSGABE / "2_streuung.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------

def grafik_kalibrierung(df: pd.DataFrame) -> None:
    if df["w_Al"].std(ddof=0) < 1e-9:
        print("  3_kalibrierung.png übersprungen: Al-Gehalt konstant, "
              "es gibt noch nichts aufzutragen.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (mittel, std, name, einheit) in zip(
            axes, [("Rho_M", "Rho_S", "Dichte", "kg/m³"),
                   ("C_M", "C_S", "Schallgeschwindigkeit", "m/s")]):
        sc = ax.scatter(df["w_Al"], df[mittel], c=df["T_M"],
                        cmap="coolwarm", s=55, edgecolor="k", lw=0.4,
                        zorder=3)
        ax.errorbar(df["w_Al"], df[mittel], yerr=df[std], fmt="none",
                    ecolor="#555", elinewidth=1, capsize=3, zorder=2)
        ax.set_xlabel("Aluminiumgehalt [Gew.-%]")
        ax.set_ylabel(f"{name} [{einheit}]")
        ax.grid(alpha=0.25, lw=0.6)
        fig.colorbar(sc, ax=ax, label="Temperatur [°C]")

    fig.suptitle("Kalibrieransicht – Messgrößen über dem Aluminiumgehalt",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(AUSGABE / "3_kalibrierung.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------

def tabelle(df: pd.DataFrame) -> None:
    print("\nZusammensetzung je Messpunkt [Gew.-%]")
    print("-" * 78)
    kopf = f"{'Nr':>4} {'Probe':>6} {'Al':>7} {'IPA':>7} {'PG':>7} " \
           f"{'MG':>7} {'H2O':>8} {'m_ges [g]':>10}"
    print(kopf)
    for _, r in df.iterrows():
        m = (r["m_SL120"] + r["m_Wasser"] + r["m_IPA"]
             + r["m_PG"] + r["m_MG"])
        print(f"{int(r.get('Nr', 0)):>4} {int(r.get('ProbeNr', 0)):>6} "
              f"{r['w_Al']:>7.3f} {r['w_IPA']:>7.3f} {r['w_PG']:>7.3f} "
              f"{r['w_MG']:>7.3f} {r['w_H2O']:>8.3f} {m:>10.3f}")

    print("\nMesswerte je Messpunkt")
    print("-" * 78)
    print(f"{'Nr':>4} {'T [°C]':>9} {'rho [kg/m³]':>13} {'±':>8} "
          f"{'c [m/s]':>11} {'±':>8} {'N':>4} {'gültig':>7}")
    for _, r in df.iterrows():
        print(f"{int(r.get('Nr', 0)):>4} {r['T_M']:>9.4f} "
              f"{r['Rho_M']:>13.4f} {r['Rho_S']:>8.4f} "
              f"{r['C_M']:>11.3f} {r['C_S']:>8.4f} "
              f"{int(r.get('N', 0)):>4} "
              f"{'ja' if r.get('Gueltig', 0) == 1 else 'NEIN':>7}")


# ---------------------------------------------------------------------

def main() -> None:
    AUSGABE.mkdir(exist_ok=True)
    df = lade()
    print(f"{len(df)} Messpunkte eingelesen")

    tabelle(df)

    print("\nGrafiken:")
    grafik_zeitreihe(df)
    print("  1_zeitreihe.png")
    grafik_streuung(df)
    print("  2_streuung.png")
    grafik_kalibrierung(df)


if __name__ == "__main__":
    main()