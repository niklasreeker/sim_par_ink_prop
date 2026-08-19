#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Rechner: Dichte und Schallgeschwindigkeit von reinem Wasser bei Normaldruck
================================================================================

Berechnet die Dichte rho [kg/m^3] und die Schallgeschwindigkeit c [m/s] von
luftfreiem, reinem Wasser (VSMOW-Isotopenzusammensetzung) beim Normaldruck
p = 101,325 kPa = 0,101325 MPa als Funktion der Temperatur.

Gueltiger Fluessigkeitsbereich bei 1 atm:  0 C  <=  t  <  99,974 C
(oberhalb von ~99,974 C siedet Wasser bei Normaldruck -> Dampfphase).

--------------------------------------------------------------------------------
VERWENDETE LITERATURQUELLEN (peer-reviewed)
--------------------------------------------------------------------------------

DICHTE
  [1] CIPM / Tanaka et al. (2001):
      M. Tanaka, G. Girard, R. Davis, A. Peuto, N. Bignell,
      "Recommended table for the density of water between 0 C and 40 C based on
       recent experimental reports", Metrologia 38 (2001) 301-309.
      -> Metrologischer Standard, kleinste Unsicherheit; Bereich 0-40 C.
      -> geschaetzte Unsicherheit ca. +/- 0,00084 kg/m^3 (rund 1 ppm).

  [2] Kell (1975):
      G. S. Kell, "Density, thermal expansivity, and compressibility of liquid
       water from 0 to 150 C", J. Chem. Eng. Data 20 (1975) 97-105.
      -> Bewaehrte Polynomgleichung, breiter Bereich 0-150 C (bei 1 atm bis Sieden).
      -> Unsicherheit im Bereich weniger ppm.

SCHALLGESCHWINDIGKEIT
  [3] Marczak (1997):
      W. Marczak, "Water as a standard in the measurements of speed of sound in
       liquids", J. Acoust. Soc. Am. 102 (1997) 2776-2779.
      -> Polynom-Refit hochwertiger Daten; Bereich 0-95 C; Streuung ca. 0,05 m/s.

  [4] Del Grosso & Mader (1972):
      V. A. Del Grosso, C. W. Mader, "Speed of sound in pure water",
      J. Acoust. Soc. Am. 52 (1972) 1442-1446.
      -> Klassische Referenzmessung; Bereich 0-95 C.

INTERNATIONALER GOLDSTANDARD (Dichte UND Schallgeschwindigkeit, optional)
  [5] IAPWS-95 / Wagner & Pruss (2002):
      W. Wagner, A. Pruss, "The IAPWS Formulation 1995 for the Thermodynamic
       Properties of Ordinary Water Substance for General and Scientific Use",
       J. Phys. Chem. Ref. Data 31 (2002) 387-535.
      -> Wird genutzt, falls das Paket 'iapws' installiert ist (pip install iapws).
         Liefert beide Groessen konsistent aus einer Zustandsgleichung.

--------------------------------------------------------------------------------
Alle drei unabhaengigen Methoden (Korrelationen + IAPWS-95) stimmen ueberein auf
  Dichte:              < ~0,003 kg/m^3
  Schallgeschwindigkeit: < ~0,06 m/s
ueber den gesamten Gueltigkeitsbereich.
================================================================================
"""

from __future__ import annotations

NORMALDRUCK_MPA = 0.101325          # Normaldruck in MPa (= 101,325 kPa = 1 atm)
T_SIEDE_1ATM = 99.9743              # Siedetemperatur bei 1 atm in C (IAPWS-95)

# Optionaler Goldstandard IAPWS-95 (nur falls installiert)
try:
    from iapws import IAPWS95
    _HAT_IAPWS = True
except Exception:
    _HAT_IAPWS = False


# =============================================================================
#  DICHTE
# =============================================================================

def dichte_cipm(t: float) -> float:
    """
    Dichte von luftfreiem Wasser [kg/m^3] nach CIPM/Tanaka (2001) [1].
    Gueltig: 0 C <= t <= 40 C. Unsicherheit ca. +/- 0,00084 kg/m^3.
    t : Temperatur in Grad Celsius.
    """
    a1 = -3.983035     # C
    a2 = 301.797       # C
    a3 = 522528.9      # C^2
    a4 = 69.34881      # C
    a5 = 999.974950    # kg/m^3
    return a5 * (1.0 - (t + a1) ** 2 * (t + a2) / (a3 * (t + a4)))


def dichte_kell(t: float) -> float:
    """
    Dichte von luftfreiem Wasser [kg/m^3] bei 1 atm nach Kell (1975) [2].
    Gueltig: 0 C <= t <= 150 C (fluessig bei 1 atm nur bis Sieden).
    t : Temperatur in Grad Celsius.
    """
    num = (999.83952
           + 16.945176 * t
           - 7.9870401e-3 * t ** 2
           - 46.170461e-6 * t ** 3
           + 105.56302e-9 * t ** 4
           - 280.54253e-12 * t ** 5)
    den = 1.0 + 16.87985e-3 * t
    return num / den


def dichte_iapws(t: float) -> float:
    """Dichte [kg/m^3] aus IAPWS-95 [5] bei Normaldruck. Benoetigt Paket 'iapws'."""
    if not _HAT_IAPWS:
        raise RuntimeError("Paket 'iapws' nicht installiert (pip install iapws).")
    return IAPWS95(T=273.15 + t, P=NORMALDRUCK_MPA).rho


def dichte(t: float, methode: str = "auto") -> float:
    """
    Beste Schaetzung der Dichte [kg/m^3] bei Normaldruck.

    methode:
      "auto"  -> IAPWS-95 falls verfuegbar, sonst beste Korrelation:
                 CIPM (0-40 C) bzw. Kell (>40 C).
      "cipm", "kell", "iapws" -> jeweilige Methode erzwingen.
    """
    _pruefe_bereich(t)
    if methode == "cipm":
        return dichte_cipm(t)
    if methode == "kell":
        return dichte_kell(t)
    if methode == "iapws":
        return dichte_iapws(t)
    if methode == "auto":
        if _HAT_IAPWS:
            return dichte_iapws(t)
        return dichte_cipm(t) if t <= 40.0 else dichte_kell(t)
    raise ValueError(f"Unbekannte Methode: {methode!r}")


# =============================================================================
#  SCHALLGESCHWINDIGKEIT
# =============================================================================

def schall_marczak(t: float) -> float:
    """
    Schallgeschwindigkeit [m/s] in reinem Wasser bei 1 atm nach Marczak (1997) [3].
    Gueltig: 0 C <= t <= 95 C.
    t : Temperatur in Grad Celsius.
    """
    return (1.402385e3
            + 5.038813 * t
            - 5.799136e-2 * t ** 2
            + 3.287156e-4 * t ** 3
            - 1.398845e-6 * t ** 4
            + 2.787860e-9 * t ** 5)


def schall_delgrosso(t: float) -> float:
    """
    Schallgeschwindigkeit [m/s] nach Del Grosso & Mader (1972) [4].
    Gueltig: 0 C <= t <= 95 C.
    t : Temperatur in Grad Celsius.
    """
    c = (1402.388, 5.03830, -5.81090e-2, 3.3432e-4, -1.47797e-6, 3.1419e-9)
    return sum(c[i] * t ** i for i in range(6))


def schall_iapws(t: float) -> float:
    """Schallgeschwindigkeit [m/s] aus IAPWS-95 [5]. Benoetigt Paket 'iapws'."""
    if not _HAT_IAPWS:
        raise RuntimeError("Paket 'iapws' nicht installiert (pip install iapws).")
    return IAPWS95(T=273.15 + t, P=NORMALDRUCK_MPA).w


def schallgeschwindigkeit(t: float, methode: str = "auto") -> float:
    """
    Beste Schaetzung der Schallgeschwindigkeit [m/s] bei Normaldruck.

    methode:
      "auto"    -> IAPWS-95 falls verfuegbar, sonst Marczak.
      "marczak", "delgrosso", "iapws" -> jeweilige Methode erzwingen.
    """
    _pruefe_bereich(t)
    if methode == "marczak":
        return schall_marczak(t)
    if methode == "delgrosso":
        return schall_delgrosso(t)
    if methode == "iapws":
        return schall_iapws(t)
    if methode == "auto":
        return schall_iapws(t) if _HAT_IAPWS else schall_marczak(t)
    raise ValueError(f"Unbekannte Methode: {methode!r}")


# =============================================================================
#  HILFSFUNKTIONEN
# =============================================================================

def _pruefe_bereich(t: float) -> None:
    if t < 0.0:
        raise ValueError(f"t = {t} C: unterhalb 0 C ist Wasser bei 1 atm gefroren "
                         "bzw. ausserhalb des Gueltigkeitsbereichs.")
    if t >= T_SIEDE_1ATM:
        raise ValueError(f"t = {t} C: bei Normaldruck siedet Wasser bei "
                         f"{T_SIEDE_1ATM} C -> keine Fluessigphase mehr.")


def bericht(t: float) -> str:
    """Formatierter Ergebnisbericht mit allen Methoden fuer eine Temperatur."""
    _pruefe_bereich(t)
    zeilen = []
    zeilen.append("=" * 60)
    zeilen.append(f"  Wasser bei t = {t:g} C, p = 101,325 kPa (Normaldruck)")
    zeilen.append("=" * 60)

    zeilen.append("  DICHTE [kg/m^3]")
    if t <= 40.0:
        zeilen.append(f"    CIPM/Tanaka 2001 : {dichte_cipm(t):11.4f}   (Standard 0-40 C)")
    zeilen.append(f"    Kell 1975        : {dichte_kell(t):11.4f}")
    if _HAT_IAPWS:
        zeilen.append(f"    IAPWS-95         : {dichte_iapws(t):11.4f}   (Referenz)")
    zeilen.append(f"    -> Empfehlung    : {dichte(t):11.4f}")

    zeilen.append("")
    zeilen.append("  SCHALLGESCHWINDIGKEIT [m/s]")
    if t <= 95.0:
        zeilen.append(f"    Marczak 1997     : {schall_marczak(t):11.3f}")
        zeilen.append(f"    Del Grosso 1972  : {schall_delgrosso(t):11.3f}")
    if _HAT_IAPWS:
        zeilen.append(f"    IAPWS-95         : {schall_iapws(t):11.3f}   (Referenz)")
    zeilen.append(f"    -> Empfehlung    : {schallgeschwindigkeit(t):11.3f}")
    zeilen.append("=" * 60)
    if not _HAT_IAPWS:
        zeilen.append("  Hinweis: 'iapws' nicht installiert -> IAPWS-95-Referenz")
        zeilen.append("           nicht verfuegbar (pip install iapws).")
    return "\n".join(zeilen)


# =============================================================================
#  KOMMANDOZEILE / INTERAKTIV
# =============================================================================

def main() -> None:
    import sys
    if len(sys.argv) > 1:
        # Nutzung:  python wasser_rechner.py 25   (oder mehrere Temperaturen)
        for arg in sys.argv[1:]:
            try:
                print(bericht(float(arg.replace(",", "."))))
                print()
            except ValueError as e:
                print(f"Fehler bei '{arg}': {e}")
        return

    print("Rechner: Dichte & Schallgeschwindigkeit von Wasser bei Normaldruck")
    print("Temperatur in Grad Celsius eingeben (0 bis 99,97 C), 'q' zum Beenden.")
    if not _HAT_IAPWS:
        print("(Tipp: 'pip install iapws' aktiviert die IAPWS-95-Referenz.)")
    while True:
        try:
            eingabe = input("\nt [C] = ").strip().replace(",", ".")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if eingabe.lower() in ("q", "quit", "exit", ""):
            break
        try:
            print(bericht(float(eingabe)))
        except ValueError as e:
            print(f"  Ungueltig: {e}")


if __name__ == "__main__":
    main()