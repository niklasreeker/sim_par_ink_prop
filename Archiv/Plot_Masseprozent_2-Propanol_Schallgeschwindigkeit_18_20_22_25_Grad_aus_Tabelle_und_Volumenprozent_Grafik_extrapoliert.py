"""
Diagramm: Schallgeschwindigkeit vs. Massenprozent 2-Propanol (IPA)

4 Geraden:
  - 18 °C, 20 °C, 22 °C: umgerechnete Geradengleichungen (x = Massenprozent)
  - 25 °C: linearer Fit aus den Tabellenwerten (298,15 K) im niedrigen
    Konzentrationsbereich (0 bis ca. 9,4 Massen-%), da nur dort der
    Verlauf annähernd linear ist.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Geradengleichungen (x in Massenprozent, y in m/s)
# ----------------------------------------------------------------------
geraden = {
    "18 °C: y = 9,93x + 1476,3": (9.93, 1476.3, "tab:blue"),
    "20 °C: y = 9,28x + 1482,5": (9.28, 1482.5, "tab:green"),
    "22 °C: y = 8,98x + 1487,8": (8.98, 1487.8, "tab:orange"),
}

# Gültigkeitsbereich der umgerechneten Gleichungen: ca. 5,5 bis 8,1 Massen-%
x_gueltig = np.linspace(5.5, 8.1, 100)

# ----------------------------------------------------------------------
# 2) Tabellenwerte bei 298,15 K (25 °C) — Massenprozent / m/s
#    (nur der niedrige Konzentrationsbereich für den linearen Fit)
# ----------------------------------------------------------------------
massenprozent_tab = np.array([0.00, 3.33, 6.35, 9.36])
schallgeschw_tab = np.array([1496.69, 1525.88, 1552.31, 1577.40])

# Linearer Fit: y = m*x + b
m_fit, b_fit = np.polyfit(massenprozent_tab, schallgeschw_tab, 1)
print(f"Fit für 25 °C: y = {m_fit:.2f}x + {b_fit:.1f}")

# ----------------------------------------------------------------------
# 3) Plot
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Geraden für 18, 20, 22 °C (durchgezogen im Gültigkeitsbereich,
# gestrichelt als Extrapolation darüber hinaus)
x_extra = np.linspace(0, 10, 100)
for label, (m, b, farbe) in geraden.items():
    plt.plot(x_extra, m * x_extra + b, linestyle="--", color=farbe,
             alpha=0.4, linewidth=1)
    plt.plot(x_gueltig, m * x_gueltig + b, color=farbe,
             linewidth=2, label=label)

# Gefittete Gerade für 25 °C
plt.plot(x_extra, m_fit * x_extra + b_fit, color="tab:red", linewidth=2,
         label=f"25 °C (Fit): y = {m_fit:.2f}x + {b_fit:.1f}")

# Tabellenpunkte für 25 °C zur Kontrolle einzeichnen
plt.scatter(massenprozent_tab, schallgeschw_tab, color="tab:red",
            zorder=5, marker="o", label="Tabellenwerte 25 °C")

# Achsen, Titel, Gitter
plt.title("Schallgeschwindigkeit in Wasser-2-Propanol-Gemischen\n"
          "(Geraden für 18/20/22 °C, gültig 5,5–8,1 Massen-%, "
          "gestrichelt = Extrapolation)")
plt.xlabel("Massenprozent 2-Propanol (Gew.-%)")
plt.ylabel("Schallgeschwindigkeit u (m/s)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 10)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("schallgeschwindigkeit_geraden.png", dpi=150)
plt.show()
