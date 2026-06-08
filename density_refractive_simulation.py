import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PHYSIKALISCHE GRUNDDATEN (bei 25°C)
# ==========================================
n_w = 1.3325  # Brechungsindex Wasser
rho_w = 0.9970  # Dichte Wasser [g/cm³]
n_ipa = 1.3753  # Brechungsindex Isopropanol
rho_ipa = 0.7850  # Dichte Isopropanol [g/cm³]
rho_al = 2.70  # Dichte Aluminiumpigment [g/cm³]

# Spezifische Refraktion (Gladstone-Dale)
R_w = (n_w - 1) / rho_w
R_ipa = (n_ipa - 1) / rho_ipa

# Molare Refraktivität pro Gramm (Lorentz-Lorenz)
L_w = ((n_w ** 2 - 1) / (n_w ** 2 + 2)) / rho_w
L_ipa = ((n_ipa ** 2 - 1) / (n_ipa ** 2 + 2)) / rho_ipa


# ==========================================
# FUNKTIONEN
# ==========================================
def get_matrix_density(w_ipa_norm):
    """
    Berechnet die reale Dichte des IPA/Wasser-Gemisches inkl. Volumenkontraktion.
    """
    p = np.polyfit([0, 0.1005, 0.2250], [0.9970, 0.9806, 0.9628], 2)
    return np.polyval(p, w_ipa_norm)


def berechne_tinte(w_al, w_ipa):
    """
    Berechnet Dichte und Brechungsindizes der Tinte.
    """
    # Matrix normieren
    w_matrix = 1.0 - w_al
    w_ipa_norm = w_ipa / w_matrix
    w_w_norm = (1.0 - w_al - w_ipa) / w_matrix

    # Dichte
    rho_matrix = get_matrix_density(w_ipa_norm)
    rho_tot = 1.0 / ((w_al / rho_al) + (w_matrix / rho_matrix))

    # Gladstone-Dale
    R_mix = w_ipa_norm * R_ipa + w_w_norm * R_w
    n_gd = R_mix * rho_matrix + 1.0

    # Lorentz-Lorenz
    L_mix = w_ipa_norm * L_ipa + w_w_norm * L_w
    n_ll = np.sqrt((1 + 2 * (L_mix * rho_matrix)) / (1 - (L_mix * rho_matrix)))

    return rho_tot, n_gd, n_ll


# ==========================================
# 1. ARBEITSPUNKTE (4er-Design)
# ==========================================
arbeitspunkte = [
    (0.005, 0.22, "0,5% Al | 22% IPA"),
    (0.022, 0.22, "2,2% Al | 22% IPA"),
    (0.005, 0.10, "0,5% Al | 10% IPA"),
    (0.022, 0.10, "2,2% Al | 10% IPA")
]

print("=== ARBEITSPUNKTE BEI 25°C ===")
for al, ipa, name in arbeitspunkte:
    rho_t, n_gd, n_ll = berechne_tinte(al, ipa)
    print(f"\n{name}")
    print(f"  Dichte: {rho_t:.4f} g/cm³")
    print(f"  n (GD): {n_gd:.4f}")
    print(f"  n (LL): {n_ll:.4f}")


# ==========================================
# 2. KONZENTRATIONSGITTER
# ==========================================
al_levels = np.linspace(0.005, 0.022, 10)
ipa_levels = np.linspace(0.10, 0.22, 10)

plt.figure(figsize=(10, 7))

# Iso-IPA Linien (rot)
for ipa in ipa_levels:
    x_vals, y_vals = [], []
    for al in al_levels:
        rho_t, n_gd, _ = berechne_tinte(al, ipa)
        x_vals.append(rho_t)
        y_vals.append(n_gd)
    plt.plot(x_vals, y_vals, 'r-', linewidth=0.6, alpha=0.8)

# Iso-Al Linien (blau)
for al in al_levels:
    x_vals, y_vals = [], []
    for ipa in ipa_levels:
        rho_t, n_gd, _ = berechne_tinte(al, ipa)
        x_vals.append(rho_t)
        y_vals.append(n_gd)
    plt.plot(x_vals, y_vals, 'b--', linewidth=0.6, alpha=0.8)


# ==========================================
# 3. ARBEITSPUNKTE MARKIEREN
# ==========================================
def mark_point(al, ipa, label, dx=0, dy=0):
    r, n, _ = berechne_tinte(al, ipa)
    plt.plot(r, n, 'ko', markersize=4)
    plt.text(r + dx, n + dy, label, fontsize=9,
             bbox=dict(facecolor='bisque', edgecolor='none', alpha=0.7))


mark_point(0.005, 0.22, "0.5% Al\n22% IPA", -0.004, -0.0005)
mark_point(0.022, 0.22, "2.2% Al\n22% IPA", 0.0005, -0.0005)
mark_point(0.005, 0.10, "0.5% Al\n10% IPA", -0.004, 0.0005)
mark_point(0.022, 0.10, "2.2% Al\n10% IPA", 0.0005, 0.0005)


# ==========================================
# 4. PLOT SETTINGS
# ==========================================
plt.xlabel("Dichte [g/cm³]", fontweight='bold')
plt.ylabel("Brechungsindex [-]", fontweight='bold')
plt.title("Konzentrationsnetz: Al + IPA + Wasser (25°C)", fontsize=14)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

plt.show()