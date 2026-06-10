import os  # <--- NEU: Wird benötigt, um den Ordner zu erstellen
import numpy as np
import matplotlib.pyplot as plt

# Import your custom calculator modules
from ink_density import InkDensityCalculator
from ink_refractive import InkRefractiveCalculator

# ==========================================
# 1. INITIALIZE CALCULATORS
# ==========================================
print("Loading empirical tables into memory...")
try:
    density_calc = InkDensityCalculator(tables_dir="tables_parameters")
    # Pass the existing density_calc to avoid loading density tables twice
    optical_calc = InkRefractiveCalculator(tables_dir="tables_parameters", density_calculator=density_calc)
except Exception as e:
    print(f"Error initializing calculators: {e}")
    print("Please ensure 'ink_density.py', 'ink_refractive.py' and the 'tables_parameters' folder are present.")
    exit()

# ==========================================
# 2. WRAPPER FUNCTION FOR INK PROPERTIES
# ==========================================
def get_ink_properties(pct_al, pct_ipa, target_temp=25):
    """
    Calculates total density and the refractive index of the liquid matrix.
    Uses the empirical tables and pseudo-binary approximations.
    """
    # 1. Total density of the ink
    rho_total = density_calc.calculate_density(pct_al=pct_al, pct_ipa=pct_ipa, target_temp=target_temp)

    # 2. Refractive index of the liquid matrix
    # The aluminium pigment does not refract light, so we normalize the
    # liquid phase to 100% to find the true concentration of the solvent matrix.
    pct_liquid_total = 100.0 - pct_al
    pct_ipa_in_liquid = (pct_ipa / pct_liquid_total) * 100.0

    n_matrix = optical_calc.calculate_refractive_index(pct_ipa=pct_ipa_in_liquid, pct_pg=0.0, target_temp=target_temp)

    return rho_total, n_matrix


# ==========================================
# 3. OPERATING POINTS (4-Point Design)
# ==========================================
# Format: (Aluminum_%, IPA_%, Label)
operating_points = [
    (0.5, 22.0, "0.5% Al | 22% IPA"),
    (2.2, 22.0, "2.2% Al | 22% IPA"),
    (0.5, 10.0, "0.5% Al | 10% IPA"),
    (2.2, 10.0, "2.2% Al | 10% IPA")
]

target_temperature = 25  # Celsius

print(f"\n=== OPERATING POINTS AT {target_temperature}°C ===")
for al_pct, ipa_pct, name in operating_points:
    rho_t, n_val = get_ink_properties(al_pct, ipa_pct, target_temperature)
    print(f"{name}:")
    print(f"  Density:          {rho_t:.4f} g/cm³")
    print(f"  Refractive Index: {n_val:.4f}")


# ==========================================
# 4. CONCENTRATION GRID CALCULATION
# ==========================================
# Define grid boundaries in percent (%)
al_levels = np.linspace(0.5, 2.2, 10)
ipa_levels = np.linspace(10.0, 22.0, 10)

plt.figure(figsize=(10, 7))

# Plot Iso-IPA lines (constant IPA, varying Aluminum) - Red Solid Lines
for ipa in ipa_levels:
    x_vals, y_vals = [], []
    for al in al_levels:
        rho_t, n_val = get_ink_properties(al, ipa, target_temperature)
        x_vals.append(rho_t)
        y_vals.append(n_val)
    plt.plot(x_vals, y_vals, 'r-', linewidth=0.6, alpha=0.8)

# Plot Iso-Al lines (constant Aluminum, varying IPA) - Blue Dashed Lines
for al in al_levels:
    x_vals, y_vals = [], []
    for ipa in ipa_levels:
        rho_t, n_val = get_ink_properties(al, ipa, target_temperature)
        x_vals.append(rho_t)
        y_vals.append(n_val)
    plt.plot(x_vals, y_vals, 'b--', linewidth=0.6, alpha=0.8)


# ==========================================
# 5. MARK OPERATING POINTS ON PLOT
# ==========================================
def mark_point(al_pct, ipa_pct, label, dx=0, dy=0):
    """Helper function to place markers and text boxes."""
    r, n = get_ink_properties(al_pct, ipa_pct, target_temperature)
    plt.plot(r, n, 'ko', markersize=4)
    plt.text(r + dx, n + dy, label, fontsize=9,
             bbox=dict(facecolor='bisque', edgecolor='none', alpha=0.7))

# Applying specific coordinate offsets for clean text placement
mark_point(0.5, 22.0, "0.5% Al\n22% IPA", dx=-0.004, dy=-0.0005)
mark_point(2.2, 22.0, "2.2% Al\n22% IPA", dx=0.0005, dy=-0.0005)
mark_point(0.5, 10.0, "0.5% Al\n10% IPA", dx=-0.004, dy=0.0005)
mark_point(2.2, 10.0, "2.2% Al\n10% IPA", dx=0.0005, dy=0.0005)


# ==========================================
# 6. PLOT SETTINGS & RENDERING
# ==========================================
plt.xlabel("Density [g/cm³]", fontweight='bold')
plt.ylabel("Refractive Index [-]", fontweight='bold')
plt.title(f"Concentration Grid: Al + IPA + Water ({target_temperature}°C)", fontsize=14)

# Custom legend for the grid lines
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='r', lw=1.5, label='Iso-IPA lines (Constant Solvent)'),
    Line2D([0], [0], color='b', lw=1.5, linestyle='--', label='Iso-Al lines (Constant Pigment)')
]
plt.legend(handles=legend_elements, loc='best')

plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

# ==========================================
# NEU: ORDNER ERSTELLEN UND PLOT SPEICHERN
# ==========================================
output_folder = "plots"
# Erstellt den Ordner, falls er noch nicht existiert (wirft keinen Fehler, wenn er schon da ist)
os.makedirs(output_folder, exist_ok=True)

# Dateipfad zusammenbauen
filepath = os.path.join(output_folder, "density_refractive.png")

print(f"\nSaving plot to {filepath}...")
# WICHTIG: savefig() muss VOR show() aufgerufen werden!
plt.savefig(filepath, dpi=300, bbox_inches='tight')

print("Rendering plot on screen...")
plt.show()