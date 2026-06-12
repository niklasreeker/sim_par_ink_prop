from Archiv.ink_sound import InkSoundCalculator

calc = InkSoundCalculator(tables_dir="../tables_parameters")
c = calc.sound_velocity(pct_al=1.82, pct_ipa=3.64, pct_pg=3.64, temperature=25)
# oder den vollen Report mit Dichte, Volumenanteilen, Warnungen:
result = calc.calculate(pct_al=1.82, pct_ipa=3.64, pct_pg=3.64, temperature=25)
print(result)

from Archiv.ink_viscosity import InkViscosityModel
model = InkViscosityModel(
    "../tables_parameters",
    suspension_model="batchelor",   # oder "einstein" / "krieger-dougherty"
    einstein_coeff=2.5,
    batchelor_coeff=7.2,
)
res = model.estimate(water=85, ipa=5, pg=7, aluminum=3, temperature_C=25)
print(res["viscosity_mPas"])   # -> ~1.68 mPa·s