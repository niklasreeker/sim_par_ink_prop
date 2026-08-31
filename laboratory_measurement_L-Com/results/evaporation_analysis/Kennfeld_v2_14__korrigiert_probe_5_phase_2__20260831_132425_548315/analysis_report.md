# Evaporation analysis

Source: Kennfeld_v2 (14)_korrigiert.csv
Sample: 5; phase: 2; duration: 193.77 min

| Model | IPA [g/h] | Water [g/h] | Total [g/h] | Density RMSE [kg/m^3] | Sound RMSE [m/s] |
|---|---:|---:|---:|---:|---:|
| none | 0.0000 | 0.0000 | 0.0000 | 0.1269 | 0.5216 |
| ipa | 0.6976 | 0.0000 | 0.6976 | 0.0396 | 0.1618 |
| mixed | 0.6925 | 0.6464 | 1.3389 | 0.0392 | 0.1591 |

## Interpretation and limitations

- Model-based estimates, not a direct measurement of vapour composition.
- Only losses after the first retained timestamp are fitted.
- Nominal additions minus entered prior losses define the starting composition.
- Al, PG and MG are assumed nonvolatile; no withdrawals, spills or unrecorded additions.
- A constant offset per sensor is fitted. Time-/composition-/temperature-dependent errors remain confounded.
- Bootstrap ranges are conditional on this calculator and the assumed initial composition.
- Stabil/Gueltig are not default filters because real evaporation itself can create a trend.
- Unknown prior evaporation: zero prior losses were assumed for at least one solvent.
- A better two-solvent fit alone is not proof that water evaporated: the model has extra free parameters.
- Calculator support: IPA sound: temperature 25.1 C is outside locally supported anchors 18-25 C at 7.4 wt%.
- Calculator support: IPA sound: temperature 25.2 C is outside locally supported anchors 18-25 C at 7.3 wt%.
- Calculator support: IPA sound: temperature 25.2 C is outside locally supported anchors 18-25 C at 7.4 wt%.
- Calculator support: IPA sound: temperature 25.3 C is outside locally supported anchors 18-25 C at 7.3 wt%.
- Calculator support: IPA sound: temperature 25.3 C is outside locally supported anchors 18-25 C at 7.4 wt%.
- Calculator support: IPA sound: temperature 25.5 C is outside locally supported anchors 18-25 C at 7.3 wt%.
- Calculator support: IPA sound: temperature 25.5 C is outside locally supported anchors 18-25 C at 7.4 wt%.
- Calculator support: IPA sound: temperature 25.6 C is outside locally supported anchors 18-25 C at 7.3 wt%.
- Calculator support: IPA sound: temperature 25.6 C is outside locally supported anchors 18-25 C at 7.4 wt%.
- Calculator support: IPA sound: temperature 25.7 C is outside locally supported anchors 18-25 C at 7.1 wt%.
- Calculator support: IPA sound: temperature 25.7 C is outside locally supported anchors 18-25 C at 7.2 wt%.
- Calculator support: IPA sound: temperature 25.7 C is outside locally supported anchors 18-25 C at 7.3 wt%.
- Calculator support: IPA sound: temperature 25.7 C is outside locally supported anchors 18-25 C at 7.4 wt%.
- Calculator support: IPA sound: temperature 25.8 C is outside locally supported anchors 18-25 C at 7.2 wt%.
- Calculator support: IPA sound: temperature 25.8 C is outside locally supported anchors 18-25 C at 7.3 wt%.
- Calculator support: IPA sound: temperature 25.8 C is outside locally supported anchors 18-25 C at 7.4 wt%.

The solvent ratio is a MASS ratio of the integrated inferred losses, not a volume or molar ratio.
All losses start at the first retained timestamp. Do not extrapolate them back before that point.
RMSE values refer to an in-sample fit with fitted channel offsets, not external validation.

![Sensor history](sensor_history.png)
![Losses](evaporation_losses_and_rates.png)
![Composition](composition_history.png)
![Diagnostics](fit_diagnostics.png)
