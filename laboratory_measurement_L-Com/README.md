# Messdaten importieren und auswerten

Die gemeinsame Messdatei liegt in `measurement_data` und heißt
`Messdaten_JJJJ-MM-TT_HH-MM-SS.csv`, zum Beispiel
`Messdaten_2026-10-03_15-42-08.csv`. Datum und Uhrzeit entsprechen der letzten
Erstellung bzw. Erweiterung in der lokalen Zeitzone des Computers.

## Normaler Ablauf

1. Neue TIA-Exporte als `Kennfeld_v2 (XX).csv` in `measurement_data` ablegen.
2. `measurement_data/messdaten_zusammenfuehren.py` starten.
3. **1 – Messdatei erweitern** auswählen (beim ersten Mal: erstellen).
4. **Enter / A** importiert alle vorhandenen Roh-Exporte; alternativ einzelne
   Dateinummern oder mehrere Nummern wie `10,11` auswählen.
5. Das gewünschte Auswertungsskript starten. Seine Dateiauswahl bietet
   standardmäßig die gemeinsame Messdatei an.

Der erste Import nimmt immer `Kennfeld_v2 (25).csv` als Basis und ergänzt die
ausgewählten späteren Exporte. Die vorhandenen Exporte können jederzeit erneut
importiert werden: Überlappende Messpunkte werden nur einmal gespeichert.
Dateien mit dem Zusatz `_korrigiert` werden beim Import nicht verwendet.

Bei einer Erweiterung wird die bestehende gemeinsame Datei vollständig und
abgesichert aktualisiert und auf den neuen Zeitstempel umbenannt. Es bleibt
eine gemeinsame CSV im Ordner. Sind keine neuen Messungen enthalten, bleiben
Dateiname und Änderungszeit unverändert. Die TIA-Originale werden nicht verändert.
Die gemeinsame Datei zum Erweitern in Excel und anderen Programmen schließen.

## Korrekturen und Format

Beim Import werden die bisherigen Korrekturen automatisch angewendet:

- Probe 3, SeqNo 132–142 am 21.08.2026: 24,60 g PG.
- Probe 11 am 16.09.2026 wird zu Probe 12; die Probe vom 09.09. bleibt 11.
- Probe 300: 8000 g Wasser, 800 g SL120 und 20 g Methylgallat.

Die Gesamtdatei behält die Messspalten und ursprünglichen SeqNo bei. Sie enthält
eine Kopfzeile und danach zeitlich sortierte Messpunkte, ohne TIA-Abschlusszeilen
und ohne die aus dem Export stammenden Auffüll-Leerzeichen. Trennzeichen ist
ein Komma, Dezimalzeichen ein Punkt, Zeichencodierung UTF-8. Sensorwerte wie
`NaN` und `+INF` bleiben erhalten; die Auswertungen können diese Zeilen wie bisher
als unbrauchbar kennzeichnen.

Messungen werden anhand von **SeqNo plus Datum und UTC-Uhrzeit** erkannt.
Dadurch bleiben nach einem Zurücksetzen der SPS auch Messungen mit erneut
vergebener SeqNo erhalten. Widersprüchliche Werte für dieselbe Messung oder
unvollständige CSV-Zeilen brechen den Import mit einer Fehlermeldung ab, bevor
die Gesamtdatei geändert wird. Abweichende Spaltensätze müssen zuerst geprüft
werden. Eine Liste fehlender SeqNo weist auf mögliche Lücken hin.

Der Import kann nur Messungen sichern, die in mindestens einem Export vorhanden
sind. Deshalb vor jeweils 2000 neuen Messpunkten erneut aus TIA exportieren.
Die Gesamtdatei selbst hat keine Begrenzung auf 2000 Zeilen. Sie enthält schon
importierte Messpunkte auch dann, wenn diese aus den TIA-Exporten verschwinden.

## Aufruf ohne Menü

Vom Projekt-Hauptverzeichnis aus:

```text
python laboratory_measurement_L-Com/measurement_data/messdaten_zusammenfuehren.py --update
python laboratory_measurement_L-Com/measurement_data/messdaten_zusammenfuehren.py --files "Kennfeld_v2 (43).csv"
python laboratory_measurement_L-Com/measurement_data/messdaten_zusammenfuehren.py --dry-run
```

Der Import benötigt nur die Python-Standardbibliothek. Das bisherige
`korrigiere_kennfeld.py` wird durch `messdaten_zusammenfuehren.py` ersetzt.

## Auswertung

`evaluation_laboratory_measurement.py`, `evaluate_model_accuracy.py`,
`evaluate_evaporation.py` und `residual_calibration_field.py` verwenden bei
der Dateiauswahl bzw. einem Eingabeordner die Gesamtdatei, sobald sie vorhanden
ist. Damit werden Gesamtdatei und überlappende Roh-Exporte nicht zusammen
ausgewertet. Explizit angegebene alte Messdateien können weiterhin verwendet
werden. Auch `results/cross_validation_field_20260911/run_cross_validation.py`
bezieht die aktuelle Datei über den Messdatenordner.

Ohne Gesamtdatei bieten die Auswertungsskripte weiterhin die bisherigen
Messdateien an. Mehrere Gesamtdateien im selben Ordner führen zu einer klaren
Fehlermeldung; Sicherungskopien daher in einem anderen Ordner aufbewahren.

Die Auswertungsskripte behalten ihre sonstigen Einstellungen, Qualitätsfilter
und Anforderungen an Verdunstungsprotokolle bei. Für neue Proben im
Kalibrierungsfeld sind weiterhin passende Protokolle erforderlich.

Die Importprüfungen können so ausgeführt werden:

```text
python -m unittest discover -s laboratory_measurement_L-Com/tests -v
```
