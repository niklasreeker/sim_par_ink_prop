# Gesamtprobe: fortlaufende Verdunstungsbilanz

Probe 6 | Kennfeld_v2 (18)_korrigiert.csv

Gesamtverlust einschliesslich Vorverlust: **5.653 g**
- Vor der ersten Messung eingegeben: 2.000 g
- Seit der ersten Messung: 3.653 g
- In Messfenstern angepasst: 3.580 g
- Seit Messbeginn in ungemessenen Zeiten angenommen: 0.073 g
- IPA: 1.577 g; Wasser: 4.077 g (modell-/annahmenabhaengig)
- Ausgewerteter Zeitraum ab erster Messung: 1.666 h
- Auswertbare Phasen: 1 von 1

## Annahmen und Interpretation

- Fortlaufende, sequenzielle Modellbilanz; keine direkte chemische Verdunstungsmessung.
- Einwaagen sind kumulative Zugaben. Al/PG/MG gelten als nichtfluechtig; Entnahmen und Verschleppung sind nicht korrigiert.
- Der eingegebene Gesamtverlust vor Messbeginn wird genau einmal als Anfangskorrektur angesetzt; seine Dauer wird nicht aus den Messdaten abgeleitet.
- Rezepturzugaben werden am ersten Rohzeitstempel der neuen Rezeptur eingebucht; der reale Zugabezeitpunkt in einer Pause ist unbekannt.
- Ungemessene Zeiten (auch ausgeschlossene Randpunkte/zu kurze Phasen) werden separat geschaetzt.
- previous setzt die letzte Segmentrate des letzten erfolgreichen Fits fort; bis dahin gilt gap-rate.
- IPA/Wasser-Anteile ungemessener Verluste sind Annahmen. Ohne Vorgabe gilt der aktuelle IPA-Anteil am IPA/Wasser-Vorrat, kein Dampfdruckmodell.
- Jede Phase hat eigene Sensoroffsets. Spruenge zwischen Rezepturen identifizieren deshalb keinen eindeutigen Pausenverlust.
- Phasen-Bootstrap ist bedingt auf die uebernommene Startmasse. Unsicherheit vorheriger Fits, Pausen und Vorverlust wird NICHT propagiert; kein Gesamt-Konfidenzintervall.
- Ratenunterschiede nach Zugaben sind beschreibend, kein kausaler Nachweis eines Zusammensetzungseffekts.
- Die Bilanz umfasst die gewaehlte CSV bis zum letzten Rohzeitstempel, nicht automatisch andere Dateien derselben Probe.
- Phase 1: PG density: temperature 23.99 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.03 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.07 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.11 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.15 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.18 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.22 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.25 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.47 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.50 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.52 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.56 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.60 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.61 C is outside the locally supported temperature range.
- Phase 1: PG density: temperature 24.67 C is outside the locally supported temperature range.
- Phase 1: PG sound: temperature 24.0 C is outside locally supported anchors 25-65 C at 7.4 wt%.
- Phase 1: PG sound: temperature 24.1 C is outside locally supported anchors 25-65 C at 7.4 wt%.
- Phase 1: PG sound: temperature 24.2 C is outside locally supported anchors 25-65 C at 7.4 wt%.
- Phase 1: PG sound: temperature 24.3 C is outside locally supported anchors 25-65 C at 7.4 wt%.
- Phase 1: PG sound: temperature 24.5 C is outside locally supported anchors 25-65 C at 7.3 wt%.
- Phase 1: PG sound: temperature 24.6 C is outside locally supported anchors 25-65 C at 7.3 wt%.
- Phase 1: PG sound: temperature 24.7 C is outside locally supported anchors 25-65 C at 7.3 wt%.

## Ausgabedateien

- sample_loss_ledger.csv: lueckenlose Zeitbilanz, getrennt nach Fit und Annahme.
- sample_time_series.csv: kumulierte Verluste, verbleibende Massen, Anteile und Sensorresiduen.
- sample_transitions.csv: Zugaben, uebernommene Verluste und Abweichung zur Nominalrezeptur.
- sample_phases.csv: Phasenraten und Aenderung gegenueber dem letzten erfolgreichen Fit.
- analysis_metadata.json: Parameter, Quellenpruefsummen, Diagnostik und bedingte Phasen-Bootstraps.

Fuer Sensitivitaet erneut mit --gap-mode zero bzw. constant und unterschiedlichen
--assumed-ipa-fraction-Werten starten. Unterschiede sind Szenariospannen, keine Konfidenzintervalle.

![Gesamtverlauf](sample_history.png)
