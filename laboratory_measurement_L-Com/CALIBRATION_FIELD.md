# Hybrides Kalibrierfeld mit Verdunstungskorrektur

## Zielmodell

Fuer jede Messgroesse wird Physik und datengetriebene Korrektur getrennt:

\[
y_{\mathrm{ges},j}(\mathbf w,T)
= f_{\mathrm{Physik},j}(\mathbf w,T)
+ \Delta_j(\mathbf w,T),
\qquad j\in\{\rho,c\}.
\]

`ink_calculator.py` liefert den physikalischen Anteil. Das Residualmodell
lernt nur den reproduzierbaren Restfehler. Fuer die aktuelle Tinte ist
\(\mathbf w=(w_\mathrm{Al},w_\mathrm{IPA},w_\mathrm{PG},w_\mathrm{H2O})\);
ein einzelnes skalares \(w\) reicht nur, wenn die Rezeptur durch genau einen
Dosierparameter eindeutig festgelegt ist.

## 1. Verdunstung zuerst in der Eingangsachse korrigieren

Die Einwaage ist die nominale Zusammensetzung. Durch Verdunstung aendert sich
die reale Zusammensetzung am Messzeitpunkt. Diese reale Zusammensetzung muss
als Eingabe fuer `ink_calculator` und fuer das Residualmodell verwendet werden.

Bei einer repraesentativen Probenentnahme gilt fuer zwei aufeinanderfolgende
Waegungen:

\[
\Delta m_{\mathrm{verd},k}
= m_{\mathrm{nachher},k-1}
+ \Delta m_{\mathrm{Zugabe},k}
- m_{\mathrm{vorher},k}.
\]

Innerhalb einer Zeile ist
\(m_{\mathrm{vorher}}-m_{\mathrm{nachher}}\) dagegen die entnommene
Probenmasse. Sie darf nicht als Verdunstung gezaehlt werden. Das Werkzeug
`estimate_evaporation.py` setzt genau diese Bilanz um und propagiert die
Komponentenmassen durch Entnahme, Zugabe und Verdunstung.

Aufruf:

```bash
python laboratory_measurement_L-Com/estimate_evaporation.py \
  --input "laboratory_measurement_L-Com/measurement_data/Kennfeld_v2 (10) korrigiert.csv"
```

Die Gravimetrie bestimmt nur den gesamten Masseverlust. Die Aufteilung auf IPA
und Wasser benoetigt eine Annahme oder eine unabhaengige Messung. Der Parameter
`--ipa-share` dokumentiert diese Annahme; `1.0` bedeutet, dass der Verlust
vollstaendig IPA zugeordnet wird. Sinnvoll ist spaeter eine Sensitivitaetsanalyse
mit mehreren Anteilen oder die Bestimmung ueber Brechungsindex/GC.

In der aktuellen Datei sind `m_vorher` und `m_nachher` in allen Zeilen null.
Damit ist die absolute Verdunstungsmasse nicht identifizierbar. Dichte- und
Schalltrends koennen Verdunstung anzeigen, duerfen aber nicht gleichzeitig zur
Verdunstungsschaetzung und zum Lernen von \(\Delta\) verwendet werden: Dann
wuerde derselbe Modellfehler doppelt angepasst.

## 2. Kalibrierpunkte erzeugen

Nach der Verdunstungskorrektur wird fuer jeden verwendbaren Messpunkt berechnet:

\[
\Delta_\rho = \rho_{\mathrm{mess}}-\rho_{\mathrm{Physik}},\qquad
\Delta_c = c_{\mathrm{mess}}-c_{\mathrm{Physik}}.
\]

Achtung: `evaluate_model_accuracy.py` verwendet bisher die umgekehrte
Vorzeichenkonvention `Simulation - Messung`. Fuer das additive Hybridmodell
muss das Residuum deshalb beim Export umgedreht oder eindeutig benannt werden.

Jeder Kalibrierpunkt sollte mindestens enthalten:

| Gruppe | Felder |
|---|---|
| Herkunft | Datei, ProbeNr, Rezepturabschnitt, Zeitstempel |
| Eingabe | korrigierte Massenanteile Al, IPA, PG, Wasser und Temperatur |
| Messung | Dichte und Schallgeschwindigkeit mit Streuung/Qualitaetsflags |
| Physik | Vorhersagen aus `ink_calculator` und verwendete Modellversion |
| Ziel | `Delta_Rho = Rho_M - Rho_Physik`, `Delta_C = C_M - C_Physik` |
| Unsicherheit | Messunsicherheit und Unsicherheit der Verdunstungskorrektur |

## 3. Erstes Residualmodell

Fuer Versuch 3 mit nur 80 zeitlich korrelierten Punkten und sechs
Rezepturabschnitten ist ein kleines, regularisiertes Modell angemessen:

1. globaler Offset (Intercept),
2. lineare und bei Bedarf quadratische Terme der Zusammensetzung und Temperatur,
3. Ridge-Regularisierung gegen Ueberanpassung.

Ein Gaussian Process oder RBF-Smoother ist erst sinnvoll, wenn mehr unabhaengige
Rezepturen und Wiederholungsversuche vorhanden sind. Zwei getrennte Modelle fuer
Dichte und Schallgeschwindigkeit sind transparenter als ein gemeinsames Modell.

Ein konstanter Offset ist Teil von \(\Delta\). Eine physikalisch interpretierbare
Groesse wie die effektive Pigmentdichte sollte nur dann im Physikmodell kalibriert
werden, wenn sie ueber mehrere Rezepturen und Versuche stabil bleibt. Freie
Pigmentdichte und flexibles Residualmodell duerfen nicht gleichzeitig denselben
Offset erklaeren.

## 4. Testen und spaeter erweitern

Die 80 Zeilen aus Versuch 3 duerfen nicht zufaellig auf Train und Test verteilt
werden. Benachbarte Zeilen derselben Rezeptur sind fast Duplikate und wuerden die
Testguete zu optimistisch machen.

Empfohlener Ablauf:

1. Modell auf Versuch 3 aufbauen und per Leave-one-recipe-section-out validieren.
2. Einen neuen Laborversuch vor dem Nachtrainieren als echten, unangetasteten
   Testdatensatz auswerten.
3. Physikmodell, reines Offsetmodell und hybrides Residualmodell mit MAE/RMSE,
   Bias und Unsicherheitsabdeckung vergleichen.
4. Erst nach dokumentierter Testauswertung die neuen Punkte an den Trainingssatz
   anhaengen, Modellversion erhoehen und erneut gruppenweise validieren.
5. Ausserhalb des durch Trainingspunkte abgedeckten Zusammensetzungs- und
   Temperaturbereichs keine unmarkierte Residualextrapolation ausgeben.

So bleibt jeder neue Versuch zunaechst ein ehrlicher Test und wird danach zu
einer kontrollierten Erweiterung des Kalibrierfelds.
