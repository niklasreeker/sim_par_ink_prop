#!/usr/bin/env python3
"""Diagnose a constant apparent evaporation selectivity; illustrate RH limits.

All humidity values below are hypothetical. No atmospheric measurements are
available. Lambda is inferred conditionally from prior inverse compositions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_evaporation_scenarios as a

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'results/composition_inversion_20261007/probe306_zusammensetzungen.csv'
OUT=HERE/'results/humidity_assumptions_20261007'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    states=pd.read_csv(SOURCE)
    intervals=[];humidity=[]
    for method,group in states.groupby('Method',sort=False):
        if not group.inversion_matches_both_measurements.all():continue
        qi=group.IPA_per_Al.to_numpy();qw=group.Water_per_Al.to_numpy()
        # Homogeneous sample withdrawal cancels in q_i=m_i/m_Al.
        # R_I/R_W=lambda*q_I/q_W integrates to ln(q_I2/q_I1)
        # =lambda*ln(q_W2/q_W1) for constant lambda within an interval.
        lam=np.log(qi[1:]/qi[:-1])/np.log(qw[1:]/qw[:-1])
        for index,value in enumerate(lam):
            intervals.append({'Method':method,'Interval':f'P{index+1}-P{index+2}',
                              'lambda_apparent':value})
        for rh_late in [0.,.25,.5,.75]:
            # Illustrative closure only: Ts=Ta, aw~1, IPA-free ambient air,
            # invariant activities and transfer ratio apart from RH.
            # lambda_app=lambda_dry/(1-RH); RH_eff is interval-representative.
            dry=lam[-1]*(1-rh_late)
            rh=1-dry/lam
            humidity.append({'Method':method,'Assumed_late_RH_pct':100*rh_late,
                             'Required_early_RH_pct':100*rh[0],
                             'Required_middle_RH_pct':100*rh[1],
                             'lambda_dry_assumed':dry})
    result=pd.DataFrame(intervals);rh=pd.DataFrame(humidity)
    result.to_csv(OUT/'scheinbare_selektivitaet.csv',index=False)
    rh.to_csv(OUT/'hypothetische_reine_feuchteerklaerung.csv',index=False)
    a.style();fig,axs=plt.subplots(1,2,figsize=(11.7,4.9))
    labels={'physics_anchor':'Calculator, P1-Anker','physics_direct':'Calculator, ohne Zusatzoffset',
            'hybrid_direct':'MG-Feld, ohne Zusatzoffset (Extrapolation)'}
    for method,g in result.groupby('Method',sort=False):
        axs[0].plot(np.arange(3),g.lambda_apparent,'o-',label=labels[method])
    axs[0].set(xticks=np.arange(3),xticklabels=['P1–P2','P2–P3','P3–P4'],
               ylabel='Scheinbarer IPA/Wasser-Selektivitätsfaktor λ',
               title='Ein konstanter Faktor beschreibt die Rückrechnungen nicht')
    axs[0].legend(frameon=False,fontsize=7)
    primary=result[result.Method.eq('physics_anchor')].lambda_apparent.to_numpy()
    late=np.linspace(0,80,101);dry=primary[2]*(1-late/100)
    axs[1].plot(late,100*(1-dry/primary[0]),label='Erforderlich im ersten Intervall',color='#244f79')
    axs[1].plot(late,100*(1-dry/primary[1]),label='Erforderlich im zweiten Intervall',color='#b7772c')
    axs[1].plot(late,late,':',color='#888',label='Unveränderte Luftfeuchte')
    axs[1].set(xlabel='Angenommene Luftfeuchte im letzten Intervall [%]',
               ylabel='Rechnerisch erforderliche Luftfeuchte [%]',ylim=(0,100),
               title='Falls allein Luftfeuchte den Faktor verändert hätte')
    axs[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Probe 306: Welche Zusatzannahmen tragen?',fontsize=15)
    fig.text(.5,.01,'Rechte Grafik: hypothetische Näherung bei gleicher Luft-/Oberflächentemperatur, aW ≈ 1, IPA-freier Luft und sonst konstanten Bedingungen.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94));a.save(fig,OUT,'selektivitaet_und_feuchte')
    primary_rh=rh[(rh.Method=='physics_anchor')&(rh.Assumed_late_RH_pct==50)].iloc[0]
    body=f'''# Was können Luftfeuchte und Zusatzannahmen beitragen?

Die Luftfeuchte liefert eine zusätzliche Randbedingung für den Wasserverlust.
Sie bestimmt die Tintenzusammensetzung nicht allein. Die vorhandene Messdatei
enthält weder Luftfeuchte noch Lufttemperatur; T_M ist die Tintentemperatur am
Sensor. Die tatsächliche Temperatur der verdunstenden Oberfläche ist damit
ebenfalls nicht unabhängig gemessen.

## Physikalische Ergänzung

Als vereinfachtes Stoffübergangsmodell kann man schreiben:

R_W = K_W [a_W(w,T_s) p_W^sat(T_s) − φ p_W^sat(T_a)]

R_IPA = K_IPA [a_IPA(w,T_s) p_IPA^sat(T_s) − p_IPA,Luft]

R_i sind positive Nettomassenverlustraten, solange die Klammern positiv sind.
K_i sind effektive Faktoren mit Geometrie, Stoffübergang und Massenumrechnung.
φ ist RH/100, T_s die Oberflächentemperatur und T_a die Lufttemperatur.
a_i sind thermodynamische Aktivitäten (bei entsprechender Näherung γ_i x_i
mit **Molenanteilen** x_i, nicht direkt Massenprozenten).
Das Modell setzt eine ausreichend repräsentative, homogene Flüssigkeitsphase
und bekannte lokale Luftbedingungen voraus. Bei mehreren Verdunstungszonen
müssen deren Beiträge summiert werden.

Höhere Luftfeuchte vermindert bei sonst gleichen Bedingungen den Wasserverlust.
Damit verschiebt sie den momentanen Verlustanteil in Richtung IPA. Für dieselbe
Tintenzusammensetzung wird ein isothermer Anstieg von c dadurch schwerer erreicht.
Raumluftfeuchte ist bei erwärmter oder rezirkulierender Prozessluft nur eine
Randbedingung für die Zuluft; lokale Luft- und Oberflächentemperaturen sowie
Wasser- und IPA-Anreicherung der Prozessluft bleiben nötig.

Primärquellen für die Beziehungen:
[COMSOL: Moist Air Properties](https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_ug_modeling.06.15.html),
[COMSOL: Evaporation and Condensation](https://doc.comsol.com/6.4/doc/com.comsol.help.chem/chem_ug_chemsptrans.08.289.html),
[COMSOL: Wasseraktivität und Gleichgewichtsdampfdruck](https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_ug_mt_features.10.04.html).

## Prüfung einer einfachen Annahme an Probe 306

Ein empirisches Ersatzmodell lautet R_IPA/R_W = λ·w_IPA/w_W.
λ beschreibt die bevorzugte Verdunstung relativ zur aktuellen Massenverteilung;
es ist ein effektiver Prozessfaktor, keine reine Stoffkonstante.
Am isothermen stationären Punkt folgt daraus

w_IPA/w_W = 1 / [λ·r_krit(w,T)].

Sind λ und weitere Rezepturverhältnisse unabhängig bekannt, entsteht eine
zusätzliche Beziehung für die Rückrechnung. r_krit bleibt von der gesuchten
Zusammensetzung abhängig; die anfänglichen 17–18:1 dürfen dabei nicht überall
als Konstante eingesetzt werden. Bei veränderlicher Temperatur muss die volle
Bilanz dc/dt = D_W R_W + D_IPA R_IPA + (∂c/∂T) dT/dt benutzt werden.

Für ein innerhalb eines Intervalls konstantes λ gilt bei homogener Probenentnahme:

λ = ln[(m_IPA/m_Al)_2/(m_IPA/m_Al)_1] /
    ln[(m_W/m_Al)_2/(m_W/m_Al)_1].

Aus den vier rückgerechneten Zuständen von Probe 306 ergeben sich unter der
P1-Rezepturannahme λ ≈ {primary[0]:.2f}, {primary[1]:.2f}, {primary[2]:.2f}.
Die anderen verwendbaren Modellvarianten zeigen dieselbe deutliche Abnahme.
Ein über den ganzen Tag unveränderter Faktor ist deshalb mit diesen
punktweisen Rückrechnungen nicht vereinbar. Die Faktoren sind aus denselben
Sensorwerten abgeleitet, keine unabhängige Validierung der Verdunstung.
Prozessänderungen, nichtideales Mischungsverhalten, IPA in der Prozessluft,
lokale Temperaturen sowie Mess-/Modellabweichungen können den effektiven
Faktor verändern. Die MG-Kalibrierfeldvariante extrapoliert; ein fehlerhafter
Fit der vierten Modellvariante wurde ausgeschlossen.

## Könnte allein Luftfeuchte diese Abnahme erklären?

Nur unter den starken Zusatzannahmen T_s = T_a, a_W ≈ 1, IPA-freie Umgebung,
unveränderte Aktivitäts-/Stoffübergangsfaktoren und je Intervall repräsentative
konstante Bedingungen gilt näherungsweise λ_app = λ_dry/(1−φ).

Dann müsste bei Probe 306 die Feuchte im ersten Intervall mindestens
{100*(1-primary[2]/primary[0]):.1f} % betragen, selbst wenn sie im letzten
Intervall auf 0 % gefallen wäre. Bei angenommenen 50 % im letzten Intervall
wären im ersten rund {primary_rh.Required_early_RH_pct:.1f} % und im zweiten
{primary_rh.Required_middle_RH_pct:.1f} % erforderlich.
Das sind **hypothetisch erforderliche Feuchten**, keine gemessenen oder
belastbar aus den Daten geschätzten Feuchten. Sie zeigen, dass eine nahezu
unveränderte Raumluftfeuchte den Faktorwechsel in diesem einfachen Modell
nicht erklärt. Gemessene Feuchteverläufe könnten diese Hypothese prüfen.

![Selektivität und hypothetische Feuchte](selektivitaet_und_feuchte.png)

## Sinnvolle Annahmen für die weitere Eingrenzung

1. Bekannte Anfangsrezeptur und dokumentierte Zugaben/Entnahmen; Verluste vor
   P1 als eigene Unsicherheit führen.
2. Gut durchmischte Tinte und homogene Probenentnahme. Bei Probe 306 laut
   Protokoll kein Druckaustrag und keine dokumentierte Nachfüllung.
3. PG, Pigment und MG als im betrachteten Versuch zurückgehalten behandeln;
   das ist eine zu prüfende Annahme über die Komponentenbilanz.
4. Dichte und Schall bei gemessener Tintentemperatur auswerten und Sensoroffsets
   über eine bekannte Referenzmischung unabhängig bestimmen.
5. Prozessabschnitte mit vergleichbarer Luftführung, Trocknung und Temperatur
   gemeinsam beschreiben. λ an diesen Bedingungen kalibrieren, statt ein
   universelles λ für Probe 12 und 306 vorauszusetzen.
6. Zeitgleiche Festkörper- und mindestens einzelne unabhängige IPA-Messungen
   als Anker verwenden. Der Tankverlust ist erst nach Inventaränderungen und
   Probenentnahmen als Gesamtverdunstung nutzbar.

Probe 12 bleibt ein Beispiel für eine temperaturbedingte Rohsignalumkehr.
Bei Probe 306 bleibt die Umkehr auch nach der bisherigen Temperaturkorrektur
erhalten. Diese beiden Mechanismen müssen getrennt modelliert werden.

Die nächste quantitative Untersuchung sollte gemessene lokale Luftfeuchte,
Lufttemperatur und möglichst Oberflächentemperatur als Eingänge verwenden und
die übrigen unbekannten Größen als Parameterbereiche behandeln. Ohne diese
Messwerte können nur ausdrücklich hypothetische Szenarien gerechnet werden.
'''
    (OUT/'einordnung.md').write_text(body,encoding='utf8')
    checks={'finite_positive_selectivities':bool(np.isfinite(result.lambda_apparent).all() and (result.lambda_apparent>0).all()),
            'late_selectivity_smaller_in_each_variant':bool(all(g.iloc[2].lambda_apparent<g.iloc[1].lambda_apparent<g.iloc[0].lambda_apparent for _,g in result.groupby('Method'))),
            'hypothetical_humidities_in_0_100':bool(((rh[['Required_early_RH_pct','Required_middle_RH_pct']]>=0)&(rh[['Required_early_RH_pct','Required_middle_RH_pct']]<=100)).all().all())}
    assert all(checks.values())
    (OUT/'manifest.json').write_text(json.dumps({'date':'2026-10-07','source_sha256':a.digest(SOURCE),
            'script_sha256':a.digest(Path(__file__)),'checks':checks,
            'caution':'No humidity measurements; humidity scenarios are hypothetical and conditional on strong simplifying assumptions',
            'lambda_definition':'(R_IPA/R_W)/(w_IPA/w_W), integrated over intervals of constant apparent lambda'},indent=2),encoding='utf8')
    print(result.to_string(index=False));print(primary_rh.to_string());print(checks)


if __name__=='__main__':main()
