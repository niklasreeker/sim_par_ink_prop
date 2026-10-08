#!/usr/bin/env python3
"""Conditional isothermal turning compositions on mass-conserving paths.

RH alone does not identify composition. Compare a process model calibrated
conditionally on early 306 inversion with a dilute aqueous Henry-law model.
Neither is a validated model of the actual heated printing apparatus.
"""
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import analyze_composition_inversion as inv

a=inv.a
HERE=Path(__file__).resolve().parent
OUT=HERE/'results/rh_40_50_estimates_20261007'
COMPS=HERE/'results/composition_inversion_20261007/probe306_zusammensetzungen.csv'
LAMBDA_SOURCE=HERE/'results/humidity_assumptions_20261007/scheinbare_selektivitaet.csv'
PSAT_W=1e5*10**(5.40221-1838.675/(298.15-31.737))
H_CP=1.2  # mol / (m3 Pa), 298.15 K, dilute IPA in water; Sander compilation
M_W=.01801528


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    study=inv.tp.Study()
    qw0=inv.RECIPE306[4]/inv.RECIPE306[0];qi0=2.
    initial=np.array([1.,qi0,2.,inv.K306,qw0])
    observed=pd.read_csv(COMPS)
    lam=pd.read_csv(LAMBDA_SOURCE)
    early=float(lam[(lam.Method=='physics_anchor')&(lam.Interval=='P1-P2')].lambda_apparent.iloc[0])

    def masses(qw,qi):return np.array([1.,float(qi),2.,inv.K306,float(qw)])
    def physical_ratio(m,rh,kr):
        pct=a.pct_from_masses(m)
        rho=study.values(m,25,'physics')[0,0]
        mol=pct[[1,2,3,4]]/np.array([60.095,76.095,184.15,18.01528])
        aw=mol[-1]/mol.sum()  # Ideal solvent mole-fraction approximation.
        assert aw>rh
        # R_i = A k_i M_i (p_i,s-p_i,air)/(RT).
        # Henry p_IPA = c_IPA/Hcp; c_IPA=rho*w_IPA/M_IPA.
        # Hence M_IPA cancels; kr is the ratio of molar gas-side k_i.
        return kr*rho*(pct[1]/100)/(H_CP*M_W*PSAT_W*(aw-rh))

    rows=[];checks=[]
    def process_path(model,rh,parameter,calibration_rh=None):
        if model=='empirical':
            dry=early*(1-calibration_rh)
            selectivity=dry/(1-rh)
            def state(qw):return masses(qw,qi0*(qw/qw0)**selectivity)
            def ratio(m):return selectivity*m[1]/m[4]
        else:
            sol=solve_ivp(lambda qw,y:[physical_ratio(masses(qw,y[0]),rh,parameter)],
                          (qw0,12),[qi0],rtol=2e-9,atol=1e-11,dense_output=True,max_step=.5)
            assert sol.success
            def state(qw):return masses(qw,sol.sol(qw)[0])
            def ratio(m):return physical_ratio(m,rh,parameter)
            dry=selectivity=np.nan
        def balance(qw):
            m=state(qw);g=study.gradient(m,25,'physics',relative_step=.0001).iloc[0]
            return g.CW_m_s_per_g+g.CI_m_s_per_g*ratio(m)
        for kind,function in [('turning',balance),('fixed95',lambda qw:ratio(state(qw))-1/19)]:
            grid=np.linspace(qw0,15,61);values=np.array([function(q) for q in grid])
            idx=np.flatnonzero(values[:-1]*values[1:]<0)
            assert len(idx), (model,rh,parameter,kind)
            j=idx[0];q=brentq(function,grid[j+1],grid[j],xtol=1e-8)
            m=state(q);pct=a.pct_from_masses(m);rate_ratio=ratio(m)
            g=study.gradient(m,25,'physics',relative_step=.0001).iloc[0]
            before,after=balance(q+.001),balance(q-.001)
            row={'Model':model,'RH_pct':100*rh,'Kind':kind,
                 'Calibration_RH_pct':np.nan if calibration_rh is None else 100*calibration_rh,
                 'gas_transfer_ratio_kIPA_kW':parameter if model=='henry' else np.nan,
                 'lambda_empirical':selectivity,'lambda_dry_empirical':dry,
                 'qWater_g_per_g_Al':q,'qIPA_g_per_g_Al':m[1],
                 'Water_fraction_in_loss':1/(1+rate_ratio),
                 'critical_water_fraction_at_state':g.critical_water_fraction,
                 'normalized_acoustic_balance':balance(q),
                 'balance_before':before,'balance_after':after,
                 'Mass_remaining_fraction':m.sum()/initial.sum(),
                 **dict(zip([c+'_pct' for c in a.COMPONENTS],pct))}
            rows.append(row)
            if kind=='turning':checks.append(bool(before<0<after and abs(balance(q))<1e-6))

    for rh in [.4,.45,.5]:
        for rhcal in [.4,.45,.5]:process_path('empirical',rh,None,rhcal)
        for kr in [.4,.5,.6]:process_path('henry',rh,kr)
        print(f'RH {100*rh:g} completed',flush=True)
    result=pd.DataFrame(rows);result.to_csv(OUT/'alle_szenarien.csv',index=False)
    turning=result[result.Kind.eq('turning')]
    summary=turning.groupby('Model')[['IPA_pct','Water_pct','Al_pct','Water_fraction_in_loss']].agg(['min','max'])
    summary.to_csv(OUT/'szenariobereiche.csv')
    centers=turning[((turning.Model=='empirical')&(turning.Calibration_RH_pct==45))|
                    ((turning.Model=='henry')&(turning.gas_transfer_ratio_kIPA_kW==.5))]
    centers.to_csv(OUT/'zentrale_szenarien.csv',index=False)

    a.style();fig,axs=plt.subplots(1,2,figsize=(11.7,4.9))
    for model,color,label in [('empirical','#244f79','Frühes 306-Verhalten fortgeschrieben'),
                               ('henry','#b7772c','Frischluft, Henry-Näherung bei 25 °C')]:
        group=turning[turning.Model.eq(model)]
        center=centers[centers.Model.eq(model)].sort_values('RH_pct')
        for ax,col in zip(axs,['IPA_pct','Water_pct']):
            bounds=group.groupby('RH_pct')[col].agg(['min','max'])
            ax.fill_between(bounds.index,bounds['min'],bounds['max'],color=color,alpha=.15)
            ax.plot(center.RH_pct,center[col],'o-',color=color,label=label)
    for ax,name in zip(axs,['IPA','Wasser']):ax.set(xlabel='Relative Luftfeuchte [%]',ylabel=name+' am isothermen Minimum [Masse-%]',xticks=[40,45,50])
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('40–50 % Luftfeuchte: Zusatzannahmen bestimmen die Abschätzung',fontsize=15)
    fig.text(.5,.01,'Massenbilanz ab nominalem P1, konstante 25 °C. Bänder sind Szenarien, keine Konfidenzintervalle. Keine Uhrzeit vorhergesagt.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));a.save(fig,OUT,'rh_zusammensetzung')

    def de(v,n=2):return f'{v:.{n}f}'.replace('.',',')
    def table(frame,cols,headers):
        return '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+(str(r[c]) if isinstance(r[c],str) else de(r[c]))+'</td>' for c in cols)+'</tr>' for _,r in frame.iterrows())+'</tbody></table>'
    central=centers.copy();central['Modell']=central.Model.map({'empirical':'Frühes 306-Verhalten fortgeschrieben','henry':'Frischluft/Henry-Näherung'})
    src=base64.b64encode((OUT/'rh_zusammensetzung.png').read_bytes()).decode()
    p3=observed[(observed.Method=='physics_anchor')&(observed.Point=='P3')].iloc[0]
    text=f'''<h1>Welche Tintenzusammensetzung folgt bei 40–50 % Luftfeuchte?</h1>
<p class="lead">Die Feuchte allein identifiziert die Zusammensetzung nicht. Bei zwei ausdrücklich festgelegten Verdunstungsmodellen ergeben sich unterschiedliche isotherme Minima. Für die reale Probe 306 bleibt die sensorbasierte Rückrechnung nahe P3 die näher an den Messdaten liegende Arbeitsschätzung.</p>
<h2>1. Ergebnis der Szenarien</h2>
<p>Alle Konzentrationen sind Masse-%. Ausgangspunkt ist die nominale Rezeptur von Probe 306 am angenommenen P1. Die absoluten Pigment-/PG-/MG-Massen bleiben erhalten, Wasser und IPA nehmen ab. Die Konzentrationen dieser zurückgehaltenen Komponenten steigen entlang jedes Pfades; sie werden nicht künstlich auf den Ausgangsprozenten festgehalten. Homogene Probenentnahmen ändern die normierten Verhältnisse q_i = m_i/m_Al unmittelbar nicht.</p>
{table(central,['RH_pct','Modell','IPA_pct','Water_pct','Al_pct','Water_fraction_in_loss'],['RH [%]','Zusatzmodell','IPA [%]','Wasser [%]','Al [%]','Wasseranteil am Verlust [0–1]'])}
<figure><img src="data:image/png;base64,{src}"><figcaption>Jeweils das erste isotherme Minimum entlang eines massenbilanzierten Pfades. <a href="rh_zusammensetzung.pdf">PDF</a>. Bänder zeigen Modellannahmen, keine statistischen Unsicherheiten.</figcaption></figure>
<h2>2. Empirische Fortschreibung des frühen 306-Verhaltens</h2>
<p>Ansatz: R_IPA/R_W = λ·w_IPA/w_W und λ = λ_dry/(1−RH), näherungsweise bei gleicher Luft-/Oberflächentemperatur, Wasseraktivität nahe 1, unverändertem Stoffübergang und IPA-freier Luft. Aus dem früheren P1–P2-Zustandsvergleich folgt bedingt λ = {de(early,3)}. Zur mittleren Kalibrierfeuchte 45 % gehört λ_dry = {de(early*.55,3)}. Für die Bandbreite wurden sowohl die unbekannte Kalibrierfeuchte als auch die spätere Feuchte zwischen 40 und 50 % variiert.</p>
<p>Bei konstantem λ integriert sich die Komponentenbilanz zu q_IPA = 2·(q_W/q_W,Start)^λ. Die Umkehr wurde entlang dieses Pfades aus den aktuellen InkCalculator-Ableitungen gesucht. Dadurch sind Rezeptur, aktueller Gehalt und angenommene Verdunstungsgeschichte miteinander vereinbar. Dieses Szenario schreibt den frühen Faktor fort; der tatsächliche spätere Versuch zeigte gerade einen anderen effektiven Faktor. Es ist keine Anpassung des gesamten Versuchs.</p>
<h2>3. Physikalische Frischluft-Näherung</h2>
<p>Zusatzannahmen: T_Luft = T_Oberfläche = 25 °C; kein IPA in der Luft; verdünntes IPA folgt näherungsweise Henry; Wasseraktivität wird als Molenanteil des Wassers im Lösungsmittel aus Wasser/IPA/PG/MG angenähert. Wechselwirkungen mit PG/MG werden nicht zusätzlich parametrisiert. Der Pigmentfeststoff gehört nicht zur Molenanteilsnormierung.</p>
<p>Verwendet: H_cp = 1,2 mol/(m³·Pa) für verdünntes IPA in Wasser bei 25 °C, Wasser-Sättigungsdampfdruck {de(PSAT_W,1)} Pa aus der NIST-Antoine-Gleichung und ein ausdrücklich angenommenes Verhältnis der molaren gasseitigen Stoffübergangsfaktoren k_IPA/k_W = 0,4–0,6, zentral 0,5. Diese Spanne ist eine Sensitivitätsannahme für ungleichen Dampftransport, keine gemessene Eigenschaft der Anlage.</p>
<p>Mit p_IPA,s = c_IPA/H_cp und p_W,s = a_W·p_W,sat gilt R_IPA/R_W = (k_IPA/k_W)·ρ·w_IPA/[H_cp·M_W·p_W,sat·(a_W−RH)], mit w_IPA als Massenbruch. Die Dichte kommt aus dem Calculator. Die Differentialgleichung dq_IPA/dq_W = R_IPA/R_W wird ab der gleichen Anfangsrezeptur integriert. Die Grenzbedingung wird ebenfalls entlang des resultierenden Pfades gesucht. Der wässrige Henry-Wert ist für die vollständige Tinte nicht validiert.</p>
<p>Quellen: <a href="https://henrys-law.org/henry/casrn/67-63-0">Sander, Henry-Konstanten für 2-Propanol</a>; zugrunde liegende Messungen u.a. <a href="https://trc.nist.gov/ThermoML/10.1021/je600567z.html">Fenclová et al. 2007</a>. Wasser-Dampfdruck: <a href="https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&amp;Mask=4">NIST WebBook</a>.</p>
<h2>4. Warum wird die Grenze von 95 % neu berechnet?</h2>
<p>Die anfänglichen ungefähr 95 % galten lokal für die Standardzusammensetzung. Hier wurde für jedes neue Gemisch D_W + D_IPA·(R_IPA/R_W) = 0 gelöst. Beim sinkenden IPA-Gehalt verschiebt sich diese Grenze. Eine bloße Vorgabe 95 % Wasser im Verlust ist daher im Allgemeinen nicht das Minimum der neuen Tinte. Die CSV enthält getrennt den Zustand mit exakt 95/5-Verlust und den tatsächlichen stationären Zustand des jeweiligen Modells. Alle ausgegebenen Minima wurden auf den Wechsel von fallender zu steigender Schallgeschwindigkeit geprüft. Eine Zeitvorhersage erfordert zusätzlich die absolute Gesamtverlustrate.</p>
<h2>5. Abgleich mit Probe 306 und Probe 12</h2>
<p>Am Messpunkt P3 ergibt die P1-verankerte Calculator-Rückrechnung {de(p3.IPA_pct)} % IPA und {de(p3.Water_pct)} % Wasser. Die anderen punktweisen Modellvarianten verschieben die absoluten Werte merklich. Als grobe Arbeitsspanne nahe diesem Sensorfenster sind deshalb etwa 2–3 % IPA und 89–91 % Wasser sinnvoller als ein präziser Einzelwert. Das ist keine statistische Konfidenzgrenze und keine Bestimmung exakt am kontinuierlichen Minimum.</p>
<p>Diese sensorbasierte Abschätzung entsteht aus Dichte, Schall und Temperatur; die angenommene Luftfeuchte legt sie nicht zusätzlich fest. Die deutlich niedrigeren IPA-Gehalte der Frischluftrechnung zeigen, dass dieses einfache Randbedingungsmodell den tatsächlichen Prozess nicht ausreichend beschreibt. Mögliche Ursachen sind andere Oberflächentemperaturen, IPA in der Prozessluft, mehrere Verdunstungszonen, geänderte Luftführung sowie unzureichende Stoffeigenschafts-/Sensorkalibrierung. Die Gründe lassen sich hier nicht eindeutig trennen.</p>
<p>Innerhalb 40–50 % Feuchte variiert der Faktor 1/(1−RH) nur um maximal 20 %. Das erklärt unter sonst gleichen Bedingungen nicht die zuvor rückgerechnete Faktorabnahme von rund 3,49 auf 0,74. Außerdem liegen die späteren MG-Gehalte teilweise außerhalb der gemessenen Calculator- und Kalibrierbereiche. Alle Warnungen stehen im Manifest.</p>
<p>Für Probe 12 bleibt am Rohminimum die frühere bedingte Schätzung von etwa 3,55 % IPA und 90,97 % Wasser bestehen. Dessen Rohsignalumkehr ist überwiegend temperaturbedingt. Die isothermen Szenarien dürfen nicht auf diesen Zeitpunkt übertragen werden.</p>
<h2>Daten</h2><p><a href="alle_szenarien.csv">Alle Szenarien und 95/5-Vergleich</a>, <a href="zentrale_szenarien.csv">Zentrale Szenarien</a>, <a href="szenariobereiche.csv">Szenariobereiche</a>, <a href="manifest.json">Annahmen, Quellen und Prüfungen</a>.</p>'''
    style='body{font:16px/1.6 system-ui;background:#f4f6f8;color:#25323b;margin:0}main{max-width:1100px;margin:30px auto;background:white;padding:36px}h1{font-size:32px;line-height:1.25}h2{margin-top:32px;color:#1b5364}.lead{padding:20px;background:#e9f4f5;font-size:20px}img{width:100%}figure{margin:25px 0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:right;padding:10px;border-bottom:1px solid #ddd}th{background:#edf2f5}a{color:#126482}@media(max-width:700px){main{padding:18px;margin:0}table{font-size:12px}}'
    (OUT/'auswertung.html').write_text('<!DOCTYPE html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Abschätzung bei 40–50 % Luftfeuchte</title><style>'+style+'</style><main>'+text+'</main></html>',encoding='utf8')
    assert all(checks)
    assert np.allclose(result[[c+'_pct' for c in a.COMPONENTS]].sum(axis=1),100)
    manifest={'date':'2026-10-07','RH_assumed_pct':[40,50],'temperature_isothermal_C':25,
              'Henry_Hcp_mol_m3_Pa':H_CP,'Water_psat_Pa':PSAT_W,
              'nominal_initial_masses_g':dict(zip(a.COMPONENTS,inv.RECIPE306.tolist())),
              'sources_sha256':{str(p.relative_to(a.ROOT)):a.digest(p) for p in [COMPS,LAMBDA_SOURCE,a.ROOT/'ink_calculator.py',Path(__file__)]},
              'checks':{'all_turns_negative_to_positive':all(checks),'compositions_sum_to_100':True,
                        'coupled_component_paths_integrated_from_same_recipe':True},
              'warnings':sorted(study.notices),
              'limits':['Conditional scenarios, not identified composition from RH',
                        'No independently measured local surface or air temperature',
                        'IPA background gas set to zero in physical scenario',
                        'Binary aqueous Henry coefficient used for multicomponent ink without validation',
                        'No absolute turning time inferred; no statistical confidence intervals']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    print(centers[['Model','RH_pct','IPA_pct','Water_pct','Al_pct','Water_fraction_in_loss']].to_string(index=False))
    print(summary.to_string());print('All turning-point checks passed',flush=True)


if __name__=='__main__':main()
