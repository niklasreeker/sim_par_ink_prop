#!/usr/bin/env python3
"""Conditional composition inversion of sample 12 and the four 306 windows.

Read-only source measurements and calibration fields. Positive changes of
water/IPA mass divided by retained pigment quantify selective losses without
mistaking increasing concentration for increasing component inventory.
"""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq
import analyze_turning_points as tp

a=tp.a
HERE=Path(__file__).resolve().parent
OUT=HERE/'results/composition_inversion_20261007'
POINT_SOURCE=a.ROOT/'plots/results_Probe306/messpunkte.csv'
RECIPE306=np.array([180.06,360.12,360.12,22.5,9002.])
K306=RECIPE306[3]/RECIPE306[0]
LABELS={'physics_anchor':'Calculator, P1-Rezepturanker',
        'physics_direct':'Calculator, ohne Zusatzoffset',
        'hybrid_anchor':'MG-Feld, P1-Anker (Lücke: Fitfehler)',
        'hybrid_direct':'MG-Feld, ohne Zusatzoffset'}


def composition(al,ipa,k=K306):
    return np.array([al,ipa,2*al,k*al,100-(3+k)*al-ipa])


def validate_windows(points):
    raw=a.rcf.load_measurements([a.SOURCE])
    raw=a.rcf.add_timestamps_and_phases(raw,'Date','UTC Time',60)
    raw=raw[raw.ProbeNr.eq(306)]
    selected=[]
    for _,p in points.iterrows():
        start=pd.Timestamp(str(p.Datum)+' '+p.Sensorfenster_Start,tz='Europe/Berlin')
        end=pd.Timestamp(str(p.Datum)+' '+p.Sensorfenster_Ende_exklusiv,tz='Europe/Berlin')
        window=raw[(raw.Measurement_Time_Local>=start)&(raw.Measurement_Time_Local<end)&raw.Gueltig.eq(1)]
        assert len(window)==p.n
        for col,target in [('Rho_M','Rho_kg_m3'),('C_M','C_m_s'),('T_M','T_C')]:
            assert np.isclose(window[col].mean(),p[target],atol=1e-8,rtol=0)
        selected.append(window.assign(Point=p.Punkt))
    pd.concat(selected).to_csv(OUT/'probe306_rohdatenfenster.csv',index=False)
    points.to_csv(OUT/'probe306_vier_messpunkte.csv',index=False)


def invert306(study,points):
    rows=[];offsets={}
    nominal=100*RECIPE306/RECIPE306.sum()
    for model in ['physics','hybrid']:
        for anchored in [True,False]:
            method=model+('_anchor' if anchored else '_direct')
            first=points.iloc[0]
            offset=first[['Rho_kg_m3','C_m_s']].to_numpy(float)-study.values(nominal,first.T_C,model)[0] if anchored else np.zeros(2)
            offsets[method]=offset.tolist()
            previous=nominal[:2].copy()
            for index,p in points.iterrows():
                target=p[['Rho_kg_m3','C_m_s']].to_numpy(float)-offset
                if index==0 and anchored:
                    comp=nominal.copy();nfev=0
                else:
                    def residual(x):
                        return (study.values(composition(*x),p.T_C,model)[0]-target)/[.1,1.]
                    solutions=[least_squares(residual,x,bounds=([.3,0],[5.5,7.]),
                                             xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=180)
                               for x in [previous,np.array([2.5,2.]),np.array([3.,3.5])]]
                    best=min(solutions,key=lambda v:np.linalg.norm(v.fun))
                    comp=composition(*best.x);nfev=best.nfev
                previous=comp[:2]
                prediction=study.values(comp,p.T_C,model)[0]+offset
                physical25=study.values(comp,25,'physics')[0]
                physicalT=study.values(comp,p.T_C,'physics')[0]
                corrected=p[['Rho_kg_m3','C_m_s']].to_numpy(float)+physical25-physicalT
                gradient=study.gradient(comp,p.T_C,model).iloc[0]
                valid=bool(np.max(np.abs(prediction-p[['Rho_kg_m3','C_m_s']].to_numpy(float)))<1e-5)
                row={'Point':p.Punkt,'Method':method,'Temperature_C':p.T_C,
                     'Rho_residual_kg_m3':prediction[0]-p.Rho_kg_m3,
                     'C_residual_m_s':prediction[1]-p.C_m_s,
                     'Rho_at25_kg_m3':corrected[0],'C_at25_m_s':corrected[1],
                     'IPA_per_Al':comp[1]/comp[0],'Water_per_Al':comp[4]/comp[0],
                     'Dry_calculated_pct':comp[0]+comp[3],
                     'Dry_measured_pct':p.Festkoerper_gemessen_pct,
                     'Dry_measured_minus_calculated_pp':p.Festkoerper_gemessen_pct-comp[0]-comp[3],
                     'inversion_matches_both_measurements':valid,
                     'critical_water_fraction_local':gradient.critical_water_fraction if gradient.stable_IDW_neighbors else np.nan,
                     'stable_IDW_neighbors':bool(gradient.stable_IDW_neighbors),
                     'outside_field_box':bool(gradient.outside_field_box),'nfev_best':nfev}
                row.update(dict(zip([c+'_pct' for c in a.COMPONENTS],comp)))
                rows.append(row)
            print(method+' complete',flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'probe306_zusammensetzungen.csv',index=False)
    intervals=[]
    for method,group in result.groupby('Method',sort=False):
        group=group.reset_index(drop=True)
        for j in range(1,4):
            before,after=group.iloc[j-1],group.iloc[j]
            di=before.IPA_per_Al-after.IPA_per_Al
            dw=before.Water_per_Al-after.Water_per_Al
            valid=bool(before.inversion_matches_both_measurements and after.inversion_matches_both_measurements)
            intervals.append({'Method':method,'Interval':before.Point+'–'+after.Point,
                              'IPA_effective_loss_per_g_Al':di,'Water_effective_loss_per_g_Al':dw,
                              'valid_endpoint_inversions':valid,
                              'Water_fraction_effective_loss':dw/(di+dw) if valid else np.nan,
                              'Water_IPA_effective_ratio':dw/di if valid else np.nan,
                              'consistent_selective_losses':bool(di>=0 and dw>=0),
                              'delta_C_at25_m_s':after.C_at25_m_s-before.C_at25_m_s,
                              'delta_IPA_concentration_pp':after.IPA_pct-before.IPA_pct})
    intervals=pd.DataFrame(intervals)
    intervals.to_csv(OUT/'probe306_intervallbilanzen.csv',index=False)
    return result,intervals,offsets


def sample12(study):
    previous=json.loads((tp.PREVIOUS/'probe12_zusammenfassung.json').read_text(encoding='utf8'))
    source=a.load_sample();m0=np.array(list(previous['recipe_masses_g'].values()))
    indices=[0,int(source.C_M.idxmin()),len(source)-1]
    rows=[]
    for fit in previous['fits']:
        if fit['mode']!='mixed':continue
        model={'physics':'physics','hybrid':'hybrid','hybrid_latest_MGfree':'sample'}[fit['model']]
        for name,index in zip(['Start','Rohminimum','Ende'],indices):
            p=source.iloc[index];m=m0.copy()
            m[1]-=fit['ipa_rate_g_h']*p.Elapsed_h
            m[4]-=fit['water_rate_g_h']*p.Elapsed_h
            comp=a.pct_from_masses(m)
            row={'Point':name,'Method':fit['model'],'Local':p.Measurement_Time_Local.isoformat(),
                 'Elapsed_h':p.Elapsed_h,'Rho_kg_m3':p.Rho_M,'C_m_s':p.C_M,'Temperature_C':p.T_M,
                 'IPA_loss_g':m0[1]-m[1],'Water_loss_g':m0[4]-m[4]}
            pred=study.values(m,p.T_M,model)[0]+[fit['Rho_offset_kg_m3'],fit['C_offset_m_s']]
            row.update({'Rho_model_residual_kg_m3':pred[0]-p.Rho_M,'C_model_residual_m_s':pred[1]-p.C_M})
            row.update(dict(zip([c+'_pct' for c in a.COMPONENTS],comp)))
            rows.append(row)
    result=pd.DataFrame(rows);result.to_csv(OUT/'probe12_bedingte_zusammensetzungen.csv',index=False)
    return result,previous


def nonunique(study):
    # An exact 95/5 instantaneous loss fraction means a ratio of 19:1.
    rows=[]
    for waterloss in [0.,10.,20.,30.]:
        def function(ipaloss):
            m=a.STANDARD.copy();m[4]-=waterloss;m[1]-=ipaloss
            return study.gradient(m,25,'physics').critical_water_IPA_ratio.iloc[0]-19.
        iloss=brentq(function,0,3.5,xtol=1e-8)
        m=a.STANDARD.copy();m[4]-=waterloss;m[1]-=iloss
        comp=a.pct_from_masses(m);v=study.values(m)[0]
        g=study.gradient(m).iloc[0]
        rows.append({'Water_loss_g_per100g':waterloss,'IPA_loss_g_per100g':iloss,
                     'critical_water_fraction':g.critical_water_fraction,
                     'Rho_kg_m3':v[0],'C_m_s':v[1],
                     **dict(zip([c+'_pct' for c in a.COMPONENTS],comp))})
    result=pd.DataFrame(rows);result.to_csv(OUT/'gleiche_95_5_grenze_verschiedene_zusammensetzungen.csv',index=False)
    return result


def figures(study,points,comps,intervals,examples):
    a.style()
    colors={'physics_anchor':'#244f79','physics_direct':'#197c92','hybrid_anchor':'#b7772c','hybrid_direct':'#7852a3'}
    x=np.arange(4)
    fig,axs=plt.subplots(1,2,figsize=(11.7,4.8))
    primary=comps[comps.Method.eq('physics_anchor')]
    for ax,raw,sd,corr,title,ylabel in zip(axs,['Rho_kg_m3','C_m_s'],['Rho_SD_kg_m3','C_SD_m_s'],['Rho_at25_kg_m3','C_at25_m_s'],['Dichte','Schallgeschwindigkeit'],['ρ [kg/m³]','c [m/s]']):
        ax.errorbar(x,points[raw],yerr=points[sd],fmt='o--',color='#88939b',label='Messung mit Fenster-SD',capsize=3)
        ax.plot(x,primary[corr],'o-',color='#244f79',label='Auf 25 °C korrigiert, bedingt')
        ax.set(xticks=x,xticklabels=['P1\n11:02','P2\n12:34','P3\n13:53','P4\n15:55'],ylabel=ylabel,title=title)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Probe 306: Vier Messfenster begrenzen den Verlauf',fontsize=15)
    fig.text(.5,.01,'Verbindungen dienen der Orientierung. Temperaturkorrektur mit dem jeweiligen P1-verankerten Calculator-Zustand.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));a.save(fig,OUT,'probe306_temperatur_und_umkehr')

    fig,axs=plt.subplots(1,2,figsize=(11.7,4.9))
    for method,group in comps.groupby('Method',sort=False):
        for ax,col in zip(axs,['IPA_pct','Water_pct']):
            values=group[col].where(group.inversion_matches_both_measurements,np.nan)
            ax.plot(x,values,'o--' if 'hybrid' in method else 'o-',color=colors[method],label=LABELS[method])
    for ax,name in zip(axs,['IPA','Wasser']):ax.set(xticks=x,xticklabels=['P1','P2','P3','P4'],ylabel=name+' [Masse-%]')
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Bedingte Rückrechnung aus Dichte, Schall und Temperatur',fontsize=15)
    fig.text(.5,.01,'Modellvarianten sind keine Konfidenzintervalle. MG-Feld bei späteren Zuständen außerhalb seiner Achsengrenzen.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));a.save(fig,OUT,'probe306_bedingte_zusammensetzung')

    fig,axs=plt.subplots(1,2,figsize=(11.7,4.8))
    for method,group in intervals.groupby('Method',sort=False):
        axs[0].plot(np.arange(3),100*group.Water_fraction_effective_loss,'o-',color=colors[method],label=LABELS[method])
    axs[0].axhline(95,color='#555',ls=':',label='95 % nur als Startorientierung')
    axs[0].set(xticks=np.arange(3),xticklabels=['P1–P2','P2–P3','P3–P4'],ylabel='Wasser im effektiven Intervallverlust [%]',title='Verluste relativ zum zurückgehaltenen Pigment')
    axs[0].legend(frameon=False,fontsize=7)
    axs[1].scatter(examples.IPA_pct,examples.Water_pct,c=examples.Water_loss_g_per100g,cmap='viridis',s=65)
    for _,p in examples.iterrows():axs[1].annotate(f'{p.Water_loss_g_per100g:g} g W-Verlust',(p.IPA_pct,p.Water_pct),xytext=(5,4),textcoords='offset points',fontsize=8)
    axs[1].set(xlabel='IPA in der Tinte [Masse-%]',ylabel='Wasser in der Tinte [Masse-%]',title='Verschiedene Zustände mit derselben 95/5-Grenze')
    axs[1].margins(.22)
    fig.suptitle('Verlustanteile und Tintenanteile unterscheiden',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94));a.save(fig,OUT,'verlust_und_zusammensetzung')


def report(points,comps,intervals,sample,previous,examples):
    def de(x,n=3):return f'{x:.{n}f}'.replace('.',',')
    def table(frame,columns,labels):
        body=''
        for _,row in frame.iterrows():
            body+='<tr>'+''.join('<td>'+html.escape(str(row[c]) if isinstance(row[c],str) else de(row[c]))+'</td>' for c in columns)+'</tr>'
        return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in labels)+'</tr></thead><tbody>'+body+'</tbody></table></div>'
    def image(name,caption):
        src=base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{src}" alt="{html.escape(caption)}"><figcaption>{caption}. <a href="{name}.pdf">PDF</a></figcaption></figure>'
    primary=comps[comps.Method.eq('physics_anchor')]
    first,last=primary.iloc[2],primary.iloc[3]
    i=intervals[(intervals.Method=='physics_anchor')&(intervals.Interval=='P3–P4')].iloc[0]
    s=sample[sample.Method.eq('hybrid_latest_MGfree')]
    ranges=sample[sample.Point.eq('Rohminimum')][['IPA_pct','Water_pct']].agg(['min','max'])
    c='''<h1>Kann man aus dem Turning Point die Tintenzusammensetzung bestimmen?</h1>
<p class="lead">Aus der 95/5-Grenze allein ergibt sich keine eindeutige Tintenzusammensetzung. Dichte und Schallgeschwindigkeit zusammen mit Temperatur, Rezepturverhältnissen und einer geprüften Kalibrierung erlauben eine bedingte Rückrechnung. Probe 306 zeigt einen deutlichen Anstieg auch nach Temperaturkorrektur; bei Probe 12 überwiegt im Rohsignal später die Erwärmung.</p>
<h2>1. Welche Information liefert 95/5?</h2>
<p>95 % Wasser im momentanen Verdunstungsverlust bedeutet R_W/(R_W+R_IPA) = 0,95. Es bedeutet weder 95 % Wasser in der verbleibenden Tinte noch einen Verlust von 95 % des anfänglichen Wassers. Bei konstanter Temperatur gilt am stationären Punkt D_W·R_W + D_IPA·R_IPA = 0. Das ist eine Bedingung für mehrere unbekannte Zustands- und Ratengrößen.</p>
<p>Für die bekannte Standardtinte wurde die lokale Grenze zu rund 95 % berechnet. Diese aus einer angenommenen Zusammensetzung berechnete Zahl kann nicht als zusätzliche unabhängige Messung derselben Zusammensetzung verwendet werden. Bei veränderlicher Temperatur kommt der Beitrag (∂c/∂T)·dT/dt hinzu. Ein Vorzeichenwechsel der Steigung ist zum Nachweis eines Minimums erforderlich.</p>
<p>Die folgenden vier Zustände entstehen durch unterschiedliche selektive Verluste aus demselben 100-g-Standardansatz. Bei allen liefert der InkCalculator bei 25 °C dieselbe lokale 95/5-Ausgleichsbedingung (exakt 19:1). Ihre Wasser- und IPA-Konzentrationen unterscheiden sich. Die weiter konzentrierten Zustände dienen der formalen Modellillustration, nicht als validierte Kalibrierpunkte.</p>'''
    c+=table(examples,['Water_loss_g_per100g','IPA_loss_g_per100g','IPA_pct','Water_pct','Rho_kg_m3','C_m_s'],['W-Verlust [g/100g]','IPA-Verlust [g/100g]','IPA [%]','Wasser [%]','ρ [kg/m³]','c [m/s]'])
    c+='''<h2>2. Was benötigt die Rückrechnung?</h2>
<p>Bei Probe 306 setzen wir PG = 2·Al und MG = (22,5/180,06)·Al. Damit bleiben zwei unabhängige Konzentrationen, beispielsweise Al und IPA. Wasser ist der Rest zu 100 %. Die beiden Messgleichungen ρ = h_ρ(Al,IPA,T) und c = h_c(Al,IPA,T) können diese beiden Konzentrationen lokal bestimmen, sofern Modell und Sensoroffsets bekannt sind und die lokale Empfindlichkeitsmatrix vollen Rang hat. Das feste Verhältnis ist eine zusätzliche Annahme, keine Folgerung aus 95/5.</p>
<p>Geprüft wurden Calculator und das MG-haltige Feld aus Proben 3–11, jeweils mit P1-Rezepturanker oder ohne zusätzlichen Offset. Beim P1-Anker wird P1 auf die angesetzte Rezeptur gesetzt und ein zeitlich konstanter Dichte-/Schalloffset abgezogen. Dadurch wird eine mögliche Verdunstung vor P1 in den Offset aufgenommen. Die Variante ohne Zusatzoffset setzt die absolute Modellkalibrierung voraus. Beide Verfahren haben unterschiedliche Annahmen; ihre Streuung ist kein statistisches Konfidenzintervall.</p>
<h2>3. Probe 306: genau vier Messfenster</h2>
<p>Quelle: die vier Fenster im Ordner plots und der Chat „Versuchsanalyse Gewichtsverlust Dichte Schallgeschwindigkeit“ im Projekt „Masterarbeit Messkennfeld“. Die gespeicherten Fenster wurden erneut mit den gültigen Rohdaten abgeglichen. Es wurden keine weiteren Rohdatenpunkte als zusätzliche unabhängige Versuchszustände verwendet.</p>'''
    c+=table(points,['Punkt','Sensor_Mitte','Rho_kg_m3','C_m_s','T_C'],['Punkt','Sensorzeit MESZ','ρ [kg/m³]','c [m/s]','T [°C]'])
    c+=image('probe306_temperatur_und_umkehr','Vier Zustände, Temperaturkorrektur auf 25 °C mit P1-verankertem Calculator')
    c+=f'''<p>P3 ist der niedrigste der vier gemessenen Werte. Das kontinuierliche Minimum und die Zusammensetzung exakt an diesem Minimum sind damit nicht lokalisiert: Für einen glatten einzelnen Talverlauf liegt die Umkehr irgendwo zwischen P2 und P4. Eine Interpolation würde zusätzliche Annahmen über den Verlauf der Verlustraten einführen.</p>
<p>Von P3 zu P4 steigt c roh um {de(points.iloc[3].C_m_s-points.iloc[2].C_m_s)} m/s. Nach der bedingten Korrektur auf 25 °C bleiben {de(last.C_at25_m_s-first.C_at25_m_s)} m/s Anstieg. Temperatur allein erklärt diese Umkehr deshalb im Calculator nicht.</p>
<p>Die folgende Tabelle ist die bedingte Calculator-Rückrechnung mit P1-Rezepturanker. Sie ist als Schätzung an den vier Sensorfenstern zu lesen, nicht als unabhängig gemessene Zusammensetzung.</p>'''
    c+=table(primary,['Point','Al_pct','IPA_pct','PG_pct','MG_pct','Water_pct','Dry_calculated_pct','Dry_measured_pct'],['Punkt','Al [%]','IPA [%]','PG [%]','MG [%]','Wasser [%]','Al+MG berechnet [%]','Festkörper gemessen [%]'])
    c+=image('probe306_bedingte_zusammensetzung','Abhängigkeit der Rückrechnung von Kalibrierung und Offsetannahme')
    c+=f'''<p><strong>Der leichte IPA-Prozentanstieg zwischen P3 und P4 ist mit weiterem IPA-Verlust vereinbar.</strong> IPA steigt hier von {de(first.IPA_pct)} auf {de(last.IPA_pct)} Masse-%, während IPA/Al von {de(first.IPA_per_Al)} auf {de(last.IPA_per_Al)} g/g fällt. Wasser/Al fällt noch stärker. Der gesamte Ansatz wird kleiner, und der prozentuale IPA-Anteil kann dadurch steigen. Die frühere Aussage im anderen Chat, dieser Anstieg bedeute zwingend IPA-Zunahme, war falsch.</p>
<p>Aus den Änderungen der auf Pigment normierten Massen ergibt sich für P3–P4 ein effektiver Wasseranteil von {de(100*i.Water_fraction_effective_loss,2)} % im Verlust. Das ist eine Intervallgröße und keine Messung des momentanen Verlustanteils am Turning Point. Bei homogenen Probenentnahmen bleibt IPA/Al beziehungsweise Wasser/Al unmittelbar gleich. Die Intervallquoten gewichten die tatsächlichen Verluste mit 1/m_Al(t); ohne Kenntnis der Entnahmezeitverläufe sind sie keine exakte ungewichtete Dampfzusammensetzung.</p>'''
    c+=table(intervals[intervals.Method.eq('physics_anchor')],['Interval','Water_fraction_effective_loss','Water_IPA_effective_ratio','delta_C_at25_m_s'],['Intervall','Effektiver Wasseranteil [0–1]','Wasser/IPA [g/g]','Δc bei 25 °C [m/s]'])
    c+=image('verlust_und_zusammensetzung','Intervallverlustanteile und Mehrdeutigkeit der 95/5-Bedingung')
    c+='''<p><strong>Grenzen der absoluten Zusammensetzung:</strong> Das bisher in plots verwendete neuere Feld aus Proben 9–11 enthält nur MG = 0. Probe 306 enthält MG; dessen punktweise Anwendung ist daher entlang der MG-Achse außerhalb des Feldes. Das hier zusätzlich geprüfte MG-haltige Feld reicht nur bis rund 0,240 % MG; spätere 306-Zustände liegen ebenfalls außerhalb seiner Achsengrenzen. Auch einzelne Calculator-Tabellen werden bei konzentrierten Zuständen extrapoliert. Die gespeicherten Modellwarnungen stehen im Manifest. Kalibrierfeldwerte sind deshalb Sensitivitätsvarianten und kein Beleg für eine präzise absolute Zusammensetzung.</p>
<p>Beim MG-Feld mit P1-Anker fand die Suche mit mehreren Startwerten für P3 keine gemeinsame genaue Lösung: Der beste Kandidat lässt rund 0,30 m/s Schallrestfehler und liegt an einem Nachbarwechsel der IDW-Interpolation. Dieser Kandidat ist in der CSV mit <em>inversion_matches_both_measurements = False</em> gekennzeichnet, wird im Zusammensetzungsdiagramm ausgelassen und nicht zur Berechnung der betreffenden Intervallquoten verwendet. Seine numerische Ableitung über den Nachbarwechsel wird ebenfalls nicht als physikalische Empfindlichkeit ausgegeben.</p>
<p>Zusätzlich liegt die P4-Festkörperprobe/Tankwägung um 15:25, das Sensorfenster dagegen um 15:52–15:57. Auch P2 ist nicht zeitgleich gewogen und beprobt. Tankverlust kann geändertes Leitungsinventar und Probenentnahmen enthalten. Die Festkörperwerte werden als unabhängiger Plausibilitätsvergleich gezeigt; sie wurden nicht gemeinsam mit den zeitversetzten Sensorwerten als exakte Gleichungen erzwungen. Bei P4 ist der berechnete Festkörpergehalt merklich höher als der frühere gemessene Wert. Das begrenzt die quantitative Aussage.</p>
<h2>4. Probe 12: bedingte Zusammensetzung am Rohminimum</h2>
<p>Probe 12 enthält kein MG. Die frühere Anpassung mit dem dazu passenden MG-freien Feld und zeitlich konstanten Wasser-/IPA-Verlustraten wurde auf Start, Rohminimum und Ende ausgewertet. Sie setzt die angesetzte Rezeptur beim ersten Sensorwert voraus. Unbekannte Verluste vor diesem Zeitpunkt und unbekannte absolute Sensoroffsets lassen sich dadurch nicht unabhängig bestimmen.</p>'''
    c+=table(s,['Point','Local','IPA_pct','Water_pct','Al_pct','PG_pct','IPA_loss_g','Water_loss_g'],['Zustand','Zeit MESZ','IPA [%]','Wasser [%]','Al [%]','PG [%]','IPA-Verlust seit Start [g]','W-Verlust seit Start [g]'])
    c+=f'''<p>Am Rohminimum ergibt diese Anpassung ungefähr {de(s.iloc[1].IPA_pct,2)} % IPA und {de(s.iloc[1].Water_pct,2)} % Wasser. Die drei geprüften Modellvarianten liefern dort IPA {de(ranges.loc['min','IPA_pct'],3)}–{de(ranges.loc['max','IPA_pct'],3)} % und Wasser {de(ranges.loc['min','Water_pct'],3)}–{de(ranges.loc['max','Water_pct'],3)} %. Das sind Modellunterschiede unter derselben Startannahme, keine Konfidenzgrenzen.</p>
<p>Vom Rohminimum bis zum Ende steigt c um {de(previous['minimum_to_end']['observed_C_change_m_s'])} m/s, während die nominal auf 25 °C korrigierte Schallgeschwindigkeit noch um {de(-previous['minimum_to_end']['C_at25_change_m_s'])} m/s sinkt. Die Umkehr im Rohsignal liefert hier somit keinen Nachweis für einen Wechsel zu 95 % Wasser im Verlust. Die Temperaturkorrektur enthält unter 25 °C teilweise Extrapolation.</p>
<h2>5. Was lässt sich belastbar ableiten?</h2>
<p>Die 95/5-Bedingung beschreibt eine lokale Bilanz der Schallbeiträge. Eine eindeutige Zusammensetzung folgt daraus erst mit weiteren Messgleichungen und geprüften Annahmen. Probe 306 unterstützt einen Wechsel zu stärker wasserbetontem Verlust im späten Intervall; die genaue momentane Verlustquote und die exakte Zusammensetzung am kontinuierlichen Minimum bleiben mit vier Zuständen offen. Probe 12 zeigt vor allem die Notwendigkeit der Temperaturkorrektur.</p>
<p>Für eine präzisere Rückrechnung sollten mindestens ein zeitgleicher unabhängiger IPA-Wert und ein Festkörperwert als Anker vorliegen. Im Bereich der erwarteten Umkehr sind zusätzliche luftarme Sensorfenster bei möglichst konstanter Temperatur sowie zeitgleiche Gesamtmassen- und Probenmessungen nötig. Damit lassen sich Offset, Zusammensetzung und Verdunstungsverhältnis voneinander trennen.</p>
<h2>Daten und Quellen</h2>
<p><a href="probe306_zusammensetzungen.csv">306: alle Modellvarianten</a>, <a href="probe306_intervallbilanzen.csv">306: Intervallbilanzen</a>, <a href="probe306_vier_messpunkte.csv">Vier Messfenster</a>, <a href="probe306_rohdatenfenster.csv">Zugehörige Rohdaten</a>, <a href="probe12_bedingte_zusammensetzungen.csv">12: bedingte Zusammensetzungen</a>, <a href="gleiche_95_5_grenze_verschiedene_zusammensetzungen.csv">Mehrdeutigkeit 95/5</a>, <a href="manifest.json">Quellen, Annahmen und Prüfungen</a>.</p>'''
    style='body{font:16px/1.6 system-ui;color:#25323b;background:#f4f6f8;margin:0}main{max-width:1100px;margin:30px auto;padding:36px;background:white}h1{font-size:32px;line-height:1.25}h2{margin-top:36px;color:#1b5364}.lead{font-size:20px;background:#e9f4f5;padding:20px}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:14px;color:#56646e}a{color:#126482}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{background:#edf2f5}@media(max-width:700px){main{padding:18px;margin:0}}'
    (OUT/'auswertung.html').write_text('<!DOCTYPE html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zusammensetzung aus Turning Points</title><style>'+style+'</style><main>'+c+'</main></html>',encoding='utf8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    study=tp.Study();points=pd.read_csv(POINT_SOURCE)
    assert len(points)==4 and list(points.Punkt)==['P1','P2','P3','P4']
    validate_windows(points)
    comps,intervals,offsets=invert306(study,points)
    sample,previous=sample12(study)
    examples=nonunique(study)
    figures(study,points,comps,intervals,examples)
    report(points,comps,intervals,sample,previous,examples)
    # Source checks and algebraic invariants, independent of display rounding.
    pct=comps[[c+'_pct' for c in a.COMPONENTS]].to_numpy()
    checks={'four_windows_match_raw_data':True,'composition_sums_to_100':bool(np.allclose(pct.sum(axis=1),100)),
            'nonnegative_compositions':bool((pct>=0).all()),
            'fixed_PG_ratio':bool(np.allclose(comps.PG_pct,2*comps.Al_pct)),
            'fixed_MG_ratio':bool(np.allclose(comps.MG_pct,K306*comps.Al_pct)),
            'same_95_5_threshold_different_compositions':bool(np.allclose(examples.critical_water_fraction,.95,atol=1e-8)),
            'physics_anchor_inversion_residual_below_1e-5':bool((np.abs(comps[comps.Method.eq('physics_anchor')][['Rho_residual_kg_m3','C_residual_m_s']])<1e-5).all().all()),
            'IPA_percent_rise_with_IPA_per_pigment_decline':bool(comps[(comps.Method=='physics_anchor')].iloc[3].IPA_pct>comps[(comps.Method=='physics_anchor')].iloc[2].IPA_pct and comps[(comps.Method=='physics_anchor')].iloc[3].IPA_per_Al<comps[(comps.Method=='physics_anchor')].iloc[2].IPA_per_Al)}
    assert all(checks.values()),checks
    source_paths=[POINT_SOURCE,a.SOURCE,a.ROOT/'ink_calculator.py',a.DEFAULT_FIELD,a.LATEST_FIELD,
                  tp.PREVIOUS/'probe12_zusammenfassung.json',Path(__file__)]
    manifest={'date':'2026-10-07','sources':{str(p.relative_to(a.ROOT)):a.digest(p) for p in source_paths},
              'recipe306_g':dict(zip(a.COMPONENTS,RECIPE306.tolist())),
              'offsets_kg_m3_and_m_s':offsets,'checks':checks,'model_warnings':sorted(study.notices),
              'chat_source':{'project':'Masterarbeit Messkennfeld','title':'Versuchsanalyse Gewichtsverlust Dichte Schallgeschwindigkeit','id':'6abfa052-ba00-83ed-8c3e-437e759b6109'},
              'scope':'Conditional inverse states; four 306 sensor windows only; no exact continuous turning time inferred',
              'limitations':['Unknown pre-P1 evaporation is absorbed by recipe anchoring',
                             'All calibration variants are extrapolated for some 306 compositions',
                             'Sample and tank weighing times differ from sensor windows',
                             'Interval loss fractions are weighted by inverse retained pigment inventory during homogeneous sampling',
                             'Model comparisons are not statistical confidence intervals']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'out':str(OUT),'306_primary':comps[comps.Method.eq('physics_anchor')].to_dict('records'),
                      '306_intervals':intervals[intervals.Method.eq('physics_anchor')].to_dict('records'),
                      '12_primary':sample[sample.Method.eq('hybrid_latest_MGfree')].to_dict('records'),
                      '95_examples':examples.to_dict('records'),'checks':checks},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
