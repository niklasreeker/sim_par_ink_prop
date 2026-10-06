#!/usr/bin/env python3
"""Study acoustic turning points under selective water/IPA mass loss.

Outputs are independent of the previous study; all source measurements,
parameter tables and residual fields are read-only. A positive loss denotes
removed mass. Ratios always refer to the instantaneous WATER / IPA loss rate.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

import analyze_evaporation_scenarios as a

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE/'results/evaporation_scenarios_20261006'
COLORS = {'physics':'#244f79', 'hybrid':'#b7772c', 'all_nodes':'#7852a3'}
MODEL_LABELS = {'physics':'InkCalculator', 'hybrid':'Kalibrierfeld, 4 Nachbarn', 'all_nodes':'Sensitivität: IDW mit allen Stützstellen'}


class Study:
    def __init__(self):
        self.calc=a.InkCalculator(str(a.ROOT/'tables_parameters'))
        self.field=a.rcf.load_model(a.DEFAULT_FIELD)
        self.latest=a.rcf.load_model(a.LATEST_FIELD)
        self.all_nodes={**self.field, 'interpolation':{**self.field['interpolation'],'neighbors':0}}
        self.models={'physics':None,'hybrid':self.field,'all_nodes':self.all_nodes,'sample':self.latest}
        self.notices=set()
        for model in [self.field,self.latest]:
            assert a.digest(a.ROOT/'ink_calculator.py')==model['calculator']['sha256']
            for name,digest in model['calculator']['table_sha256'].items():
                assert a.digest(a.ROOT/'tables_parameters'/name)==digest

    def values(self,masses,temp=25.,model='physics'):
        m=np.atleast_2d(np.asarray(masses,float))
        pct=a.pct_from_masses(m)
        v,notices=a.physics(self.calc,pct,temp)
        self.notices.update(notices)
        field=self.models[model]
        if field is not None:
            v+=a.corrections(field,pct)[['A_Rho_kg_m3','A_C_m_s']].to_numpy()
        return v

    def topology(self,masses,model):
        field=self.models[model]
        if field is None:
            return None
        nodes=pd.DataFrame(field['nodes'])
        axes=a.rcf.model_composition_axes(field)
        x=nodes[axes].to_numpy(float)
        target=a.pct_from_masses(masses)[:,[a.rcf.FIELD_AXES.index(axis) for axis in axes]]
        dist=np.linalg.norm((target[:,None,:]-x[None,:,:])/field['interpolation']['scale'],axis=2)
        k=int(field['interpolation'].get('neighbors',4))
        k=len(nodes) if k<=0 else min(k,len(nodes))
        return np.sort(np.argsort(dist,axis=1)[:,:k],axis=1)

    def gradient(self,masses,temp=25.,model='physics',relative_step=.001):
        m=np.atleast_2d(np.asarray(masses,float));n=len(m)
        t=np.broadcast_to(temp,(n,)).copy()
        h=relative_step*m.sum(axis=1)/100
        parts=[]
        steps=[]
        one_sided=[]
        for j in [4,1]:
            hi=h.copy()
            minus=m.copy();plus=m.copy()
            boundary=m[:,j]<2*hi
            # At a depleted stock only a derivative from feasible additions
            # is available. No negative mass is introduced.
            hi=np.where(boundary,h,np.minimum(h,m[:,j]/4))
            minus[:,j]-=np.where(boundary,0,hi)
            plus[:,j]+=hi
            parts.extend([minus,plus]);steps.append(hi);one_sided.append(boundary)
        stack=np.vstack(parts)
        v=self.values(stack,np.tile(t,4),model)
        d=[]
        topo=self.topology(stack,model)
        stable=np.ones(n,bool)
        for j in range(2):
            denominator=np.where(one_sided[j],steps[j],2*steps[j])
            d.append((v[(2*j)*n:(2*j+1)*n]-v[(2*j+1)*n:(2*j+2)*n])/denominator[:,None])
            if topo is not None:
                stable &= np.all(topo[(2*j)*n:(2*j+1)*n]==topo[(2*j+1)*n:(2*j+2)*n],axis=1)
        vt=self.values(np.vstack([m,m]),np.r_[t+.01,t-.01],'physics')
        beta=(vt[:n]-vt[n:])/.02
        cw,ci=d[0][:,1],d[1][:,1]
        ratio=np.divide(-ci,cw,out=np.full(n,np.nan),where=(cw>0)&(ci<0))
        fcrit=np.divide(-ci,cw-ci,out=np.full(n,np.nan),where=(cw-ci)!=0)
        outside=np.zeros(n,bool)
        if self.models[model] is not None:
            outside=a.corrections(self.models[model],a.pct_from_masses(m)).Outside_Bounding_Box.to_numpy()
        return pd.DataFrame({'CW_m_s_per_g':cw,'CI_m_s_per_g':ci,'RhoW_kg_m3_per_g':d[0][:,0],
                             'RhoI_kg_m3_per_g':d[1][:,0],'CT_m_s_K':beta[:,1],
                             'RhoT_kg_m3_K':beta[:,0],'critical_water_IPA_ratio':ratio,
                             'critical_water_fraction':fcrit,'stable_IDW_neighbors':stable,
                             'outside_field_box':outside})


def path_masses(ratio,loss):
    e=np.asarray(loss,float)
    m=np.tile(a.STANDARD,(len(e),1))
    m[:,1]-=e/(1+ratio);m[:,4]-=e*ratio/(1+ratio)
    assert np.allclose(m.sum(axis=1),100-e)
    return m


def ratio_limit(r):
    return min(a.STANDARD[1]*(1+r),a.STANDARD[4]*(1+r)/r if r>0 else np.inf)


def default_style():
    a.style()
    plt.rcParams.update({'axes.titlesize':11,'axes.labelsize':10})


def initial_study(study,out):
    rows=[]
    for temp in [20.,25.,30.]:
        for model in ['physics','hybrid','all_nodes']:
            row=study.gradient(a.STANDARD,temp,model).iloc[0].to_dict()
            rows.append({'Temperature_C':temp,'model':model,**row})
    result=pd.DataFrame(rows)
    result.to_csv(out/'start_grenzverhaeltnisse.csv',index=False)
    steps=[]
    for model in ['physics','hybrid','all_nodes']:
        for step in [.01,.001,.0001]:
            row=study.gradient(a.STANDARD,25,model,step).iloc[0].to_dict()
            steps.append({'model':model,'step_g_per100g':step,**row})
    pd.DataFrame(steps).to_csv(out/'ableitung_schrittweiten.csv',index=False)

    f=np.linspace(0,1,1001)
    fig,axs=plt.subplots(1,2,figsize=(11.7,4.8))
    for model in ['physics','hybrid','all_nodes']:
        row=result[(result.model==model)&(result.Temperature_C==25)].iloc[0]
        slope=row.CI_m_s_per_g*(1-f)+row.CW_m_s_per_g*f
        axs[0].plot(100*f,slope,color=COLORS[model],label=MODEL_LABELS[model],lw=1.8)
        axs[0].scatter([100*row.critical_water_fraction],[0],color=COLORS[model],s=24)
        axs[1].plot(100*f,slope,color=COLORS[model],lw=1.8)
        axs[1].axvline(100*row.critical_water_fraction,color=COLORS[model],ls=':',lw=1)
    for ax in axs:
        ax.axhline(0,color='#333',lw=1)
        ax.set(xlabel='Wasseranteil an der verdunsteten Masse [%]',ylabel='dc/dE [m/s je g Verlust aus 100 g Anfangstinte]')
    axs[0].set_title('Gesamtbereich: 50/50 liegt deutlich im fallenden Bereich')
    axs[1].set_title('Vergrößerung der lokalen Umschlaggrenze')
    axs[1].set_xlim(92,98);axs[1].set_ylim(-.24,.27)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Standardtinte bei konstanten 25 °C: Wann steigt oder fällt c?',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.95))
    a.save(fig,out,'verlustverhaeltnis_start')
    return result,steps


def phase_map(study,out):
    water=np.linspace(0,20,61);ipa=np.linspace(0,3.5,61)
    wi,ii=np.meshgrid(water,ipa,indexing='ij')
    m=np.tile(a.STANDARD,(wi.size,1));m[:,4]-=wi.ravel();m[:,1]-=ii.ravel()
    c=study.values(m,25,'physics')[:,1].reshape(wi.shape)
    cw,ci=np.gradient(c,water,ipa,edge_order=2)
    critical=-ci/cw
    fraction=-ci/(cw-ci)
    outside=a.corrections(study.field,a.pct_from_masses(m)).Outside_Bounding_Box.to_numpy().reshape(wi.shape)
    frame=pd.DataFrame({'IPA_loss_g_per100g':ii.ravel(),'Water_loss_g_per100g':wi.ravel(),
                        'C_m_s':c.ravel(),'CW_grid_m_s_g':cw.ravel(),'CI_grid_m_s_g':ci.ravel(),
                        'critical_water_IPA_ratio':critical.ravel(),'critical_water_fraction':fraction.ravel(),
                        'outside_field_box':outside.ravel()})
    frame.to_csv(out/'grenzverhaeltnis_kennfeld.csv',index=False)
    fig,ax=plt.subplots(figsize=(9.5,6.4))
    mesh=ax.pcolormesh(ii,wi,critical,shading='auto',cmap='viridis')
    levels=np.arange(12,40,2)
    contours=ax.contour(ii,wi,critical,levels=levels,colors='white',linewidths=.8)
    ax.clabel(contours,fmt='%g',fontsize=8)
    ax.contour(ii,wi,outside.astype(float),levels=[.5],colors='#e28625',linewidths=2)
    ax.scatter([0],[0],s=45,color='#dd573a',zorder=5)
    ax.set(xlabel='Bereits verdunstetes IPA [g je 100 g Anfangstinte]',
           ylabel='Bereits verdunstetes Wasser [g je 100 g Anfangstinte]',
           title='Die momentane Grenze hängt von der verbleibenden Zusammensetzung ab')
    fig.colorbar(mesh,ax=ax,label='Kritische momentane Wasser-/IPA-Verlustrate [g/g]')
    fig.text(.5,.02,'InkCalculator bei 25 °C. Orange: Grenzen des Kalibrierfelds. Das Raster zeigt Zustände, keine Zeitentwicklung.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    a.save(fig,out,'grenzverhaeltnis_zusammensetzung')
    return frame


def directional(study,r,e,model='physics'):
    h=.001
    masses=path_masses(r,np.array([e-h,e+h]))
    c=study.values(masses,25,model)[:,1]
    return float((c[1]-c[0])/(2*h))


def path_study(study,out):
    ratios=[1.,4.,10.,17.,17.5,17.65,17.7,17.71,17.72,17.75,18.,20.,30.]
    frames=[];roots=[]
    baseline=study.values(a.STANDARD)[0]
    for r in ratios:
        loss=np.linspace(0,min(20,.999*ratio_limit(r)),301)
        masses=path_masses(r,loss)
        values=study.values(masses)
        frame=pd.DataFrame({'ratio_Water_IPA':r,'loss_g_per100g':loss,'C_m_s':values[:,1],
                            'delta_C_m_s':values[:,1]-baseline[1],'Rho_kg_m3':values[:,0]})
        for j,name in enumerate(a.COMPONENTS):
            frame[f'{name}_wt_pct']=a.pct_from_masses(masses)[:,j]
        slopes=np.gradient(values[:,1],loss,edge_order=2)
        frame['dc_dE_grid_m_s_g']=slopes
        frames.append(frame)
        crosses=np.flatnonzero(slopes[:-1]*slopes[1:]<0)
        for i in crosses:
            lo=loss[max(0,i-1)];hi=loss[min(len(loss)-1,i+2)]
            gl,gh=directional(study,r,lo),directional(study,r,hi)
            if gl*gh>=0:
                continue
            e=brentq(lambda value:directional(study,r,value),lo,hi,xtol=1e-7)
            m=path_masses(r,[e]);pct=a.pct_from_masses(m)[0]
            y=study.values(m)[0]
            grad=study.gradient(m).iloc[0]
            h=.03
            cv=study.values(path_masses(r,[e-h,e,e+h]))[:,1]
            curvature=(cv[2]-2*cv[1]+cv[0])/h**2
            roots.append({'ratio_Water_IPA':r,'type':'minimum' if gl<0<gh else 'maximum',
                          'loss_g_per100g':e,'IPA_loss_g_per100g':e/(r+1),'Water_loss_g_per100g':e*r/(r+1),
                          'C_m_s':float(y[1]),'delta_C_from_start_m_s':float(y[1]-baseline[1]),
                          'curvature_m_s_g2':float(curvature),'critical_ratio_at_point':float(grad.critical_water_IPA_ratio),
                          'Rho_change_kg_m3':float(y[0]-baseline[0]),
                          **{f'{name}_wt_pct':float(pct[j]) for j,name in enumerate(a.COMPONENTS)}})
    paths=pd.concat(frames,ignore_index=True)
    paths.to_csv(out/'isotherme_pfade.csv',index=False)
    root_frame=pd.DataFrame(roots)
    root_frame.to_csv(out/'isotherme_umkehrpunkte.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11.7,5))
    for r in [1,4,10,18,20,30]:
        part=paths[paths.ratio_Water_IPA==r]
        axs[0].plot(part.loss_g_per100g,part.delta_C_m_s,label=f'{r:g}:1',lw=1.8)
    for r in [17.5,17.65,17.7,17.71,17.72,17.75,18]:
        part=paths[paths.ratio_Water_IPA==r]
        line=axs[1].plot(part.loss_g_per100g,part.delta_C_m_s,label=f'{r:g}:1',lw=1.6)[0]
        for root in [x for x in roots if x['ratio_Water_IPA']==r]:
            axs[1].scatter(root['loss_g_per100g'],root['delta_C_from_start_m_s'],color=line.get_color(),marker='o',s=30)
    for ax in axs:
        ax.axhline(0,color='#777',lw=.7)
        ax.set(xlabel='Gesamtverlust [g je 100 g Anfangstinte]',ylabel='Änderung der Schallgeschwindigkeit [m/s]')
        ax.legend(title='Wasser : IPA',frameon=False,fontsize=8,ncol=2)
    axs[0].set_title('Konstantes Verhältnis der verdunsteten Massen')
    axs[1].set_title('Nahe der Grenze: kleine echte Minima sind möglich')
    fig.suptitle('Isotherme Verläufe bei 25 °C mit unveränderten Al-/PG-/MG-Massen',fontsize=15)
    fig.text(.5,.01,'InkCalculator. Kreise markieren rechnerische Minima. Ihre sehr geringe Tiefe ist keine experimentell abgesicherte Vorhersage.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95))
    a.save(fig,out,'isotherme_verlaeufe')

    # Derivatives of the saved 4-neighbor correction are only meaningful
    # within a fixed neighbor set; switches are explicitly exported.
    loss=np.linspace(0,20,401);r=17.7;masses=path_masses(r,loss)
    frames=[]
    fig,axs=plt.subplots(1,2,figsize=(11.7,5))
    for model in ['physics','hybrid','all_nodes']:
        values=study.values(masses,25,model)
        topo=study.topology(masses,model)
        change=np.zeros(len(loss),bool)
        if topo is not None:
            change[1:]=np.any(topo[1:]!=topo[:-1],axis=1)
        delta=values[:,1]-values[0,1]
        axs[0].plot(loss,delta,color=COLORS[model],label=MODEL_LABELS[model],lw=1.6)
        if model=='hybrid':
            for e in loss[change]:
                axs[0].axvline(e,color=COLORS[model],ls=':',lw=.7,alpha=.5)
        grad=study.gradient(masses,25,model)
        valid=grad.stable_IDW_neighbors.to_numpy(bool)
        axs[1].plot(loss,np.where(valid,grad.critical_water_IPA_ratio,np.nan),color=COLORS[model],lw=1.7)
        frames.append(pd.DataFrame({'model':model,'loss_g_per100g':loss,'delta_C_m_s':delta,
                                    'IDW_neighbor_switch':change,**{col:grad[col].to_numpy() for col in grad.columns}}))
    axs[0].set(xlabel='Gesamtverlust [g je 100 g Anfangstinte]',ylabel='Änderung von c [m/s]',title='Nahe Null ist die Kalibrierungsstruktur entscheidend')
    axs[1].axhline(r,color='#555',ls='--',label='Vorgegebenes Verhältnis 17,7:1')
    axs[1].set(xlabel='Gesamtverlust [g je 100 g Anfangstinte]',ylabel='Kritische Wasser-/IPA-Verlustrate [g/g]',title='Sprünge der Nachbarauswahl erzeugen keine sicheren Umkehrpunkte')
    axs[0].legend(frameon=False,fontsize=8);axs[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Kalibrierungssensitivität entlang des Pfads Wasser:IPA = 17,7:1',fontsize=15)
    fig.text(.5,.01,'IDW mit allen Stützstellen ist eine Sensitivitätsrechnung, keine neu validierte Kalibrierung. Gepunktete Linien: Nachbarwechsel.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95))
    a.save(fig,out,'kalibrierungssensitivitaet')
    pd.concat(frames,ignore_index=True).to_csv(out/'kalibrierungssensitivitaet.csv',index=False)
    return paths,roots


def turning_locus(study,out):
    """Ratios whose fixed-ratio path is stationary after a given loss."""
    rows=[]
    for e in np.linspace(0,20,41):
        r=brentq(lambda value:directional(study,value,e),15,20,xtol=1e-9)
        m=path_masses(r,[e]);pct=a.pct_from_masses(m)[0]
        h=.03
        c=study.values(path_masses(r,[e-h,e,e+h]))[:,1]
        curvature=(c[2]-2*c[1]+c[0])/h**2
        rows.append({'loss_g_per100g':float(e),'ratio_Water_IPA':float(r),
                     'IPA_loss_g_per100g':float(e/(r+1)),'Water_loss_g_per100g':float(e*r/(r+1)),
                     'curvature_m_s_g2':float(curvature),
                     **{f'{name}_wt_pct':float(pct[j]) for j,name in enumerate(a.COMPONENTS)}})
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'umkehrgrenze_feste_verhaeltnisse.csv',index=False)
    return {'first_20g_minimum_ratio_lower':float(frame.ratio_Water_IPA.iloc[-1]),
             'first_20g_minimum_ratio_upper':float(frame.ratio_Water_IPA.iloc[0]),
             'locus_ratio_monotone':bool(np.all(np.diff(frame.ratio_Water_IPA)<0)),
             'locus_curvature_positive':bool((frame.curvature_m_s_g2>0).all())}


def changing_ratio_example(study,out):
    loss=np.linspace(0,7,701)
    f=.5+(.985-.5)*(.5+.5*np.tanh((loss-3)/.55))
    ipa=cumulative_trapezoid(1-f,loss,initial=0)
    water=loss-ipa
    masses=np.tile(a.STANDARD,(len(loss),1));masses[:,1]-=ipa;masses[:,4]-=water
    values=study.values(masses)
    grad=study.gradient(masses)
    g=grad.CI_m_s_per_g.to_numpy()*(1-f)+grad.CW_m_s_per_g.to_numpy()*f
    crossings=np.flatnonzero((g[:-1]<0)&(g[1:]>0))
    i=int(crossings[0]);e=float(np.interp(0,g[i:i+2],loss[i:i+2]))
    records=pd.DataFrame({'loss_g_per100g':loss,'IPA_loss_g_per100g':ipa,'Water_loss_g_per100g':water,
                          'instant_water_fraction':f,'instant_water_IPA_ratio':f/(1-f),'critical_ratio':grad.critical_water_IPA_ratio,
                          'dc_dE_m_s_g':g,'C_m_s':values[:,1]})
    records.to_csv(out/'beispiel_aenderndes_verhaeltnis.csv',index=False)
    loss2=np.linspace(0,15,601)
    ipa2=np.minimum(loss2/2,a.STANDARD[1]);water2=loss2-ipa2
    masses2=np.tile(a.STANDARD,(len(loss2),1));masses2[:,1]-=ipa2;masses2[:,4]-=water2
    c2=study.values(masses2)[:,1]
    pd.DataFrame({'loss_g_per100g':loss2,'IPA_loss_g_per100g':ipa2,'Water_loss_g_per100g':water2,'C_m_s':c2}).to_csv(out/'beispiel_IPA_erschoepfung.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(11.7,5))
    axs[0].plot(loss,values[:,1],color='#244f79',lw=2)
    axs[0].axvline(e,color='#c16b18',ls='--',label=f'Umkehr bei E ≈ {e:.2f} g/100 g')
    axs[0].set(xlabel='Gesamtverlust [g je 100 g Anfangstinte]',ylabel='Schallgeschwindigkeit [m/s]',title='Wasseranteil im Verlust steigt von 50 auf 98,5 %')
    axs[0].legend(frameon=False,fontsize=8)
    axs[1].plot(loss2,c2,color='#197c92',lw=2)
    axs[1].axvline(2*a.STANDARD[1],color='#c16b18',ls='--',label='IPA-Vorrat erschöpft')
    axs[1].set(xlabel='Gesamtverlust [g je 100 g Anfangstinte]',ylabel='Schallgeschwindigkeit [m/s]',title='Zuerst 50/50, nach IPA-Erschöpfung nur Wasser')
    axs[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Zwei isotherme Ursachen für ein Minimum: Änderung der Verlustraten',fontsize=15)
    fig.text(.5,.01,'Vorgegebene Beispiele bei 25 °C. Keine Rekonstruktion des Verdunstungsmechanismus von Probe 12.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95))
    a.save(fig,out,'isotherme_minimum_beispiele')
    return {'changing_ratio_turn_E_g_per100g':e,'stock_exhaustion_E_g_per100g':2*a.STANDARD[1]}


def local_derivative(time,values,half_width_min):
    time=np.asarray(time,float);values=np.asarray(values,float)
    slopes=[];fitted=[]
    for t in time:
        mask=abs(time-t)<=half_width_min/60+1e-7
        coef=np.polyfit(time[mask]-t,values[mask],2)
        slopes.append(coef[-2]);fitted.append(coef[-1])
    return np.array(slopes),np.array(fitted)


def zero_crossings(time,values):
    i=np.flatnonzero((values[:-1]<0)&(values[1:]>0))
    return [float(np.interp(0,values[j:j+2],time[j:j+2])) for j in i]


def sample_study(study,out):
    provenance=json.loads((PREVIOUS/'manifest.json').read_text(encoding='utf-8'))
    assert provenance['source_sha256']==a.digest(a.SOURCE)
    assert provenance['calculator_sha256']==a.digest(a.ROOT/'ink_calculator.py')
    assert provenance['sample_plot_field_sha256']==a.digest(a.LATEST_FIELD)
    prior=json.loads((PREVIOUS/'probe12_zusammenfassung.json').read_text(encoding='utf-8'))
    data=pd.read_csv(PREVIOUS/'probe12_ausgewertet.csv')
    time=data.Elapsed_h.to_numpy()
    local=pd.to_datetime(data.Measurement_Time_UTC,utc=True).dt.tz_convert('Europe/Berlin')
    masses0=np.array([prior['recipe_masses_g'][c] for c in a.COMPONENTS])
    ri=prior['fit']['ipa_rate_g_h'];rw=prior['fit']['water_rate_g_h'];total=ri+rw
    masses=np.tile(masses0,(len(data),1));masses[:,1]-=ri*time;masses[:,4]-=rw*time
    grad=study.gradient(masses,data.T_M.to_numpy(),'sample')
    chemical=grad.CI_m_s_per_g.to_numpy()*ri+grad.CW_m_s_per_g.to_numpy()*rw
    tcrit=-chemical/grad.CT_m_s_K.to_numpy()
    rawmin=int(data.C_M.argmin())
    c25=study.values(masses,25,'sample')[:,1]
    selection=[];all_derivatives=[]
    for width in [10,20,30]:
        dt,tf=local_derivative(time,data.T_M,width)
        dc,cf=local_derivative(time,data.C_M,width)
        predicted=chemical+grad.CT_m_s_K.to_numpy()*dt
        crosses=zero_crossings(time,dc)
        predicted_crosses=zero_crossings(time,predicted)
        def nearest(crosses):
            return min(crosses,key=lambda t:abs(t-time[rawmin])) if crosses else None
        empirical=nearest(crosses);model=nearest(predicted_crosses)
        stamp=lambda t: None if t is None else (local.iloc[0]+pd.Timedelta(hours=t)).isoformat()
        selection.append({'half_window_min':width,'observed_slope_zero_local':stamp(empirical),'model_balance_zero_local':stamp(model),
                          'all_observed_negative_to_positive_crossings_local':[stamp(t) for t in crosses],
                          'all_model_negative_to_positive_crossings_local':[stamp(t) for t in predicted_crosses]})
        for i in range(len(data)):
            all_derivatives.append({'local':local.iloc[i].isoformat(),'elapsed_h':time[i],'half_window_min':width,
                                     'T_rate_C_h':dt[i],'observed_C_rate_m_s_h':dc[i],
                                     'chemical_C_rate_m_s_h':chemical[i],'temperature_C_rate_m_s_h':grad.CT_m_s_K.iloc[i]*dt[i],
                                     'predicted_C_rate_m_s_h':predicted[i],'critical_heating_rate_C_h':tcrit[i]})
        if width==20:
            main_dt,main_dc,main_predicted=dt,dc,predicted
    frame=pd.DataFrame(all_derivatives)
    frame.to_csv(out/'probe12_ableitungen.csv',index=False)
    grad.to_csv(out/'probe12_lokale_empfindlichkeiten.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(10.5,7.2),sharex=True)
    axs[0].plot(local,main_dt,color='#b44b4b',lw=2,label='Gemessene Erwärmungsrate, lokal geglättet')
    axs[0].plot(local,tcrit,color='#197c92',lw=2,label='Erwärmung, die den Zusammensetzungseffekt kompensiert')
    axs[0].set(ylabel='Temperaturänderung [°C/h]',title='Wird die erforderliche Erwärmung überschritten, steigt c')
    axs[1].plot(local,main_dc,color='#244f79',lw=2,label='Ableitung der gemessenen Schallgeschwindigkeit')
    axs[1].plot(local,main_predicted,color='#b7772c',ls='--',lw=1.7,label='Bilanz aus Verlusten und Erwärmung')
    axs[1].axhline(0,color='#333',lw=1)
    axs[1].set(ylabel='Änderung von c [m/s pro Stunde]',xlabel='Uhrzeit (MESZ)',title='Minimum: Wechsel der Ableitung von negativ nach positiv')
    for ax in axs:
        ax.axvline(local.iloc[rawmin],color='#888',ls=':',label='Rohdatenminimum 13:42')
        ax.legend(frameon=False,fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=local.dt.tz))
    fig.suptitle('Probe 12: Turning Point als Gleichgewicht zweier Änderungsbeiträge',fontsize=15)
    fig.text(.5,.01,'Lokale quadratische Fits im Fenster ±20 min. Verlustaufteilung aus vorherigem Fit; keine unabhängig gemessenen Verdunstungsraten.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95))
    a.save(fig,out,'probe12_umkehrbedingung')

    point=grad.iloc[rawmin]
    f=np.linspace(0,1,201);heating=np.linspace(-.15,.6,201)
    ff,tt=np.meshgrid(f,heating)
    c_rate=total*(point.CI_m_s_per_g*(1-ff)+point.CW_m_s_per_g*ff)+point.CT_m_s_K*tt
    fig,ax=plt.subplots(figsize=(9.5,6))
    scale=max(abs(c_rate.min()),abs(c_rate.max()))
    im=ax.pcolormesh(100*ff,tt,c_rate,shading='auto',cmap='RdBu_r',vmin=-scale,vmax=scale)
    zero=ax.contour(100*ff,tt,c_rate,levels=[0],colors='black',linewidths=2)
    ax.clabel(zero,fmt={0:'dc/dt = 0'},fontsize=10)
    fraction=rw/total
    ax.scatter([100*fraction],[tcrit[rawmin]],s=65,color='#ffbf47',edgecolor='#333',label='Grenze bei angepasstem Wasser/IPA-Verhältnis')
    ax.axvline(100*fraction,color='#555',ls=':',lw=1)
    ax.set(xlabel='Momentaner Wasseranteil an der verdunsteten Masse [%]',ylabel='Erwärmungsrate [°C/h]',
           title=f'Probe 12, Zustand am Rohminimum: Gesamtverlustrate {total:.2f} g/h')
    fig.colorbar(im,ax=ax,label='Änderung von c [m/s pro Stunde]')
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    fig.text(.5,.01,'Negative Werte: c fällt. Positive Werte: c steigt. Verwendet werden die Zusammensetzung und das MG-freie Kalibrierfeld der Probe.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    a.save(fig,out,'temperatur_verlust_grenze')
    summary={'instant_water_IPA_ratio_fitted':rw/ri,'instant_water_fraction_fitted':rw/total,
             'fitted_IPA_rate_g_h':ri,'fitted_Water_rate_g_h':rw,'fitted_total_rate_g_h':total,
             'critical_heating_rate_C_h_range':[float(tcrit.min()),float(tcrit.max())],
             'critical_heating_rate_at_raw_min_C_h':float(tcrit[rawmin]),
             'isothermal_critical_ratio_at_raw_min':float(point.critical_water_IPA_ratio),
             'chemical_C_rate_at_raw_min_m_s_h':float(chemical[rawmin]),
             'CT_at_raw_min_m_s_K':float(point.CT_m_s_K),
             'isothermal_C_model_change_m_s':float(c25[-1]-c25[0]),
             'IDW_derivative_neighbor_changes':int((~grad.stable_IDW_neighbors).sum()),
             'smoothing_sensitivity':selection}
    (out/'probe12_umkehr_zusammenfassung.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def write_report(out,initial,roots,example,sample,study,locus):
    def de(v,d=3):
        return f'{float(v):.{d}f}'.replace('.',',')
    def table(headers,rows):
        return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def image(name,caption):
        src=base64.b64encode((out/(name+'.png')).read_bytes()).decode('ascii')
        return f'<figure><img src="data:image/png;base64,{src}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}. <a href="{name}.pdf">PDF herunterladen</a></figcaption></figure>'
    p=initial[(initial.model=='physics')&(initial.Temperature_C==25)].iloc[0]
    h=initial[(initial.model=='hybrid')&(initial.Temperature_C==25)].iloc[0]
    rows=[]
    for _,r in initial.iterrows():
        rows.append([de(r.Temperature_C,0),MODEL_LABELS[r.model],de(r.CW_m_s_per_g,4),de(r.CI_m_s_per_g,4),de(r.critical_water_IPA_ratio,2),de(100*r.critical_water_fraction,2)])
    rootrows=[[de(r['ratio_Water_IPA'],2),r['type'],de(r['loss_g_per100g'],3),de(r['IPA_loss_g_per100g'],3),de(r['Water_loss_g_per100g'],3),de(r['delta_C_from_start_m_s'],5),de(r['curvature_m_s_g2'],6)] for r in roots]
    smoothing=[]
    for r in sample['smoothing_sensitivity']:
        observed='—' if r['observed_slope_zero_local'] is None else pd.Timestamp(r['observed_slope_zero_local']).strftime('%H:%M')
        model='—' if r['model_balance_zero_local'] is None else pd.Timestamp(r['model_balance_zero_local']).strftime('%H:%M')
        smoothing.append([f"±{r['half_window_min']} min",observed,model])
    content=f'''<h1>Wann kehrt sich die Änderung der Schallgeschwindigkeit um?</h1>
<p class="subtitle">Untersuchung vom 06.10.2026 mit InkCalculator, Kalibrierfeld und Probe 12. Bezug: absolute verdunstete Massen.</p>
<p class="lead">Nahe der Standardtinte bei konstanten 25 °C muss die momentane Wasserverlustrate etwa 17–18-mal so hoch sein wie die IPA-Verlustrate, damit die Schallgeschwindigkeit steigt. Das entspricht ungefähr 95 % Wasser in der verdunsteten Masse. Diese Grenze ist lokal und modellabhängig.</p>
<h2>1. Die lokale Umschlaggrenze</h2>
<p>Der Ansatz enthält je 100 g: 1,81 g Al/Pigment, 3,63 g IPA, 3,63 g PG, 0,22 g MG und 90,71 g Wasser. Al/Pigment, PG und MG behalten ihre absoluten Massen. Nach jedem Verlust werden alle Massenanteile aus der verbleibenden Gesamtmasse neu berechnet. „Al“ bezeichnet im Calculator die ganze gekapselte Pigmentphase.</p>
<p>Die Empfindlichkeiten D<sub>W</sub> = ∂c/∂E<sub>W</sub> und D<sub>I</sub> = ∂c/∂E<sub>IPA</sub> beziehen sich auf entfernte Gramm, während die übrigen absoluten Komponentenmassen konstant bleiben. Sie enthalten somit auch die Neunormierung von Al, PG und MG. Sie sind nicht einfach die Steigungen binärer IPA-/Wassertabellen.</p>
<p>Bei 25 °C liefert der InkCalculator: D<sub>W</sub> = +{de(p.CW_m_s_per_g,4)} m/s je g Wasserverlust und D<sub>I</sub> = {de(p.CI_m_s_per_g,4)} m/s je g IPA-Verlust aus dem anfänglich 100-g-Ansatz. Ein Gramm IPA-Verlust wirkt akustisch daher etwa {de(p.critical_water_IPA_ratio,2)}-mal stärker als ein Gramm Wasserverlust, mit entgegengesetztem Vorzeichen.</p>
<div class="formula">dc/dt = D<sub>W</sub> · R<sub>W</sub> + D<sub>I</sub> · R<sub>IPA</sub> + (∂c/∂T) · dT/dt</div>
<p>R<sub>W</sub> und R<sub>IPA</sub> sind positive momentane Verlustraten in g/h. Bei konstanter Temperatur gilt:</p>
<div class="formula">r<sub>krit</sub> = R<sub>W</sub>/R<sub>IPA</sub> = −D<sub>I</sub>/D<sub>W</sub><br>r &lt; r<sub>krit</sub>: c sinkt. r &gt; r<sub>krit</sub>: c steigt.</div>
<p>Die Ungleichungen setzen D<sub>W</sub> &gt; 0 und D<sub>I</sub> &lt; 0 voraus. Das gilt im hier untersuchten Bereich nahe der Ausgangsrezeptur. Bei Gleichheit ist nur die momentane Steigung null. Ein echtes Minimum erfordert einen Vorzeichenwechsel von negativ zu positiv.</p>
{image('verlustverhaeltnis_start','Änderung von c als Funktion des Wasseranteils im momentanen Massenverlust')}
<p>Mit dem gespeicherten MG-haltigen Feld aus Proben 3–11 beträgt die lokale Grenze {de(h.critical_water_IPA_ratio,2)}:1 beziehungsweise {de(100*h.critical_water_fraction,2)} % Wasser im Verlust. Der InkCalculator allein ergibt {de(p.critical_water_IPA_ratio,2)}:1 beziehungsweise {de(100*p.critical_water_fraction,2)} %. Diese Zahlen sind keine allgemeine Stoffkonstante und keine experimentell bestimmte Dampfzusammensetzung.</p>
{table(['T [°C]','Modell','D_W [m/s/g]','D_IPA [m/s/g]','Wasser/IPA kritisch [g/g]','Wasser im Verlust [%]'],rows)}
<p>Die Rechnung bei 20 °C verwendet in Teilen der PG-Tabellen eine Temperaturextrapolation. Die Rechnung „alle Stützstellen“ ändert die Interpolationsstruktur ausschließlich für eine Sensitivitätsprüfung. Sie ist keine neu validierte Kalibrierung.</p>
<h2>2. Momentane Rate und bereits verlorene Masse sind verschiedene Größen</h2>
<p>r<sub>krit</sub> betrifft R<sub>W</sub>/R<sub>IPA</sub> am aktuellen Zustand. Das bisherige Verhältnis E<sub>W</sub>/E<sub>IPA</sub> kann anders sein. Ein Ansatz kann zuerst viel IPA verlieren und erst später überwiegend Wasser. Er steigt dann bereits wieder, obwohl c noch unter seinem Anfangswert liegt. Ebenso ist c = c<sub>Start</sub> eine andere Bedingung als dc/dt = 0.</p>
<p>Bei festem Verlustverhältnis r und Gesamtverlust E gilt E<sub>IPA</sub> = E/(1+r), E<sub>W</sub> = rE/(1+r). Die Zeit eines Umkehrpunkts lässt sich erst mit dem Gesamtverlustverlauf E(t) bestimmen. Bei konstanter Gesamtverlustrate R ist t<sub>*</sub> = E<sub>*</sub>/R. E und R müssen sich auf dieselbe Ansatzmasse beziehen.</p>
{image('grenzverhaeltnis_zusammensetzung','Grenzverhältnis im Raum bereits verdunsteter Wasser- und IPA-Massen')}
<p>Das Raster umfasst 0–20 g Wasser- und 0–3,5 g IPA-Verlust je 100 g Anfangstinte. Es zeigt lokale Zustände bei 25 °C. Die Ableitungen des Rasters werden mit zentralen Differenzen der Modellwerte angenähert; genaue Grenzwerte und Umkehrpunkte werden separat mit kleineren Schritten gerechnet. Orange markiert die Achsengrenzen des gespeicherten Kalibrierfelds. Innerhalb dieser Grenzen ist eine vollständige lokale Abdeckung nicht automatisch belegt.</p>
<h2>3. Echte isotherme Minima</h2>
<p>Auch bei zeitlich konstantem Verlustverhältnis kann ein Minimum entstehen: Die Zusammensetzung verändert sich, dadurch verschiebt sich r<sub>krit</sub>, und das vorgegebene r kann die Grenze kreuzen. Die separat berechnete Umkehrgrenze liegt für die ersten 20 g Gesamtverlust im engen Bereich <strong>{de(locus['first_20g_minimum_ratio_lower'],3)}:1 bis {de(locus['first_20g_minimum_ratio_upper'],3)}:1</strong>. Die berechnete Grenzkurve ist monoton, und die Krümmung an ihren stationären Zuständen ist positiv. An den Randwerten liegt der stationäre Punkt am Anfang beziehungsweise bei 20 g Verlust; ein inneres Minimum verlangt ein Verhältnis dazwischen.</p>
{image('isotherme_verlaeufe','Isotherme Pfade bei fest vorgegebenem Verlustverhältnis')}
{table(['Wasser:IPA [g/g]','Typ','Gesamtverlust [g/100g]','IPA-Verlust [g/100g]','Wasserverlust [g/100g]','c−c_Start [m/s]','Krümmung [m/s/g²]'],rootrows)}
<p>Diese Minima sind sehr flach. Ihre Tiefe ist mit den Modellabweichungen des Kalibrierfelds und teilweise mit der gespeicherten Kurzzeitstreuung der Messung vergleichbar oder kleiner. Die genaue Position ist deshalb eine formale Modellvorhersage, kein belastbar nachgewiesener experimenteller Turning Point.</p>
{image('kalibrierungssensitivitaet','Abhängigkeit sehr flacher Verläufe von der Kalibrierinterpolation')}
<p>Das ursprüngliche Feld verwendet die vier nächsten Stützstellen. Beim Wechsel dieser Gruppe kann A(w) springen. Eine Differenz über einen solchen Sprung ist keine physikalische Ableitung. Diese Stellen sind in den Daten gekennzeichnet. Schon innerhalb einer festen Nachbargruppe können die Gradienten des Residualfelds den Verlauf nahe dc/dt = 0 stark verändern. Ein Umkehrpunkt sollte deshalb unter kleineren Rechenschritten, verschiedenen plausiblen Kalibrierungen und experimentellen Wiederholungen erhalten bleiben.</p>
<p>Deutlichere isotherme Minima entstehen, wenn sich das Verhältnis der momentanen Verlustraten verändert:</p>
{image('isotherme_minimum_beispiele','Vorgegebene Beispiele mit wechselndem Verlustverhältnis und IPA-Erschöpfung')}
<p>Links steigt der Wasseranteil im Verlust kontinuierlich von 50 auf 98,5 %. Die Bilanz wechselt bei ungefähr {de(example['changing_ratio_turn_E_g_per100g'],2)} g Gesamtverlust je 100 g Ansatz von fallend zu steigend. Rechts verdunstet zuerst IPA/Wasser 50/50; bei 7,26 g Gesamtverlust ist der anfängliche IPA-Vorrat erschöpft. Danach ist ausdrücklich nur weiterer Wasserverlust vorgegeben. Das erzeugt ein Minimum mit einem Wechsel der einseitigen Steigungen. Ein konstantes 50/50-Verhältnis darf nach IPA-Erschöpfung nicht fortgesetzt werden.</p>
<h2>4. Charakterisierung eines Umkehrpunkts</h2>
<p>Ein Minimum ist ein Wechsel von dc/dt &lt; 0 zu dc/dt &gt; 0, ein Maximum der umgekehrte Wechsel. Für einen glatten Verlauf erfüllt ein Minimum dc/dt = 0 und d²c/dt² &gt; 0. Ein mathematischer Wendepunkt wäre dagegen ein Wechsel der Krümmung; er muss keine Richtungsumkehr enthalten.</p>
<p>Ein sinnvoller Datensatz für einen Turning Point enthält: verbleibende Zusammensetzung, T und dT/dt, momentane Wasser- und IPA-Verlustraten, kumulative Wasser- und IPA-Verluste, c und ρ, Steigungen vor/nach dem Ereignis, Krümmung beziehungsweise Breite des Minimums, Messauflösung und Modellunterstützung. Ein Minimum mit fast null Krümmung ist zeitlich schlecht lokalisierbar.</p>
<p>Die Dichte ist eine zusätzliche Prüfung. Für die Standardtinte ergibt das isotherme akustische Grenzverhältnis weiterhin einen positiven Dichteanstieg: im Calculator ungefähr {de((p.RhoW_kg_m3_per_g*p.critical_water_IPA_ratio+p.RhoI_kg_m3_per_g)/(1+p.critical_water_IPA_ratio),3)} kg/m³ je g Gesamtverlust aus 100 g Anfangstinte. Ein Minimum von c verlangt daher kein Minimum der Dichte.</p>
<div class="formula">[ dρ/dt − (∂ρ/∂T)·dT/dt ] = (∂ρ/∂E_W)·R_W + (∂ρ/∂E_IPA)·R_IPA<br>[ dc/dt − (∂c/∂T)·dT/dt ] = D_W·R_W + D_IPA·R_IPA</div>
<p>Dieses Gleichungssystem kann die beiden Raten schätzen, falls die Empfindlichkeitsmatrix hinreichend bestimmt ist. Es bleibt eine modellgestützte Inversion: Änderungen der Partikelverteilung, Sensorabweichungen oder ungeprüfte Feldgradienten können als Verlust interpretiert werden. Eine gravimetrische oder chemische Kontrolle ist daher für quantitative Dampfanteile nötig.</p>
<h2>5. Temperatur und Probe 12</h2>
<p>Bei nicht konstanter Temperatur lautet die Umschlaggrenze:</p>
<div class="formula">D_W·R_W + D_IPA·R_IPA + (∂c/∂T)·dT/dt = 0<br>(dT/dt)_krit = −(D_W·R_W + D_IPA·R_IPA)/(∂c/∂T)</div>
<p>Erwärmung senkt den benötigten Wasseranteil, solange ∂c/∂T &gt; 0. Kühlung erhöht ihn. Ein Minimum kann deshalb bei unveränderten Verlustanteilen allein durch eine veränderte Erwärmungsrate entstehen.</p>
{image('temperatur_verlust_grenze','Lokale Grenzlinie aus Wasseranteil im Verlust und Erwärmungsrate bei Probe 12')}
<p>Die vorherige Anpassung von Probe 12 ergibt modellabhängig R<sub>IPA</sub> = {de(sample['fitted_IPA_rate_g_h'])} g/h und R<sub>W</sub> = {de(sample['fitted_Water_rate_g_h'])} g/h. Das Verhältnis beträgt nur {de(sample['instant_water_IPA_ratio_fitted'],2)}:1, entsprechend {de(100*sample['instant_water_fraction_fitted'],1)} % Wasser im Verlust. Für eine isotherme Umkehr wäre am Zustand des Rohminimums ein Verhältnis von etwa {de(sample['isothermal_critical_ratio_at_raw_min'],1)}:1 nötig. Isotherm fällt die angepasste Modellkurve über den Tag um {de(-sample['isothermal_C_model_change_m_s'])} m/s.</p>
<p>Der negative Zusammensetzungsbeitrag am Rohminimum beträgt {de(sample['chemical_C_rate_at_raw_min_m_s_h'])} m/s pro Stunde. ∂c/∂T beträgt dort {de(sample['CT_at_raw_min_m_s_K'])} m/(s·K). Eine Erwärmung von ungefähr <strong>{de(sample['critical_heating_rate_at_raw_min_C_h'],3)} °C/h</strong> genügt in dieser Anpassung, um beide Beiträge auszugleichen. Über den gesamten Messverlauf variiert dieser Schwellenwert zwischen {de(sample['critical_heating_rate_C_h_range'][0],3)} und {de(sample['critical_heating_rate_C_h_range'][1],3)} °C/h.</p>
{image('probe12_umkehrbedingung','Lokale Ableitungen der Messung und die modellgestützte Umkehrbedingung')}
{table(['Glättungsfenster','Nullstelle dc/dt, Messung','Nullstelle der Modellbilanz'],smoothing)}
<p>Die Tabelle zeigt jeweils den negativen-zu-positiven Wechsel, der dem Rohdatenminimum am nächsten liegt. Das Rohminimum bleibt 13:42:29 MESZ. Die Nullstelle einer geglätteten Ableitung hängt vom Fenster und von kleineren realen Temperaturschwankungen ab. Sie sollte als Zeitbereich angegeben werden. Die vollständigen Übergänge stehen in der JSON-Zusammenfassung. Die PG-Temperaturkorrektur unter 25 °C enthält Extrapolation; die Verlustraten sind keine unabhängigen Messwerte.</p>
<h2>6. Konkrete experimentelle Untersuchung</h2>
<ol><li><strong>Lokale Empfindlichkeiten messen.</strong> Bei festem T und unveränderter Pigment-/PG-/MG-Masse mehrere Ansätze um den Standardpunkt herstellen. Wasser beispielsweise um ±0,5 g und IPA um ±0,05 g je 100 g Ausgangsansatz variieren. Die jeweils neue Gesamtmasse verwenden. Daraus D_W und D_IPA lokal durch Regression bestimmen. Mindestens drei unabhängig hergestellte Wiederholungen je Zustand; Reihenfolge variieren und dieselben Rühr-/Pumpbedingungen verwenden.</li>
<li><strong>Grenzverhältnis prüfen.</strong> Danach Ansätze mit definiertem Gesamtverlust und Wasser:IPA-Verhältnissen etwa 10:1, 15:1, 17:1, 18:1, 20:1 und 25:1 herstellen. Die tatsächlichen Restmassen vorgeben statt natürliche Verdunstung auf dieses Verhältnis zu zwingen. Signifikanz der Änderung gegenüber Wiederholungsstreuung und Temperaturunsicherheit prüfen. Für sehr flache Minima ist eine Ableitungsschätzung aus mehreren Punkten aussagekräftiger als ein einzelner c-Vergleich.</li>
<li><strong>Natürliche Verdunstung verfolgen.</strong> Den Ansatz kontinuierlich wiegen, Temperatur und c/ρ gleichzeitig protokollieren und IPA unabhängig in entnommenen Proben oder im aufgefangenen Kondensat bestimmen. Entnahmen in der Massenbilanz korrigieren. So erhält man R_W(t) und R_IPA(t), ohne sie ausschließlich aus c und ρ zurückzurechnen.</li>
<li><strong>Temperatureffekt trennen.</strong> Einen Verdunstungsversuch thermostatiert durchführen und einen zweiten mit dokumentiertem Temperaturverlauf. Die Bilanzgrenze über dT/dt prüfen. Schon 0,01 °C Temperaturfehler entspricht bei Probe 12 ungefähr 0,017 m/s; das ist größer als manche formalen isothermen Minima der Modellkurven.</li></ol>
<p>Die wichtigsten Unsicherheiten sind die lokale Temperaturkorrektur, schwach belegte Zusammensetzungsgradienten, Nachbarwechsel des Kalibrierfelds und langsame Änderungen der Pigmentverteilung. Die hier verglichenen Modelle bilden Sensitivitäten ab, keine statistischen Konfidenzintervalle.</p>
<h2>7. Daten und Reproduzierbarkeit</h2>
<p><a href="start_grenzverhaeltnisse.csv">Startgrenzen</a>, <a href="ableitung_schrittweiten.csv">Schrittweitenprüfung</a>, <a href="grenzverhaeltnis_kennfeld.csv">Zusammensetzungsraster</a>, <a href="isotherme_pfade.csv">Isotherme Pfade</a>, <a href="isotherme_umkehrpunkte.csv">Umkehrpunkte</a>, <a href="umkehrgrenze_feste_verhaeltnisse.csv">Grenzkurve für konstante Verhältnisse</a>, <a href="kalibrierungssensitivitaet.csv">Kalibrierungssensitivität</a>, <a href="probe12_ableitungen.csv">Messableitungen</a>, <a href="probe12_umkehr_zusammenfassung.json">Probe-12-Zusammenfassung</a>, <a href="manifest.json">Quellen und Annahmen</a>. Alle sieben Abbildungen liegen zusätzlich als PNG und PDF vor.</p>
<p><a href="verification.json">Numerische Verifikation</a>: unabhängige Ableitung über die Massenanteile, Skalierung der Ansatzmasse, Schrittweiten, Vorzeichenwechsel der Minima, Massenbilanzen und Quellenprüfung.</p>
<p>Primäre Grundlage: <a href="https://github.com/niklasreeker/sim_par_ink_prop/blob/main/ink_calculator.py">InkCalculator</a> und die gespeicherten Residualfelder des Laborordners. Die Feldannahme 66 % IPA/34 % Wasser betrifft den Aufbau von A(w); sie wird nicht als Verlustverhältnis der neuen Pfade verwendet. Dennoch hängt die Kalibrierkorrektur von dieser Aufbauannahme ab. Die aktuelle Untersuchung verändert weder die Rohdaten noch die Felder.</p>'''
    doc='''<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Turning Points der Schallgeschwindigkeit</title><style>body{font:16px/1.6 system-ui,Segoe UI,sans-serif;color:#25323b;background:#f4f6f8;margin:0}main{max-width:1100px;margin:32px auto;padding:36px;background:white;border-radius:8px}h1{font-size:32px;line-height:1.25}h2{font-size:24px;margin-top:40px;color:#1b5364}.subtitle,figcaption{color:#56646e;font-size:14px}.lead{font-size:20px;background:#e9f4f5;padding:20px;border-left:4px solid #197c92}.formula{font-size:19px;background:#f2f4f7;padding:14px}figure{margin:24px 0}img{width:100%;height:auto}a{color:#126482}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:right;padding:10px;border-bottom:1px solid #d6dde2}th:first-child,td:first-child{text-align:left}th{background:#edf2f5}tbody tr:nth-child(even){background:#f8fafb}li{margin:12px 0}@media(max-width:700px){main{padding:18px;margin:0}h1{font-size:27px}}@media print{body{background:white}main{padding:0;margin:0}figure,table{break-inside:avoid}}</style></head><body><main>'''+content+'</main></body></html>'
    (out/'untersuchung.html').write_text(doc,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/turning_points_20261006')
    parser.add_argument('--finalize-existing',action='store_true',help='Reuse previously exported curves and add the stationary-point locus.')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    default_style();study=Study()
    if args.finalize_existing:
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['calculator_sha256']==a.digest(a.ROOT/'ink_calculator.py')
        assert manifest['field_sha256']==a.digest(a.DEFAULT_FIELD)
        assert manifest['sample_field_sha256']==a.digest(a.LATEST_FIELD)
        assert manifest['measurement_source_sha256']==a.digest(a.SOURCE)
        initial=pd.read_csv(out/'start_grenzverhaeltnisse.csv')
        roots=[r for r in manifest['isothermal_roots'] if abs(r['delta_C_from_start_m_s'])>1e-8]
        pd.DataFrame(roots).to_csv(out/'isotherme_umkehrpunkte.csv',index=False)
        paths=pd.read_csv(out/'isotherme_pfade.csv')
        paths=paths[~np.isclose(paths.ratio_Water_IPA,17.71765,rtol=0,atol=1e-8)]
        paths.to_csv(out/'isotherme_pfade.csv',index=False)
        locus=turning_locus(study,out)
        manifest.update({'isothermal_roots':roots,'stationary_locus':locus,'script_sha256':a.digest(Path(__file__))})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        write_report(out,initial,roots,manifest['examples'],manifest['sample12'],study,locus)
        print(json.dumps(locus,indent=2),flush=True)
        return
    initial,steps=initial_study(study,out)
    print('Initial sensitivities completed',flush=True)
    grid=phase_map(study,out)
    print('Composition map completed',flush=True)
    paths,roots=path_study(study,out)
    locus=turning_locus(study,out)
    print('Isothermal paths and calibration comparison completed',flush=True)
    examples=changing_ratio_example(study,out)
    sample=sample_study(study,out)
    manifest={'initial_masses_g_per100g':dict(zip(a.COMPONENTS,a.STANDARD.tolist())),
              'temperature_isothermal_C':25,'ratio_definition':'instantaneous water loss rate / IPA loss rate, both by mass',
              'derivative_definition':'change of c upon removing one component, all other component masses held fixed, then renormalized',
              'calculator_sha256':a.digest(a.ROOT/'ink_calculator.py'),'field_sha256':a.digest(a.DEFAULT_FIELD),
              'sample_field_sha256':a.digest(a.LATEST_FIELD),'measurement_source_sha256':a.digest(a.SOURCE),
              'script_sha256':a.digest(Path(__file__)),'warnings':sorted(study.notices),
              'isothermal_roots':roots,'examples':examples,'sample12':sample,'stationary_locus':locus,
              'limits':['Local sensitivity threshold, not an independently measured vapor composition',
                         '4-neighbor IDW is not globally differentiable; topology switches exported',
                         'All-node IDW is an unvalidated sensitivity alternative',
                         'No constant threshold or clock time across evolving compositions',
                         'Sample rates inherited from prior model fit; no new independent mass data']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    write_report(out,initial,roots,examples,sample,study,locus)
    print(json.dumps({'output':str(out),'initial':initial.to_dict('records'),'roots':roots,'sample12':sample},indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
