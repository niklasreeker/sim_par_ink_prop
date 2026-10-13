#!/usr/bin/env python3
"""Mass-conserving scenario plots and temperature analysis of sample 12.

Run from any directory. Inputs are read-only. All files are written to
results/evaporation_scenarios_20261006 by default. The saved residual field
is used through its canonical schema-v3 implementation, not the older
schema-v1/v2 adapter in evaluate_evaporation.py.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
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
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import residual_calibration_field as rcf
from ink_calculator import InkCalculator

DEFAULT_FIELD = HERE / 'results/calibration_field/probe_3_4_5_6_7_8_9_10_11__Kennfeld_v2_21__korrigiert__20260911_132130/calibration_field.json'
LATEST_FIELD = HERE / 'results/calibration_field/probe_9_10_11__Kennfeld_v2_23__korrigiert__20260928_110809/calibration_field.json'
SOURCE = HERE / 'measurement_data/Messdaten_2026-10-05_16-50-09.csv'
COMPONENTS = ['Al', 'IPA', 'PG', 'MG', 'Water']
STANDARD = np.array([1.81, 3.63, 3.63, .22, 90.71])  # g per 100 g initial ink
COLORS = {'water': '#197c92', 'ipa': '#c16b18', 'equal': '#7852a3'}
LABELS = {'water': 'Nur Wasser verdunstet', 'ipa': 'Nur IPA verdunstet', 'equal': 'IPA und Wasser 50/50 (Massen)'}
_FIELD_CACHE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct_from_masses(masses):
    masses = np.asarray(masses, dtype=float)
    if np.any(masses < -1e-8) or np.any(masses.sum(axis=-1) <= 0):
        raise ValueError('Negative stock or zero total mass')
    result = 100 * masses / masses.sum(axis=-1, keepdims=True)
    assert np.allclose(result.sum(axis=-1), 100, atol=1e-9)
    return result


def scenario_masses(mode, loss):
    loss = np.asarray(loss)
    masses = np.tile(STANDARD, (len(loss), 1))
    fractions = {'water': (0., 1.), 'ipa': (1., 0.), 'equal': (.5, .5)}
    fi, fw = fractions[mode]
    masses[:, 1] -= fi*loss
    masses[:, 4] -= fw*loss
    assert np.allclose(masses[:, [0, 2, 3]], STANDARD[[0, 2, 3]])
    assert np.allclose(masses.sum(axis=1), 100-loss)
    if mode == 'equal':
        assert np.allclose(STANDARD[1]-masses[:, 1], STANDARD[4]-masses[:, 4])
    return masses


def physics(calc, pct, temperature):
    temperatures = np.broadcast_to(temperature, (len(pct),))
    result = []
    notices = set()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', RuntimeWarning)
        for w, t in zip(pct, temperatures):
            kw = dict(al=w[0], ipa=w[1], pg=w[2], mg=w[3], temperature=float(t))
            rho = 1000*calc.density(**kw)
            sound = calc.sound_calc.calculate(pct_al=w[0], pct_ipa=w[1], pct_pg=w[2], pct_mg=w[3], temperature=float(t))
            result.append([rho, sound.sound_velocity])
            notices.update(sound.warnings)
        notices.update(str(c.message) for c in captured)
    values = np.asarray(result)
    assert np.isfinite(values).all()
    return values, sorted(notices)


def corrections(field, pct):
    # Vectorized form of the canonical schema-v3 equation. Check against the
    # repository implementation before using each cached field.
    key = id(field)
    if key not in _FIELD_CACHE:
        nodes = pd.DataFrame(field['nodes'])
        axes = rcf.model_composition_axes(field)
        coordinates = nodes[axes].to_numpy(float)
        indexes = [rcf.FIELD_AXES.index(axis) for axis in axes]
        _FIELD_CACHE[key] = (nodes, axes, coordinates, indexes, False)
    nodes, axes, coordinates, axis_indexes, checked = _FIELD_CACHE[key]
    targets = np.asarray(pct)[:, axis_indexes]
    scale = np.array(field['interpolation']['scale'])
    distances = np.linalg.norm((coordinates[None,:,:]-targets[:,None,:])/scale,axis=2)
    nearest = distances.argmin(axis=1)
    requested = int(field['interpolation'].get('neighbors',4))
    count = len(nodes) if requested <= 0 else min(requested,len(nodes))
    indexes = np.argsort(distances,axis=1)[:,:count]
    d = np.take_along_axis(distances,indexes,axis=1)
    geometric = 1/np.maximum(d,1e-12)**float(field['interpolation'].get('power',2))
    exact = distances[np.arange(len(targets)),nearest] < 1e-12
    geometric[exact,:] = 0
    geometric[exact,0] = 1
    result = {}
    for value, error, name in [('A_Rho_kg_m3','A_Rho_SE_kg_m3','A_Rho_kg_m3'),('A_C_m_s','A_C_SE_m_s','A_C_m_s')]:
        v = nodes[value].to_numpy(float)[indexes]
        se = pd.to_numeric(nodes[error],errors='coerce').to_numpy(float)[indexes]
        valid = np.isfinite(se)&(se>=0)
        fallback = np.ones(len(targets))
        for i in range(len(targets)):
            retained = valid[i].copy()
            if exact[i]:
                retained[1:]=False
            if retained.any():
                fallback[i]=np.median(se[i,retained])
        se = np.where(valid,se,fallback[:,None])
        weights = geometric/np.maximum(se,np.maximum(fallback*.25,1e-9)[:,None])**2
        weights /= weights.sum(axis=1,keepdims=True)
        estimate = np.sum(weights*v,axis=1)
        result[name]=estimate
        result[name.replace('A_','A_Uncertainty_')]=np.sqrt(np.sum(weights*((v-estimate[:,None])**2+se**2),axis=1))
    low,high = coordinates.min(axis=0),coordinates.max(axis=0)
    tol=1e-10*np.maximum(1,np.maximum(abs(low),abs(high)))
    outside=(targets<low-tol)|(targets>high+tol)
    result.update({'Outside_Bounding_Box':outside.any(axis=1),
                   'Outside_Axes':[[axis.removesuffix('_wt_pct') for axis,flag in zip(axes,row) if flag] for row in outside],
                   'Nearest_Normalized_Distance':distances[np.arange(len(targets)),nearest],
                   'Nearest_Node':nearest})
    frame=pd.DataFrame(result)
    if not checked:
        for i in sorted(set([0,len(pct)//2,len(pct)-1])):
            reference=rcf.idw_residual(field,*pct[i,:3],mg=pct[i,3])
            for name in ['A_Rho_kg_m3','A_C_m_s','A_Uncertainty_Rho_kg_m3','A_Uncertainty_C_m_s']:
                assert np.isclose(frame[name].iloc[i],reference[name],atol=1e-10), name
        _FIELD_CACHE[key]=(nodes,axes,coordinates,axis_indexes,True)
    return frame


def save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2,
                         'pdf.fonttype': 42, 'figure.dpi': 120})


def scenario_table(mode, loss, calc, field, latest, temperature):
    masses = scenario_masses(mode, loss)
    pct = pct_from_masses(masses)
    values, notices = physics(calc, pct, temperature)
    corr = corrections(field, pct)
    other = corrections(latest, pct)
    table = pd.DataFrame({'Scenario': mode, 'Loss_g_per_100g_initial': loss,
                          'Temperature_C': temperature, 'Remaining_total_g': masses.sum(axis=1)})
    for j, comp in enumerate(COMPONENTS):
        table[f'm_{comp}_g'] = masses[:, j]
        table[f'{comp}_wt_pct'] = pct[:, j]
    for j, (prefix, a) in enumerate([('Rho', 'A_Rho_kg_m3'), ('C', 'A_C_m_s')]):
        unit = 'kg_m3' if j == 0 else 'm_s'
        table[f'{prefix}_Physics_{unit}'] = values[:, j]
        table[f'{prefix}_Hybrid_{unit}'] = values[:, j] + corr[a].to_numpy()
        table[f'{prefix}_Hybrid_latest_MGfree_{unit}'] = values[:, j] + other[a].to_numpy()
        table[a] = corr[a]
    table['Outside_field_box'] = corr.Outside_Bounding_Box
    table['Outside_axes'] = corr.Outside_Axes.map(lambda v: ','.join(v))
    table['Nearest_normalized_distance'] = corr.Nearest_Normalized_Distance
    table['Nearest_node'] = corr.Nearest_Node
    # A bounding box is only a screening condition, not proof of local support.
    table['MG_density_extrapolated'] = table.MG_wt_pct > .30
    return table, notices


def split_hybrid(ax, table, x, column, color):
    inside = ~table.Outside_field_box.to_numpy(bool)
    y = table[column].to_numpy()
    # Show the entire IDW prediction faintly, and solid where within the box.
    ax.plot(x, y, '--', color=color, lw=1.7, label='InkCalculator + Kalibrierfeld (extrapoliert)')
    ax.plot(x, np.where(inside, y, np.nan), '-', color=color, lw=2.2,
            label='InkCalculator + Kalibrierfeld (innerhalb der Grenzen)')


def scenario_plot(table, mode, out, temperature):
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.5))
    if mode == 'water':
        x = table.Water_wt_pct
        xlabel = 'Verbleibender Wasser-Massenanteil [%]'
    else:
        x = table.IPA_wt_pct
        xlabel = 'Verbleibender IPA-Massenanteil [%]'
    for ax, prefix, unit, ylabel in zip(axs, ['Rho','C'], ['kg_m3','m_s'], ['Dichte [kg/m³]', 'Schallgeschwindigkeit [m/s]']):
        ax.plot(x, table[f'{prefix}_Physics_{unit}'], color='#5b6470', lw=1.7, label='InkCalculator')
        split_hybrid(ax, table, x, f'{prefix}_Hybrid_{unit}', COLORS[mode])
        ax.scatter([x.iloc[0]], [table[f'{prefix}_Hybrid_{unit}'].iloc[0]], color=COLORS[mode], s=35, zorder=5)
        ax.invert_xaxis()
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if mode == 'water':
            ax.set_xlim(90.9, 88.2)
    if mode == 'water':
        note = '0–20 g Wasserverlust je 100 g Anfangstinte. Weiterer Verlauf in wasser_gesamtbereich.'
    elif mode == 'ipa':
        note = 'Bis zur vollständigen IPA-Abnahme: 3,63 g Verlust je 100 g Anfangstinte.'
    else:
        note = 'Stopp bei IPA = 0: 3,63 g IPA + 3,63 g Wasser verdunstet. Restwasser: 93,90 Massen-%.'
    fig.suptitle(f'{LABELS[mode]} bei {temperature:g} °C', fontsize=15, y=.98)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.07), ncol=1, frameon=False, fontsize=9)
    fig.text(.5, .025, note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.28,1,.92))
    save(fig, out, 'szenario_'+mode)


def full_water_plot(table, out, temperature):
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.5))
    for ax, prefix, unit, ylabel in zip(axs, ['Rho','C'], ['kg_m3','m_s'], ['Dichte [kg/m³]', 'Schallgeschwindigkeit [m/s]']):
        ax.plot(table.Water_wt_pct, table[f'{prefix}_Physics_{unit}'], color='#5b6470', label='InkCalculator')
        split_hybrid(ax, table, table.Water_wt_pct, f'{prefix}_Hybrid_{unit}', COLORS['water'])
        ax.invert_xaxis()
        ax.set(xlabel='Verbleibender Wasser-Massenanteil [%]', ylabel=ylabel)
    fig.suptitle(f'Nur Wasserverlust: formaler Gesamtbereich bei {temperature:g} °C', fontsize=15)
    fig.text(.5, .025, 'Hohe Konzentrationen: Extrapolation. MG > 0,30 % und fehlende Ternärdaten begrenzen die physikalische Aussage.', ha='center', fontsize=9)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.28,1,.92))
    save(fig, out, 'wasser_gesamtbereich')


def comparison_plot(tables, out, temperature):
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.7))
    for mode, table in tables.items():
        for ax, prefix, unit, ylabel in zip(axs, ['Rho','C'], ['kg_m3','m_s'], ['Änderung der Dichte [kg/m³]', 'Änderung der Schallgeschwindigkeit [m/s]']):
            column = f'{prefix}_Hybrid_{unit}'
            ax.plot(table.Loss_g_per_100g_initial, table[column]-table[column].iloc[0], color=COLORS[mode], lw=2.2, label=LABELS[mode])
            physical = table[f'{prefix}_Physics_{unit}']
            ax.plot(table.Loss_g_per_100g_initial, physical-physical.iloc[0], '--', color=COLORS[mode], alpha=.7, lw=1.2)
            ax.set(xlabel='Verdunstete Masse [g je 100 g Anfangstinte]', ylabel=ylabel)
            ax.axhline(0, color='#858585', lw=.7)
    fig.suptitle(f'Verdunstung bei {temperature:g} °C: Vergleich nahe der Ausgangsrezeptur', fontsize=15)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,-.02), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, -.07, 'Durchgezogen: mit Kalibrierfeld (teils Extrapolation). Gestrichelt: InkCalculator. Kleine Sprünge stammen aus der IDW-Korrektur.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95))
    save(fig, out, 'verdunstung_vergleich')


def load_sample():
    data = rcf.load_measurements([SOURCE])
    data = data[data.ProbeNr == 12].copy()
    data = rcf.add_timestamps_and_phases(data, 'Date','UTC Time',60)
    data = data.sort_values('Measurement_Time_UTC').reset_index(drop=True)
    if data.Experiment_Date_Local.nunique() != 1:
        raise ValueError('Sample has multiple dates: analyze separately')
    used = np.isfinite(data[['Rho_M','C_M','T_M']]).all(axis=1)
    used &= data.SensOK.eq(1) & data.Stabil.eq(1) & data.Gueltig.eq(1) & data.N.gt(0)
    used &= data.Rho_S.between(0,.05) & data.C_S.between(0,.2)
    data['Used'] = used
    assert data[rcf.MASS_COLUMNS].drop_duplicates().shape[0] == 1, 'Separate recipe changes first'
    data['Elapsed_h'] = (data.Measurement_Time_UTC-data.Measurement_Time_UTC.iloc[0]).dt.total_seconds()/3600
    return data


def fit_sample(calc, field, sample, mode, hybrid=True, fixed_rates=None):
    selected = sample[sample.Used]
    row = selected.iloc[0]
    masses0 = np.array([.2*row.m_SL120, .4*row.m_SL120+row.m_IPA,
                        .4*row.m_SL120+row.m_PG, row.m_MG, row.m_Wasser])
    time = selected.Elapsed_h.to_numpy()
    observed = selected[['Rho_M','C_M']].to_numpy()
    temp = selected.T_M.to_numpy()
    pairs = {'none': [], 'water': [4], 'ipa': [1], 'equal': [1,4], 'mixed': [1,4]}

    def unpack(params):
        if mode == 'none':
            return np.array([0.,0.])
        if mode == 'water':
            return np.array([0.,params[0]])
        if mode == 'ipa':
            return np.array([params[0],0.])
        if mode == 'equal':
            return np.array([params[0]/2,params[0]/2])
        return np.asarray(params)

    def predict(params):
        rates = unpack(params)
        masses = np.tile(masses0,(len(time),1))
        masses[:,1] -= rates[0]*time
        masses[:,4] -= rates[1]*time
        pct = pct_from_masses(masses)
        values,_ = physics(calc,pct,temp)
        if hybrid:
            corr = corrections(field,pct)
            values += corr[['A_Rho_kg_m3','A_C_m_s']].to_numpy()
        return values, pct

    sigma = np.array([.01,.05])  # model-discrepancy scales, not SEM

    def residual(params):
        pred,_ = predict(params)
        offset = np.mean(observed-pred,axis=0)
        return ((pred+offset-observed)/sigma).ravel()

    if fixed_rates is not None:
        ri,rw=fixed_rates
        params=np.array([] if mode=='none' else [rw] if mode=='water' else [ri] if mode=='ipa' else [ri+rw] if mode=='equal' else [ri,rw])
    elif mode == 'none':
        params = np.array([])
    else:
        upper = {'water': [30.], 'ipa': [.98*masses0[1]/time[-1]],
                 'equal': [1.96*min(masses0[1],masses0[4])/time[-1]],
                 'mixed': [.98*masses0[1]/time[-1],30.]}[mode]
        start = {'water':[2.], 'ipa':[.6], 'equal':[2.], 'mixed':[.6,2.]}[mode]
        # IDW neighbor changes make the inverse objective discontinuous. In
        # the constrained 50/50 case check multiple distinct starts so an
        # isolated poor local minimum is not presented as the best fit.
        starts=[[.6],[1.0],[1.4],start] if mode=='equal' and hybrid else [start]
        solutions=[least_squares(residual,s,bounds=(np.zeros(len(start)),upper),diff_step=1e-4,max_nfev=60) for s in starts]
        solutions=[s for s in solutions if s.success]
        if not solutions:
            raise RuntimeError(f'Fit did not converge: {mode}')
        solved=min(solutions,key=lambda s:np.sum(s.fun**2))
        params = solved.x
    pred, pct = predict(params)
    offset = np.mean(observed-pred,axis=0)
    rates = unpack(params)
    rmse = np.sqrt(np.mean((pred+offset-observed)**2,axis=0))
    result = {'mode':mode,'model':'hybrid' if hybrid else 'physics',
              'ipa_rate_g_h':float(rates[0]),'water_rate_g_h':float(rates[1]),
              'Rho_offset_kg_m3':float(offset[0]),'C_offset_m_s':float(offset[1]),
              'Rho_RMSE_kg_m3':float(rmse[0]),'C_RMSE_m_s':float(rmse[1]),
              'weighted_RMSE':float(np.sqrt(np.mean(((pred+offset-observed)/sigma)**2))),
              'fitted_IPA_loss_g':float(rates[0]*time[-1]),'fitted_Water_loss_g':float(rates[1]*time[-1])}
    return result,pred+offset,pct


def sample_analysis(calc, field, latest, out, previous_fits=None):
    data = load_sample()
    selected = data[data.Used].copy()
    first = selected.iloc[0]
    masses0 = np.array([.2*first.m_SL120,.4*first.m_SL120+first.m_IPA,
                        .4*first.m_SL120+first.m_PG,first.m_MG,first.m_Wasser])
    pct0 = pct_from_masses(masses0[None,:])[0]
    fixed = np.tile(pct0,(len(selected),1))
    actualT,_ = physics(calc,fixed,selected.T_M.to_numpy())
    ref25,_ = physics(calc,fixed,25.)
    deltaT = actualT-ref25
    selected['C_at25_nominal_m_s'] = selected.C_M-deltaT[:,1]
    selected['Rho_at25_nominal_kg_m3'] = selected.Rho_M-deltaT[:,0]
    selected['C_temperature_effect_vs25_m_s'] = deltaT[:,1]
    fits=[]
    def fitting(model,mode,hybrid=True):
        label='physics' if not hybrid else 'hybrid_latest_MGfree' if model is latest else 'hybrid'
        known=[] if previous_fits is None or mode=='equal' else [r for r in previous_fits if r['mode']==mode and r['model']==label]
        rates=None if not known else [known[0]['ipa_rate_g_h'],known[0]['water_rate_g_h']]
        return fit_sample(calc,model,selected,mode,hybrid,fixed_rates=rates)
    for mode in ['none','water','ipa','equal','mixed']:
        result,_,_ = fitting(field,mode,True)
        fits.append(result)
        print('Completed sample fit:',mode,flush=True)
    result,pred,pct = fitting(latest,'mixed',True)
    result['model']='hybrid_latest_MGfree'
    fits.append(fitting(field,'mixed',False)[0])
    fits.append(result)
    pd.DataFrame(fits).to_csv(out/'probe12_modellvergleich.csv',index=False)
    currentT,notes = physics(calc,pct,selected.T_M.to_numpy())
    current25,_ = physics(calc,pct,25.)
    corr = corrections(latest,pct)
    selected['C_at25_fittedcomposition_m_s'] = selected.C_M-(currentT[:,1]-current25[:,1])
    selected['Rho_at25_fittedcomposition_kg_m3'] = selected.Rho_M-(currentT[:,0]-current25[:,0])
    for j,comp in enumerate(COMPONENTS):
        selected[f'Fit_{comp}_wt_pct'] = pct[:,j]
    selected['C_fit_m_s']=pred[:,1]
    selected['Rho_fit_kg_m3']=pred[:,0]
    # Exact additive decomposition, centred on the observed minimum.
    k = int(np.argmin(selected.C_M.to_numpy()))
    selected['C_temp_delta_from_min_m_s'] = currentT[:,1]-current25[:,1]-(currentT[k,1]-current25[k,1])
    selected['C_composition_delta_from_min_m_s'] = current25[:,1]+corr.A_C_m_s.to_numpy()-(current25[k,1]+corr.A_C_m_s.iloc[k])
    selected['C_model_delta_from_min_m_s'] = selected.C_temp_delta_from_min_m_s+selected.C_composition_delta_from_min_m_s
    selected['C_observed_delta_from_min_m_s'] = selected.C_M-selected.C_M.iloc[k]
    selected.to_csv(out/'probe12_ausgewertet.csv',index=False)
    data.to_csv(out/'probe12_messauswahl.csv',index=False)

    time=selected.Measurement_Time_Local
    fig,axs=plt.subplots(2,2,figsize=(11.7,8.2),sharex=True)
    axs[0,0].plot(time,selected.C_M,color='#244f79',lw=2,label='Gemessen')
    axs[0,0].plot(time,pred[:,1],'--',color='#b7772c',label='Konstante Verdunstungsraten + gemessene Temperatur')
    axs[0,0].set(ylabel='Schallgeschwindigkeit [m/s]',title='Rohmessung: Abnahme, Minimum, Anstieg')
    axs[0,1].plot(time,selected.T_M,color='#b44b4b',lw=2)
    axs[0,1].set(ylabel='Temperatur [°C]',title='Gemessene Temperatur')
    axs[1,0].plot(time,selected.C_at25_nominal_m_s,color='#197c92',lw=2,label='Korrektur mit nominaler Zusammensetzung')
    axs[1,0].plot(time,selected.C_at25_fittedcomposition_m_s,'--',color='#7852a3',lw=1.7,label='Korrektur mit angepasster Zusammensetzung')
    axs[1,0].set(ylabel='Schallgeschwindigkeit bei 25 °C [m/s]',title='Nach Temperaturkorrektur bleibt die Abnahme')
    axs[1,1].plot(time,selected.Rho_M,color='#244f79',lw=1.6,label='Gemessen')
    axs[1,1].plot(time,selected.Rho_at25_nominal_kg_m3,color='#197c92',lw=2,label='Auf 25 °C korrigiert')
    axs[1,1].set(ylabel='Dichte [kg/m³]',title='Temperatur verdeckt einen Teil des Dichteanstiegs')
    for ax in axs.flat:
        ax.axvline(time.iloc[k],color='#888',ls=':',lw=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=time.dt.tz))
        ax.xaxis.set_major_locator(mdates.HourLocator(tz=time.dt.tz))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    for ax in [axs[0,0],axs[1,0],axs[1,1]]:
        ax.legend(fontsize=8,frameon=False,loc='best')
    fig.suptitle('Probe 12 am 16.09.2026: Zusammensetzung und Temperatureffekt',fontsize=15)
    fig.text(.5,.015,'Uhrzeit: Europe/Berlin (MESZ). Minimum um 13:42. Keine Rezepturzugaben; alle 141 Messpunkte mit gültigen Qualitätsflags.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    save(fig,out,'probe12_temperaturanalyse')

    fig,ax=plt.subplots(figsize=(10,4.8))
    for col,label,color in [('C_temp_delta_from_min_m_s','Temperaturbeitrag','#b44b4b'),
                            ('C_composition_delta_from_min_m_s','Zusammensetzungsbeitrag (angepasstes Modell)','#197c92'),
                            ('C_model_delta_from_min_m_s','Summe beider Beiträge','#b7772c'),
                            ('C_observed_delta_from_min_m_s','Gemessene Änderung','#244f79')]:
        ax.plot(time.iloc[k:],selected[col].iloc[k:],label=label,color=color,lw=2)
    ax.axhline(0,color='#777',lw=.8)
    ax.set(ylabel='Änderung seit dem Minimum [m/s]',xlabel='Uhrzeit (MESZ)',title='Probe 12: Warum die Schallgeschwindigkeit nach 13:42 wieder steigt')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=time.dt.tz))
    ax.legend(frameon=False,fontsize=9)
    fig.tight_layout()
    save(fig,out,'probe12_beitraege')

    last=selected.iloc[-1]
    minimum=selected.iloc[k]
    def metrics(row):
        return {'local':row.Measurement_Time_Local.isoformat(),
                'C_m_s':float(row.C_M),'T_C':float(row.T_M),'Rho_kg_m3':float(row.Rho_M),
                'C_at25_nominal_m_s':float(row.C_at25_nominal_m_s)}
    beta=[]
    for temp in selected.T_M:
        plus,_=physics(calc,pct0[None,:],temp+.01)
        minus,_=physics(calc,pct0[None,:],temp-.01)
        beta.append((plus[0,1]-minus[0,1])/.02)
    summary={'count_raw':len(data),'count_used':len(selected),'recipe_masses_g':dict(zip(COMPONENTS,masses0.tolist())),
             'nominal_composition_wt_pct':dict(zip(COMPONENTS,pct0.tolist())),
             'first':metrics(first if 'C_at25_nominal_m_s' in first else selected.iloc[0]),
             'minimum':metrics(minimum),'last':metrics(last),
             'dc_dT_range_m_s_K':[min(beta),max(beta)],
             'minimum_to_end':{'temperature_change_C':float(last.T_M-minimum.T_M),
                               'observed_C_change_m_s':float(last.C_M-minimum.C_M),
                               'nominal_temperature_contribution_m_s':float(deltaT[-1,1]-deltaT[k,1]),
                               'C_at25_change_m_s':float(last.C_at25_nominal_m_s-minimum.C_at25_nominal_m_s),
                               'fitted_temperature_contribution_m_s':float(selected.C_temp_delta_from_min_m_s.iloc[-1]),
                               'fitted_composition_contribution_m_s':float(selected.C_composition_delta_from_min_m_s.iloc[-1])},
             'fit':result,'fits':fits,'warnings':notes}
    (out/'probe12_zusammenfassung.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def write_report(out, summary, endpoints, field, temperature):
    def de(value, digits=3):
        return f'{float(value):.{digits}f}'.replace('.',',')
    def image(name, title):
        encoded=base64.b64encode((out/(name+'.png')).read_bytes()).decode('ascii')
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}. <a href="{name}.pdf">PDF</a></figcaption></figure>'
    def table(headers, rows):
        return '<div class="table"><table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    change=summary['minimum_to_end']
    start_values=next(row for row in endpoints if row['Point']=='start')
    first=summary['first']; minimum=summary['minimum']; last=summary['last']
    measurement_rows=[]
    for name,values in [('Messbeginn',first),('Schallminimum',minimum),('Messende',last)]:
        measurement_rows.append([name,pd.Timestamp(values['local']).strftime('%H:%M:%S'),de(values['C_m_s']),de(values['T_C']),de(values['Rho_kg_m3']),de(values['C_at25_nominal_m_s'])])
    fit_rows=[]
    for row in summary['fits']:
        fit_rows.append([{'none':'Nur Temperatur','water':'Wasserverlust','ipa':'IPA-Verlust','equal':'50/50','mixed':'Freie IPA-/Wasserraten'}[row['mode']],
                         {'hybrid':'Feld Proben 3–11','physics':'InkCalculator','hybrid_latest_MGfree':'Feld Proben 9–11'}[row['model']],
                         de(row['ipa_rate_g_h']),de(row['water_rate_g_h']),de(row['Rho_RMSE_kg_m3'],4),de(row['C_RMSE_m_s'],4)])
    end_rows=[]
    for row in endpoints:
        if row['Point']=='end':
            end_rows.append([LABELS[row['Scenario']],de(row['Loss_g_per_100g_initial'],2),de(row['IPA_wt_pct']),de(row['Water_wt_pct']),de(row['Rho_Hybrid_kg_m3']),de(row['C_Hybrid_m_s']),row['Outside_axes'] or 'keine'])
    bounds=pd.DataFrame(field['nodes'])[field['composition_axes']].agg(['min','max'])
    bounds_rows=[[col.removesuffix('_wt_pct'),de(bounds.loc['min',col]),de(bounds.loc['max',col])] for col in bounds.columns]
    contents=f'''<h1>Verdunstung der Tinte und Tagesverlauf von Probe 12</h1>
<p class="subtitle">Auswertung vom 06.10.2026. Quelle: GitHub-Repository sim_par_ink_prop, Ordner laboratory_measurement_L-Com.</p>
<p class="lead">Der Wiederanstieg der Schallgeschwindigkeit von Probe 12 lässt sich durch die Erwärmung erklären. Nach Korrektur auf 25 °C bleibt der fallende Grundtrend bestehen. Ein Wechsel von IPA- zu Wasserverdunstung ist dafür nicht erforderlich.</p>
<h2>1. Die drei Verdunstungsszenarien</h2>
<p>Ausgangsbasis sind 100 g Tinte: 1,81 g Al/Pigment, 3,63 g IPA, 3,63 g PG, 0,22 g MG und 90,71 g Wasser. Im Calculator bezeichnet „Al“ die gesamte gekapselte Pigmentphase. Alle Szenarien werden bei konstanten {temperature:g} °C gerechnet. Al/Pigment, PG und MG behalten ihre absoluten Massen.</p>
<p>Nach dem Verlust E gilt für jede Komponente: <strong>wᵢ = 100 · mᵢ,verbleibend / (100 − E)</strong>. Bei reinem Wasserverlust bleibt die absolute IPA-Masse konstant, ihr prozentualer Anteil steigt. Bei reinem IPA-Verlust steigt der Wasser-Massenanteil. Bei 50/50 werden für jede verdunstete Gesamtmasse E jeweils E/2 Wasser und IPA abgezogen.</p>
{image('verdunstung_vergleich','Vergleich bei gleicher verdunsteter Gesamtmasse')}
{table(['Verdunstung','Dichte nahe dem Start','Schallgeschwindigkeit nahe dem Start'],[['Nur Wasser','steigt schwach','steigt'],['Nur IPA','steigt deutlich','sinkt'],['50/50 nach Masse','steigt','sinkt']])}
<p>Die durchgezogenen Vergleichskurven enthalten die Kalibrierkorrektur. Die gestrichelten Vergleichskurven zeigen den InkCalculator allein. In den Einzelgrafiken ist die Kalibrierkurve außerhalb der gespeicherten Achsengrenzen gestrichelt. Kleine Sprünge der Hybridkurven entstehen aus dem Wechsel der IDW-Nachbarn und sind keine physikalischen Übergänge.</p>
{image('szenario_water','Reiner Wasserverlust, 0 bis 20 g je 100 g Anfangstinte')}
{image('szenario_ipa','Reiner IPA-Verlust bis zur vollständigen Abnahme des IPA-Vorrats')}
{image('szenario_equal','Gleiche verdunstete IPA- und Wassermassen')}
<p><strong>Grenze des 50/50-Szenarios:</strong> Der Vorrat von 3,63 g IPA begrenzt den Gesamtverlust auf 7,26 g je 100 g Anfangstinte. Danach liegen 0 g IPA, 87,08 g Wasser und insgesamt 92,74 g Tinte vor. Der Wasser-Massenanteil beträgt 93,897 %. Obwohl Wasser verdunstet, steigt somit sein prozentualer Anteil von 90,710 % auf 93,897 %. Eine Fortsetzung mit negativen IPA-Massen ist ausgeschlossen.</p>
{table(['Endpunkt','Gesamtverlust [g/100 g]','IPA [%]','Wasser [%]','Dichte hybrid [kg/m³]','c hybrid [m/s]','Außerhalb Feldachsen'],end_rows)}
<p>Startwerte bei {temperature:g} °C: InkCalculator {de(start_values['Rho_Physics_kg_m3'])} kg/m³ und {de(start_values['C_Physics_m_s'])} m/s; mit Feld Proben 3–11: {de(start_values['Rho_Hybrid_kg_m3'])} kg/m³ und {de(start_values['C_Hybrid_m_s'])} m/s. Die Startwerte sind Modellvorhersagen für die gewünschte Standardtinte, keine Messwerte von Probe 12.</p>
<h2>2. Probe 12 am 16.09.2026</h2>
<p>Die aktuelle Messdatei enthält für Probe 12 genau 141 Messpunkte von 11:24:29 bis 16:04:29 MESZ. Die Einwaagen bleiben konstant: 110 g SL120 und 1099,99 g Wasser, keine weiteren Zugaben und <strong>kein MG</strong>. SL120 wird zu 22 g Pigment, 44 g IPA und 44 g PG aufgeteilt. Der nominale Ansatz wiegt 1209,99 g. Damit weicht die Probe von der oben gewünschten Standardtinte mit MG ab.</p>
<p>Alle 141 Punkte erfüllen SensOK = Stabil = Gueltig = 1; N = 100. Pumpe_pct bleibt 100. Keine Messlücke, Rezepturänderung oder Änderung dieser Flags liegt am Minimum vor. Die geringe gespeicherte Kurzzeitstreuung spricht gegen einen einzelnen groben Ausreißer als Erklärung des gesamten Verlaufs.</p>
{table(['Punkt','Uhrzeit MESZ','c gemessen [m/s]','T [°C]','ρ gemessen [kg/m³]','c auf 25 °C [m/s]'],measurement_rows)}
{image('probe12_temperaturanalyse','Rohmessung, Temperatur und modellgestützte Korrektur auf 25 °C')}
<p>Die Temperatur erreicht ihr Minimum bereits um 12:02 Uhr bei 23,780 °C. Die Schallgeschwindigkeit erreicht ihr Minimum erst um 13:42 Uhr bei 1543,320 m/s. Die Temperatur steigt zu diesem Zeitpunkt schon. Der spätere Schallanstieg entsteht, wenn der positive Temperaturbeitrag den negativen Zusammensetzungsbeitrag überwiegt.</p>
<p>Für die nominale Zusammensetzung berechnet der InkCalculator im gemessenen Temperaturbereich ∂c/∂T = {de(summary['dc_dT_range_m_s_K'][0],2)} bis {de(summary['dc_dT_range_m_s_K'][1],2)} m/(s·K). Die Korrektur benutzt die vollständige Modellkurve, nicht einen pauschalen Wasserkoeffizienten:</p>
<p class="formula">c₂₅(t) = cₘₑₛₛ(t) − [cCalculator(w, T(t)) − cCalculator(w, 25 °C)]</p>
<p>Vom Schallminimum bis Messende steigt T um {de(change['temperature_change_C'])} °C. Bei nominaler Zusammensetzung würde das c um {de(change['nominal_temperature_contribution_m_s'])} m/s erhöhen. Tatsächlich steigt c nur um {de(change['observed_C_change_m_s'])} m/s. Der auf 25 °C korrigierte Wert <strong>sinkt in diesem Intervall um {de(-change['C_at25_change_m_s'])} m/s</strong>. Der grobe Knick verschwindet damit im temperaturkorrigierten Verlauf. Kleine Restschwankungen bleiben erhalten.</p>
{image('probe12_beitraege','Additive Beiträge nach dem Schallminimum im angepassten Modell')}
<p>Auch ein Modell mit zeitlich konstanten, nichtnegativen IPA- und Wasserverlustraten reproduziert den Wiederanstieg, wenn es die gemessene Temperatur verwendet. Für die dargestellte Probe ohne MG wird das neueste MG-freie Feld aus Proben 9–11 verwendet. Die Grafik zerlegt dieses Modell exakt in Temperaturänderung und Änderung der Zusammensetzung einschließlich A(w). Die Aufteilung der Verluste ist dadurch nicht unabhängig gemessen.</p>
<h2>3. Was die Messung über die Verdunstung aussagt</h2>
<p>Der temperaturkorrigierte Dichteanstieg zusammen mit dem temperaturkorrigierten Schallabfall passt zu IPA-Verlust bzw. IPA-haltiger Mischverdunstung. Reiner Wasserverlust würde nahe der Ausgangsrezeptur bei gleicher Temperatur beide Größen erhöhen und erklärt den Schallabfall deshalb nicht. Die Signale bestimmen jedoch weder das Dampfverhältnis noch die tatsächlichen Verluste eindeutig.</p>
<p>Die folgende Gegenrechnung passt konstante Raten an beide Messkanäle an. Jeder Kanal erhält einen konstanten Offset, sodass absolute Modellabweichungen nicht als Verdunstung ausgelegt werden. Die Gewichtung verwendet Modellfehlerskalen von 0,01 kg/m³ und 0,05 m/s. Die Raten sind bedingt auf die nominalen Anfangsmassen, auf unveränderte Pigment-/PG-Massen und auf das ausgewählte Feld. Ein Vorverlust vor 11:24 Uhr wurde mangels unabhängiger Angabe nicht angesetzt. Die Gewichte sind keine Messunsicherheiten oder Konfidenzintervalle.</p>
{table(['Szenario','Modell','IPA-Rate [g/h]','Wasser-Rate [g/h]','RMSE ρ [kg/m³]','RMSE c [m/s]'],fit_rows)}
<p>Unterschiede zwischen Calculator und den beiden Kalibrierfeldern zeigen die Modellabhängigkeit der invers geschätzten Raten. Zur unabhängigen Bestimmung des IPA/Wasser-Verhältnisses wären eine Massenbilanz und eine Analyse der Zusammensetzung nötig. Eine zeitliche Partikelumverteilung oder andere langsame Sensoreffekte sind durch die vorhandenen Flags allein nicht vollständig ausgeschlossen.</p>
<h2>4. Kalibrierung und Gültigkeitsgrenzen</h2>
<p>Für die Standardtinte wird das gespeicherte Feld aus Proben 3–11 verwendet. Es besitzt 51 Stützstellen und enthält MG. Das neueste Feld aus Proben 9–11 besitzt 24 MG-freie Stützstellen. Es dient zusätzlich zur Empfindlichkeitsprüfung der MG-freien Probe 12. Die Prüfsummen des Calculators und aller vom Feld benannten Parameterdateien stimmen mit den gespeicherten Feldern überein.</p>
{table(['Achse [Massen-%]','Minimum im Feld Proben 3–11','Maximum im Feld Proben 3–11'],bounds_rows)}
<p>Innerhalb der Achsengrenzen zu liegen beweist keine dichte lokale Abdeckung oder Zugehörigkeit zur konvexen Hülle. A(w) wird mit skalierten inversen Abstandsgewichten interpoliert und kann beim Wechsel der vier ausgewählten Nachbarn kleine Sprünge erzeugen. Die Felder enthalten keine unabhängig gelernte Temperaturkorrektur. Diese kommt vollständig aus dem InkCalculator.</p>
<p>Die Stützstellen wurden unter der Annahme 66 % IPA und 34 % Wasser im Massenverlust aufgebaut, mit einem rührzahlabhängigen Gesamtverlust. Diese Annahme wird nicht auf die drei neuen Szenarien übertragen; sie beeinflusst aber die gespeicherte Kalibrierkorrektur. Temperaturwerte von Probe 12 unter 25 °C liegen für Teile der PG-Tabellen außerhalb der lokal belegten Temperaturanker. Der Calculator extrapoliert dort. Die Schlussfolgerung zum dominierenden Temperatureffekt ist daher modellgestützt.</p>
{image('wasser_gesamtbereich','Formaler Gesamtbereich bis zum vollständigen Wasserverlust')}
<p>Der Gesamtbereich der Wasserverdunstung dient als formale Modellfortsetzung. Bei hohen Konzentrationen sind weder das Kalibrierfeld noch die Annahme unabhängiger binärer IPA-/PG-Beiträge ausreichend belegt. Oberhalb 0,30 % MG extrapoliert außerdem die angepasste MG-Dichte. Maxima und Krümmungen in diesem Bereich sind keine gesicherten experimentellen Vorhersagen.</p>
<h2>5. Dateien und Quellen</h2>
<p>Alle Abbildungen liegen als PNG und PDF im selben Ordner. Berechnete Massen, Massenanteile, Modellwerte und Kennzeichnungen: <a href="szenario_water.csv">Wasser-CSV</a>, <a href="szenario_ipa.csv">IPA-CSV</a>, <a href="szenario_equal.csv">50/50-CSV</a>, <a href="wasser_gesamtbereich.csv">Wasser-Gesamtbereich</a>. Probe 12: <a href="probe12_ausgewertet.csv">Ausgewertete Messpunkte</a>, <a href="probe12_modellvergleich.csv">Modellvergleich</a>, <a href="probe12_zusammenfassung.json">Zahlen und Zusammenfassung</a>. Quellenprüfsummen und Rechenannahmen stehen in <a href="manifest.json">manifest.json</a>.</p>
<p>Primäre Daten und Modelle: <a href="https://github.com/niklasreeker/sim_par_ink_prop/blob/main/laboratory_measurement_L-Com/measurement_data/Messdaten_2026-10-05_16-50-09.csv">GitHub-Messdatei</a>, <a href="https://github.com/niklasreeker/sim_par_ink_prop/blob/main/ink_calculator.py">InkCalculator</a>, <a href="https://github.com/niklasreeker/sim_par_ink_prop/blob/main/laboratory_measurement_L-Com/residual_calibration_field.py">Kalibrierfeld-Implementierung</a>. Die Temperaturreferenz des Calculators beruht auf Marczak (1997), <a href="https://doi.org/10.1121/1.420332">Water as a standard in the measurements of speed of sound in liquids</a>. Das <a href="https://resource.npl.co.uk/acoustics/techguides/soundpurewater/underlying-phys.html">National Physical Laboratory</a> beschreibt diese Wasserreferenz. Die konkreten Zahlen oben stammen aus dem Repository und dieser Auswertung.</p>'''
    document='''<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Verdunstung und Probe 12</title><style>
body{font:16px/1.6 system-ui,Segoe UI,sans-serif;color:#25323b;background:#f4f6f8;margin:0}main{max-width:1100px;margin:32px auto;padding:36px;background:white;border-radius:8px}h1{font-size:32px;line-height:1.25}h2{font-size:24px;margin-top:40px;color:#1b5364}.subtitle,figcaption{color:#56646e;font-size:14px}.lead{font-size:20px;background:#e9f4f5;padding:20px;border-left:4px solid #197c92}.formula{font-size:19px;background:#f2f4f7;padding:14px}figure{margin:24px 0}img{width:100%;height:auto}a{color:#126482}.table{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:right;padding:10px;border-bottom:1px solid #d6dde2}th:first-child,td:first-child{text-align:left}th{background:#edf2f5}tbody tr:nth-child(even){background:#f8fafb}@media(max-width:700px){main{padding:18px;margin:0}h1{font-size:27px}}@media print{body{background:white}main{padding:0;margin:0}figure,table{break-inside:avoid}}
</style></head><body><main>'''+contents+'</main></body></html>'
    (out/'auswertung.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results/evaporation_scenarios_20261006')
    parser.add_argument('--temperature',type=float,default=25.)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    style()
    field=rcf.load_model(DEFAULT_FIELD)
    latest=rcf.load_model(LATEST_FIELD)
    for model in [field,latest]:
        provenance=model['calculator']
        assert digest(ROOT/'ink_calculator.py') == provenance['sha256'], 'Calculator differs from field'
        for name,sha in provenance['table_sha256'].items():
            assert digest(ROOT/'tables_parameters'/name)==sha, 'Table differs from field'
    calc=InkCalculator(str(ROOT/'tables_parameters'))
    tables={}
    notices={}
    for mode,limit in [('water',20.),('ipa',STANDARD[1]),('equal',2*STANDARD[1])]:
        table,note=scenario_table(mode,np.linspace(0,limit,401),calc,field,latest,args.temperature)
        table.to_csv(out/f'szenario_{mode}.csv',index=False)
        tables[mode]=table
        notices[mode]=note
        scenario_plot(table,mode,out,args.temperature)
    full,note=scenario_table('water',np.linspace(0,STANDARD[4],501),calc,field,latest,args.temperature)
    full.to_csv(out/'wasser_gesamtbereich.csv',index=False)
    notices['water_full']=note
    full_water_plot(full,out,args.temperature)
    common={mode:table[table.Loss_g_per_100g_initial<=STANDARD[1]].copy() for mode,table in tables.items()}
    comparison_plot(common,out,args.temperature)
    summary=sample_analysis(calc,field,latest,out)
    rows=[]
    for mode,table in tables.items():
        for location in [0,-1]:
            row=table.iloc[location]
            rows.append({'Scenario':mode,'Point':'start' if location==0 else 'end',
                         **{col:row[col] for col in ['Loss_g_per_100g_initial','Water_wt_pct','IPA_wt_pct','Al_wt_pct','PG_wt_pct','MG_wt_pct','Rho_Physics_kg_m3','Rho_Hybrid_kg_m3','C_Physics_m_s','C_Hybrid_m_s','Outside_field_box','Outside_axes']}})
    pd.DataFrame(rows).to_csv(out/'szenario_eckwerte.csv',index=False)
    manifest={'repository':'https://github.com/niklasreeker/sim_par_ink_prop','measurement_source':str(SOURCE),'source_sha256':digest(SOURCE),
              'field':str(DEFAULT_FIELD),'field_sha256':digest(DEFAULT_FIELD),'sensitivity_field':str(LATEST_FIELD),
              'sample_plot_field':str(LATEST_FIELD),'sample_plot_field_sha256':digest(LATEST_FIELD),
              'analysis_script_sha256':digest(Path(__file__)),
              'calculator_sha256':digest(ROOT/'ink_calculator.py'),'initial_masses_g_per100g':dict(zip(COMPONENTS,STANDARD.tolist())),
              'temperature_C':args.temperature,'model_warnings':notices,
              'interpretation':'Field box screening is not proof of interpolation inside the convex hull. IDW can switch neighbors and generate local slope artifacts. Fits estimate rates conditionally, not an independent evaporation measurement.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    write_report(out,summary,rows,field,args.temperature)
    print(json.dumps({'output':str(out),'sample12':summary},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
