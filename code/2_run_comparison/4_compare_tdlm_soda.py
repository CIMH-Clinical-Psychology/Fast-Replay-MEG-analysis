#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:59:41 2026

@author: simon.kern
"""
import sys; sys.path.append('..')
import joblib
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import settings
from mne_bids import BIDSPath
from settings import layout_MEG, layout_3T
from meg_utils import plotting
from meg_utils.plotting import savefig
from meg_utils.misc import long_df_to_array


bids_base_MEG = BIDSPath(
    root=layout_MEG.derivatives['derivatives'].root,
    datatype='results',
    subject='group',
    task='main',
    check=False
)

bids_base_3T = BIDSPath(
    root=layout_3T.derivatives['derivatives'].root,
    datatype='results',
    subject='group',
    task='main',
    check=False
)

#%% load data
pkl_seq = bids_base_MEG.copy().update(processing='trials', suffix='sequenceness', extension='.pkl.gz')
res = joblib.load(pkl_seq)
sf_mean = res['sf_mean']
sf_trials = res['sf_trials']

pkl_slopes = str(bids_base_3T.copy().update(processing='soda', suffix='slopes', extension='.pkl.gz'))
df_slopes = joblib.load(pkl_slopes)
df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

#%% effect size comparison: TDLM/MEG vs SODA/fMRI

# TDLM/MEG group mean effect sizes
df_tdlm = pd.DataFrame()
df_tdlm_subj = pd.DataFrame()
for interval in settings.intervals_MEG:
    sf = sf_mean[interval][:, 0, :]
    expected_lag = settings.exp_lag[interval]
    x = sf[:, expected_lag]
    n = len(x)
    d = pg.compute_effsize(x, y=0)
    ci_lo, ci_hi = pg.compute_esci(d, nx=n, ny=n, paired=True)
    df_tdlm = pd.concat([df_tdlm, pd.DataFrame({
        'interval': interval, 'cohens d': d,
        'ci_lo': ci_lo, 'ci_hi': ci_hi}, index=[0])])
    # per-subject: Cohen's d across their trials
    sf_t = sf_trials[interval][:, :, 0, 1:]
    for i in range(sf_t.shape[0]):
        trials_at_lag = sf_t[i, :, expected_lag - 1]
        d_subj = pg.compute_effsize(trials_at_lag, y=0)
        df_tdlm_subj = pd.concat([df_tdlm_subj, pd.DataFrame({
            'interval': interval, 'cohens d': d_subj}, index=[0])])

# SODA/fMRI group mean effect sizes (fwd + bkw, excluding 2048)
soda_intervals = [iv for iv in settings.intervals_3T if iv != 2048]
df_soda = pd.DataFrame()
df_soda_subj = pd.DataFrame()
for interval in soda_intervals:
    df_sel = df_mean[df_mean.interval == interval]
    slopes = long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])
    for period in ['fwd', 'bkw']:
        tr_range = settings.exp_tr[interval][period]
        lo, hi = tr_range[0] - 1, tr_range[1]
        slope_sign = -1 if period == 'bkw' else 1
        slopes_peak = slopes[:, lo:hi].mean(-1) * slope_sign
        n = len(slopes_peak)
        d = pg.compute_effsize(slopes_peak, y=0)
        ci_lo, ci_hi = pg.compute_esci(d, nx=n, ny=n, paired=True)
        df_soda = pd.concat([df_soda, pd.DataFrame({
            'interval': interval, 'cohens d': d,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'period': period}, index=[0])])
    # per-subject: Cohen's d across their trials
    for subj in df_slopes.subject.unique():
        df_s = df_slopes[(df_slopes.interval == interval) & (df_slopes.subject == subj)]
        for period in ['fwd', 'bkw']:
            tr_range = settings.exp_tr[interval][period]
            lo, hi = tr_range[0] - 1, tr_range[1]
            slope_sign = -1 if period == 'bkw' else 1
            trial_slopes = np.array([g.sort_values('tr')['slope'].values
                                     for _, g in df_s.groupby('trial')])
            if len(trial_slopes) == 0:
                continue
            peak = trial_slopes[:, lo:hi].mean(-1) * slope_sign
            d_subj = pg.compute_effsize(peak, y=0)
            df_soda_subj = pd.concat([df_soda_subj, pd.DataFrame({
                'interval': interval, 'cohens d': d_subj, 'period': period}, index=[0])])

# plot side by side
fig, axs = plt.subplots(1, 2, figsize=[10, 4])

# left: TDLM / MEG
ax = axs[0]
df_tdlm = df_tdlm.reset_index(drop=True)
x_pos = np.arange(len(df_tdlm))
ax.bar(x_pos, df_tdlm['cohens d'].values,
       yerr=[df_tdlm['cohens d'].values - df_tdlm['ci_lo'].values,
             df_tdlm['ci_hi'].values - df_tdlm['cohens d'].values],
       capsize=4, color=[settings.palette_wittkuhn2[i] for i in range(len(df_tdlm))],
       edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(df_tdlm['interval'].values)
ax.set_xlabel('interval (ms)')
ax.set_ylabel("Cohen's d")
ax.set_title('TDLM / MEG')
ax.axhline(0, linestyle='--', alpha=0.5, c='black')
for i, interval in enumerate(settings.intervals_MEG):
    ys = df_tdlm_subj[df_tdlm_subj.interval == interval]['cohens d'].values
    ax.scatter(np.full(len(ys), i), ys, color='gray', alpha=0.3, s=10, zorder=3)

# right: SODA / fMRI with fwd/bkw as grouped bars
ax = axs[1]
bar_width = 0.35
hatches = {'fwd': '', 'bkw': '//'}
for j, period in enumerate(['fwd', 'bkw']):
    df_p = df_soda[df_soda.period == period].reset_index(drop=True)
    x_pos = np.arange(len(df_p)) + j * bar_width
    ax.bar(x_pos, df_p['cohens d'].values,
           width=bar_width,
           yerr=[df_p['cohens d'].values - df_p['ci_lo'].values,
                 df_p['ci_hi'].values - df_p['cohens d'].values],
           capsize=3, color=[settings.palette_wittkuhn2[i] for i in range(len(df_p))],
           edgecolor='black', linewidth=0.5,
           hatch=hatches[period], label=period)
ax.set_xticks(np.arange(len(soda_intervals)) + bar_width / 2)
ax.set_xticklabels(soda_intervals)
ax.set_xlabel('interval (ms)')
ax.set_ylabel("Cohen's d")
ax.set_title('SODA / fMRI')
ax.axhline(0, linestyle='--', alpha=0.5, c='black')
for j, period in enumerate(['fwd', 'bkw']):
    for i, interval in enumerate(soda_intervals):
        ys = df_soda_subj[(df_soda_subj.interval == interval) &
                          (df_soda_subj.period == period)]['cohens d'].values
        ax.scatter(np.full(len(ys), i + j * bar_width), ys,
                   color='gray', alpha=0.3, s=10, zorder=3)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='none', edgecolor='black', hatch=hatches[p], label=p)
                  for p in ['fwd', 'bkw']]
ax.legend(handles=legend_handles)

plotting.normalize_lims(axs)
fig.suptitle("Group mean effect size (Cohen's d with 95% CI)")
fig.tight_layout()
savefig(fig, settings.plot_dir + '/figures/compare_effect_sizes.png')

#%% power contours: TDLM/MEG vs SODA/fMRI (ttest, 3×4, excl. 2048)

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

power_levels   = [0.5, 0.6, 0.7, 0.8, 0.9]
contour_colors = ['#8b0000', '#f46d43', '#fdae61', '#d62728', '#fee08b']
contour_widths = [1, 1, 1, 3, 1]
power_intervals = [32, 64, 128, 512]  # exclude 2048

# load ttest power data
pkl_tdlm_power = bids_base_MEG.copy().update(processing='ttest', suffix='powercontour',
                                              extension='.pkl.gz')
df_tdlm_power = joblib.load(pkl_tdlm_power.fpath)

pkl_soda_power = bids_base_3T.copy().update(processing='sodattest', suffix='powercontour',
                                             extension='.pkl.gz')
df_soda_power = joblib.load(pkl_soda_power.fpath)

def _contour_ax(ax, df_power, interval, period=None, ylabel=True):
    df_sel = df_power[df_power.interval == interval]
    if period is not None:
        df_sel = df_sel[df_sel.period == period]
    df_pivot = df_sel.pivot_table(index='n_trials', columns='n_samples',
                                  values='power', aggfunc='mean')
    X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
    Z = df_pivot.values
    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')
    for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
        ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)
    ax.set_xticks(np.arange(0, int(X.max()) + 10, 10))
    ax.set_xlabel('n participants')
    ax.set_ylabel('n trials' if ylabel else '')
    return cf

# two separate GridSpecs: MEG top, SODA bottom, with a gap in between
fig = plt.figure(figsize=[16, 14])
gs_meg  = GridSpec(1, 4, figure=fig, top=0.93, bottom=0.62, wspace=0.3)
gs_soda = GridSpec(2, 4, figure=fig, top=0.50, bottom=0.06, hspace=0.45, wspace=0.3)
ax_meg  = [fig.add_subplot(gs_meg[0, i])  for i in range(4)]
ax_fwd  = [fig.add_subplot(gs_soda[0, i]) for i in range(4)]
ax_bkw  = [fig.add_subplot(gs_soda[1, i]) for i in range(4)]

cf = None
for i, interval in enumerate(power_intervals):
    cf = _contour_ax(ax_meg[i],  df_tdlm_power, interval,           ylabel=(i == 0))
    cf = _contour_ax(ax_fwd[i],  df_soda_power, interval, 'fwd',    ylabel=(i == 0))
    cf = _contour_ax(ax_bkw[i],  df_soda_power, interval, 'bkw',    ylabel=(i == 0))
    ax_meg[i].set_title(f'{interval} ms')
    ax_fwd[i].set_title(f'{interval} ms')
    ax_bkw[i].set_title('')  # suppress duplicate titles on lower row

# row y-labels
ax_meg[0].set_ylabel('n trials')
ax_fwd[0].set_ylabel('fwd\nn trials')
ax_bkw[0].set_ylabel('bkw\nn trials')

# bold section titles above each group
fig.text(0.5, 0.955, 'TDLM / MEG', ha='center', va='bottom',
         fontsize=13, fontweight='bold', transform=fig.transFigure)
fig.text(0.5, 0.525, 'SODA / fMRI', ha='center', va='bottom',
         fontsize=13, fontweight='bold', transform=fig.transFigure)

# power level legend on first panel
legend_handles = [Line2D([], [], color=col, linewidth=lw, label=f'{int(lvl*100)}%')
                  for lvl, col, lw in zip(power_levels, contour_colors, contour_widths)]
ax_meg[0].legend(handles=legend_handles, title='power', loc='upper right', frameon=True)

# shared colorbar on right
cax = fig.add_axes([0.94, 0.06, 0.015, 0.87])
fig.colorbar(cf, cax=cax, label='power')

savefig(fig, settings.plot_dir + '/figures/compare_power_contours.png', tight=False)

#%% 80% power lines collapsed across speed conditions (all 3 methods, 1×3)

from glob import glob
import seaborn as sns

method_colors = sns.color_palette('tab10', 3)
method_labels = ['t-test', 'perm. t-test', 'cluster']

def _load_ttest_power(bids_base, processing):
    f = bids_base.copy().update(processing=processing, suffix='powercontour',
                                extension='.pkl.gz')
    return joblib.load(f.fpath) if f.fpath.is_file() else None

def _load_perm_power(bids_base, processing):
    template = bids_base.copy().update(processing=processing, suffix='powercontour',
                                       task='NSAMPLES', extension='.pkl.gz')
    files = sorted(glob(str(template.fpath).replace('task-NSAMPLES', 'task-*')))
    return pd.concat([joblib.load(f) for f in files], ignore_index=True) if files else None

# load all methods for TDLM/MEG and SODA/fMRI
tdlm_methods = [
    _load_ttest_power(bids_base_MEG, 'ttest'),
    _load_perm_power( bids_base_MEG, 'signflip'),
    _load_perm_power( bids_base_MEG, 'cluster'),
]
soda_methods = [
    _load_ttest_power(bids_base_3T, 'sodattest'),
    _load_perm_power( bids_base_3T, 'sodasignflip'),
    _load_perm_power( bids_base_3T, 'sodacluster'),
]

def _draw_80_line(ax, df, period=None, color='black', label=''):
    """Mean power across speed conditions (excl. 2048), draw 80% iso-line."""
    df = df[df.interval != 2048]
    if period is not None:
        df = df[df.period == period]
    df_mean = df.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
    df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples', values='power')
    X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
    Z = df_pivot.values
    ax.contour(X, Y, Z, levels=[0.8], colors=[color], linewidths=2)
    ax.plot([], [], color=color, linewidth=2, label=label)  # proxy for legend

# two GridSpecs: TDLM left, SODA fwd+bkw right
fig = plt.figure(figsize=[14, 4.5])
gs_tdlm = GridSpec(1, 1, figure=fig, left=0.07, right=0.33, top=0.82, bottom=0.15)
gs_soda = GridSpec(1, 2, figure=fig, left=0.40, right=0.95, top=0.82, bottom=0.15,
                   wspace=0.3)
ax_tdlm = fig.add_subplot(gs_tdlm[0, 0])
ax_fwd  = fig.add_subplot(gs_soda[0, 0])
ax_bkw  = fig.add_subplot(gs_soda[0, 1], sharey=ax_fwd, sharex=ax_fwd)

for df, color, label in zip(tdlm_methods, method_colors, method_labels):
    if df is not None:
        _draw_80_line(ax_tdlm, df, color=color, label=label)

for df, color, label in zip(soda_methods, method_colors, method_labels):
    if df is not None:
        _draw_80_line(ax_fwd, df, period='fwd', color=color, label=label)
        _draw_80_line(ax_bkw, df, period='bkw', color=color, label=label)

for ax, title in [(ax_tdlm, ''), (ax_fwd, 'fwd'), (ax_bkw, 'bkw')]:
    ax.set_xlabel('n participants')
    ax.set_ylabel('n trials')
    ax.set_title(title)
    ax.legend(frameon=True)

fig.text(0.20, 0.88, 'TDLM / MEG',  ha='center', fontsize=12, fontweight='bold')
fig.text(0.675, 0.88, 'SODA / fMRI', ha='center', fontsize=12, fontweight='bold')
fig.suptitle('80% power (mean across speed conditions)', y=1.01)
savefig(fig, settings.plot_dir + '/figures/compare_power_80.png')
