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
from matplotlib.pyplot import GridSpec
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

# SODA/fMRI group mean effect sizes (onset + offset, excluding 2048)
soda_intervals = [iv for iv in settings.intervals_3T if iv != 2048]
df_soda = pd.DataFrame()
df_soda_subj = pd.DataFrame()
for interval in soda_intervals:
    df_sel = df_mean[df_mean.interval == interval]
    slopes = long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])
    for period in ['onset', 'offset']:
        tr_range = settings.exp_tr[interval][period]
        lo, hi = tr_range[0] - 1, tr_range[1]
        slope_sign = -1 if period == 'offset' else 1
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
        for period in ['onset', 'offset']:
            tr_range = settings.exp_tr[interval][period]
            lo, hi = tr_range[0] - 1, tr_range[1]
            slope_sign = -1 if period == 'offset' else 1
            trial_slopes = np.array([g.sort_values('tr')['slope'].values
                                     for _, g in df_s.groupby('trial')])
            if len(trial_slopes) == 0:
                continue
            peak = trial_slopes[:, lo:hi].mean(-1) * slope_sign
            d_subj = pg.compute_effsize(peak, y=0)
            df_soda_subj = pd.concat([df_soda_subj, pd.DataFrame({
                'interval': interval, 'cohens d': d_subj, 'period': period}, index=[0])])

#%% effect size: plot side by side
fig, axs = plt.subplots(1, 2, figsize=[10, 5])

# left: TDLM / MEG
ax = axs[0]
bar_width = 0.4

ax.grid('on', axis='y', alpha=0.3)
df_tdlm = df_tdlm.reset_index(drop=True)
x_pos = np.arange(len(df_tdlm))
ax.bar(x_pos, df_tdlm['cohens d'].values, width=bar_width,
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

vals = df_tdlm["cohens d"]
print(f'TDLM mean d: {vals.mean():.2f}, range: {vals.min():.2f} - {vals.max():.2f}')

# right: SODA / fMRI with onset/offset as grouped bars
ax = axs[1]
ax.grid('on', axis='y', alpha=0.3)

# bar_width = 0.35
hatches = {'onset': '', 'offset': '//'}
for j, period in enumerate(['onset', 'offset']):
    df_p = df_soda[df_soda.period == period].reset_index(drop=True)
    x_pos = np.arange(len(df_p)) + j * bar_width
    ax.bar(x_pos, df_p['cohens d'].values,
           width=bar_width,
           yerr=[df_p['cohens d'].values - df_p['ci_lo'].values,
                 df_p['ci_hi'].values - df_p['cohens d'].values],
           capsize=3, color=[settings.palette_wittkuhn2[i] for i in range(len(df_p))],
           edgecolor='white', linewidth=0.5,
           hatch=hatches[period], label=period)
ax.set_xticks(np.arange(len(soda_intervals)) + bar_width / 2)
ax.set_xticklabels(soda_intervals)
ax.set_xlabel('interval (ms)')
ax.set_ylabel("Cohen's d")
ax.set_title('SODA / fMRI')
ax.axhline(0, linestyle='--', alpha=0.5, c='black')
for j, period in enumerate(['onset', 'offset']):
    for i, interval in enumerate(soda_intervals):
        ys = df_soda_subj[(df_soda_subj.interval == interval) &
                          (df_soda_subj.period == period)]['cohens d'].values
        ax.scatter(np.full(len(ys), i + j * bar_width), ys,
                   color='gray', alpha=0.3, s=10, zorder=3)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='none', edgecolor='black', hatch=hatches[p], label=p)
                  for p in ['onset', 'offset']]
ax.legend(handles=legend_handles, title='period')

for period, df_tmp in df_soda.groupby('period'):
    vals = df_tmp['cohens d']
    print(f'SODA {period} mean d: {vals.mean():.2f}, range: {vals.min():.2f} - {vals.max():.2f}')


plotting.normalize_lims(axs)
fig.suptitle("Group mean effect size\n(Cohen's d)")
fig.tight_layout()
savefig(fig, settings.plot_dir + '/figures/compare_effect_sizes.png')


#%% 80% power lines collapsed across speed conditions (all 3 methods, 1×3)

from glob import glob
import seaborn as sns

method_colors = sns.color_palette('tab10', 3)
method_labels = ['t-test', 'perm. t-test', 'cluster']

def _load_ttest_power(bids_base, processing):
    f = bids_base.copy().update(processing=processing, suffix='powercontour',
                                extension='.pkl.gz')
    df = joblib.load(f.fpath)
    if 'soda' in processing:
        df['method'] = processing
        mapping = {'fwd': 'onset', 'bkw': 'offset'}
        df['period'] = df.period.apply(lambda x: mapping[x])
        df = df[df.interval!=2048]
    return df

def _load_perm_power(bids_base, processing):
    template = bids_base.copy().update(processing=processing, suffix='powercontour',
                                       task='NSAMPLES', extension='.pkl.gz')
    files = sorted(glob(str(template.fpath).replace('task-NSAMPLES', 'task-*')))
    df = pd.concat([joblib.load(f) for f in files], ignore_index=True)

    if 'soda' in processing:
        df['method'] = processing
        mapping = {'fwd': 'onset', 'bkw': 'offset'}
        df['period'] = df.period.apply(lambda x: mapping[x])
        df = df[df.interval!=2048]
    return df

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

# two GridSpecs: TDLM left, SODA onset+offset right
fig, (ax_tdlm, ax_soda) = plt.subplots(1, 2, figsize=[10, 5])

for df, color, label in zip(tdlm_methods, method_colors, method_labels):
    _draw_80_line(ax_tdlm, df, color=color, label=label)

for df, color, label in zip(soda_methods, method_colors, method_labels):
    _draw_80_line(ax_soda, df, period='offset', color=color, label=label)

for ax, title in [(ax_tdlm, 'TDLM\nforward sequenceness'), (ax_soda, 'SODA\noffset period')]:
    ax.set_xlabel('bootstrapped sample size')
    ax.set_ylabel('bootstrapped trials per participant')
    ax.set_title(title)
    ax.legend(title='80% power', frameon=True)

fig.suptitle('80% power of statistical methods')
savefig(fig, settings.plot_dir + '/figures/compare_power_80.png')
