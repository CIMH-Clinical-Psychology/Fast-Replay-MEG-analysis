#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 2026

Create power contour plots for SODA as a function of trial numbers and sample size.

Uses ttest_1samp on the mean slope per period (fwd and bkw assessed separately).
For each period the mean slope across the expected TRs is tested against zero.

@author: simon.kern
"""

import os
os.environ['NO_PRELOADING'] = '1'
import sys; sys.path.append('..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import settings
from meg_utils.misc import make_seed, to_long_df
from mne_bids import BIDSPath
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
from itertools import product

from settings import subjects_3T as subjects
from settings import intervals_3T as intervals


def bootstrap(combined, n_trials, n_samples, n_draws=1000, tqdm_loop=None):
    """Bootstrap both participants and trials simultaneously.

    For each draw: sample n_samples participants (with replacement),
    then for each sampled participant sample n_trials trials (with replacement),
    average to get one value per participant, run ttest_1samp.

    Parameters
    ----------
    combined : array [n_subj, n_trials_orig]
        Mean slope in one period per subject and trial.
    n_trials : int
        Number of trials to resample per participant.
    n_samples : int
        Number of participants to resample.
    n_draws : int
        Number of bootstrap repetitions.

    Returns
    -------
    ps : np.ndarray [n_draws]
        p-value for each draw.
    """
    combined = np.atleast_2d(combined)
    n_subj, n_trials_orig = combined.shape
    seed = make_seed(n_trials, n_samples)
    rng = np.random.default_rng(seed)

    # pre-draw all indices
    subj_idx = rng.integers(0, n_subj, (n_draws, n_samples))
    trial_idx = rng.integers(0, n_trials_orig, (n_draws, n_samples, n_trials))

    ps = []
    for draw in range(n_draws):
        sampled = np.array([combined[subj_idx[draw, s],
                                     trial_idx[draw, s]].mean()
                            for s in range(n_samples)])
        _, p = ttest_1samp(sampled, 0)
        ps.append(p)
        if tqdm_loop is not None:
            tqdm_loop.update()
    return np.array(ps)


#%% run bootstrap power analysis with ttest

bids_base = BIDSPath(
    root=settings.bids_dir_3T + '/derivatives',
    datatype='results',
    subject='group',
    task='main',
    check=False
)

pkl_slopes = str(bids_base.copy().update(processing='soda', suffix='slopes',
                                          extension='.pkl.gz'))
pkl_out = bids_base.copy().update(processing='sodattest', suffix='powercontour',
                                   extension='.pkl.gz')

assert os.path.isfile(pkl_slopes), f'File not found, did 1b script run? {pkl_slopes}'

max_trials = 64
max_samples = 80
n_draws = 1000

df_slopes = joblib.load(pkl_slopes)
n_subj = len(df_slopes.subject.unique())

pool = Parallel(-1)

n_trials_range = range(2, max_trials + 1)
n_samples_range = range(2, max_samples + 1)
df = pd.DataFrame()

df_mean = df_slopes.groupby(['subject', 'period', 'interval', 'trial']).mean(True).reset_index()

params = list(product(n_trials_range, n_samples_range))

for iv in intervals:
    # extract mean slopes for fwd and bkw periods
    for period in ['bkw', 'fwd']:
        df_sel = df_mean[(df_mean.interval==iv) & (df_mean.period==period)]
        df_sel = df_sel.sort_values(['subject', 'trial'])
        slopes = df_sel.slope.values.reshape([len(subjects), -1])
        ps = pool(delayed(bootstrap)(slopes, nt, ns, n_draws=n_draws)
                   for nt, ns in tqdm(params))

        ps = np.array(ps).reshape([len(n_trials_range), len(n_samples_range), n_draws])
        power = (ps < 0.05).mean(-1)

        df_iv = to_long_df(power, value_name='power',
                       columns=['n_trials', 'n_samples'],
                       n_trials=n_trials_range, n_samples=n_samples_range)
        df_iv['interval'] = iv
        df_iv['period'] = period
        df = pd.concat([df, df_iv])

joblib.dump(df, pkl_out.fpath)

#%% plotting
import matplotlib.pyplot as plt
from meg_utils.plotting import savefig
from matplotlib.lines import Line2D

df_power = joblib.load(pkl_out)
intervals_plot = df_power.interval.unique()

n_iv = len(intervals_plot)
fig, axs = plt.subplots(2, n_iv, figsize=[4 * n_iv, 8])

power_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
contour_colors = ['#8b0000', '#f46d43', '#fdae61', '#d62728', '#fee08b']
contour_widths = [1, 1, 1, 3, 1]

for j, period in enumerate(['fwd', 'bkw']):
    for i, interval in enumerate(intervals_plot):
        ax = axs[j, i]
        df_sel = df_power[(df_power.interval == interval) &
                          (df_power.period == period)]

        if len(df_sel) == 0:
            ax.set_title(f'{interval=} ms, {period}\n(no data)')
            continue

        df_pivot = df_sel.pivot_table(index='n_trials', columns='n_samples',
                                      values='power', aggfunc='mean')
        X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
        Z = df_pivot.values

        cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')

        for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
            ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)

        ax.set_xticks(np.arange(0, 80, 10))
        ax.set(title=f'{interval=} ms, {period}',
               xlabel='bootstrapped sample size',
               ylabel='bootstrapped trials per participant' if i == 0 else '')
        legend_handles = [Line2D([], [], color=col, linewidth=lw,
                                 label=f'{int(lvl*100)}%')
                          for lvl, col, lw in zip(power_levels, contour_colors,
                                                  contour_widths)]
        ax.legend(handles=legend_handles, title='power', loc='upper right',
                  frameon=True)
fig.suptitle('SODA power contour: bootstrapped participants × trials (ttest)')

plt.pause(0.1)
fig.tight_layout(rect=[0, 0, 0.92, 1])
plt.pause(0.1)
fig.subplots_adjust(right=0.92)
cax = fig.add_axes([0.93, 0.175, 0.015, 0.6])
fig.colorbar(cf, cax=cax, label='power', shrink=0.5)
savefig(fig, settings.plot_dir + '/figures/soda_power_contour_ttest.png', tight=False)

#%% mean across all speed conditions
df_mean = df_power.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples', values='power')
X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
Z = df_pivot.values

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')

for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
    ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)

ax.set_xticks(np.arange(0, 80, 10))
ax.set(title='',
       xlabel='bootstrapped sample size',
       ylabel='bootstrapped trials per participant')
legend_handles = [Line2D([], [], color=col, linewidth=lw, label=f'{int(lvl*100)}%')
                  for lvl, col, lw in zip(power_levels, contour_colors, contour_widths)]
ax.legend(handles=legend_handles, title='power', loc='upper right', frameon=True)
fig.suptitle('SODA power contour\nacross speed conditions (ttest)')

fig.colorbar(cf, ax=ax, label='power')
savefig(fig, settings.plot_dir + '/figures/soda_power_contour_ttest_mean.png', tight=False)
