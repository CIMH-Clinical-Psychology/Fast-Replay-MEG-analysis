#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:07:28 2026

Create power contour plots as a function of trial numbers and sample size.

This script uses a simple ttest on the mean sequenceness values around
the expected time lag

@author: simon.kern
"""

import os
os.environ['NO_PRELOADING'] = '1'
import sys; sys.path.append('..')
import argparse
import tdlm
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import settings
from meg_utils.misc import make_seed, to_long_df
from mne_bids import BIDSPath
from mne.stats import permutation_t_test
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed


def bootstrap(sf, n_trials, n_samples, n_draws=1000, tqdm_loop=None):
    """Bootstrap both participants and trials simultaneously.

    For each draw: sample n_samples participants (with replacement),
    then for each sampled participant sample n_trials trials (with replacement),
    run signflip test per participant, collect p-values.

    Parameters
    ----------
    sf : array [n_subj, n_trials_orig, n_lags]
        Trial-level sequenceness for one interval.
    n_trials : int
        Number of trials to resample per participant.
    n_samples : int
        Number of participants to resample.
    n_draws : int
        Number of bootstrap repetitions.

    Returns
    -------
    ps : np.ndarray [n_draws, n_lags]
        p-value per lag for each draw.
    """
    n_subj, n_trials_orig, n_features = sf.shape
    seed = make_seed(n_trials, n_samples)
    rng = np.random.default_rng(seed)

    # pre-draw all indices
    subj_idx = rng.integers(0, n_subj, (n_draws, n_samples))
    trial_idx = rng.integers(0, n_trials_orig, (n_draws, n_samples, n_trials))

    ps = []
    for draw in range(n_draws):
        # for each sampled participant, average over sampled trials -> mean sequenceness
        sf_sampled = np.full((n_samples, n_features), np.nan)
        for s in range(n_samples):
            sf_sampled[s] = sf[subj_idx[draw, s],
                                                trial_idx[draw, s]].mean(axis=0)
        # group-level permutation t-test on the mean sequenceness
        _, p = ttest_1samp(sf_sampled, 0)
        # p, t_obs, t_perms = permutation_t_test(sf_sampled, n_permutations=1014,
        #                                        seed=rng, verbose=False)
        ps.append(p)
        if tqdm_loop is not None:
            tqdm_loop.update()
    return np.squeeze(ps)

#%% run bootstrap power analysis with ttest
from itertools import product
subjects = [f'sub-{i:02d}' for i in range(1, 31)]

bids_base = BIDSPath(
    root=settings.bids_dir_meg + '/derivatives',
    datatype='results',
    subject='group',
    task='main',
    check=False
)

pkl_seq = bids_base.copy().update(processing='trials', suffix='sequenceness',
                                  extension='.pkl.gz')
pkl_out = bids_base.copy().update(processing='ttest', suffix='powercontour',
                                   extension='.pkl.gz')

assert pkl_seq.fpath.is_file(), f'File not found, did 1a script run? {pkl_seq.fpath}'

max_trials = 48   # n_trials * 3
max_samples = 90  # n_subjects *3
n_draws = 1000

sf_trials = joblib.load(pkl_seq)['sf_trials']

pool = Parallel(-1)

n_trials = range(2, max_trials + 1)
n_samples = range(2, max_samples + 1)
df = pd.DataFrame()

#
params = list(product(n_trials, n_samples))

for iv in settings.intervals_MEG:
    sf = sf_trials[iv][:, :, 0, 1:]  # [n_subj, n_trials, n_lags], drop lag=0

    # get mean sequenceness around the expected peak +- 10 ms
    lag_exp = settings.exp_lag[iv]
    sf_peak = sf[:, :, lag_exp-2:lag_exp+1, None].mean(-2)

    ps = pool(delayed(bootstrap)(sf_peak, nt, ns, n_draws=n_draws)
               for nt, ns in tqdm(params))
    # ps is a list of len(params) arrays, each of shape [n_draws]
    # product iterates trials first, then samples -> reshape accordingly
    ps = np.array(ps).reshape([len(n_trials), len(n_samples), n_draws])
    power = (ps < 0.05).mean(-1)

    df_iv = to_long_df(power, value_name='power',
                       columns=['n_trials', 'n_samples'],
                       n_trials=n_trials, n_samples=n_samples)
    df_iv['interval'] = iv

    df = pd.concat([df, df_iv])


joblib.dump(df, pkl_out.fpath)
#%% plotting
import matplotlib.pyplot as plt
from meg_utils.plotting import savefig
from matplotlib.lines import Line2D

df_power = joblib.load(pkl_out)
intervals = df_power.interval.unique()

fig, axs = plt.subplots(1, len(intervals), figsize=[4 * len(intervals), 4])

for i, interval in enumerate(intervals):
    ax = axs[i]
    df_sel = df_power[df_power.interval == interval]

    if len(df_sel) == 0:
        ax.set_title(f'{interval=} ms\n(no data)')
        continue

    # pivot to 2D grid for contour plot
    df_pivot = df_sel.pivot_table(index='n_trials', columns='n_samples',
                                  values='power', aggfunc='mean')
    X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
    Z = df_pivot.values

    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')

    # contour lines at key power thresholds with graded colours
    power_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    contour_colors = ['#8b0000', '#f46d43', '#fdae61', '#d62728', '#fee08b']
    contour_widths = [1, 1, 1, 3, 1]
    for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
        ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)

    ax.set_xticks(np.arange(0, 90, 10))
    ax.set(title=f'{interval=} ms',
           xlabel='bootstrapped sample size',
           ylabel='bootstrapped trials per participant' if i==0 else '')
    legend_handles = [Line2D([], [], color=col, linewidth=lw,
                             label=f'{int(lvl*100)}%')
                      for lvl, col, lw in zip(power_levels, contour_colors,
                                              contour_widths)]
    ax.legend(handles=legend_handles, title='power', loc='upper right',
              frameon=True)
fig.suptitle('Power contour: boostrapped participants × trials (ttest)')

plt.pause(0.1)
fig.tight_layout(rect=[0, 0, 0.92, 1])
plt.pause(0.1)
fig.subplots_adjust(right=0.92)
cax = fig.add_axes([0.93, 0.175, 0.015, 0.6])
fig.colorbar(cf, cax=cax, label='power', shrink=0.5)
savefig(fig, settings.plot_dir + '/figures/power_contour_ttest.png', tight=False)

#%% mean across all speed conditions
df_mean = df_power.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples', values='power')
X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
Z = df_pivot.values

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')

power_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
contour_colors = ['#8b0000', '#f46d43', '#fdae61', '#d62728', '#fee08b']
contour_widths = [1, 1, 1, 3, 1]
for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
    ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)

ax.set_xticks(np.arange(0, 90, 10))
ax.set(title='',
       xlabel='bootstrapped sample size',
       ylabel='bootstrapped trials per participant')
legend_handles = [Line2D([], [], color=col, linewidth=lw, label=f'{int(lvl*100)}%')
                  for lvl, col, lw in zip(power_levels, contour_colors, contour_widths)]
ax.legend(handles=legend_handles, title='power', loc='upper right', frameon=True)
fig.suptitle('Power contour\nacross speed conditions (ttest)')

# plt.pause(0.1)
# fig.tight_layout(rect=[0, 0, 0.92, 1])
# plt.pause(0.1)
fig.colorbar(cf, ax=ax,  label='power')
savefig(fig, settings.plot_dir + '/figures/power_contour_ttest_mean.png', tight=False)
