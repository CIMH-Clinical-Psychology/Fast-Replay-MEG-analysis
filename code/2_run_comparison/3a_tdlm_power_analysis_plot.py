#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 2026

All power contour plotting for TDLM power analysis.
Loads results from all three statistical methods (signflip, ttest, cluster)
and produces:
  - per-interval contour plots for each method  (1×4)
  - mean-across-intervals contour plot per method  (1×1)
  - side-by-side comparison across methods  (1×3)

@author: simon.kern
"""

import os
os.environ['NO_PRELOADING'] = '1'
import sys; sys.path.append('..')
import numpy as np
import pandas as pd
import joblib
import settings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from meg_utils.plotting import savefig
from mne_bids import BIDSPath
from glob import glob

intervals = [32, 64, 128, 512]

bids_base = BIDSPath(
    root=settings.bids_dir_meg + '/derivatives',
    datatype='results',
    subject='group',
    task='main',
    check=False
)

# shared plotting style
power_levels  = [0.5, 0.6, 0.7, 0.8, 0.9]
contour_colors = ['#8b0000', '#f46d43', '#fdae61', '#d62728', '#fee08b']
contour_widths = [1, 1, 1, 3, 1]


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_pkl_method(processing):
    """Load pkl.gz results split by n_samples (signflip / cluster)."""
    template = bids_base.copy().update(processing=processing,
                                        suffix='powercontour',
                                        task='NSAMPLES',
                                        extension='.pkl.gz')
    pattern = str(template.fpath).replace('task-NSAMPLES', 'task-*')
    files = sorted(glob(pattern))
    if not files:
        return None
    return pd.concat([joblib.load(f) for f in files], ignore_index=True)


def _load_ttest():
    """Load ttest results (single CSV file)."""
    pkl_out = bids_base.copy().update(processing='ttest', suffix='powercontour',
                                       extension='.pkl.gz')
    if not pkl_out.fpath.is_file():
        return None
    return joblib.load(pkl_out.fpath)


# ── shared plot helpers ───────────────────────────────────────────────────────

def _contour_ax(ax, Z, X, Y, i=0):
    """Draw filled contour + threshold lines on ax. Returns cf handle."""
    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')
    for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
        ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)
    ax.set_xticks(np.arange(0, int(X.max()) + 10, 10))
    legend_handles = [Line2D([], [], color=col, linewidth=lw,
                             label=f'{int(lvl*100)}%')
                      for lvl, col, lw in zip(power_levels, contour_colors,
                                              contour_widths)]
    ax.legend(handles=legend_handles, title='power', loc='upper right',
              frameon=True)
    ax.set(xlabel='bootstrapped sample size',
           ylabel='bootstrapped trials per participant' if i == 0 else '')
    return cf


def _add_colorbar(fig, cf, axs):
    plt.pause(0.1)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    plt.pause(0.1)
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.93, 0.175, 0.015, 0.6])
    fig.colorbar(cf, cax=cax, label='power')


# ── per-method plots ──────────────────────────────────────────────────────────

def plot_per_interval(df_power, method_name, savepath):
    """1×4 contour plot, one panel per speed condition."""
    fig, axs = plt.subplots(1, len(intervals), figsize=[4 * len(intervals), 4])
    cf = None
    for i, interval in enumerate(intervals):
        ax = axs[i]
        df_sel = df_power[df_power.interval == interval]
        if len(df_sel) == 0:
            ax.set_title(f'{interval=} ms\n(no data)')
            continue
        df_pivot = df_sel.pivot_table(index='n_trials', columns='n_samples',
                                      values='power', aggfunc='mean')
        X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
        Z = df_pivot.values
        cf = _contour_ax(ax, Z, X, Y, i)
        ax.set_title(f'{interval=} ms')
    fig.suptitle(f'Power contour: bootstrapped participants × trials ({method_name})')
    if cf is not None:
        _add_colorbar(fig, cf, axs)
    savefig(fig, savepath, tight=False)
    print(f'Saved {savepath}')


def plot_mean(df_power, method_name, savepath):
    """1×1 contour plot, mean power across all speed conditions."""
    df_mean = df_power.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
    df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples', values='power')
    X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
    Z = df_pivot.values

    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    cf = _contour_ax(ax, Z, X, Y, i=0)
    fig.suptitle(f'Power contour: mean across speed conditions ({method_name})')
    fig.colorbar(cf, ax=ax, label='power')
    savefig(fig, savepath, tight=False)
    print(f'Saved {savepath}')


# ── comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(datasets):
    """1×N side-by-side mean-power contour, one panel per method."""
    n = len(datasets)
    fig, axs = plt.subplots(1, n, figsize=[6 * n, 5], sharey=True, sharex=True)
    if n == 1:
        axs = [axs]
    cf = None
    for i, (method_name, df_power) in enumerate(datasets):
        ax = axs[i]
        df_mean = df_power.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
        df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples',
                                        values='power')
        X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
        Z = df_pivot.values
        cf = _contour_ax(ax, Z, X, Y, i)
        ax.set_title(method_name)
    fig.suptitle('Power contour: mean across speed conditions')
    if cf is not None:
        _add_colorbar(fig, cf, axs)
    savepath = settings.plot_dir + '/figures/power_contour_methods_comparison.png'
    savefig(fig, savepath, tight=False)
    print(f'Saved {savepath}')


# ── 80% power curve comparison ────────────────────────────────────────────────

def plot_80_curve(datasets):
    """Plot the 80% power contour line for each method on the same axes.

    For each method, the mean power across speed conditions is computed,
    then the 80% iso-power line is drawn.
    """
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    colors = sns.color_palette()

    for i, (method_name, df_power) in enumerate(datasets):
        df_mean = df_power.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
        df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples',
                                        values='power')
        X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
        Z = df_pivot.values
        cs = ax.contour(X, Y, Z, levels=[0.8], colors=[colors[i]], linewidths=2)
        cs.collections[0].set_label(method_name)

    ax.legend(frameon=True)
    ax.set(xlabel='bootstrapped sample size',
           ylabel='bootstrapped trials per participant',
           title='80% power curve (mean across speed conditions)')
    fig.tight_layout()
    savepath = settings.plot_dir + '/figures/power_80_curve_comparison.png'
    savefig(fig, savepath)
    print(f'Saved {savepath}')


# ── main ──────────────────────────────────────────────────────────────────────

all_methods = [
    ('permutation t-test', _load_pkl_method('signflip')),
    ('ttest',              _load_ttest()),
    ('cluster',            _load_pkl_method('cluster')),
]

fnames = {
    'permutation t-test': ('power_contour_signflip',      'power_contour_signflip_mean'),
    'ttest':              ('power_contour_ttest',          'power_contour_ttest_mean'),
    'cluster':            ('power_contour_cluster',        'power_contour_cluster_mean'),
}

available = []
for method_name, df in all_methods:
    if df is None:
        print(f'WARNING: no data found for "{method_name}", skipping')
        continue
    available.append((method_name, df))
    stem_iv, stem_mean = fnames[method_name]
    plot_per_interval(df, method_name,
                      settings.plot_dir + f'/figures/{stem_iv}.png')
    plot_mean(df, method_name,
              settings.plot_dir + f'/figures/{stem_mean}.png')

assert len(available) > 0, 'No results found for any method'
plot_comparison(available)
plot_80_curve(available)
print('Done')
