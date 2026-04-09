#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 2026

All power contour plotting for SODA power analysis.
Loads results from all three statistical methods (signflip, ttest, cluster)
and produces:
  - per-interval contour plots for each method
    (1×N for ttest, 2×N for signflip/cluster with onset/offset rows)
  - mean-across-intervals contour plot per method  (1×1 or 1×2)
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

from settings import intervals_3T as intervals

intervals = intervals[:-1]

bids_base = BIDSPath(
    root=settings.bids_dir_3T + '/derivatives',
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
    df = pd.concat([joblib.load(f) for f in files], ignore_index=True)
    df['method'] = processing
    mapping = {'fwd': 'onset', 'bkw': 'offset'}
    df['period'] = df.period.apply(lambda x: mapping[x])
    df = df[df.interval!=2048]
    return df


def _load_ttest():
    """Load ttest results (single pkl.gz file)."""
    pkl_out = bids_base.copy().update(processing='sodattest',
                                       suffix='powercontour',
                                       extension='.pkl.gz')
    if not pkl_out.fpath.is_file():
        return None
    df = joblib.load(pkl_out.fpath)
    df['method']= 't-test'
    mapping = {'fwd': 'onset', 'bkw': 'offset'}
    df['period'] = df.period.apply(lambda x: mapping[x])
    df = df[df.interval!=2048]
    return df


# ── shared plot helpers ───────────────────────────────────────────────────────

def _contour_ax(ax, Z, X, Y, i=0):
    """Draw filled contour + threshold lines on ax. Returns cf handle."""
    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')
    for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
        ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)
    ax.set_xticks(np.arange(0, int(X.max()) + 10, 10))
    ax.set(xlabel='bootstrapped sample size',
           ylabel='bootstrapped trials per participant' if i == 0 else '')
    return cf


def _add_colorbar_and_legend(fig, cf, axs):
    plt.pause(0.1)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    plt.pause(0.1)
    fig.subplots_adjust(right=0.92)
    # figure-level legend above colorbar
    legend_handles = [Line2D([], [], color=col, linewidth=lw,
                             label=f'{int(lvl*100)}%')
                      for lvl, col, lw in zip(power_levels, contour_colors,
                                              contour_widths)]
    fig.legend(handles=legend_handles, title='power',
               loc='upper right', bbox_to_anchor=(0.99, 0.88), frameon=True)
    cax = fig.add_axes([0.93, 0.1, 0.015, 0.5])
    fig.colorbar(cf, cax=cax, label='power')


# ── per-method plots ──────────────────────────────────────────────────────────

def plot_per_interval(df_power, method_name, savepath):
    """Contour plot, one panel per speed condition.

    If data has a 'period' column (onset/offset), creates 2×N layout.
    Otherwise creates 1×N layout.
    """
    has_period = 'period' in df_power.columns
    n_iv = len(intervals)

    if has_period:
        fig, axs = plt.subplots(2, n_iv, figsize=[4 * n_iv, 8])
        periods = ['onset', 'offset']
    else:
        fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4])
        axs = axs[np.newaxis, :]  # make 2D for uniform indexing
        periods = [None]

    cf = None
    for j, period in enumerate(periods):
        for i, interval in enumerate(intervals):
            ax = axs[j, i]
            df_sel = df_power[df_power.interval == interval]
            if period is not None:
                df_sel = df_sel[df_sel.period == period]

            assert len(df_sel) > 0
                # title = f'{interval} ms'
                # if period is not None:
                #     title += f', {period}'
                # ax.set_title(f'{title}\n(no data)')
                # continue

            df_pivot = df_sel.pivot_table(index='n_trials', columns='n_samples',
                                          values='power', aggfunc='mean')
            X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
            Z = df_pivot.values
            cf = _contour_ax(ax, Z, X, Y, i)

            title = f'{interval} ms'
            if period is not None:
                title += f', {period}'
            ax.set_title(title)
            ax.set_ylim([2, 60])
            ax.set_xlim([2, 80])

    fig.suptitle(f'SODA power contour ({method_name})')
    if has_period:
        row_labels = ['onset period', 'offset period']
        for j, label in enumerate(row_labels):
            axs[j, 0].annotate(label,
                                xy=(0, 0.5), xycoords='axes fraction',
                                xytext=(-60, 0), textcoords='offset points',
                                va='center', ha='center', rotation=90,
                                fontsize=16, fontweight='bold')
    else:
        axs[0, 0].annotate('SODA on fMRI',
                             xy=(0, 0.5), xycoords='axes fraction',
                             xytext=(-60, 0), textcoords='offset points',
                             va='center', ha='center', rotation=90,
                             fontsize=16, fontweight='bold')
    if cf is not None:
        _add_colorbar_and_legend(fig, cf, axs)
    savefig(fig, savepath, tight=False)
    print(f'Saved {savepath}')


def plot_mean(df_power, method_name, savepath):
    """Mean power across all speed conditions (and periods if present)."""
    df_mean = df_power.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
    df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples', values='power')
    X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
    Z = df_pivot.values

    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    cf = _contour_ax(ax, Z, X, Y, i=0)
    fig.suptitle(f'SODA power contour: mean across speed conditions ({method_name})')
    fig.colorbar(cf, ax=ax, label='power')
    savefig(fig, savepath, tight=False)
    print(f'Saved {savepath}')


def plot_mean_per_period(df_power, method_name, savepath):
    """1×2 mean power contour, one panel per period (onset/offset)."""
    if 'period' not in df_power.columns:
        return

    fig, axs = plt.subplots(1, 2, figsize=[12, 5], sharey=True, sharex=True)
    cf = None
    for i, period in enumerate(['onset', 'offset']):
        ax = axs[i]
        df_sel = df_power[df_power.period == period]
        df_mean = df_sel.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
        df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples',
                                        values='power')
        X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
        Z = df_pivot.values
        cf = _contour_ax(ax, Z, X, Y, i)
        ax.set_title(f'{period}')
    fig.suptitle(f'SODA power contour: mean across speed conditions ({method_name})')
    if cf is not None:
        _add_colorbar_and_legend(fig, cf, axs)
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
    fig.suptitle('SODA power contour: mean across speed conditions')
    if cf is not None:
        _add_colorbar_and_legend(fig, cf, axs)
    savepath = settings.plot_dir + '/figures/soda_power_contour_methods_comparison.png'
    savefig(fig, savepath, tight=False)
    print(f'Saved {savepath}')


# ── 80% power curve comparison ────────────────────────────────────────────────

def plot_80_curve(datasets):
    """Plot the 80% power contour line for each method, onset and offset separately.

    For each method, the mean power across speed conditions is computed
    per period, then the 80% iso-power line is drawn.
    """
    fig, axs = plt.subplots(1, 2, figsize=[14, 6], sharey=True, sharex=True)
    colors = sns.color_palette()

    for j, period in enumerate(['onset', 'offset']):
        ax = axs[j]
        for i, (method_name, df_power) in enumerate(datasets):
            df_sel = df_power[df_power.period == period]
            df_mean = df_sel.groupby(['n_trials', 'n_samples'])['power'].mean().reset_index()
            df_pivot = df_mean.pivot_table(index='n_trials', columns='n_samples',
                                            values='power')
            X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
            Z = df_pivot.values
            cs = ax.contour(X, Y, Z, levels=[0.8], colors=[colors[i]], linewidths=2)
            cs.collections[0].set_label(method_name)

        ax.legend(frameon=True)
        ax.set(xlabel='bootstrapped sample size',
               ylabel='bootstrapped trials per participant' if j == 0 else '',
               title=f'{period}')

    fig.suptitle('SODA 80% power curve (mean across speed conditions)')
    fig.tight_layout()
    savepath = settings.plot_dir + '/figures/soda_power_80_curve_comparison.png'
    savefig(fig, savepath)
    print(f'Saved {savepath}')


# ── main ──────────────────────────────────────────────────────────────────────

all_methods = [
    ('permutation t-test', _load_pkl_method('sodasignflip')),
    ('ttest',              _load_ttest()),
    ('cluster',            _load_pkl_method('sodacluster')),
]

fnames = {
    'permutation t-test': ('soda_power_contour_signflip',
                           'soda_power_contour_signflip_mean',
                           'soda_power_contour_signflip_mean_period'),
    'ttest':              ('soda_power_contour_ttest',
                           'soda_power_contour_ttest_mean',
                           'soda_power_contour_ttest_mean_period'),
    'cluster':            ('soda_power_contour_cluster',
                           'soda_power_contour_cluster_mean',
                           'soda_power_contour_cluster_mean_period'),
}

df_all = pd.concat([dfx[1] for dfx in all_methods])
plot_per_interval(df_all, 'mean', settings.plot_dir + f'/figures/soda_power_contours_mean.png')
asd
available = []
for method_name, df in all_methods:
    if df is None:
        print(f'WARNING: no data found for "{method_name}", skipping')
        continue
    available.append((method_name, df))
    stem_iv, stem_mean, stem_period = fnames[method_name]
    plot_per_interval(df, method_name,
                      settings.plot_dir + f'/figures/soda_{stem_iv}.png')
    plot_mean(df, method_name,
              settings.plot_dir + f'/figures/{stem_mean}.png')
    if stem_period is not None:
        plot_mean_per_period(df, method_name,
                             settings.plot_dir + f'/figures/soda_{stem_period}.png')

assert len(available) > 0, 'No results found for any method'
plot_comparison(available)
plot_80_curve(available)
print('Done')
