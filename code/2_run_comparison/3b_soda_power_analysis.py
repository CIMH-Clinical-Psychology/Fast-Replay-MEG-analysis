#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 2026

Create power contour plots for SODA as a function of trial numbers and sample size.
Tests both forward and backward periods separately.

Cluster usage (one job per n_samples value, iterates over all n_trials internally):
    python 3b_soda_power_analysis.py --max_trials 32 --n_samples 15
    python 3b_soda_power_analysis.py --max_trials 32 --n_samples 15 --interval 2048

Without arguments, collects all precomputed results and plots contour maps.

@author: simon.kern
"""

import os
os.environ['NO_PRELOADING'] = '1'
import sys; sys.path.append('..')
import argparse
import tdlm
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import settings
from meg_utils.misc import make_seed, long_df_to_array
from mne_bids import BIDSPath

from settings import subjects_3T as subjects
from settings import intervals_3T as intervals

bids_base = BIDSPath(
    root=settings.bids_dir_3T + '/derivatives',
    datatype='results',
    subject='group',
    task='main',
    check=False
)

pkl_slopes = str(bids_base.copy().update(processing='soda', suffix='slopes',
                                          extension='.pkl.gz'))


def bootstrap(mean_slopes, n_trials, n_samples, n_draws=1000):
    """Bootstrap both participants and trials simultaneously.

    For each draw: sample n_samples participants (with replacement),
    then for each sampled participant sample n_trials trials (with replacement),
    average to get one value per participant, run tdlm.signflit_test.

    Parameters
    ----------
    mean_slopes : array [n_subj, n_trials_orig]
        Mean slope in the expected period per subject and trial.
    n_trials : int
        Number of trials to resample per participant.
    n_samples : int
        Number of participants to resample.
    n_draws : int
        Number of bootstrap repetitions.

    Returns
    -------
    ps : list of float
        p-value for each draw.
    """
    mean_slopes = np.atleast_3d(mean_slopes)
    n_subj, n_trials_orig, n_features = mean_slopes.shape
    seed = make_seed(n_trials, n_samples)
    rng = np.random.default_rng(seed)

    # pre-draw all indices
    subj_idx = rng.integers(0, n_subj, (n_draws, n_samples))
    trial_idx = rng.integers(0, n_trials_orig, (n_draws, n_samples, n_trials))

    ps = []
    for draw in range(n_draws):
        slopes_sampled = np.full((n_samples, n_features), np.nan)
        for s in range(n_samples):
            slopes_sampled[s] = mean_slopes[subj_idx[draw, s],
                                            trial_idx[draw, s]].mean(axis=0)
        p, t_obs, t_perms = tdlm.signflit_test(slopes_sampled, rng=rng)
        ps.append(p)
    return ps


def run_power_contour(max_trials, n_samples, interval=None, n_draws=1000,
                      overwrite=False):
    """Compute power for all n_trials in [2, max_trials] at fixed n_samples."""
    assert os.path.isfile(pkl_slopes), \
        f'File not found, did 1b script run? {pkl_slopes}'

    intervals_to_run = [interval] if interval is not None else intervals

    # check which intervals need computing
    out_dir = bids_base.copy().update(processing='sodapowercontour',
                                      suffix='power', extension='.csv')
    out_dir.mkdir(True)

    df_slopes = joblib.load(pkl_slopes)
    n_subj = len(df_slopes.subject.unique())

    n_total = len(intervals_to_run) * 2 * (max_trials - 1)
    tqdm_loop = tqdm(total=n_total)

    for iv in intervals_to_run:
        for period, flip_sign in [('fwd', False), ('bkw', True)]:
            out_path = str(out_dir.fpath).replace(
                '.csv', f'_interval-{iv}_period-{period}_nsamples-{n_samples}.csv')

            if os.path.isfile(out_path) and not overwrite:
                print(f'{n_samples=} - {iv} ms {period} exists already, no overwrite')
                tqdm_loop.update(max_trials - 1)
                continue
            sel = (df_slopes.interval==iv) & (df_slopes.period==period)
            df_sel = df_slopes[sel].sort_values(['subject', 'trial', 'tr'])
            n_trs = len(df_sel.tr.unique())
            slopes = df_sel.slope.values.reshape([n_subj, -1, n_trs])
            mean_slopes = slopes.mean(2)

            # flip sign so test is always to right tail
            if period=='bkw':
                mean_slopes *= -1

            rows = []
            tqdm_loop.set_description(f'{iv} ms {period}, {n_samples=}')
            for n_trials in range(2, max_trials + 1):
                ps = bootstrap(mean_slopes, n_trials, n_samples, n_draws=n_draws)
                power = (np.array(ps) < 0.05).mean()
                rows.append({'n_trials': n_trials, 'n_samples': n_samples,
                             'interval': iv, 'period': period, 'power': power})
                tqdm_loop.update()

            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_path, index=False)
            print(f'Saved {out_path}')


def plot_results():
    """Collect all precomputed results and create power contour plots."""
    import matplotlib.pyplot as plt
    from meg_utils.plotting import savefig
    from glob import glob
    from matplotlib.lines import Line2D

    out_dir = bids_base.copy().update(processing='sodapowercontour',
                                      suffix='power', extension='.csv')
    pattern = str(out_dir.fpath).replace('.csv', '_interval-*_period-*_nsamples-*.csv')
    files = sorted(glob(pattern))
    assert len(files) > 0, f'No result files found matching {pattern}'

    df_power = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    n_iv = len(intervals)
    fig, axs = plt.subplots(2, n_iv, figsize=[4 * n_iv, 8])

    power_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    contour_colors = ['#8b0000', '#f46d43', '#fdae61', '#d62728', '#fee08b']
    contour_widths = [1, 1, 1, 3, 1]

    cf = None
    for i, interval in enumerate(intervals):
        for j, period in enumerate(['fwd', 'bkw']):
            ax = axs[j, i]
            df_sel = df_power[(df_power.interval == interval) &
                              (df_power.period == period)]

            if len(df_sel) == 0:
                ax.set_title(f'{interval=} ms, {period}\n(no data)')
                continue

            # pivot to 2D grid for contour plot
            df_pivot = df_sel.pivot_table(index='n_trials', columns='n_samples',
                                          values='power', aggfunc='mean')
            X, Y = np.meshgrid(df_pivot.columns.values, df_pivot.index.values)
            Z = df_pivot.values

            cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 1, 21), cmap='Blues')

            # contour lines at key power thresholds with graded colours
            for lvl, col, lw in zip(power_levels, contour_colors, contour_widths):
                ax.contour(X, Y, Z, levels=[lvl], colors=[col], linewidths=lw)

            ax.set_xticks(np.arange(0, ax.get_xlim()[1], 10))
            ax.set(title=f'{interval=} ms, {period}',
                   xlabel='bootstrapped sample size',
                   ylabel='bootstrapped trials per participant' if i == 0 else '')

            legend_handles = [Line2D([], [], color=col, linewidth=lw,
                                     label=f'{int(lvl*100)}%')
                              for lvl, col, lw in zip(power_levels, contour_colors,
                                                      contour_widths)]
            ax.legend(handles=legend_handles, title='power', loc='upper right',
                      frameon=True)

    fig.suptitle('SODA power contour: bootstrapped participants × trials')

    plt.pause(0.1)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    plt.pause(0.1)
    fig.subplots_adjust(right=0.92)
    if cf is not None:
        cax = fig.add_axes([0.93, 0.175, 0.015, 0.6])
        fig.colorbar(cf, cax=cax, label='power')
    savefig(fig, settings.plot_dir + '/figures/soda_power_contour_combined.png', tight=False)
    print(f'Plotted {len(files)} result files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SODA power analysis')
    parser.add_argument('--max_trials', type=int, default=None,
                        help='Max number of trials to iterate over (2..max_trials)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of participants to resample (fixed per job)')
    parser.add_argument('--n_draws', type=int, default=1000,
                        help='Number of bootstrap repetitions (default: 1000)')
    parser.add_argument('--interval', type=int, default=None,
                        choices=intervals,
                        help='Run only this interval (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing result files')
    args = parser.parse_args()

    if args.max_trials is not None and args.n_samples is not None:
        run_power_contour(args.max_trials, args.n_samples, interval=args.interval,
                          n_draws=args.n_draws, overwrite=args.overwrite)
    else:
        plot_results()
