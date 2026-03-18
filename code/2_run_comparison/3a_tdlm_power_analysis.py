#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:07:28 2026

Create power contour plots as a function of trial numbers and sample size.

Cluster usage (one job per n_samples value, iterates over all n_trials internally):
    python 3a_tdlm_power_analysis.py --max_trials 64 --n_samples 15
    python 3a_tdlm_power_analysis.py --max_trials 64 --n_samples 15 --interval 32

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
from meg_utils.misc import make_seed
from mne_bids import BIDSPath

intervals = [32, 64, 128, 512]
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


def bootstrap(sf_trials_interval, n_trials, n_samples, n_draws=1000):
    """Bootstrap both participants and trials simultaneously.

    For each draw: sample n_samples participants (with replacement),
    then for each sampled participant sample n_trials trials (with replacement),
    run signflip test per participant, collect p-values.

    Parameters
    ----------
    sf_trials_interval : array [n_subj, n_trials_orig, n_lags]
        Trial-level sequenceness for one interval.
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
    n_subj, n_trials_orig, n_features = sf_trials_interval.shape
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
            sf_sampled[s] = sf_trials_interval[subj_idx[draw, s],
                                                trial_idx[draw, s]].mean(axis=0)
        # group-level signflip test on the mean sequenceness
        p, t_obs, t_perms = tdlm.signflit_test(sf_sampled, rng=rng)
        ps.append(p)
    return ps


def run_power_contour(max_trials, n_samples, interval=None, n_draws=1000,
                      overwrite=False):
    """Compute power for all n_trials in [2, max_trials] at fixed n_samples."""
    assert pkl_seq.fpath.is_file(), f'File not found, did 1a script run? {pkl_seq.fpath}'

    intervals_to_run = [interval] if interval is not None else intervals

    # check which intervals need computing
    out_dir = bids_base.copy().update(processing='powercontour',
                                      suffix='power', extension='.csv')
    out_dir.mkdir(True)

    res = joblib.load(pkl_seq)
    sf_trials = res['sf_trials']

    tqdm_loop = tqdm(total=len(intervals_to_run)*(max_trials-2))
    for iv in intervals_to_run:
        sf = sf_trials[iv][:, :, 0, 1:]  # [n_subj, n_trials, n_lags], drop lag=0
        out_path  = str(out_dir.fpath).replace('.csv',
                                                f'_interval-{iv}_nsamples-{n_samples}.csv')
        if os.path.isfile(out_path) and not overwrite:
            print (f'{n_samples=} - {iv} ms  exists already, no overwrite')
            continue

        rows = []
        tqdm_loop.set_description(f'{iv} ms, {n_samples=}')
        for n_trials in tqdm(range(2, max_trials + 1)):
            ps = bootstrap(sf, n_trials, n_samples, n_draws=n_draws)
            power = (np.array(ps) < 0.05).mean()
            rows.append({'n_trials': n_trials, 'n_samples': n_samples,
                         'interval': iv, 'power': power})
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

    out_dir = bids_base.copy().update(processing='powercontour',
                                      suffix='power', extension='.csv')
    pattern = str(out_dir.fpath).replace('.csv', '_interval-*_nsamples-*.csv')
    files = sorted(glob(pattern))
    assert len(files) > 0, f'No result files found matching {pattern}'

    df_power = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

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
    fig.suptitle('Power contour: boostrapped participants × trials')

    plt.pause(0.1)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    plt.pause(0.1)
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.93, 0.175, 0.015, 0.6])
    fig.colorbar(cf, cax=cax, label='power', shrink=0.5)
    savefig(fig, settings.plot_dir + '/figures/power_contour_combined.png', tight=False)
    print(f'Plotted {len(files)} result files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDLM power analysis')
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
