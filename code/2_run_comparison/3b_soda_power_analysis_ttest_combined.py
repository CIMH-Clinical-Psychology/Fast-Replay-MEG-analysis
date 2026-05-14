#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 2026

Create power contour plots for SODA as a function of trial numbers and sample size.

Uses ttest_1samp on the COMBINED onset/offset metric:
    combined = 0.5 * mean(slope at onset TRs) + 0.5 * (-mean(slope at offset TRs))
The offset is sign-flipped so a positive combined value reflects evidence for
sequential structure in both phases simultaneously.

Cluster usage (one job per n_samples value, iterates over all n_trials internally):
    python 3b_soda_power_analysis_ttest_combined.py --max_trials 64 --n_samples 15

Without arguments, prints a message to use the plot script.

@author: simon.kern
"""

import os
os.environ['NO_PRELOADING'] = '1'
import sys; sys.path.append('..')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import settings
from meg_utils.misc import make_seed
from mne_bids import BIDSPath
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed

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


def bootstrap(combined, n_trials, n_samples, n_draws=1000):
    """Bootstrap both participants and trials simultaneously.

    For each draw: sample n_samples participants (with replacement),
    then for each sampled participant sample n_trials trials (with replacement),
    average to get one value per participant, run ttest_1samp.

    Parameters
    ----------
    combined : array [n_subj, n_trials_orig]
        Combined onset/offset metric per subject and trial.
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

    subj_idx = rng.integers(0, n_subj, (n_draws, n_samples))
    trial_idx = rng.integers(0, n_trials_orig, (n_draws, n_samples, n_trials))

    ps = []
    for draw in range(n_draws):
        sampled = np.array([combined[subj_idx[draw, s],
                                     trial_idx[draw, s]].mean()
                            for s in range(n_samples)])
        _, p = ttest_1samp(sampled, 0)
        ps.append(p)
    return np.array(ps)


def run_power_contour(max_trials, n_samples, n_draws=1000, overwrite=False):
    """Compute power for all n_trials in [2, max_trials] at fixed n_samples."""
    assert os.path.isfile(pkl_slopes), \
        f'File not found, did 1b script run? {pkl_slopes}'

    pkl_out = bids_base.copy().update(processing='sodattestcombined',
                                      suffix='powercontour',
                                      task=f'{n_samples}',
                                      extension='.pkl.gz')

    if pkl_out.fpath.is_file() and not overwrite:
        print(f'{n_samples=} exists already, skipping')
        return

    df_slopes = joblib.load(pkl_slopes)
    n_subj = len(df_slopes.subject.unique())

    pool = Parallel(-1)
    trial_numbers = range(2, max_trials + 1)

    # per-(subject, trial) mean slope within each period
    df_mean = df_slopes.groupby(['subject', 'period', 'interval', 'trial']).mean(True).reset_index()

    df = pd.DataFrame()

    for iv in intervals:
        print(f'starting with {iv=} for {n_samples=}')

        df_on = df_mean[(df_mean.interval == iv) & (df_mean.period == 'onset')].sort_values(['subject', 'trial'])
        df_off = df_mean[(df_mean.interval == iv) & (df_mean.period == 'offset')].sort_values(['subject', 'trial'])
        slopes_on = df_on.slope.values.reshape([n_subj, -1])
        slopes_off = df_off.slope.values.reshape([n_subj, -1])
        combined = 0.5 * slopes_on + 0.5 * (-slopes_off)

        ps = pool(delayed(bootstrap)(combined, nt, n_samples, n_draws=n_draws)
                  for nt in tqdm(trial_numbers))
        ps = np.array(ps)  # [n_trial_sizes, n_draws]
        power = (ps < 0.05).mean(-1)

        df_tmp = pd.DataFrame({'n_trials': trial_numbers, 'n_samples': n_samples,
                               'interval': iv, 'period': 'combined', 'power': power})
        df = pd.concat([df, df_tmp], ignore_index=True)

    df.attrs['method'] = 'scipy.stats.ttest_1samp on combined metric'
    df.attrs['overwrite'] = overwrite
    df.attrs['n_draws'] = n_draws
    df.attrs['max_trials'] = max_trials
    df.attrs['n_samples'] = n_samples
    df.attrs['__file__'] = locals().get('__file__')

    joblib.dump(df, pkl_out.fpath)
    print(f'Saved {pkl_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SODA power analysis (ttest, combined)')
    parser.add_argument('--max_trials', type=int, default=64,
                        help='Max number of trials to iterate over (2..max_trials)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of participants to resample (fixed per job)')
    parser.add_argument('--n_draws', type=int, default=1000,
                        help='Number of bootstrap repetitions (default: 1000)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing result files')
    args = parser.parse_args()

    if args.n_samples is not None:
        run_power_contour(args.max_trials, args.n_samples,
                          n_draws=args.n_draws, overwrite=args.overwrite)
    else:
        print('For plotting, run: python 3b_soda_power_analysis_plot.py')
