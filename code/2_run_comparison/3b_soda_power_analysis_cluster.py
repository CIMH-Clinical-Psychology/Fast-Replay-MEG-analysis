#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 2026

Create power contour plots for SODA as a function of trial numbers and sample size.

Uses cluster permutation test on the full slope curve. Backward-period TRs are
sign-flipped so both fwd and bkw are tested for positive effects simultaneously.
A draw is counted as significant only if a cluster overlaps with the expected
fwd or bkw TR window. Power is reported separately for each period.

Cluster usage (one job per n_samples value, iterates over all n_trials internally):
    python 3b_soda_power_analysis_cluster.py --max_trials 32 --n_samples 15

Without arguments, prints a message to use the plot script.

@author: simon.kern
"""

import os
os.environ['NO_PRELOADING'] = '1'
import sys; sys.path.append('..')
import argparse
import warnings
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import settings
from meg_utils.misc import make_seed
from mne_bids import BIDSPath
from mne.stats import permutation_cluster_1samp_test
from joblib import Parallel, delayed

warnings.filterwarnings('ignore', message='No cluster found.*')

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


def bootstrap(slopes, n_trials, n_samples, tr_fwd, tr_bkw, n_draws=512):
    """Bootstrap both participants and trials simultaneously.

    For each draw: sample n_samples participants (with replacement),
    then for each sampled participant sample n_trials trials (with replacement),
    run a cluster permutation test. A draw is significant for fwd/bkw only if
    any cluster overlaps with that period's TR window.

    Parameters
    ----------
    slopes : array [n_subj, n_trials_orig, n_trs]
        Trial-level slopes for one interval (bkw TRs already sign-flipped).
    n_trials : int
        Number of trials to resample per participant.
    n_samples : int
        Number of participants to resample.
    tr_fwd : list [start, end]
        1-based TR range for forward period.
    tr_bkw : list [start, end]
        1-based TR range for backward period.
    n_draws : int
        Number of bootstrap repetitions.

    Returns
    -------
    ps_fwd : np.ndarray [n_draws]
        Per-draw p-value for fwd period.
    ps_bkw : np.ndarray [n_draws]
        Per-draw p-value for bkw period.
    """
    n_subj, n_trials_orig, n_features = slopes.shape
    seed = make_seed(n_trials, n_samples)
    rng = np.random.default_rng(seed)

    # pre-draw all indices
    subj_idx = rng.integers(0, n_subj, (n_draws, n_samples))
    trial_idx = rng.integers(0, n_trials_orig, (n_draws, n_samples, n_trials))

    fwd_window = set(range(tr_fwd[0]-1, tr_fwd[1]))
    bkw_window = set(range(tr_bkw[0]-1, tr_bkw[1]))

    ps_fwd = []
    ps_bkw = []
    for draw in range(n_draws):
        # for each sampled participant, average over sampled trials
        sampled = slopes[subj_idx[draw, :, None], trial_idx[draw]].mean(1)

        # cluster permutation test; tail=1 because bkw signs are already flipped
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='No clusters*',
                                    category=RuntimeWarning)
            t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                sampled, tail=1, n_permutations=512, seed=rng, verbose=False,
                n_jobs=1)

        # check which clusters overlap with fwd and bkw windows
        p_fwd = 1.0
        p_bkw = 1.0
        for cluster, cp in zip(clusters, cluster_p_values):
            cluster_trs = set(cluster[0])
            if fwd_window & cluster_trs:
                p_fwd = min(p_fwd, cp)
            if bkw_window & cluster_trs:
                p_bkw = min(p_bkw, cp)

        ps_fwd.append(p_fwd)
        ps_bkw.append(p_bkw)
    return np.array(ps_fwd), np.array(ps_bkw)


def run_power_contour(max_trials, n_samples, n_draws=512, overwrite=False):
    """Compute power for all n_trials in [2, max_trials] at fixed n_samples."""
    assert os.path.isfile(pkl_slopes), \
        f'File not found, did 1b script run? {pkl_slopes}'

    pkl_out = bids_base.copy().update(processing='sodacluster',
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

    df = pd.DataFrame()

    for iv in intervals:
        tr_fwd = settings.exp_tr[iv]['fwd']
        tr_bkw = settings.exp_tr[iv]['bkw']

        # extract all slopes for this interval: [n_subj, n_trials, n_trs]
        df_sel = df_slopes[df_slopes.interval == iv].sort_values(
            ['subject', 'trial', 'tr'])
        n_trs = len(df_sel.tr.unique())
        slopes = df_sel.slope.values.reshape([n_subj, -1, n_trs]).copy()

        # flip sign of bkw TRs so both periods test for positive
        slopes[:, :, tr_bkw[0]-1:tr_bkw[1]] *= -1

        results = pool(delayed(bootstrap)(slopes, nt, n_samples, tr_fwd, tr_bkw,
                                          n_draws=n_draws)
                       for nt in tqdm(trial_numbers))

        rows = []
        for n_trials, (p_fwd, p_bkw) in zip(trial_numbers, results):
            rows.append({'n_trials': n_trials, 'n_samples': n_samples,
                         'interval': iv, 'period': 'fwd',
                         'power': (p_fwd < 0.05).mean()})
            rows.append({'n_trials': n_trials, 'n_samples': n_samples,
                         'interval': iv, 'period': 'bkw',
                         'power': (p_bkw < 0.05).mean()})

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # add some metadata
    df.attrs['method'] = 'mne.stats.permutation_cluster_1samp_test'
    df.attrs['mne.version'] = str(mne.__version__)
    df.attrs['overwrite'] = overwrite
    df.attrs['n_draws'] = n_draws
    df.attrs['max_trials'] = max_trials
    df.attrs['n_samples'] = n_samples
    df.attrs['__file__'] = locals().get('__file__')

    joblib.dump(df, pkl_out.fpath)
    print(f'Saved {pkl_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SODA power analysis (cluster)')
    parser.add_argument('--max_trials', type=int, default=60,
                        help='Max number of trials to iterate over (2..max_trials)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of participants to resample (fixed per job)')
    parser.add_argument('--n_draws', type=int, default=512,
                        help='Number of bootstrap repetitions (default: 512)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing result files')
    args = parser.parse_args()

    if args.n_samples is not None:
        run_power_contour(args.max_trials, args.n_samples,
                          n_draws=args.n_draws, overwrite=args.overwrite)
    else:
        print('For plotting, run: python 3b_soda_power_analysis_plot.py')
