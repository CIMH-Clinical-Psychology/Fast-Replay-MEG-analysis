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
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import settings
from meg_utils.misc import make_seed, to_long_df
from mne_bids import BIDSPath
from mne.stats import permutation_t_test
from joblib import Parallel, delayed


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


def bootstrap(sf, n_trials, n_samples, n_draws=1000):
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
        sf_sampled = sf[subj_idx[draw, :, None], trial_idx[draw]].mean(1)
        # group-level permutation t-test on the mean sequenceness
        _, p, _ = permutation_t_test(sf_sampled, n_permutations=1024,
                                     seed=rng, verbose=False, n_jobs=1)
        ps.append(p)
    return np.array(ps)


def run_power_contour(max_trials, n_samples, n_draws=1000, overwrite=False):
    """Compute power for all n_trials in [2, max_trials] at fixed n_samples."""
    assert pkl_seq.fpath.is_file(), f'File not found, did 1a script run? {pkl_seq.fpath}'

    pkl_out = bids_base.copy().update(processing='signflip',
                                      suffix='powercontour',
                                      task=f'{n_samples}',
                                      extension='.pkl.gz')

    if pkl_out.fpath.is_file() and not overwrite:
        print(f'{n_samples=} exists already, skipping')
        return

    res = joblib.load(pkl_seq)
    sf_trials = res['sf_trials']

    pool = Parallel(n_jobs=-1)
    n_trials = range(2, max_trials + 1)

    df = pd.DataFrame()

    for iv in settings.intervals_MEG:
        print(f'starting with {iv=} for {n_samples=} ')

        sf = sf_trials[iv][:, :, 0, 1:]  # [n_subj, n_trials, n_lags], drop lag=0

        ps = pool(delayed(bootstrap)(sf, nt, n_samples) for nt in tqdm(n_trials))
        ps = np.array(ps)

        # take p values at the expected time lag+-10ms
        lag_exp = settings.exp_lag[iv]

        # p at expected lag,
        # we removed lag 0, so idx= -1, then +- 1
        # then take minimum p value in that area
        p_exp = (ps[:, :, lag_exp-2:lag_exp+1]).min(-1)

        # now look how many bootstrap samples were significant
        # across trial sizes
        power = (np.array(p_exp) < 0.05).mean(-1)

        df_iv = pd.DataFrame({'n_trials': n_trials, 'n_samples': n_samples,
                              'interval': iv, 'power': power})
        df = pd.concat([df, df_iv], ignore_index=True)

    # add some metadata
    df.attrs['method'] = 'mne.stats.permutation_t_test'
    df.attrs['mne.version'] = str(mne.__version__)
    df.attrs['overwrite'] = overwrite
    df.attrs['n_draws'] = n_draws
    df.attrs['max_trials'] = max_trials
    df.attrs['n_samples'] = n_samples
    df.attrs['__file__'] = locals().get('__file__')

    joblib.dump(df, pkl_out.fpath)
    print(f'Saved {pkl_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDLM power analysis (signflip)')
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
        print('For plotting, run: python 3a_tdlm_power_analysis_plot.py')
