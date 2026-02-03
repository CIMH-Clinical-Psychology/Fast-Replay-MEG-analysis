#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:38:18 2024

SLURM-ready version to run cross-validation on the MEG localizer data
using a fixed grid of L1 regularization values.

@author: simon.kern
"""

import os
import sys; sys.path.append('..')
import json
from tqdm import tqdm
import pandas as pd
from meg_utils import decoding
from bids_utils import layout_MEG as layout
from bids_utils import load_localizer, load_fast_images, make_bids_fname
from meg_utils.decoding import cross_validation_across_time, LogisticRegressionOvaNegX
from meg_utils import misc
from mne_bids import BIDSPath
import numpy as np
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%% Settings

# Default time windows for different conditions
TIME_WINDOWS = {
    'slow': {'tmin': -0.2, 'tmax': 0.8},  # localizer images
    'fast': {'tmin': -0.2, 'tmax': 0.5},  # sequence/fast images
}

# Fixed grid of L1 values
L1_VALUES = [0.1, 0.5] + list(np.arange(1, 10.5, 0.5)) + [12, 14, 16, 18, 20, 25, 30, 40, 50]

ex_per_fold = 1  # lower values = more cross-val folds

#%% Calculate the best regularization parameter


def _run_one(subject, C, data_x, data_y, tmin, tmax):

    np.random.seed(misc.make_seed(subject, C))
    clf = LogisticRegressionOvaNegX(l1_ratio=1, C=C,
                                    solver='liblinear', max_iter=1000)

    df, probas = cross_validation_across_time(
        data_x, data_y, subj=subject, n_jobs=2, tmin=tmin, tmax=tmax,
        ex_per_fold=ex_per_fold, clf=clf, verbose=False, return_probas=True
    )
    df = df.groupby('timepoint').mean(True).reset_index()
    del df['fold']

    df_proba = misc.to_long_df(probas, ['trial', 'timepoint', 'class_idx'],
                                class_idx=[0, 1, 2, 3, 4],
                                trial={'trial_idx': np.arange(len(data_y)),
                                       'trial_label': data_y},
                                timepoint=df.timepoint.sort_values(),
                                value_name='proba')
    df_proba = df_proba.sort_values(['trial_idx', 'timepoint', 'class_idx'])

    df_proba['subject'] = subject
    df_proba['C'] = np.round(C, 3)
    df['subject'] = subject
    df['C'] = np.round(C, 3)

    return df, df_proba


def run_gridsearch_l1(subject, condition='slow', overwrite=False):

    # Validate condition parameter
    if condition not in ['slow', 'fast']:
        raise ValueError(f"condition must be 'slow' or 'fast', got '{condition}'")

    subject = f'{int(subject):02d}'

    # Set acquisition name and time windows based on condition
    tmin = TIME_WINDOWS[condition]['tmin']
    tmax = TIME_WINDOWS[condition]['tmax']

    path_acc_pkl = BIDSPath(
        root=layout.derivatives['derivatives'].root,
        datatype='results',
        subject=subject,
        task='main',
        acquisition=condition,
        processing='gridsearch',
        extension='.csv.pkl.gz',
        suffix='accuracy',
        check=False
    )
    path_proba_pkl = BIDSPath(
        root=layout.derivatives['derivatives'].root,
        datatype='results',
        subject=subject,
        task='main',
        acquisition=condition,
        processing='gridsearch',
        extension='.csv.pkl.gz',
        suffix='probas',
        check=False
    )

    if os.path.isfile(path_acc_pkl.fpath) and os.path.isfile(path_proba_pkl.fpath):
        print(f'{subject=} {condition=} already processed: {path_proba_pkl}')
        if overwrite:
            print('overwriting...')
        else:
            return
    else:
        path_proba_pkl.mkdir(True)

    # Load data based on condition
    if condition == 'slow':
        data_x, data_y, df_beh = load_localizer(subject=subject, tmin=tmin, tmax=tmax, verbose=False)
        interval_times = (df_beh.interval_time*1000).values.astype(int)
    else:  # condition == 'fast'
        data_x, data_y, df_beh = load_fast_images(subject=subject, tmin=tmin, tmax=tmax, verbose=False)
        interval_times = df_beh.interval_time.values.astype(int)

    data_x, data_y = decoding.stratify(data_x, data_y)

    # Run grid search over fixed L1 values
    print(f'{subject=} {condition=}: running {len(L1_VALUES)} L1 values')
    res = Parallel(n_jobs=-1)(
        delayed(_run_one)(subject, C, data_x, data_y, tmin, tmax)
        for C in tqdm(L1_VALUES, desc=f'{subject} L1 values')
    )

    df_acc = pd.concat([r[0] for r in res], ignore_index=True)
    df_proba = pd.concat([r[1] for r in res], ignore_index=True)

    df_acc['interval_time'] = interval_times
    df_proba['interval_time'] = interval_times

    df_acc.to_pickle(path_acc_pkl)
    df_proba.to_pickle(path_proba_pkl)

    # Save metadata
    clf = LogisticRegressionOvaNegX(l1_ratio=1, C=0, solver='liblinear', max_iter=1000)

    metadata_acc_file = path_acc_pkl.copy().update(extension='.json')
    metadata_acc = {
        'timepoint': f'timepoint after {condition} image onset',
        'accuracy': 'cross-validated accuracy',
        'subject': subject,
        'condition': condition,
        'tmin': tmin,
        'tmax': tmax,
        'C': 'l1 regularization value of LogReg',
        'ex_per_fold': ex_per_fold,
        'clf': str(clf),
        'l1_values': L1_VALUES,
    }
    metadata_acc_file.fpath.write_text(json.dumps(metadata_acc, indent=4))

    metadata_proba_file = path_proba_pkl.copy().update(extension='.json')
    metadata_proba = {
        'timepoint': f'timepoint after {condition} image onset',
        'accuracy': 'cross-validated accuracy',
        'subject': subject,
        'condition': condition,
        'tmin': tmin,
        'tmax': tmax,
        'C': 'l1 regularization value of LogReg',
        'ex_per_fold': ex_per_fold,
        'clf': str(clf),
        'l1_values': L1_VALUES,
        'proba': 'probability estimate of classifier from fold excluding this trial',
        'trial_idx': 'trial idx, should be but not guaranteed to be continuous or in any order',
        'class_idx': 'class that the probability estimate belongs to',
        'trial_label': 'the true class of the trial (i.e. class_idx==trial_label is the true class)',
    }
    metadata_proba_file.fpath.write_text(json.dumps(metadata_proba, indent=4))

    best_C = df_acc.C[df_acc.accuracy.argmax()]

    # Also save best l1 value easily accessible
    res_file = path_acc_pkl.copy().update(
        extension='txt',
        processing=None,
        acquisition=condition,
        task=None,
        suffix='best-l1',
    )
    res_file.fpath.write_text(str(best_C))

    print(f'{subject=} {condition=}: best C = {best_C}')

    return


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='L1 regularization grid search for MEG decoding'
    )
    parser.add_argument("--subjects",
                        type=lambda s: [f'{int(x):02d}' for x in s.split(',')],
                        help='comma-separated list of subjects',
                        default=list(range(1, 31))
                        )
    parser.add_argument('--condition',
                        type=str,
                        choices=['slow', 'fast'],
                        default=None,
                        help="condition: 'slow' for localizer images, 'fast' for sequence images (default: both)")
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite results if exist')
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else ['slow', 'fast']

    print(f'L1 values to evaluate: {L1_VALUES}')
    print(f'Total: {len(L1_VALUES)} values')

    for condition in conditions:
        print(f'\nRunning L1 grid search for condition: {condition}')
        print(f'Time window: {TIME_WINDOWS[condition]["tmin"]} to {TIME_WINDOWS[condition]["tmax"]}s')

        for subject in tqdm(args.subjects, f'running subjects cross-validation ({condition})'):
            run_gridsearch_l1(subject, condition=condition, overwrite=args.overwrite)
