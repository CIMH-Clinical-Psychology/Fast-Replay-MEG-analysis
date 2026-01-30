#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:38:18 2024

SLURM-ready version to run cross-validation on the MEG localizer data

@author: simon.kern
"""

import os
import sys; sys.path.append('..')
import mne
import json
import logging
import sklearn
from tqdm import tqdm
import pandas as pd
from meg_utils import plotting, decoding
from bids import BIDSLayout
from bids_utils import layout_MEG as layout
from bids_utils import load_localizer, load_fast_images, make_bids_fname
from meg_utils.decoding import cross_validation_across_time, LogisticRegressionOvaNegX
from meg_utils import misc
import settings
from mne_bids import BIDSPath
from mne_bids import update_sidecar_json
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
import telegram_send

plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
# Ignore only the specific convergence warning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#%% Settings

# Default time windows for different conditions
TIME_WINDOWS = {
    'slow': {'tmin': -0.2, 'tmax': 0.8},  # localizer images
    'fast': {'tmin': -0.2, 'tmax': 0.5},  # sequence/fast images
}

clf = LogisticRegressionOvaNegX(l1_ratio=1, C=0, solver='liblinear', max_iter=1000)

# we have a grid of values and evaluate them. Then we choose the peak
# and run an evaluation of the values around the peak.
# repeat n times and get an optimal value, similar to iterative halving

n_iterations = 5                    # zoom in this many times
values_per_iter = 11                # evaluate 11 values per iteration
bounds_init = [0.0001, 100.0001]    # start with these bounds
ex_per_fold = 1                     # lower values = more cross-val folds
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
    acquisition = 'localizer' if condition == 'slow' else 'sequence'
    tmin = TIME_WINDOWS[condition]['tmin']
    tmax = TIME_WINDOWS[condition]['tmax']

    path_acc_pkl = BIDSPath(
        root=layout.derivatives['derivatives'].root,
        datatype='results',
        subject=subject,
        task=condition,
        acquisition=acquisition,
        processing='gridsearch',
        extension='.csv.pkl.gz',
        suffix='accuracy',
        check=False
    )
    path_proba_pkl = BIDSPath(
        root=layout.derivatives['derivatives'].root,
        datatype='results',
        subject=subject,
        task=condition,
        acquisition=acquisition,
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
        data_x, data_y, _ = load_localizer(subject=subject, tmin=tmin, tmax=tmax, verbose=False)
    else:  # condition == 'fast'
        data_x, data_y, _ = load_fast_images(subject=subject, tmin=tmin, tmax=tmax, verbose=False)

    data_x, data_y = decoding.stratify(data_x, data_y)

    # collect all results here
    df_acc = pd.DataFrame()
    df_proba = pd.DataFrame()

    bounds = bounds_init.copy()

    for iteration in tqdm(range(n_iterations), desc='iterations'):

        l1_values = np.linspace(*bounds, values_per_iter)
        res =  Parallel(n_jobs=-1)(delayed(_run_one)(subject, C, data_x, data_y, tmin, tmax) for C in l1_values)
        df_acc_iter = pd.concat([r[0] for r in res], ignore_index=True)
        df_proba_iter = pd.concat([r[1] for r in res], ignore_index=True)

        df_acc_iter['iteration'] = iteration
        df_proba_iter['iteration'] = iteration

        # calculate bounds for next iteration
        df_mean = df_acc_iter.groupby(['timepoint', 'C']).mean(True).reset_index()
        best_c = df_mean.C[df_mean.accuracy.argmax()]
        bound_min = l1_values[max(np.argmin(abs(l1_values-best_c))-1, 0)]
        bound_max = l1_values[min(np.argmin(abs(l1_values-best_c))+1, len(l1_values)-1)]
        bounds = [bound_min, bound_max]
        print(f'{subject=} {best_c=}, new {bounds=}')

        df_acc = pd.concat([df_acc, df_acc_iter], ignore_index=True)
        df_proba = pd.concat([df_proba, df_proba_iter], ignore_index=True)

    df_acc.to_pickle(path_acc_pkl)
    df_proba.to_pickle(path_proba_pkl)

    metadata_acc_file =  path_acc_pkl.copy().update(extension='.json')
    metadata_acc = {'timepoint': f'timepoint after {condition} image onset',
                    'accuracy': 'cross-validated accuracy',
                    'subject': subject,
                    'condition': condition,
                    'acquisition': acquisition,
                    'tmin': tmin,
                    'tmax': tmax,
                    'C': 'l1 regularization value of LogReg',
                    'ex_per_fold': ex_per_fold,
                    'clf': str(clf),
                    'bounds_init': bounds_init,
                    'iteration': 'in which zooming/halving iteration it was calculated',
                    }
    metadata_acc_file.fpath.write_text(json.dumps(metadata_acc, indent=4))

    metadata_proba_file =  path_proba_pkl.copy().update(extension='.json')
    metadata_proba = {'timepoint': f'timepoint after {condition} image onset',
                    'accuracy': 'cross-validated accuracy',
                    'subject': subject,
                    'condition': condition,
                    'acquisition': acquisition,
                    'tmin': tmin,
                    'tmax': tmax,
                    'C': 'l1 regularization value of LogReg',
                    'ex_per_fold': ex_per_fold,
                    'clf': str(clf),
                    'bounds_init': bounds_init,
                    'proba': 'probability estimate of classifier from fold excluding this trial',
                    'trial_idx': 'trial idx, should be but not garantueed to be continuous or in any order',
                    'class_idx': 'class that the probability estimate belongs to',
                    'trial_label': 'the true class of the trial (i.e. class_idx==trial_label is the true class)',
                    'iteration': 'in which zooming/halving iteration it was calculated',
                    }
    metadata_proba_file.fpath.write_text(json.dumps(metadata_proba, indent=4))

    best_C = df_acc.C[df_acc.accuracy.argmax()]

    # also save best l1 value easily accessible
    res_file = path_acc_pkl.copy().update(extension='txt',
                                          processing=None,
                                          task=None,
                                          suffix='best-l1',
                                          acquisition=None)
    res_file.fpath.write_text(str(best_C))


    # metadata_acc_file =  path_acc_pkl.copy().update(extension='.json')

    return




if __name__=='__main__':

    # subject specified? run only that one
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

    for condition in conditions:
        print(f'Running L1 grid search for condition: {condition}')
        print(f'Time window: {TIME_WINDOWS[condition]["tmin"]} to {TIME_WINDOWS[condition]["tmax"]}s')

        for subject in tqdm(args.subjects, f'running subjects cross-validation ({condition})'):
            run_gridsearch_l1(subject, condition=condition, overwrite=args.overwrite)
