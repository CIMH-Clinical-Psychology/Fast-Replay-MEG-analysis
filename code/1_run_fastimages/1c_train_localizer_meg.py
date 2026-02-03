#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:38:18 2024

Train final localizer decoders on MEG data using optimal parameters.

This script loads the gridsearch results from 1a, determines the best C value
and timepoint (both on average and per subject), and trains decoders with
different parameter combinations:
    - "mean": using mean best L1 and timepoint across subjects
    - "subj": using subject-specific best L1 and timepoint

Input: Gridsearch CSVs from 1a, preprocessed MEG data
Output: Final classifier per participant (saved as pkl)

@author: simon.kern
"""

import os
import sys; sys.path.append('..')
import logging
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from meg_utils import decoding
from meg_utils.decoding import LogisticRegressionOvaNegX
from bids_utils import layout_MEG as layout
from bids_utils import load_localizer, make_bids_fname
import settings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%% Settings

tmin = -0.2
tmax = 0.8
ex_per_fold = 8

#%% Load gridsearch CSVs from 1a

deriv = layout.derivatives['derivatives']
files_acc = deriv.get(task='main',
                      acquisition='slow',
                      proc='gridsearch',
                      extension='.csv.pkl.gz',
                      suffix='accuracy',
                      invalid_filters='allow')

assert len(files_acc) == 30, f'Expected 30 files, got {len(files_acc)}'

print(f'Loading {len(files_acc)} accuracy files...')
df_acc = pd.concat([pd.read_pickle(f) for f in tqdm(files_acc, desc='loading csvs')],
                   ignore_index=True)

#%% Determine best C and timepoint per subject

# Get unique timepoints (assumes same for all subjects)
timepoints = np.sort(df_acc.timepoint.unique())

best_params = {}
for subj, df_subj in df_acc.groupby('subject'):
    # Best C: highest mean accuracy across all timepoints
    mean_acc_per_C = df_subj.groupby('C').accuracy.mean()
    best_C = mean_acc_per_C.idxmax().round()

    # Best timepoint: highest mean accuracy across all C values
    mean_acc_per_t = df_subj.groupby('timepoint').accuracy.mean()
    best_t = mean_acc_per_t.idxmax()

    # Also get the timepoint index
    best_t_idx = np.where(timepoints == best_t)[0][0]

    best_params[subj] = {
        'best_C': best_C,
        'best_t': best_t,
        'best_t_idx': best_t_idx,
    }

# Calculate mean best values across subjects
mean_best_C = np.round(2*np.mean([p['best_C'] for p in best_params.values()]))/2
mean_best_t = np.mean([p['best_t'] for p in best_params.values()])
mean_best_t_idx = int(np.round(np.mean([p['best_t_idx'] for p in best_params.values()])))

print(f'\nBest parameters:')
print(f'  Mean best C: {mean_best_C:.4f}')
print(f'  Mean best timepoint: {mean_best_t:.4f} (idx={mean_best_t_idx})')
print(f'\nPer-subject best C values:')
for subj, params in sorted(best_params.items()):
    print(f'  {subj}: C={params["best_C"]:.4f}, t={params["best_t"]:.4f}')

#%% Train decoders with different parameter combinations

# We train 4 variants per subject:
# 1. mean_C_mean_t: mean C, mean timepoint
# 2. subj_C_mean_t: subject-specific C, mean timepoint
# 3. mean_C_subj_t: mean C, subject-specific timepoint
# 4. subj_C_subj_t: subject-specific C, subject-specific timepoint

decoder_variants = ['mean', 'subj']

for subject in tqdm(layout.subjects, desc='training classifiers'):
    # Load data
    data_x, data_y, epochs = load_localizer(subject=subject, tmin=tmin, tmax=tmax, verbose=False)

    if len(set(np.bincount(data_y))) != 1:
        logging.info(f'stratifying for {subject=} as unequal classes found')
        data_x, data_y = decoding.stratify(data_x, data_y)

    subj_params = best_params[subject]

    # Define parameter combinations
    param_combos = {
        'mean': {
            'C': mean_best_C,
            't_idx': mean_best_t_idx,
            't': mean_best_t,
        },
        'subj': {
            'C': subj_params['best_C'],
            't_idx': subj_params['best_t_idx'],
            't': subj_params['best_t'],
        },
    }

    for variant_name, params in param_combos.items():
        C = params['C']
        t_idx = params['t_idx']
        t = params['t']

        # Create and train classifier
        clf_base = LogisticRegression(l1_ratio=1.0, C=C, solver='liblinear', max_iter=1000)
        clf = LogisticRegressionOvaNegX(clf_base, C=C)

        clf.fit(data_x[:, :, t_idx], data_y)

        # Save classifier
        metadata = {
            'subject': subject,
            'variant': variant_name,
            'C': C,
            't': t,
            't_idx': t_idx,
            'data_y': data_y.tolist(),
            'n_trials': len(data_y),
            'n_classes': len(np.unique(data_y)),
        }

        # Save as latest for this variant
        clf_pkl_latest = make_bids_fname('latest', modality='clf',
                                          subject=f'sub-{subject}',
                                          suffix=f'clf-{variant_name}')
        decoding.save_clf(clf, clf_pkl_latest, metadata=metadata)

        print(f'  {subject} [{variant_name}]: C={C:.4f}, t={t:.4f} -> {clf_pkl_latest}')

#%% Save summary of best parameters

df_params = pd.DataFrame([
    {'subject': subj, **params}
    for subj, params in best_params.items()
])

params_file = make_bids_fname('latest', modality='results', suffix='best_params')
df_params.to_csv(params_file + '.csv', index=False)
print(f'\nSaved best parameters to: {params_file}')

print('\nDone!')
