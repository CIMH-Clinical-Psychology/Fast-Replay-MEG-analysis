#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:30:00 2026

Temporal generalization and transfer analysis for MEG data.

This script creates cross-validated temporal generalization matrices (TGMs)
that show how decoders trained at one time point generalize to other time points.
Supports both within-condition generalization and cross-condition transfer.

Input: Preprocessed MEG localizer and/or fast images data
Output: Accuracy heatmaps as PKL files per subject and L1 parameter

@author: simon.kern
"""

import os
import sys; sys.path.append('..')
import json
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from sklearn.exceptions import ConvergenceWarning

from meg_utils import decoding
from meg_utils.decoding import LogisticRegressionOvaNegX
from bids_utils import layout_MEG as layout
from bids_utils import load_localizer, load_fast_images
import settings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%% Settings

TIME_WINDOWS = {
    'slow': {'tmin': -0.2, 'tmax': 0.8},
    'fast': {'tmin': -0.2, 'tmax': 0.5},
}

ex_per_fold = 8


def run_generalization(subject, l1_value, source='slow', target='fast', overwrite=False):
    """
    Run temporal generalization or transfer analysis for a given subject.

    Parameters
    ----------
    subject : str or int
        Subject identifier
    l1_value : float
        L1 regularization parameter (C value for LogisticRegression)
    source : str, default='slow'
        Source condition: 'slow' for localizer, 'fast' for sequence images
    target : str, default='fast'
        Target condition: 'slow' for localizer, 'fast' for sequence images
    overwrite : bool, default=False
        Whether to overwrite existing results

    Returns
    -------
    heatmap : ndarray
        Accuracy heatmap (n_source_times x n_target_times)
    """

    if source not in ['slow', 'fast']:
        raise ValueError(f"source must be 'slow' or 'fast', got '{source}'")
    if target not in ['slow', 'fast']:
        raise ValueError(f"target must be 'slow' or 'fast', got '{target}'")

    subject = f'{int(subject):02d}'

    source_acq = 'localizer' if source == 'slow' else 'sequence'
    target_acq = 'localizer' if target == 'slow' else 'sequence'

    tmin_source = TIME_WINDOWS[source]['tmin']
    tmax_source = TIME_WINDOWS[source]['tmax']
    tmin_target = TIME_WINDOWS[target]['tmin']
    tmax_target = TIME_WINDOWS[target]['tmax']


    path_csv = BIDSPath(
        root=layout.derivatives['derivatives'].root,
        datatype='results',
        subject=subject,
        task='main',
        processing=f'{source}2{target}',
        description=f'C{l1_value:.3f}',
        suffix='heatmap',
        extension='.csv',
        check=False
    )

    if os.path.isfile(path_csv.fpath):
        print(f'{subject=} {source=}->{target=} C={l1_value:.6f} already processed: {path_csv.basename}')
        if not overwrite:
            return
        else:
            print('overwriting...')
    else:
        path_csv.mkdir(True)

    # Load source data
    print(f'Loading source data ({source}) for {subject=}...')
    if source == 'slow':
        source_x, source_y, _ = load_localizer(subject=subject, tmin=tmin_source, tmax=tmax_source, verbose=False)
    elif source == 'fast':
        source_x, source_y, _ = load_fast_images(subject=subject, tmin=tmin_source, tmax=tmax_source, verbose=False)

    source_x, source_y = decoding.stratify(source_x, source_y)

    # Create classifier
    clf = LogisticRegressionOvaNegX(
        l1_ratio=1,
        C=l1_value,
        solver='liblinear',
        max_iter=1000
    )

    if source == target:
        # Within-condition temporal generalization
        print(f'Computing generalization for {subject=}, {source=}, C={l1_value:.6f}...')
        heatmap = decoding.decoding_heatmap_generalization(
            clf=clf,
            data_x=source_x,
            data_y=source_y,
            ex_per_fold=ex_per_fold,
            n_jobs=-1,
        )
        timepoints_train = np.linspace(tmin_source, tmax_source, source_x.shape[-1])
        timepoints_test = timepoints_train
    else:
        # Cross-condition transfer
        print(f'Loading target data ({target}) for {subject=}...')
        if target == 'slow':
            target_x, target_y, _ = load_localizer(subject=subject, tmin=tmin_target, tmax=tmax_target, verbose=False)
        else:
            target_x, target_y, _ = load_fast_images(subject=subject, tmin=tmin_target, tmax=tmax_target, verbose=False)

        target_x, target_y = decoding.stratify(target_x, target_y)

        print(f'Computing transfer {source}->{target} for {subject=}, C={l1_value:.6f}...')
        heatmap = decoding.decoding_heatmap_transfer(
            clf=clf,
            data_x=source_x,
            data_y=source_y,
            test_x=target_x,
            test_y=target_y,
        )
        timepoints_train = np.linspace(tmin_source, tmax_source, source_x.shape[-1])
        timepoints_test = np.linspace(tmin_target, tmax_target, target_x.shape[-1])

    # Save heatmap as CSV with timepoints as index and columns
    df_heatmap = pd.DataFrame(
        heatmap,
        index=timepoints_train,
        columns=timepoints_test
    )
    df_heatmap.index.name = f'train_time ({source})'
    df_heatmap.columns.name = f'test_time ({target})'
    df_heatmap.to_csv(path_csv.fpath)
    print(f'Saved heatmap to: {path_csv.fpath}')

    # Save metadata JSON
    metadata_file = path_csv.copy().update(extension='.json')
    metadata = {
        'description': f'Temporal generalization/transfer heatmap ({source}->{target})',
        'subject': subject,
        'source': source,
        'target': target,
        'source_acquisition': source_acq,
        'target_acquisition': target_acq,
        'C': l1_value,
        'l1_ratio': 1.0,
        'ex_per_fold': ex_per_fold,
        'clf': str(clf),
        'tmin_source': tmin_source,
        'tmax_source': tmax_source,
        'tmin_target': tmin_target,
        'tmax_target': tmax_target,
        'timepoints_train': timepoints_train.tolist(),
        'timepoints_test': timepoints_test.tolist(),
        'n_source_trials': int(len(source_y)),
        'n_classes': int(len(np.unique(source_y))),
        'heatmap_shape': list(heatmap.shape),
        'mean_accuracy': float(heatmap.mean()),
        'max_accuracy': float(heatmap.max()),
    }

    if source == target:
        metadata['diagonal_accuracy'] = float(np.diag(heatmap).mean())

    metadata_file.fpath.write_text(json.dumps(metadata, indent=4))

    return heatmap


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Temporal generalization and transfer analysis for MEG data'
    )
    parser.add_argument('--subjects',
                        type=lambda s: [f'{int(x):02d}' for x in s.split(',')],
                        help='Comma-separated list of subjects (e.g., "1,2,3")',
                        default= [f'{i:02d}' for i in range(1, len(layout.subjects) + 1)])
    parser.add_argument(
        '--l1',
        type=float,
        help='L1 regularization parameter (C value)',
        default=4.64
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['slow', 'fast'],
        default=None,
        help="Source condition: 'slow' or 'fast' (default: all combinations)"
    )
    parser.add_argument(
        '--target',
        type=str,
        choices=['slow', 'fast'],
        default=None,
        help="Target condition: 'slow' or 'fast' (default: all combinations)"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing results'
    )

    args = parser.parse_args()

    # Determine combinations to run
    if args.source is None and args.target is None:
        combinations = [
            ('slow', 'slow'),
            ('fast', 'fast'),
            ('slow', 'fast'),
            ('fast', 'slow'),
        ]
    elif args.source is None or args.target is None:
        parser.error('Both --source and --target must be specified together, or neither')
    else:
        combinations = [(args.source, args.target)]

    print(f'Running generalization/transfer analysis:')
    print(f'  Subjects: {args.subjects}')
    print(f'  L1 parameter: {args.l1}')
    print(f'  Combinations: {combinations}')
    print(f'  Overwrite: {args.overwrite}')

    for source, target in combinations:
        print(f'\n=== Processing {source} -> {target} ===')
        for subject in tqdm(args.subjects, desc=f'{source}->{target}'):
            run_generalization(
                    subject=subject,
                    l1_value=args.l1,
                    source=source,
                    target=target,
                    overwrite=args.overwrite
                    )


    print('\nDone!')
