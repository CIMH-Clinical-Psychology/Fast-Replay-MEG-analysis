# -*- coding: utf-8 -*-
"""
Extract trial-level classifier probability matrices for MEG and fMRI fast
sequence trials, together with the stimulus order (sequence) of each trial.

Outputs one HDF5 file per subject under
    <repo>/sequence_predictions/MEG/sub-XX.h5
    <repo>/sequence_predictions/fMRI/sub-XX.h5

Each subject file contains the following datasets (all gzip compressed):

    probas           (n_trials, n_timepoints, 5)  float32  class probabilities
    sequences        (n_trials, 5)                int      class indices, 0..4
    sequence_labels  (n_trials, 5)                str      'face','house',...
    intervals        (n_trials,)                  int      ISI in ms
    trial_ids        (n_trials,)                  int      original trial index

…and these file-level attributes:

    modality      : 'MEG' | 'fMRI'
    subject       : 'sub-XX'
    categories    : ['face','house','cat','shoe','chair']  (column order of probas)
    sfreq         : 100              (MEG only, Hz)
    tmin, tmax    : seconds          (MEG only, relative to text cue)
    tr_duration   : 1.25             (fMRI only, seconds)
    n_trs         : 13               (fMRI only)
    description   : free-text description of the time axis

Load with (Python):
    import h5py
    with h5py.File('sequence_predictions/MEG/sub-01.h5', 'r') as f:
        probas = f['probas'][:]                # (n_trials, T, 5)
        seq    = f['sequences'][:]             # (n_trials, 5)
        labels = f['sequence_labels'][:].astype(str)
        cats   = list(f.attrs['categories'])

@author: Simon.Kern
"""
import sys; sys.path.append('..')

import os
import h5py
import numpy as np
from tqdm import tqdm

import bids_utils
import settings
from settings import subjects_MEG, subjects_3T
from settings import intervals_MEG
from meg_utils.decoding import predict_proba_along


#%% output paths

REPO_ROOT = os.path.abspath(os.path.join(settings.script_dir, '..'))
OUT_DIR_MEG = os.path.join(REPO_ROOT, 'sequence_predictions', 'MEG')
OUT_DIR_FMRI = os.path.join(REPO_ROOT, 'sequence_predictions', 'fMRI')
os.makedirs(OUT_DIR_MEG, exist_ok=True)
os.makedirs(OUT_DIR_FMRI, exist_ok=True)

# gzip compression options — universal across HDF5 readers
H5_KW = dict(compression='gzip', compression_opts=4)
# h5py needs explicit variable-length-string dtype for fixed string arrays
STR_DTYPE = h5py.string_dtype(encoding='utf-8')


def _write_subject(path, probas, sequences, sequence_labels, intervals,
                   trial_ids, attrs):
    with h5py.File(path, 'w') as f:
        f.create_dataset('probas', data=probas.astype(np.float32), **H5_KW)
        f.create_dataset('sequences', data=sequences.astype(np.int16), **H5_KW)
        f.create_dataset('sequence_labels',
                         data=np.asarray(sequence_labels, dtype=object),
                         dtype=STR_DTYPE, **H5_KW)
        f.create_dataset('intervals', data=intervals.astype(np.int32), **H5_KW)
        f.create_dataset('trial_ids', data=trial_ids.astype(np.int32), **H5_KW)
        for k, v in attrs.items():
            f.attrs[k] = v

asdf
#%% MEG: extract per-trial probability matrices

MEG_SFREQ = 100  # Hz, 1 timepoint = 10 ms
MEG_TMIN = 3.1
MEG_TMAX = 3 + (0.512 + 0.1) * 5

for subject in tqdm(subjects_MEG, desc='MEG: extracting probas'):

    # mean classifier (matches 1a_run_tdlm_meg.py)
    clf = bids_utils.load_latest_classifier(subject, which='mean')

    # data_x: (n_trials, n_channels, n_timepoints); beh: list of per-trial DataFrames
    data_x, _, beh = bids_utils.load_fast_sequences(subject, intervals=intervals_MEG)

    # probas: (n_trials, n_timepoints, n_classes)
    probas = predict_proba_along(clf, data_x, -1)

    sequences = np.array([df.trigger.values.astype(int) for df in beh])
    sequence_labels = np.array(
        [[settings.categories[int(t)] for t in df.trigger.values] for df in beh]
    )
    intervals_arr = np.array([int(df.interval_time.values[0]) for df in beh])
    trial_ids = np.array([int(df.idx.values[0]) for df in beh])

    out_path = os.path.join(OUT_DIR_MEG, f'sub-{subject}.h5')
    _write_subject(
        out_path, probas, sequences, sequence_labels, intervals_arr, trial_ids,
        attrs={
            'modality': 'MEG',
            'subject': f'sub-{subject}',
            'categories': list(settings.categories),
            'sfreq': MEG_SFREQ,
            'tmin': MEG_TMIN,
            'tmax': MEG_TMAX,
            'description': ('Per-trial classifier probabilities from MEG '
                            'fast-sequence trials. Probability columns follow '
                            'the `categories` attribute. Time axis: tmin..tmax '
                            'at sfreq Hz, aligned to the text cue (first '
                            'sequence image starts at 3.3 s).'),
        })


#%% fMRI: extract per-trial probability matrices

FMRI_TR = settings.tr_duration  # 1.25 s
FMRI_N_TRS = 13

for subject in tqdm(subjects_3T, desc='fMRI: extracting probas'):

    try:
        df_seq = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long')
        df_beh = bids_utils.load_trial_data_3T(subject, condition='sequence')
    except (FileNotFoundError, AssertionError) as e:
        print(f'  skipping sub-{subject}: {e}')
        continue

    df_seq = df_seq[df_seq['class'].isin(settings.categories)]

    probas_list, seqs_int, seqs_lab, intervals_list, trial_ids = [], [], [], [], []

    for (i, df_trial), (j, beh) in zip(df_seq.groupby('trial'),
                                       df_beh.groupby('trial'), strict=True):
        assert i == j

        seq_labels = list(beh.stim_label.values[0])           # ['face','house',...]
        seq_idx = [settings.categories.index(x) for x in seq_labels]

        # sort by class (in settings.categories order), then by tr_onset
        df_trial = df_trial.sort_values(
            ['class', 'tr_onset'],
            key=lambda x: ([settings.categories.index(y) for y in x]
                           if 'str' in str(x.dtype) else x))
        probas = np.array(df_trial.probability.values.reshape(
            [FMRI_N_TRS, len(settings.categories)], order='F'))

        probas_list.append(probas)
        seqs_int.append(seq_idx)
        seqs_lab.append(seq_labels)
        intervals_list.append(int(round(df_trial.tITI.unique()[0] * 1000)))
        trial_ids.append(int(i))

    if not probas_list:
        print(f'  no trials for sub-{subject}, skipping')
        continue

    out_path = os.path.join(OUT_DIR_FMRI, f'sub-{subject}.h5')
    _write_subject(
        out_path,
        np.stack(probas_list, axis=0),
        np.array(seqs_int, dtype=int),
        np.array(seqs_lab),
        np.array(intervals_list, dtype=int),
        np.array(trial_ids, dtype=int),
        attrs={
            'modality': 'fMRI',
            'subject': f'sub-{subject}',
            'categories': list(settings.categories),
            'tr_duration': FMRI_TR,
            'n_trs': FMRI_N_TRS,
            'description': ('Per-trial classifier probabilities from 3T fMRI '
                            'fast-sequence trials. Probability columns follow '
                            'the `categories` attribute. Time axis: 13 TRs '
                            '(1.25 s each) aligned to sequence onset.'),
        })


#%% sanity check

if __name__ == '__main__':
    for out_dir in [OUT_DIR_MEG, OUT_DIR_FMRI]:
        files = sorted(f for f in os.listdir(out_dir) if f.endswith('.h5'))
        if not files:
            continue
        with h5py.File(os.path.join(out_dir, files[0]), 'r') as f:
            print(f"\n{f.attrs['modality']}: {len(files)} files | first={files[0]}")
            print(f"  probas {f['probas'].shape} {f['probas'].dtype}, "
                  f"intervals {sorted(set(f['intervals'][:].tolist()))}")
