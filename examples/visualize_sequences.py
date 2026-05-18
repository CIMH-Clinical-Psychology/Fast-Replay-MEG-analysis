#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize per-position class probability time courses for MEG and fMRI fast
sequence trials, using only the HDF5 files in ../sequence_predictions/MEG/
and ../sequence_predictions/fMRI/.

Reproduces the per-position trace plot from
    code/1_run_fastimages/3_run_visualize_sequences.py
without any project-specific imports — just h5py, numpy, pandas, matplotlib,
seaborn.

For each ISI, probabilities are normalised within trial (proba / proba.mean(0))
and then reordered so that line k = "probability of whatever stimulus was
shown at serial position k".

@author: Simon.Kern
"""
import os
import glob
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#%% config

HERE = os.path.dirname(os.path.abspath(__file__))
MEG_DIR = os.path.normpath(os.path.join(HERE, '..', 'sequence_predictions', 'MEG'))
FMRI_DIR = os.path.normpath(os.path.join(HERE, '..', 'sequence_predictions', 'fMRI'))

# Wittkuhn et al. 2021 palette (one colour per serial position)
PALETTE = ["#f5191c", "#e78f0a", "#eacb2b", "#7cba96", "#3b99b1"]


#%% load MEG h5 files → long-format DataFrame
# Each row holds the normalised probability of the *stimulus shown at a given
# serial position* at one timepoint of one trial of one subject.

rows = []
for path in sorted(glob.glob(os.path.join(MEG_DIR, 'sub-*.h5'))):
    with h5py.File(path, 'r') as f:
        probas = f['probas'][:]            # (n_trials, n_t, 5)
        sequences = f['sequences'][:]      # (n_trials, 5) class indices
        intervals = f['intervals'][:]      # (n_trials,) ISI in ms
        sfreq = float(f.attrs['sfreq'])    # 100 Hz
    # epoch starts 200 ms before first image → t=0 at first image onset
    time = np.arange(probas.shape[1]) * (1000 / sfreq) - 200
    subject = os.path.basename(path)[:-3]

    for k in range(probas.shape[0]):
        p_norm = probas[k] / probas[k].mean(axis=0, keepdims=True)
        # column j → probability of the stimulus shown at serial position j
        p_pos = p_norm[:, sequences[k]]
        rows.append(pd.DataFrame({
            'time': np.tile(time, 5),
            'proba': p_pos.ravel(order='F'),
            'position': np.repeat(np.arange(1, 6), len(time)),
            'interval': int(intervals[k]),
            'subject': subject,
        }))
df_meg = pd.concat(rows, ignore_index=True)


#%% load fMRI h5 files → long-format DataFrame

rows = []
for path in sorted(glob.glob(os.path.join(FMRI_DIR, 'sub-*.h5'))):
    with h5py.File(path, 'r') as f:
        probas = f['probas'][:]            # (n_trials, 13, 5)
        sequences = f['sequences'][:]
        intervals = f['intervals'][:]
    time = np.arange(1, probas.shape[1] + 1)   # TR index, 1..13
    subject = os.path.basename(path)[:-3]

    for k in range(probas.shape[0]):
        p_norm = probas[k] / probas[k].mean(axis=0, keepdims=True)
        p_pos = p_norm[:, sequences[k]]
        rows.append(pd.DataFrame({
            'time': np.tile(time, 5),
            'proba': p_pos.ravel(order='F'),
            'position': np.repeat(np.arange(1, 6), len(time)),
            'interval': int(intervals[k]),
            'subject': subject,
        }))
df_fmri = pd.concat(rows, ignore_index=True)


#%% aggregate per (interval, position, time, subject) for group-level lines

df_meg_mean = (df_meg.groupby(['time', 'interval', 'position', 'subject'])
                     .mean(numeric_only=True).reset_index())
df_fmri_mean = (df_fmri.groupby(['time', 'interval', 'position', 'subject'])
                       .mean(numeric_only=True).reset_index())


#%% plot — 2 rows (MEG / fMRI) × N ISI columns

intervals_meg = sorted(df_meg.interval.unique())
# skip 2048 ms for fMRI (slow control condition, hard to fit on same x-axis)
intervals_fmri = [iv for iv in sorted(df_fmri.interval.unique()) if iv != 2048]
n_cols = max(len(intervals_meg), len(intervals_fmri))

fig, axs = plt.subplots(2, n_cols, figsize=[4 * n_cols, 6])

# --- MEG row ---
for i, iv in enumerate(intervals_meg):
    ax = axs[0, i]
    sns.lineplot(df_meg_mean[df_meg_mean.interval == iv],
                 x='time', y='proba', hue='position',
                 palette=PALETTE, ax=ax, legend=False)
    # image-onset markers: image p at p * SOA = p * (ISI + 100 ms)
    for pos in range(5):
        ax.axvline(pos * (iv + 100), linewidth=0.5, color='black', alpha=0.3)
    ax.set(xlabel='ms after first image onset',
           ylabel='MEG\nnormed probability' if i == 0 else '',
           title=f'{iv + 100} ms')

# --- fMRI row ---
for i, iv in enumerate(intervals_fmri):
    ax = axs[1, i]
    sns.lineplot(df_fmri_mean[df_fmri_mean.interval == iv],
                 x='time', y='proba', hue='position',
                 palette=PALETTE, ax=ax, legend=False)
    # image-onset markers, expressed in TR units (TR_dur = 1.25 s)
    for pos in range(5):
        t_onset = pos * (iv / 1000 + 0.1) / 1.25 + 1.25 / 2
        ax.axvline(t_onset, linewidth=0.5, color='black', alpha=0.3)
    ax.set(xlabel='TR after sequence onset',
           ylabel='fMRI\nnormed probability' if i == 0 else '',
           title=f'{iv + 100} ms')

# blank unused fMRI columns (fMRI may have fewer ISIs than MEG)
for j in range(len(intervals_fmri), n_cols):
    axs[1, j].axis('off')

# shared legend on the right
handles = [Line2D([0], [0], color=c, lw=2, label=f'position {k+1}')
           for k, c in enumerate(PALETTE)]
handles.append(Line2D([0], [0], color='black', lw=0.8, alpha=0.5,
                      label='image onset'))
fig.legend(handles=handles, title='serial position',
           loc='center right', bbox_to_anchor=(0.99, 0.5))

sns.despine(fig)
fig.tight_layout(rect=[0, 0, 0.88, 1])
