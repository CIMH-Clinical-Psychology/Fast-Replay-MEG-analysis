#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal example: run TDLM on MEG and SODA on fMRI using the per-subject HDF5
files in ../sequence_predictions/MEG/ and ../sequence_predictions/fMRI/.

Both methods take a *per-trial* probability matrix (time × classes) and the
*sequence* of stimuli that were presented in that trial. The h5 files already
contain everything we need — no project-specific settings or BIDS access.

Output:
    A 2-panel figure (MEG sequenceness curves, fMRI slope curves) summarising
    the group-level results per ISI, plus a brief printed summary.

Dependencies: numpy, h5py, matplotlib, seaborn, tdlm, soda.

@author: Simon.Kern
"""
import os
import glob
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tdlm
from tdlm.utils import seq2tf, num2char
import soda


#%% config

HERE = os.path.dirname(os.path.abspath(__file__))
MEG_DIR = os.path.normpath(os.path.join(HERE, '..', 'sequence_predictions', 'MEG'))
FMRI_DIR = os.path.normpath(os.path.join(HERE, '..', 'sequence_predictions', 'fMRI'))


#%% MEG — TDLM (Temporally Delayed Linear Modelling)
# TDLM looks for sequential structure by regressing the probability of state X
# at time t against the probability of state Y at time t+lag, for each pair in
# the expected sequence (forward) and the reversed sequence (backward).
# tdlm.compute_1step returns sequenceness curves over a range of lags.

# Container keyed by ISI → list of (n_lags+1,) arrays, one per trial × subject
sf_by_iv = {}
sb_by_iv = {}

for path in sorted(glob.glob(os.path.join(MEG_DIR, 'sub-*.h5'))):
    with h5py.File(path, 'r') as f:
        probas = f['probas'][:]            # (n_trials, n_t, 5)
        sequences = f['sequences'][:]      # (n_trials, 5) class indices
        intervals = f['intervals'][:]      # (n_trials,) ISI in ms
    subject_id = int(os.path.basename(path)[4:6])  # 'sub-07.h5' → 7

    for k in range(probas.shape[0]):
        # divide each class probability by its mean across time (remove bias)
        proba = probas[k] / probas[k].mean(axis=0, keepdims=True)
        iv = int(intervals[k])

        # Build a 5×5 transition-frequency matrix from the trial's sequence.
        # tdlm wants a character string ('ABCDE' = sequence of states 0..4).
        tf = seq2tf(''.join(num2char(sequences[k])))

        # Maximum lag of interest: up to (ISI + 100 ms) × 1.5 in 10-ms samples.
        # Prevents picking up the next-but-one image as a sequence lag.
        max_lag = int(((iv / 10) + 10) * 1.5)
        # Restrict to the 5 displayed images + max_lag — don't analyse buffer.
        length = int((iv + 100) * 5) // 10 + max_lag

        sf, sb = tdlm.compute_1step(proba[:length, :], tf,
                                    n_shuf=100, max_lag=max_lag,
                                    rng=subject_id)
        sf_by_iv.setdefault(iv, []).append(sf[0])  # sf[0] = unshuffled
        sb_by_iv.setdefault(iv, []).append(sb[0])

# stack to arrays of shape (n_trials_total, n_lags+1)
sf_by_iv = {iv: np.stack(v) for iv, v in sf_by_iv.items()}
sb_by_iv = {iv: np.stack(v) for iv, v in sb_by_iv.items()}


#%% fMRI — SODA (Slope Order Dynamic Analysis)
# SODA fits a linear slope to the probabilities of the sequence stimuli (in
# their presented order) at each TR. A positive slope at early TRs ("onset")
# means forward order; a negative slope at later TRs ("offset") means the
# reverse pattern.

slopes_by_iv = {}

for path in sorted(glob.glob(os.path.join(FMRI_DIR, 'sub-*.h5'))):
    with h5py.File(path, 'r') as f:
        probas = f['probas'][:]            # (n_trials, 13, 5)
        sequences = f['sequences'][:]      # (n_trials, 5) class indices
        intervals = f['intervals'][:]      # (n_trials,) ISI in ms

    for k in range(probas.shape[0]):
        proba = probas[k] / probas[k].mean(axis=0, keepdims=True)
        iv = int(intervals[k])
        # soda.compute_slopes returns one slope per TR (1..13).
        slopes = soda.compute_slopes(proba, order=list(sequences[k]))
        slopes_by_iv.setdefault(iv, []).append(slopes)

slopes_by_iv = {iv: np.stack(v) for iv, v in slopes_by_iv.items()}


#%% plot — MEG sequenceness curves and fMRI slope curves

fig, axs = plt.subplots(1, 2, figsize=[12, 4.5])

# MEG forward sequenceness, averaged across trials, per ISI
ax = axs[0]
for iv in sorted(sf_by_iv):
    lags_ms = np.arange(sf_by_iv[iv].shape[1]) * 10  # 10 ms per sample
    ax.plot(lags_ms, sf_by_iv[iv].mean(0), label=f'{iv + 100} ms')
ax.axhline(0, color='black', alpha=0.3, linestyle='--')
ax.set(title='MEG — TDLM forward sequenceness',
       xlabel='time lag (ms)', ylabel='sequenceness')
ax.legend(title='speed', fontsize=9)

# fMRI slopes per TR, averaged across trials, per ISI (skip 2048 ms control)
ax = axs[1]
for iv in sorted(slopes_by_iv):
    if iv == 2048:
        continue
    trs = np.arange(1, slopes_by_iv[iv].shape[1] + 1)
    ax.plot(trs, slopes_by_iv[iv].mean(0), label=f'{iv + 100} ms')
ax.axhline(0, color='black', alpha=0.3, linestyle='--')
ax.set(title='fMRI — SODA slopes', xlabel='TR', ylabel='slope')
ax.legend(title='speed', fontsize=9)

sns.despine(fig)
fig.tight_layout()
