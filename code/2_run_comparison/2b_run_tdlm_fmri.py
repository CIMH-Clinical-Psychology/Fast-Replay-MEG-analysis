# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run TDLM (Temporally Delayed Linear Modelling) on fMRI data

@author: Simon.Kern
"""
import os
import settings
import bids_utils
from tqdm import tqdm
from bids import BIDSLayout
import numpy as np
import pandas as pd
import mne
from meg_utils import decoding, plotting, sigproc
import tdlm

#%% settings
subjects = [f'{i:02d}' for i in range(1, 41)]
normalization = 'lambda x: x/x.mean(0)'

layout = BIDSLayout(settings.bids_dir_3T)

#%% TR: TDLM on super trials
intervals = ['0.032', '0.128', '0.064', '0.512', '2.048']
categories = ['cat', 'chair', 'face', 'house', 'shoe']

sf        = {interval: [] for interval in intervals}
sb        = {interval: [] for interval in intervals}
sf_single = {interval: [] for interval in intervals}
sb_single = {interval: [] for interval in intervals}

for subject in tqdm(subjects):
    df_probas = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long',
                                                classifier=categories)
    df_seq = bids_utils.load_trial_data_3T(subject)

    intervals = df_probas.tITI.unique()
    tf = tdlm.seq2tf('ABCDE')

    probas = {interval: [] for interval in intervals}
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial')):
        assert t1==t2
        interval = df_s.tITI.values[0]
        seq_labels = list(df_s.stim_label.values[0])
        proba = [df_p[df_p.classifier==label].probability.values for label in seq_labels]
        probas[interval] += [proba]

    for interval in intervals:
        # supertrial: TDLM on mean proba across trials
        sf_s, sb_s = tdlm.compute_1step(np.mean(probas[interval], axis=0).T, tf=tf, max_lag=3)
        sf[interval] += [sf_s]
        sb[interval] += [sb_s]
        # single trial: TDLM on each trial separately, average sequenceness
        results = [tdlm.compute_1step(np.array(p).T, tf=tf, max_lag=3) for p in probas[interval]]
        sf_single[interval] += [np.mean([r[0] for r in results], axis=0)]
        sb_single[interval] += [np.mean([r[1] for r in results], axis=0)]

plot_intervals = [iv for iv in intervals if iv < 1.0]  # exclude 2048 ms
xticks     = np.arange(0, 40, 10)
xticklabels = [0, 1.25, 2.5, 3.75]
iv_labels  = [int(iv * 1000) for iv in sorted(plot_intervals)]

# 1×1: all 32–512 ms overlaid (supertrials)
fig, ax = plt.subplots(1, 1, figsize=[6, 4])
for i, interval in enumerate(sorted(plot_intervals)):
    tdlm.plot_sequenceness(sf[interval], sb[interval],
                           maxlag=3, which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           ax=ax, clear=False, plot95=False, plotmax=i==0)
ax.set_title('TDLM fMRI – supertrials (32–512 ms)')
ax.legend(iv_labels, title='interval (ms)')
ax.set_xticks(xticks, xticklabels)
ax.set_xlabel('time lag (s)')
ax.set_ylim(-3, 3)

# 1×2: single trials (left) vs supertrials (right)
fig, axs = plt.subplots(1, 2, figsize=[12, 4], sharey=True)
for i, interval in enumerate(sorted(plot_intervals)):
    tdlm.plot_sequenceness(sf_single[interval], sb_single[interval],
                           maxlag=3, which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           ax=axs[0], clear=False, plot95=False, plotmax=i==0)
    tdlm.plot_sequenceness(sf[interval], sb[interval],
                           maxlag=3, which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           ax=axs[1], clear=False, plot95=False, plotmax=i==0)
for ax, title in zip(axs, ['single trials', 'supertrials']):
    ax.set_title(title)
    ax.legend(iv_labels, title='interval (ms)')
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel('time lag (s)')
axs[0].set_ylabel('sequenceness')
fig.suptitle('TDLM fMRI – single trials vs supertrials')


#%% TR: interpolated TDLM on super trials
intervals = ['0.032', '0.128', '0.064', '0.512', '2.048']
categories = ['cat', 'chair', 'face', 'house', 'shoe']
df = pd.DataFrame()
sf = {interval: [] for interval in intervals}
sb = {interval: [] for interval in intervals}



for subject in tqdm(subjects):
    df_probas = load_decoding_3T(subject,
                                test_set='test-seq_long',
                                mask='cv')
    df_probas = df_probas[df_probas.classifier.isin(categories)]
    df_probas = df_probas[df_probas['class'].isin(categories)]
    df_seq = bids_utils.load_sequences_3T(subject)

    intervals = df_probas.tITI.unique()
    sf_subj = {interval: [] for interval in intervals}
    sb_subj = {interval: [] for interval in intervals}

    tf = tdlm.seq2tf('ABCDE')

    # extract the sequences
    probas = {interval:[] for interval in intervals}
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial')):
        assert t1==t2
        interval = df_s.tITI.values[0]
        seq_labels = list(df_s.stim_label.values[0])
        proba = [df_p[df_p.classifier==label].probability.values for label in seq_labels]
        probas[interval] += [proba]

    for i, interval in enumerate(intervals):
        mean_proba = np.mean(probas[interval], axis=0).T
        probas_fitted = []
        plt.figure()
        for proba in mean_proba.T:
            amp_guess   = proba.max()             # peak height
            shift_guess = (np.arange(13)*1.25)[np.argmax(proba)]  # peak location
            initial_params = {
                "frequency": [(0.01, 0.2), 0.1],                 # fixed at 10 Hz
                "amplitude": [(amp_guess, 2 * amp_guess), amp_guess],      # start at peak
                "loc":     [(0, 13*1.25), shift_guess],
                "baseline":  [(0, 0), 0.0],                        # pinned to 0
            }
            fitted_t, fitted, _= sigproc.fit_curve(proba, model=sigproc.curves.sine_truncated,
                                            curve_params=initial_params,
                                            plot_fit=True)
            probas_fitted += [fitted]

        probas_fitted = np.transpose(probas_fitted)
        max_lag = int(float(interval)*1000/10 + 10)
        sf_subj, sb_subj = tdlm.compute_1step(probas_fitted, tf=tf, max_lag=max_lag)
        sf[interval] += [sf_subj]
        sb[interval] += [sb_subj]



fig, ax = plt.subplots(1, 1)
for i, interval in enumerate(intervals):
    # plt.figure()
    tdlm.plot_sequenceness(sf[interval], sb[interval],
                           maxlag=5,
                           which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           # ax=ax,
                           clear=False,
                           plot95=False,
                           plotmax=False)
    plt.title(interval)
ax.set_title('TDLM on model fit')
ax.legend([32, '_', '_', 64, 128, 512, 2048], title='interval')
ax.set_xticks(np.arange(0, 50, 10), [0, 1, 2, 3, 4])
ax.set_xlabel('time lag (seconds)')
plt.ylim(-2, 2)

#%% decoding accuracy vs TDLM sequenceness (fwd / bkw, mean across speed conditions)

import scipy.stats as stats
import matplotlib.pyplot as plt

dec_acc = np.array([bids_utils.get_decoding_accuracy_3T(s) for s in subjects])

# sequenceness at lag 1 TR (index 0), mean across 32–512 ms conditions
seq_fwd = np.mean([np.array([sf[iv][i][0] for i in range(len(subjects))])
                   for iv in plot_intervals], axis=0)
seq_bkw = np.mean([np.array([sb[iv][i][0] for i in range(len(subjects))])
                   for iv in plot_intervals], axis=0)

fig, axs = plt.subplots(1, 2, figsize=[10, 4])
for ax, seq, direction in zip(axs, [seq_fwd, seq_bkw], ['fwd', 'bkw']):
    r, p = stats.pearsonr(dec_acc, seq)
    ax.scatter(dec_acc, seq, color='steelblue', alpha=0.6, edgecolors='white', linewidths=0.5)
    m, b = np.polyfit(dec_acc, seq, 1)
    x_line = np.linspace(dec_acc.min(), dec_acc.max(), 100)
    ax.plot(x_line, m * x_line + b, color='steelblue', linewidth=1.5,
            linestyle='--' if p > 0.05 else '-')
    ax.set_xlabel('decoding accuracy')
    ax.set_ylabel('sequenceness (lag 1 TR)')
    ax.set_title(f'{direction}  r={r:.2f}, p={p:.3f}')

fig.suptitle('Decoding accuracy vs TDLM sequenceness (mean across 32–512 ms)')
fig.tight_layout()
