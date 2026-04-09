# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run TDLM (Temporally Delayed Linear Modelling) on fMRI data

@author: Simon.Kern
"""
import sys; sys.path.append('..')
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
import matplotlib.pyplot as plt
from mne.stats import permutation_t_test, permutation_cluster_1samp_test
from meg_utils import misc

#%% settings
subjects = [f'{i:02d}' for i in range(1, 41)]
normalization = 'lambda x: x/x.mean(0)'

layout = BIDSLayout(settings.bids_dir_3T)

#%% TR: TDLM on fMRI data individual trials and supertrials
intervals = ['0.032', '0.128', '0.064', '0.512', '2.048']
categories = ['cat', 'chair', 'face', 'house', 'shoe']

sf        = {interval: [] for interval in intervals}  # averaged trials
sb        = {interval: [] for interval in intervals}
sf_single = {interval: [] for interval in intervals}  # individual trials
sb_single = {interval: [] for interval in intervals}

for subject in tqdm(subjects):
    df_probas = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long',
                                                classifier=categories)
    df_seq = bids_utils.load_trial_data_3T(subject)

    tf = tdlm.seq2tf('ABCDE')

    probas = {interval: [] for interval in intervals}
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial')):
        assert t1==t2
        interval = str(df_s.tITI.values[0])
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

#%% plotting
plot_intervals = intervals[:-1]  # exclude 2048 ms
xticks     = [0, 10, 16.5, 20, 30]
xticklabels = [0, 1.25, 2.048, 2.5, 3.75]
iv_labels  = plot_intervals

# 1×1: all 32–512 ms overlaid (supertrials)
fig, axs = plt.subplots(1, 2, figsize=[14, 4])

ax = axs[0]
for i, interval in enumerate(sorted(plot_intervals)):
    tdlm.plot_sequenceness(sf[interval], sb[interval],
                           maxlag=3, which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           ax=ax, clear=False, plot95=False,
                           plotmax=i==0)

ax.set_title('32 to 512 ms condition')
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], color=settings.palette_wittkuhn2[i], lw=2)
                  for i in range(len(plot_intervals))]
ax.legend(legend_handles, [f'{int(float(x)*1000)} ms' for x in sorted(plot_intervals)],  loc='lower left')
ax.set_xticks(xticks, xticklabels)
ax.axhline(0, linestyle='--', c='gray', alpha=0.3)
ax.set_xlabel('time lag (s)')
# ax.set_ylim(-1.2, 3)

# signflip and cluster permutation tests per interval on axs[0]
ax = axs[0]
xlim = ax.get_xlim()
for i, interval in enumerate(sorted(plot_intervals)):
    sx = np.array(sf[interval])[:, 0, 1:]

    # signflip permutation t-test
    _, pvals, _ = permutation_t_test(sx, seed=i, verbose=False)
    clusters = misc.get_clusters(pvals < 0.05, start=1)
    bounds = [b for significant, b in clusters if significant]
    for b1, b2 in bounds:
        ax.axvspan(max(b1*10 - 5, xlim[0]), min(b2*10 + 5, xlim[1]), alpha=0.3,
                   color=settings.palette_wittkuhn2[i], ymin=0.95, ymax=1,
                   label=f'signflip-perm p<0.05' if i == 0 else '')

    # cluster-based permutation test
    _, cl, cl_pvals, _ = permutation_cluster_1samp_test(sx, seed=i, verbose=False)
    sig_clusters = [c[0] for c, p in zip(cl, cl_pvals) if p < 0.05]
    for cl in sig_clusters:
        b1, b2 = cl[0], cl[-1]
        ax.axvspan(max(b1*10 + 5, xlim[0]), min(b2*10 + 15, xlim[1]), alpha=0.5, hatch='///',
                   color=settings.palette_wittkuhn2[i], ymin=0.9, ymax=0.95,
                   label=f'cluster-perm p<0.05' if i == 0 else '')

# 1×2: single trials (left) vs supertrials (right) of 2048er condition
interval = '2.048'
tdlm.plot_sequenceness(sf_single[interval], sb_single[interval],
                       maxlag=3, which=['fwd'],
                       color=settings.palette_wittkuhn2[-2],
                       ax=axs[1], clear=False, plot95=False, plotmax=1)
# tdlm.plot_sequenceness(sf[interval], sb[interval],
#                        maxlag=3, which=['fwd'],
#                        color=settings.palette_wittkuhn2[-2],
#                        ax=axs[2], clear=False, plot95=False, plotmax=1)

for ax, title, sx in zip(axs[1:2],
                         ['2048 ms condition', 'averaged trials\n2048 ms condition'],
                         [sf_single['2.048'], sf['2.048']]):
    ax.set_title(title)
    ax.legend(['2048 ms'], title='interval (ms)')
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel('time lag (s)')
    ax.axhline(0, linestyle='--', c='gray', alpha=0.3)

    # expected time lag for 2048 ms condition (in plot units: 10 = 1 TR = 1.25s)
    expected_lag = 2.048 / 1.25 * 10
    ax.axvspan(expected_lag - 1, expected_lag + 1, color='black', alpha=0.15,
               label='expected lag', ymax=0.9)

    # signflip permutation t-test
    xlim = ax.get_xlim()
    sx_data = np.array(sx)[:, 0, 1:]
    _, pvals, _ = permutation_t_test(sx_data, seed=0, verbose=False)
    clusters = misc.get_clusters(pvals < 0.05, start=1)
    bounds = [b for significant, b in clusters if significant]
    for b1, b2 in bounds:
        ax.axvspan(max(b1*10 - 5, xlim[0]), min(b2*10 + 5, xlim[1]), alpha=0.3,
                   color=settings.palette_wittkuhn2[-2], ymin=0.95, ymax=1,
                   label='signflip-perm p<0.05')

    # cluster-based permutation test
    _, cl, cl_pvals, _ = permutation_cluster_1samp_test(sx_data, seed=0, verbose=False)
    sig_clusters = [c[0] for c, p in zip(cl, cl_pvals) if p < 0.05]
    for cl in sig_clusters:
        b1, b2 = cl[0], cl[-1]
        ax.axvspan(max(b1*10 + 5, xlim[0]), min(b2*10 + 15, xlim[1]), alpha=0.5, hatch='///',
                   color=settings.palette_wittkuhn2[-2], ymin=0.9, ymax=0.95,
                   label='cluster-perm p<0.05')

axs[0].set_ylabel('forward sequenceness')
axs[1].set_ylabel('forward sequenceness')
fig.suptitle('TDLM on fMRI')


by_label = {}
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(l, h)
del by_label['fwd']
fig.legend(by_label.values(), by_label.keys(), loc='upper right',
           bbox_to_anchor=(0.99, 1), ncol=2, fontsize='small')

plotting.savefig(fig, settings.plot_dir + '/figures/TDLM_fMRI_overview.png')
