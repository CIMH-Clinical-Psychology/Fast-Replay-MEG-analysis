# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run SODA (Slope Order Dynamic Analysis) on fMRI data

@author: Simon.Kern
"""

import os
import settings
from settings import layout_3T as layout
import bids_utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import mne
from meg_utils import decoding, plotting, sigproc
from meg_utils.plotting import savefig
from bids import BIDSLayout
import soda
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend
#%% settings
subjects = [f'{i:02d}' for i in range(1, 41)]
normalization = 'lambda x: x/x.mean(0)'

layout = BIDSLayout(settings.bids_dir_3T)
layout_decoding = BIDSLayout(settings.bids_dir_3T_decoding, validate=False)

#%% run SODA on sequence trials

df_slope = pd.DataFrame()
for subject in tqdm(subjects):

    # read file
    df_seq = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long')
    df_seq = df_seq[df_seq['class'].isin(settings.categories)]

    df_beh = bids_utils.load_trial_data_3T(subject, condition='sequence')


    df_subj = pd.DataFrame()
    for (i, df_trial), (j, beh) in zip(df_seq.groupby('trial'), df_beh.groupby('trial'), strict=True):
        assert i==j

        stim_order = [settings.categories.index(x) for x in beh.stim_label.values[0]]

        # make sure sorting is same as the category number
        df_trial = df_trial.sort_values(['class', 'tr_onset'], key=lambda x:
                                        [settings.categories.index(y) for y in x]
                                        if 'str' in str(x.dtype) else x)

        probas = np.array(df_trial.probability.values.reshape([13, 5], order='F'))

        probas = eval(normalization)(probas)
        slopes = soda.compute_slopes(probas, order=stim_order)

        df_tmp = pd.DataFrame({'slope': slopes,
                               'timepoint': sorted(df_trial.tr_onset.unique()),
                               'tr': np.arange(1, 14),
                               'interval': df_trial.tITI.unique()[0],
                               'trial': df_trial.trial.values[0],
                               'subject': subject})

        df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    df_slope = pd.concat([df_slope, df_subj], ignore_index=True)

# df_slope = df_slope.groupby(['tr', 'interval', 'subject']).mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
df_slope.timepoint = df_slope.timepoint.round(1)
sns.lineplot(df_slope, x='tr', y='slope', hue='interval',
             palette=settings.palette_wittkuhn2, ax=ax)

savefig(fig, settings.plot_dir + '/figures/soda_slopes_3T.png')


#%% slopes of participants as heatmap

fig, axs = plt.subplots(2, 3, figsize=[12, 8])
axs.flat[-1].axis('off')
intervals = df_slope.interval.unique()

for i, (interval, df_subj) in enumerate(df_slope.groupby('interval')):

    df = df_subj.groupby(['tr', 'subject']).mean().reset_index()

    slopes = np.squeeze([df_subj.sort_values('timepoint').slope for _, df_subj in df.groupby('subject')])

    ax = axs.flat[i]
    ax.clear()
    sns.heatmap(pd.DataFrame(slopes, columns=np.arange(1, 14), index=subjects),
                cmap='RdBu_r', ax=ax)
    ax.set(ylabel='subject', xlabel='tr', title=f'{interval=} ms')

    # h = ax.imshow(sf_isi, aspect='auto', cmap='RdBu_r')
    ax.set_xticks(np.arange(13), np.arange(1, 14))
    ax.set_yticks(np.arange(len(subjects))[::5], subjects[::5])

plotting.normalize_lims(axs, which='v')

fig.suptitle('Mean slopes for all participants')
savefig(fig, settings.plot_dir + '/figures/slopes_heatmap.png')

#%% heatmap for trials of selected participants
np.random.seed(0)
subjects_rnd = sorted(np.random.choice(subjects, 6, replace=False))

fig, axs = plt.subplots(2, 3, figsize=[12, 8])
axs.flat[-1].axis('off')

df_sel = df_slope[(df_slope.interval==0.512) & df_slope.subject.isin(subjects_rnd)]

for i, (subject, df_subj) in enumerate(df_sel.groupby('subject')):

    slopes = [x.slope.values for _, x in df_subj.sort_values(['tr', 'trial']).groupby('trial')]
    slopes = np.squeeze(slopes)

    ax = axs.flat[i]
    ax.clear()
    sns.heatmap(pd.DataFrame(slopes, columns=np.arange(1, 14), index=np.arange(1, 16)),
                cmap='RdBu_r', ax=ax)
    ax.set(ylabel='trial', xlabel='tr', title=f'{subject}')

    # h = ax.imshow(sf_isi, aspect='auto', cmap='RdBu_r')
    ax.set_xticks(np.arange(13), np.arange(1, 14))

plotting.normalize_lims(axs, which='v')

fig.suptitle('Slopes of all trials for selected participants')
savefig(fig, settings.plot_dir + '/figures/slopes_heatmap_trials.png')
