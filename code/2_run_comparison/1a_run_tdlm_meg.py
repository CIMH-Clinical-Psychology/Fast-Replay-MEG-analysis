# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run TDLM (Temporally Delayed Linear Modelling) on MEG data

@author: Simon.Kern
"""

import mne
from tqdm import tqdm
import pandas as pd
import bids_utils
import settings
from settings import layout_MEG as layout
import numpy as np
import seaborn as sns
from meg_utils import plotting
import matplotlib.pyplot as plt
from meg_utils.plotting import savefig, normalize_lims
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tdlm
from tdlm.utils import seq2tf, num2char
from scipy.stats import zscore
#%% settings

normalization = 'lambda x: x/x.mean(0)'


stop
#%% TDLM

intervals = [32, 64, 128, 512]

df = pd.DataFrame()
sf = {interval: [] for interval in intervals}
sb = {interval: [] for interval in intervals}

fig_subj, axs_subj = plt.subplots(2, 2, figsize=[12, 8])
axs_subj = axs_subj.flatten()

# zscore = lambda x, axis, nan_policy:x

for subject in tqdm(layout.subjects):

    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    tf_trials = [seq2tf(''.join(num2char(df_trial.trigger.values))) for df_trial in beh]

    sf_subj = {interval: [] for interval in intervals}
    sb_subj = {interval: [] for interval in intervals}

    for proba, df_trial in zip(tqdm(probas), beh, strict=True):
        proba = eval(normalization)(proba)

        # transition matrix for this specific trial
        seq_trial = num2char(df_trial.trigger.values)
        tf = seq2tf(''.join(seq_trial))
        interval = df_trial.interval_time.values[0]

        # only calculate up to the length that they have actually been shown!
        # else we are analysing the buffer period already +200ms for safety
        length = int((interval+100)*5)//10 + 20  # assuming 100 Hz sfreq
        max_lag = int(((interval/10) + 10)*1.5)

        sf_trial, sb_trial = tdlm.compute_1step(proba[:length, :], tf, n_shuf=100, max_lag=max_lag)
        sf_subj[interval] += [zscore(sf_trial, axis=-1, nan_policy='omit')]
        sb_subj[interval] += [zscore(sb_trial, axis=-1, nan_policy='omit')]

    for interval in intervals:
        sf[interval] += [np.nanmean(sf_subj[interval], 0)]
        sb[interval] += [np.nanmean(sb_subj[interval], 0)]

    for i, interval in enumerate(sf_subj):
        ax = axs_subj[i]
        tdlm.plotting.plot_sequenceness(sf_subj[interval], sb_subj[interval],
                                        which=['fwd', 'bkw'], ax=axs_subj[i])
        ax.set_title(f'{interval=} ms')
        fig_subj.suptitle(f'{subject=}')
    savefig(fig_subj, f'{settings.plot_dir}/sequence/meg_fast_images_sequenceness_{subject}.png')



for i, interval in enumerate(sf_subj):
    ax = axs_subj[i]
    tdlm.plotting.plot_sequenceness(sf[interval], sb[interval],
                                    which=['fwd', 'bkw'],
                                    ax=axs_subj[i], plot95=False)
    axs_subj[2].set_ylim([-1.5, 1.9])
    ax.vlines(interval+100, *ax.get_ylim(), color='black', linewidth=4, alpha=0.3)
    if interval==512:
        ax.set_xticks(np.arange(50, 800, 100))
    ax.set_xticks(list(ax.get_xticks()) + [int(interval+100)])

    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_title(f'{interval=} ms')
    # Apply updated ticks
    fig_subj.suptitle('MEG: Sequenceness during fast sequence presentation')
    plt.pause(0.1)

savefig(fig_subj, f'{settings.plot_dir}/fast_images_sequenceness_all.png')

#%% participants as heatmap


fig, axs = plt.subplots(2, 2, figsize=[12, 8])

for i, interval in enumerate(sf_subj):
    sf_isi = np.array(sf[interval])[:, 0, :]
    sb_isi = np.array(sb[interval])[:, 0, :]

    ax = axs.flat[i]
    ax.clear()
    max_lag = sf_isi.shape[-1]
    # h = ax.imshow(sf_isi, aspect='auto', cmap='RdBu_r')
    df = pd.DataFrame(sf_isi, columns=np.arange(0, max_lag*10, 10), index=layout.subjects)
    sns.heatmap(df, ax=ax, cmap='RdBu_r')
    ax.set_xticks(np.arange(0, max_lag)[::5 if interval<500 else 10], np.arange(0, max_lag*10, 10)[::5 if interval<500 else 10])
    ax.set_yticks(np.arange(len(layout.subjects))[::2], layout.subjects[::2])
    ax.set(ylabel='subject', xlabel='time lag', title=f'{interval=} ms')

plotting.normalize_lims(axs, which='v')

fig.suptitle('Forward sequenceness across participants')
savefig(fig, settings.plot_dir + '/figures/sequenceness_heatmap.png')

#%% heatmap for trials
np.random.seed(0)
subjects_rnd = sorted(np.random.choice(layout.subjects, 6, replace=False))

fig, axs = plt.subplots(2, 3, figsize=[10, 6])

for i, subject in enumerate(tqdm(subjects_rnd)):
    subject = str(subject)
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject, intervals=[32])

    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    tf_trials = [seq2tf(''.join(num2char(df_trial.trigger.values))) for df_trial in beh]

    sf_subj = []
    sb_subj = []

    for proba, df_trial in zip(tqdm(probas), beh, strict=True):
        proba = eval(normalization)(proba)

        # transition matrix for this specific trial
        seq_trial = num2char(df_trial.trigger.values)
        tf = seq2tf(''.join(seq_trial))
        interval = df_trial.interval_time.values[0]

        # only calculate up to the length that they have actually been shown!
        # else we are analysing the buffer period already +200ms for safety
        length = int((interval+100)*5)//10 + 20  # assuming 100 Hz sfreq
        max_lag = int(((interval/10) + 10)*1.5)

        sf_trial, sb_trial = tdlm.compute_1step(proba[:length, :], tf, n_shuf=0, max_lag=max_lag)
        sf_subj += [zscore(sf_trial, axis=-1, nan_policy='omit').squeeze()]
        sb_subj += [zscore(sb_trial, axis=-1, nan_policy='omit').squeeze()]

    sf_subj = np.array(sf_subj)
    sb_subj = np.array(sb_subj)

    df = pd.DataFrame(sf_subj, columns=np.arange(0, max_lag*10+10, 10))
    ax = axs.flat[i]
    sns.heatmap(df, cmap='RdBu_r', ax=ax)
    ax.set(xlabel='time lag', ylabel='trial', title=f'{subject=}')

plotting.normalize_lims(axs, 'v')
fig.suptitle('Forward sequenceness across participants for 32 ms condition')
savefig(fig, settings.plot_dir + '/figures/sequenceness_trial_level.png')

#%% correlate peak decoding accuracy with sequenceness?


#%% TDLM on super trials
import tdlm
from tdlm.utils import seq2tf, num2char

normalization = 'lambda x: x/x.mean(0)'
# normalization = 'lambda x: zscore(x, axis=0)'

intervals = [32, 64, 128, 512]

df = pd.DataFrame()


# fig_subj, axs_subj = plt.subplots(2, 2, figsize=[12, 8])
# axs_subj = axs_subj.flatten()
zscore = lambda x, axis, nan_policy:x

sf1 = {interval: [] for interval in intervals}
sb1 = {interval: [] for interval in intervals}
sf2 = {interval: [] for interval in intervals}
sb2 = {interval: [] for interval in intervals}
for subject in tqdm(settings.layout.subjects):
    if subject in ['05', '12', '15', '16', '20', '21', '22', '27', '28', '29']:
        continue

    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    tf_trials = [seq2tf(''.join(num2char(df_trial.trigger.values))) for df_trial in beh]

    # group trials by sequence
    seqs = set([''.join(df.sequence.values) for df in beh])
    trials_grouped = {seq:[p for p, df in zip(probas, beh) if seq==''.join(df['sequence'].values)] for seq in seqs}

    sf1_subj = {interval: [] for interval in intervals}
    sb1_subj = {interval: [] for interval in intervals}
    sf2_subj = {interval: [] for interval in intervals}
    sb2_subj = {interval: [] for interval in intervals}
    for seq, trials in trials_grouped.items():
        trials_mean = np.mean(trials, 0)
        tf = seq2tf(seq)
        interval =[df.interval_time.iloc[0] for df in beh if seq==''.join(df['sequence'].values)][0]

        # length = int((interval+100)*5)//10 + 20  # assuming 100 Hz sfreq
        max_lag = int((interval+100)/10) + 10
        sf1_trial, sb1_trial = tdlm.compute_1step(trials_mean, tf, n_shuf=None,
                                                  max_lag=max_lag)
        sf2_trial, sb2_trial = tdlm.compute_2step(trials_mean, tf, n_shuf=None,
                                                  max_lag=max_lag)
        sf1_subj[interval] += [zscore(sf1_trial, axis=-1, nan_policy='omit')]
        sb1_subj[interval] += [zscore(sb1_trial, axis=-1, nan_policy='omit')]
        sf2_subj[interval] += [zscore(sf2_trial, axis=-1, nan_policy='omit')]
        sb2_subj[interval] += [zscore(sb2_trial, axis=-1, nan_policy='omit')]

    for interval in intervals:
        sf1[interval] += [np.nanmean(sf1_subj[interval], 0)]
        sb1[interval] += [np.nanmean(sb1_subj[interval], 0)]
        sf2[interval] += [np.nanmean(sf2_subj[interval], 0)]
        sb2[interval] += [np.nanmean(sb2_subj[interval], 0)]

    # for i, interval in enumerate(sf_subj):
    #     ax = axs_subj[i]
    #     tdlm.plotting.plot_sequenceness(sf1_subj[interval], sb1_subj[interval],
    #                                     which=['fwd', 'bkw'], ax=axs_subj[i])
    #     ax.set_title(f'{interval=} ms')
    #     plt.pause(0.1)

    # fig_subj.tight_layout()
    # savefig(fig_subj, f'{settings.plot_dir}/sequence/fast_images_sequenceness_supertrials_{subject}.png')


fig_subj, axs_subj = plt.subplots(2, 2, figsize=[12, 8])
axs_subj = axs_subj.flatten()
for i, interval in enumerate(intervals):
    ax = axs_subj[i]
    tdlm.plotting.plot_sequenceness(sf1[interval], sb1[interval],
                                    which=['fwd', 'bkw'], ax=axs_subj[i])
    ax.set_title(f'{interval=} ms')
    plt.pause(0.1)


fig, ax = plt.subplots(1, 1)
for i, interval in enumerate(intervals):
    tdlm.plot_sequenceness(sf1[interval], sb1[interval],
                           maxlag=5,
                           which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           ax=ax,
                           clear=False,
                           plot95=False,
                           plotmax=False)

ax.hlines(-0.6, 0, 700, linestyles='--', color=settings.palette_wittkuhn2[0], linewidth=1.5, alpha=0.6, label='_')
ax.hlines(0.6, 0, 700, linestyles='--', color=settings.palette_wittkuhn2[0], linewidth=1.5, alpha=0.6, label='_')

ax.set_title('')
ax.legend([32, 64, 128, 512], title='interval')
ax.set_xticks(np.arange(0, 750, 100), np.arange(0, 750, 100))
ax.set_xlabel('time lag (ms)')

fig_subj, axs_subj = plt.subplots(2, 2, figsize=[12, 8])
axs_subj = axs_subj.flatten()
for i, interval in enumerate(intervals):
    ax = axs_subj[i]
    tdlm.plotting.plot_sequenceness(sf2[interval], sb2[interval],
                                    which=['fwd', 'bkw'], ax=axs_subj[i])
    ax.set_title(f'{interval=} ms')
    fig_subj.suptitle('All subjects, fast images sequenceness 2-step')
    plt.pause(0.1)
    fig_subj.tight_layout()
    fig_subj.savefig(f'{settings.plot_dir}/fast_images_sequenceness_supertrials_all-2step.png')
