#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:40:24 2024

@author: simon.kern
"""
import mne
from tqdm import tqdm
import pandas as pd
import bids_utils
import settings
from settings import layout
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from meg_utils.plotting import savefig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


for subj in ['01', '23', '03', '04', '09', '13', '26']:
    # somehow, these give errors when loading, so leave them out for now
    # should be included in the final calculation
    # if you see this in a PR, please let me knowm ;-)
    if subj in layout.subjects:
        layout.subjects.remove(subj)
stop
#%% individual images probability visualized

df = pd.DataFrame()

times = np.arange(71) *10 -200

fig, ax = plt.subplots(figsize=[8, 5])
fig_cm, ax_cm = plt.subplots(figsize=[6, 5])

confmats = []

for subject in tqdm(settings.layout.subjects):

    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, df_beh = bids_utils.load_fast_images(subject)

    # run classifier across trial
    probas = np.swapaxes([clf.predict_proba(data_x[:, :, t]) for t in range(data_x.shape[-1])], 0, 1)
    # probas /= probas.mean(0).mean(0)

    timepoint_idx = np.where(times == 150)[0][0]  # Find index for 150ms
    predictions = np.argmax(probas[:, timepoint_idx, :], axis=1)  # Predicted labels at 150ms

    # Compute confusion matrix
    confmat = confusion_matrix(data_y, predictions)

    confmats.append(confmat)
    ax_cm.clear()

    cmd = ConfusionMatrixDisplay(confmat, display_labels=list(settings.img_trigger.values()))
    cmd.plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(f'ConfMat {subject=}')
    savefig(fig_cm, f'{settings.plot_dir}/confmat/confmat_fast_images_{subject}.png', despine=False)

    timepoint = np.repeat(times, probas.shape[-1])

    df_subj = pd.DataFrame()
    for i, (proba, y) in enumerate(zip(probas, data_y, strict=True)):
        label = ['other']*5
        label[y] = 'target'
        df_tmp = pd.DataFrame({'label': np.hstack([label]* probas.shape[1]),
                               'timepoint': timepoint,
                               'proba': proba.ravel(),
                               'stimulus': settings.img_trigger[y],
                               'interval': df_beh.iloc[i].interval_time,
                               'subject':subject})
        df_subj = pd.concat([df_subj, df_tmp])

    ax.clear()
    df_subj = df_subj.groupby(['label', 'timepoint', 'stimulus', 'interval', 'subject']).mean().reset_index()
    sns.lineplot(df_subj, x='timepoint', y='proba', hue='interval',style='label',
                 palette='muted', ax=ax)
    ax.set_title(f'Fast images decoding {subject=}')
    fig.savefig(settings.plot_dir + f'fast_images_decoding_{subject}.png')
    plt.pause(0.1)
    df = pd.concat([df, df_subj], ignore_index=True)

ax_cm.clear()
cmd = ConfusionMatrixDisplay(np.mean(confmats, 0), display_labels=list(settings.img_trigger.values()))
cmd.plot(ax=ax_cm, colorbar=False)
ax_cm.set_title(f'ConfMat {len(confmats)=}')
savefig(fig_cm, f'{settings.plot_dir}/confmat_fast_images_all.png', despine=False)

ax.clear()
sns.lineplot(df, x='timepoint', y='proba', hue='interval',style='label',
             palette='muted', ax=ax)
ax.set_title(f'Fast images decoding, n={len(df.subject.unique())}')
fig.savefig(settings.plot_dir + f'fast_images_decoding_all.png')
plt.pause(0.1)

#%% full sequence probabilities
from scipy.stats import zscore

normalization = 'lambda x: np.log(x)/np.log(x).mean(0)'
normalization = 'lambda x: x/x.mean(0)'
# normalization = 'lambda x: zscore(x, axis=0)'


df = pd.DataFrame()
for subject in tqdm(settings.layout.subjects):

    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, df_beh = bids_utils.load_fast_sequences(subject)

    probas = np.swapaxes([clf.predict_proba(data_x[:, :, t]) for t in range(data_x.shape[-1])], 0, 1)
    df_subj = pd.DataFrame()
    timepoint = np.hstack([np.arange(data_x.shape[-1])*10]*5)-200
    stimulus = np.repeat(np.arange(5), probas.shape[1])
    probas = eval(normalization)(probas)

    for proba, df_trial in zip(probas, df_beh):
        # proba /= proba.mean(0)
        # proba = np.log(proba)
        pos_idx = [list(df_trial.trigger).index(i) for i in  range(5)]
        position = np.repeat(pos_idx, probas.shape[1])
        df_tmp = pd.DataFrame({'proba': proba.ravel('F'),
                               'stimulus': stimulus,
                               'timepoint': timepoint,
                               'position': position,
                               'interval': df_trial.interval_time.iloc[0],
                               'subject': subject})
        df_subj = pd.concat([df_subj, df_tmp])
    df_subj = df_subj.groupby(['timepoint', 'interval', 'position']).mean(True).reset_index()
    # sns.lineplot(df_subj, x='timepoint', y='proba', hue='interval')
    df = pd.concat([df, df_subj], ignore_index=True)

fig, axs = plt.subplots(2, 2, figsize=[18, 8])
axs = axs.flatten()

for i, interval in enumerate(df.interval.unique()):
    df_sel = df[df.interval==interval]
    ax = axs[i]
    sns.lineplot(df_sel, x='timepoint', y='proba', hue='position', palette='muted', ax=ax)
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim())
    # ax.set_xlim(-200, interval*5+750)
    ax.set_title(f'{interval=}ms\n{normalization=}')
    plt.pause(0.1)

fig.tight_layout()
plt.pause(0.1)
fig.savefig(settings.plot_dir + f'fast_images_sequence_zscore.png')

#%% decode individual fast images

normalization = 'lambda x: x/x.mean(0)'
# normalization = 'lambda x: zscore(x, axis=0)'


df = pd.DataFrame()
for subject in tqdm(settings.layout.subjects):

    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, df_beh = bids_utils.load_fast_sequences(subject)

    probas = np.swapaxes([clf.predict_proba(data_x[:, :, t]) for t in range(data_x.shape[-1])], 0, 1)
    df_subj = pd.DataFrame()
    timepoint = np.hstack([np.arange(data_x.shape[-1])*10]*5)-200
    position = np.repeat(np.arange(5), probas.shape[1])
    for proba, df_trial in zip(probas, df_beh):
        # proba /= proba.mean(0)
        proba = eval(normalization)(proba)
        stimulus = np.repeat(df_trial.trigger, probas.shape[1])
        df_tmp = pd.DataFrame({'proba': proba.ravel('F'),
                               'stimulus': stimulus,
                               'timepoint': timepoint,
                               'position': position,
                               'interval': df_trial.interval_time.iloc[0],
                               'subject': subject})
        df_subj = pd.concat([df_subj, df_tmp])
    df_subj = df_subj.groupby(['timepoint', 'interval', 'position', 'stimulus']).mean(True).reset_index()
    # sns.lineplot(df_subj, x='timepoint', y='proba', hue='interval')
    df = pd.concat([df, df_subj], ignore_index=True)


#%% TDLM
import tdlm
from tdlm.utils import seq2tf, num2char

normalization = 'lambda x: x/x.mean(0)'
# normalization = 'lambda x: zscore(x, axis=0)'

intervals = [32, 64, 128, 512]

df = pd.DataFrame()
sf = {interval: [] for interval in intervals}
sb = {interval: [] for interval in intervals}

fig_subj, axs_subj = plt.subplots(2, 2, figsize=[12, 8])
axs_subj = axs_subj.flatten()
zscore = lambda x, axis, nan_policy:x
for subject in tqdm(settings.layout.subjects):
    if subject in ['05', '12', '15', '16', '20', '21', '22', '27', '28', '29']:
        continue

    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    tf_trials = [seq2tf(''.join(num2char(df_trial.trigger.values))) for df_trial in beh]

    sf_subj = {interval: [] for interval in intervals}
    sb_subj = {interval: [] for interval in intervals}

    for proba, df_trial in zip(tqdm(probas), beh, strict=True):
        proba = eval(normalization)(proba)
        # asd
        # pos_idx = [list(df_trial.trigger).index(i) for i in  range(5)]

        # transition matrix for this specific trial
        seq_trial = num2char(df_trial.trigger.values)
        tf = seq2tf(''.join(seq_trial))
        interval = df_trial.interval_time.values[0]

        # only calculate up to the length that they have actually been shown!
        # else we are analysing the buffer period already +200ms for safety
        length = int((interval+100)*5)//10 + 20  # assuming 100 Hz sfreq
        max_lag = int(interval/10) + 10
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
        plt.pause(0.1)
    fig_subj.tight_layout()
    savefig(fig_subj, f'{settings.plot_dir}/sequence/fast_images_sequenceness_{subject}.png')



for i, interval in enumerate(sf_subj):
    ax = axs_subj[i]
    tdlm.plotting.plot_sequenceness(sf[interval], sb[interval],
                                    which=['fwd', 'bkw'], ax=axs_subj[i])
    ax.set_title(f'{interval=} ms')
    fig_subj.suptitle('All subjects, fast images sequenceness')
    plt.pause(0.1)
    fig_subj.tight_layout()
    fig_subj.savefig(f'{settings.plot_dir}/fast_images_sequenceness_all.png')


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

for max_true_trans in [0, 1, 2, 3, 4]:
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
                                                      max_lag=max_lag, max_true_trans=max_true_trans)
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
    fig_subj.suptitle(f'All subjects, fast images sequenceness 1-step \n{max_true_trans=}')
    fig_subj.tight_layout()
    fig_subj.savefig(f'{settings.plot_dir}/max_true_trans_{max_true_trans}_fast_images_sequenceness_supertrials_all.png')



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
