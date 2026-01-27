#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:40:24 2024

simply visualize the fast sequences both for MEG and fMRI

@author: simon.kern
"""
import mne
from tqdm import tqdm
import pandas as pd
import bids_utils
import settings
from settings import layout_3T, layout_MEG
import numpy as np
import seaborn as sns
from meg_utils import plotting
import matplotlib.pyplot as plt
from meg_utils.plotting import savefig, normalize_lims
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
stop


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

plt.rcParams.update({'font.size':14})
fig, axs = plt.subplots(1, 4, figsize=[16, 4])
axs = axs.flatten()

for i, interval in enumerate(df.interval.unique()):
    df_sel = df[df.interval==interval]
    ax = axs[i]
    sns.lineplot(df_sel, x='timepoint', y='proba', hue='position', palette=settings.palette_wittkuhn1,
                 ax=ax, legend=False)
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(), linewidth=0.5, color='black', alpha=0.3)
    # ax.set_xlim(-200, interval*5+750)
    if i>0:
        ax.set_ylabel('')
    for ax in axs:
        ax.set_xticks(np.arange(0, 3000, 500), np.arange(0, 3000, 500)/1000)
        ax.set_xlabel('timepoint (s)')
    ax.set_title(f'{int(interval)} ms')
    plt.pause(0.1)

fig.legend(
    ['1', '_', '2', '_', '3', '_', '4', '_', '5'],
    title='item position',
    loc='center right',
    bbox_to_anchor=(0.96, 0.6),  # 1.02 puts it just outside the right edge, 0.5 is vertical center
    bbox_transform=fig.transFigure
)
plotting.normalize_lims(axs)

asd
fig.tight_layout(rect=[0, 0, 0.86, 1])  # leave space on the right for the legend

fig.savefig(settings.plot_dir + f'fast_images_sequence_zscore.png')



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
