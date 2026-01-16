#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 09:27:44 2025

in this script we simply visualize the raw probabilities across time for the
fast sequences trials, i.e. sequence item 1-5

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
from meg_utils import plotting, misc
import matplotlib.pyplot as plt
from meg_utils.plotting import savefig, normalize_lims
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend
# ---- load data -----
stop
#%% MEG full sequences probability
from scipy.stats import zscore

normalization = 'lambda x: x/x.mean(0)'
df_meg = pd.DataFrame()

for subject in tqdm(layout_MEG.subjects):

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
                               'interval': df_trial.interval_time.iloc[0]})
        df_subj = pd.concat([df_subj, df_tmp])
    df_subj = df_subj.groupby(['timepoint', 'interval', 'position']).mean(True).reset_index()
    df_subj['subject'] = subject
    # sns.lineplot(df_subj, x='timepoint', y='proba', hue='interval')
    df_meg = pd.concat([df_meg, df_subj], ignore_index=True)

#%% MEG individual fast images decoding
df_meg_single = pd.DataFrame()

for subject in tqdm(layout_MEG.subjects):

    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, df_beh = bids_utils.load_fast_images(subject)

    preds = np.swapaxes([clf.predict(data_x[:, :, t]) for t in range(data_x.shape[-1])], 0, 1)

    accs = (preds.T==data_y).T

    df_subj = misc.to_long_df(accs, columns=['trial', 'timepoint'],
                              value_name='accuracy',
                              timepoint=np.arange(-200, 510, 10),
                              trial={'interval': df_beh.interval_time.astype(int),
                                     'serial_position': df_beh.serial_position.astype(int)})
    df_subj = df_subj.groupby(['timepoint', 'interval', 'serial_position']).mean().reset_index()
    df_subj['subject'] = subject

    df_meg_single = pd.concat([df_meg_single, df_subj], ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
ax.hlines(0.2, -200, 500, linestyle='--', color='black', alpha=0.5, label='chance')
ax.vlines(0, 0.1, 0.5,color='black', label='image onset')
sns.lineplot(df_meg_single, x='timepoint', y='accuracy', hue='interval',
             palette=settings.palette_wittkuhn2)
savefig(fig, settings.plot_dir + '/fast-sequence_individual_items.png')

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
ax.hlines(0.2, -200, 500, linestyle='--', color='black', alpha=0.5, label='chance')
ax.vlines(0, 0.1, 0.5,color='black', label='image onset')
sns.lineplot(df_meg_single, x='timepoint', y='accuracy', hue='serial_position',
             palette=settings.palette_wittkuhn2)
savefig(fig, settings.plot_dir + '/fast-sequence_individual_items-by-position.png')

#%% 3T full sequences probability
subjects = [f'{i:02d}' for i in range(1, 41)]

categories = list(settings.trigger_translation.values())

df_3T = pd.DataFrame()

for subject in tqdm(subjects):
    df_probas = bids_utils.load_decoding_3T(subject,
                                test_set='test-seq_long',
                                mask='cv',
                                classifier=categories)
    df_probas = df_probas[df_probas['class'].isin(categories)].reset_index()
    df_seq = bids_utils.load_trial_data_3T(subject, condition='sequence')

    # extract the sequences
    dfs = []
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial'), strict=True):
        assert t1==t2
        seq_labels = list(df_s.stim_label.values[0])
        df_p['serial_position'] = df_p.classifier.apply(lambda x: seq_labels.index(x))
        dfs += [df_p]

    df_subj = pd.concat(dfs)
    df_subj = df_subj.groupby(['tITI', 'serial_position', 'seq_tr']).mean(True).reset_index()
    # for iti, df_iti in df_subj.groupby('tITI'):
    #     plt.figure()
    #     sns.lineplot(df_iti, x='seq_tr', y='probability', hue='serial_position')

    df_3T = pd.concat([df_3T, df_subj], ignore_index=True)

#%% plotting
fig, axs = plt.subplots(5, 2, figsize=[8, 14])
sns.despine(fig)

for i, (isi, df_iti) in enumerate(df_meg.groupby('interval')):
    ax = axs[i, 0]
    sns.lineplot(df_iti, x='timepoint', y='proba', hue='position',
                 palette=settings.palette_wittkuhn1,
                 ax=ax, legend=False)
    ax.vlines(np.arange(5)* (100+isi), *ax.get_ylim(), linewidth=0.5, color='black', alpha=0.3)
    # ax.set_xlim(-200, interval*5+750)
    ax.set_xlabel('')
    ax.set_ylim([0.5, 2])
    ax.set_ylabel('probability')

    # for ax in axs.flat:
    #     ax.set_xticks(np.arange(0, 3000, 500), np.arange(0, 3000, 500)/1000)
    ax.set_title(f'ISI {int(isi)} ms')

for i, (isi, df_iti) in enumerate(df_3T.groupby('tITI')):
    ax = axs[i, 1]
    plot = sns.lineplot(df_iti, x='seq_tr', y='probability', hue='serial_position',
                 palette=settings.palette_wittkuhn1,
                 ax=ax, legend=None)
    ax.set_xticks(np.arange(1, 14, 2))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'ISI {int((float(isi)*1000))} ms')
    plt.pause(0.1)

# plotting.normalize_lims(axs[:, 0])
plotting.normalize_lims(axs[:, 1])

axs[4, 0].set_xlabel('ms after stim onset')
axs[4, 1].set_xlabel('TR after stim onset')


# # Draw the legend outside the rightmost axis
# fig.legend(
#     ['1', '_', '2', '_', '3', '_', '4', '_', '5'],
#     title='item position',
#     loc='center right',
#     bbox_to_anchor=(0.96, 0.6),  # 1.02 puts it just outside the right edge, 0.5 is vertical center
#     bbox_transform=fig.transFigure
# )

ax = axs[-1, 0]
ax.text(
    0.5, 0.5,
    'not recorded',
    ha='center', va='center'
)

# remove xticks and yticks
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout(rect=[0, 0, 0.88, 1])  # leave space on the right for the legend
plotting.savefig(fig, settings.plot_dir + f'/figures/fast_sequences_probabilities.png')
