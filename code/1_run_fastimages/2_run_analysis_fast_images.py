#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:40:24 2024

simply visualize the individual images of the fast images

@author: simon.kern
"""
import sys; sys.path.append('..')
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
from meg_utils import misc

round_to_base = lambda data, base: np.round(data / base) * base

subjects_fmri = [f'{i:02d}' for i in range(1, 41)]
subjects_fmri = [f'{i:02d}' for i in range(1, 31)]

#%% MEG accuracies
df_meg_acc = pd.DataFrame()
df_meg_proba = pd.DataFrame()

for subject in tqdm(layout_MEG.subjects):

    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, df_beh = bids_utils.load_fast_images(subject)

    proba = np.swapaxes([clf.predict_proba(data_x[:, :, t]) for t in range(data_x.shape[-1])], 0, 1)
    preds = np.argmax(proba, -1)

    accs = (preds.T==data_y).T

    df_subj = misc.to_long_df(accs, columns=['trial', 'timepoint'],
                              value_name='accuracy',
                              timepoint=np.arange(-200, 510, 10),
                              trial={'interval': df_beh.interval_time.astype(int),
                                     'serial_position': df_beh.serial_position.astype(int),
                                     'stimulus': df_beh.trigger.astype(int)})
    df_subj = df_subj.groupby(['timepoint', 'interval', 'serial_position', 'stimulus']).mean().reset_index()
    df_subj['subject'] = subject

    df_proba = misc.to_long_df(proba, columns=['trial', 'timepoint', 'classifier'],
                               value_name='probability', classifier=settings.categories,
                               timepoint=np.arange(-200, 510, 10),
                               trial={'interval': df_beh.interval_time.astype(int),
                                      'serial_position': df_beh.serial_position.astype(int),
                                      'stimulus': df_beh.trigger.astype(int)})
    df_proba['subject'] = subject

    df_meg_acc = pd.concat([df_meg_acc, df_subj], ignore_index=True)
    df_meg_proba = pd.concat([df_meg_proba, df_proba], ignore_index=True)

df_meg_acc['stimulus'] = df_meg_acc['stimulus'].apply(lambda x: settings.categories[x])
df_meg_proba['stimulus'] = df_meg_proba['stimulus'].apply(lambda x: settings.categories[x])

#%% fMRI data loading
df_fmri_proba = pd.DataFrame()
for subject in tqdm(subjects_fmri):
    df_subj = bids_utils.load_decoding_fast_images_3T(subject)
    df_fmri_proba = pd.concat([df_fmri_proba, df_subj], ignore_index=True)

# remove slow condition (this is not the oddball)
df_fmri_proba = df_fmri_proba[df_fmri_proba.interval<2048]

#%% normalize probabilities
# we need to normalize the probabilities per subject, per classifier

for (subj, cat), df_sel in df_meg_proba.groupby(['subject', 'classifier']):
    mean_proba = df_sel['probability'].mean()
    df_sel['probability'] = df_sel['probability'] / mean_proba

for (subj, cat), df_sel in df_fmri_proba.groupby(['subject', 'classifier']):
    mean_proba = df_sel['probability'].mean()
    df_sel['probability'] = df_sel['probability'] / mean_proba

df_fmri_proba['timepoint'] = df_fmri_proba['tr_onset']
df_fmri_proba['stimulus'] = df_fmri_proba['stim']

#%% plot probability variants

fig, axs = plt.subplots(2, 2, figsize=[12, 8])

for i, df in enumerate([df_meg_proba, df_fmri_proba]):
    cond = ['MEG', 'fMRI'][i]

    df = df.copy()  # prevent overwriting of original values
    if cond=='MEG':  # convert to seconds
        df.timepoint = df.timepoint/1000
    elif cond=='fMRI':  # round to prevent jitter
        df.timepoint = round_to_base(df.timepoint, 0.5)

    # only show from 0 to 10 seconds (MEG goes until 0.5 anyway)
    df = df[df.timepoint>=0]
    df = df[df.timepoint<=10]

    # only subselect classifiers for the current image being shown
    df = df[df.stimulus==df.classifier]
    ax = axs[i, 0]
    sns.lineplot(df, x='timepoint', y='probability', hue='serial_position',
                 palette=settings.palette_wittkuhn2, ax=ax)
    ax.legend(title='sequence position', loc='upper right')

    ax = axs[i, 1]
    sns.lineplot(df, x='timepoint', y='probability', hue='interval',
                 palette=settings.palette_wittkuhn2, ax=ax)
    ax.legend(title='interval', loc='upper right')

    axs[i, 0].set(title='Serial Position')
    axs[i, 1].set(title='Interval Speed')

    axs[i, 0].set(ylabel=f'{cond}\nProbability (normalized)',
                  xlabel='time after stim onset (s)')
    axs[i, 1].set(ylabel=f'{cond}\nProbability (normalized)',
                  xlabel='time after stim onset (s)')
fig.suptitle('Decoding Fast Images Individually')
plotting.savefig(fig, f'{settings.plot_dir}/figures/fast_images_decoding.png')
asd
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
