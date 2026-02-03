#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 13:24:40 2026

Visualize the results of the l1 value gridsearch

@author: simon.kern
"""
import os
import sys; sys.path.append('..')
import mne
import json
import logging
import sklearn
from tqdm import tqdm
import pandas as pd
from meg_utils import plotting, decoding
from bids import BIDSLayout
from bids_utils import layout_MEG as layout
from bids_utils import load_localizer, load_fast_images, make_bids_fname
from meg_utils.decoding import cross_validation_across_time, LogisticRegressionOvaNegX
from meg_utils import misc
import settings
from settings import plot_dir
from mne_bids import BIDSPath
from mne_bids import update_sidecar_json
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning
import telegram_send

plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend


#%% first load all CSVs

condition = 'slow'
deriv = layout.derivatives['derivatives']
files_proba = deriv.get(task='main',
                 acquisition=condition,
                 proc='gridsearch',
                 extension='.csv.pkl.gz',
                 suffix='probas',
                 invalid_filters='allow')

files_acc = deriv.get(task='main',
                 acquisition=condition,
                 proc='gridsearch',
                 extension='.csv.pkl.gz',
                 suffix='accuracy',
                 invalid_filters='allow')

assert len(files_acc)==30
assert len(files_proba)==30

df_acc = pd.concat([pd.read_pickle(f) for f in files_acc], ignore_index=True)
df_proba = pd.concat([pd.read_pickle(f) for f in tqdm(files_proba, 'loading csvs')], ignore_index=True)

#%% calculate the mean best C value per participant

best_l1s = []
best_ts = []
subjs = ['avg']
for subj, df_subj in df_acc.groupby('subject'):
    mean_acc = df_subj.groupby('C').accuracy.mean().reset_index()
    best_l1 = mean_acc.C[mean_acc.accuracy.argmax()]
    best_t_idx= df_subj[df_subj.C==best_l1].accuracy.argmax()
    best_t = df_subj[df_subj.C==best_l1].timepoint.iloc[best_t_idx]
    best_l1s += [best_l1]
    best_ts += [best_t]
    subjs += [subj]

best_l1s.insert(0, np.mean(best_l1s))
best_ts.insert(0, np.mean(best_t))

fig, axs = plt.subplots(2, 1, figsize=[8, 6])

ax = axs[0]
sns.barplot(y=best_l1s, x=subjs, ax=ax)
ax.set(xlabel='participant', ylabel='best l1 value (C)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.axhline(best_l1s[0], c='k', label='average', linestyle='--')
ax.legend()

ax = axs[1]
sns.barplot(y=best_ts, x=subjs, ax=ax)
ax.set(xlabel='participant', ylabel='best decoding timepoint')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.axhline(best_ts[0], c='k', label='average', linestyle='--')
ax.legend()

plotting.savefig(fig, f'{plot_dir}/localizer_l1-timepoints.png')

#%% Plot mean decoding accuracy across all participants (using their best L1)

# Build accuracy timecourses per subject using their best L1
subjects_sorted = sorted(df_acc.subject.unique())
timepoints = sorted(df_acc.timepoint.unique())
n_subjs = len(subjects_sorted)

acc_per_subj = {}
for i, (subj, df_subj) in enumerate(df_acc.groupby('subject')):
    best_l1 = best_l1s[i + 1]  # +1 because index 0 is 'avg'
    df_best = df_subj[df_subj.C == best_l1]
    acc_per_subj[subj] = df_best.set_index('timepoint')['accuracy']

# Stack all accuracy timecourses into array
acc_matrix = np.array([acc_per_subj[subj].reindex(timepoints).values
                       for subj in subjects_sorted])

mean_acc = np.nanmean(acc_matrix, axis=0)
sem_acc = np.nanstd(acc_matrix, axis=0) / np.sqrt(n_subjs)

fig, ax = plt.subplots(figsize=[10, 6])

# Plot individual traces (light)
for i, subj in enumerate(subjects_sorted):
    ax.plot(timepoints, acc_matrix[i], 'b-', alpha=0.2, linewidth=0.8)

# Plot mean with SEM shading
ax.plot(timepoints, mean_acc, 'b-', linewidth=2, label='Mean')
ax.fill_between(timepoints, mean_acc - sem_acc, mean_acc + sem_acc,
                color='b', alpha=0.3, label='SEM')

ax.axhline(1/5, color='gray', linestyle='--', label='chance (1/5)')
ax.axvline(0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('Timepoint (s)')
ax.set_ylabel('Decoding Accuracy')
ax.set_title('Mean decoding accuracy across participants (each with their best L1)')
ax.set_ylim([0, 1])
ax.legend()
plt.tight_layout()
plt.savefig('decoding_accuracy_mean_best_l1.png', dpi=150)
plotting.savefig(fig, f'{plot_dir}/localizer_decoding-all.png')

#%% Peak decoding accuracy vs L1 value (rounded to 1 decimal)

# Round L1 values to 1 decimal place
df_acc_rounded = df_acc.copy()
df_acc_rounded['C_rounded'] = df_acc_rounded['C'].round(1)

# Find the best timepoint (peak) per subject
best_timepoint_per_subj = {}
for subj, df_subj in df_acc_rounded.groupby('subject'):
    # Average accuracy across L1 values per timepoint, find peak
    mean_acc_per_tp = df_subj.groupby('timepoint').accuracy.mean()
    best_timepoint_per_subj[subj] = mean_acc_per_tp.idxmax()

# Extract accuracy at peak timepoint for each L1 value per subject
peak_acc_data = []
for subj, df_subj in df_acc_rounded.groupby('subject'):
    best_tp = best_timepoint_per_subj[subj]
    df_peak = df_subj[df_subj.timepoint == best_tp]
    for _, row in df_peak.iterrows():
        peak_acc_data.append({
            'subject': subj,
            'C_rounded': row['C_rounded'],
            'accuracy': row['accuracy']
        })

df_peak = pd.DataFrame(peak_acc_data)

# Plot using seaborn
fig, ax = plt.subplots(figsize=[10, 6])
sns.lineplot(data=df_peak, x='C_rounded', y='accuracy', ax=ax,
             errorbar='se', marker='o')
ax.axhline(1/5, color='gray', linestyle='--', label='chance (1/5)')
ax.set_xlabel('L1 regularization value (C, rounded)')
ax.set_ylabel('Peak decoding accuracy')
ax.set_title('Peak decoding accuracy vs L1 value')
ax.legend()
plt.tight_layout()
plt.savefig('peak_accuracy_vs_l1.png', dpi=150)

#%% Heatmap per participant: L1 value (y) x timepoint (x) with accuracy as color

# Determine global vmin/vmax across all participants
vmin = 0.2
vmax = 0.8

subjects_sorted = sorted(df_acc.subject.unique())

fig, ax = plt.subplots(figsize=[12, 8])

for i, subj in enumerate(tqdm(subjects_sorted, desc='Creating heatmaps')):
    df_subj = df_acc[df_acc.subject == subj]

    # Pivot to create heatmap matrix: rows=C values, cols=timepoints
    heatmap_data = df_subj.pivot_table(index='C', columns='timepoint', values='accuracy')
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Create figure
    ax.clear()
    sns.heatmap(heatmap_data, ax=ax, cmap='viridis', vmin=vmin, vmax=vmax,
                cbar=True if i==0 else False, cbar_kws={'label': 'Decoding Accuracy'})

    ax.set_title(f'Subject {subj} - Decoding accuracy (vmin={vmin:.2f}, vmax={vmax:.2f})')
    ax.set_xlabel('Timepoint (ms)')
    ax.set_ylabel('C (L1 regularization)')

    plotting.savefig(fig, f'{plot_dir}/heatmap_l1/accuracy_heatmap_{subj}.png')

# Also create mean heatmap across all participants
heatmap_mean = df_acc.pivot_table(index='C', columns='timepoint', values='accuracy', aggfunc='mean')
heatmap_mean = heatmap_mean.sort_index(ascending=False)

ax.clear()
sns.heatmap(heatmap_mean, ax=ax, cmap='viridis', vmin=vmin, vmax=vmax, cbar=False,
            cbar_kws={'label': 'Decoding Accuracy'})

ax.set_title(f'Mean decoding accuracy across participants (n={len(subjects_sorted)})')
ax.set_xlabel('Timepoint (ms)')
ax.set_ylabel('C (L1 regularization)')

plotting.savefig(fig, f'{plot_dir}/heatmap_l1.png')
plt.close(fig)

#%% FIGURE plot probabilities per class

# Use best C value (average across participants)
best_C = round(best_l1s[0]*2)/2
df_proba_best = df_proba[df_proba.C == best_C]

# Get unique trial labels from the proba dataframe
trial_labels = sorted(df_proba_best.trial_label.unique())
n_labels = len(trial_labels)

fig, axs = plt.subplots(2, 3, figsize=[14, 8])
sns.despine(fig)
axs = axs.flatten()

for i, trial_label in enumerate(trial_labels):
    ax = axs[i]
    # Get stimulus name from settings if available
    try:
        stim_name = settings.trigger_translation.get(trial_label, str(trial_label))
    except:
        stim_name = str(trial_label)

    # Filter data for this trial label
    df_stim = df_proba_best[df_proba_best.trial_label == trial_label]

    # Plot probability timecourse for each predicted class
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.7, label='chance')
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    sns.lineplot(data=df_stim, x='timepoint', y='proba', hue='class_idx',
                 errorbar='se', ax=ax, legend=(i == 0))
    ax.set(title=f'True: {stim_name}', ylabel='Probability', xlabel='ms after onset')
    ax.set_ylim([0, 1])

# Last subplot: mean probability for correct class (target) vs others
ax = axs[-1]

# Create target column: 1 if class_idx matches trial_label, 0 otherwise
df_proba_target = df_proba_best.copy()
df_proba_target['is_target'] = df_proba_target['trial_label'] == df_proba_target['class_idx']
df_proba_target['class_type'] = df_proba_target['is_target'].map({True: 'Target', False: 'Other'})

# Average probability for target vs other classes
ax.axhline(0.2, color='gray', linestyle='--', alpha=0.7, label='chance')
ax.axvline(0, color='k', linestyle='-', alpha=0.3)
sns.lineplot(data=df_proba_target, x='timepoint', y='proba', hue='class_type',
             errorbar='se', ax=ax, palette={'Target': 'green', 'Other': 'red'})
ax.set(title='Target vs Other classes', ylabel='Probability', xlabel='ms after onset')
ax.set_ylim([0, 1])
ax.legend()

plt.suptitle('Class probabilities over time (mean across participants)')
plt.tight_layout()
plt.savefig('localizer_MEG_probabilities.png', dpi=150)


#%% SUPPL: plot accuracy per participant

# Use the already loaded df_acc, filter by best C value
df_acc_best = df_acc[df_acc.C == best_C]

fig, axs = plt.subplots(6, 5, figsize=[18, 14], sharex=True, sharey=True, dpi=70)
fig.suptitle('Decoding accuracy per participant')
sns.despine(fig)

for i, (subject, df_subj) in enumerate(tqdm(df_acc_best.groupby('subject'))):
    ax = axs.flat[i]
    ax.vlines(0, 0, 0.75, color='black')
    sns.lineplot(df_subj, x='timepoint', y='accuracy', legend=False, ax=ax)
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([-100, 0, 100, 200, 300, 400, 500, 600])
    ax.grid('on')
    ax.set(title=f'{subject}')

for ax in axs.flat[30:]: ax.set_axis_off()
plotting.normalize_lims(axs.flat[:30])
plotting.savefig(fig, settings.plot_dir + '/supplement/fastimages_accuracies.png')
#%% create GIF of classifiers
import imageio
from joblib import Parallel, delayed
from meg_utils.decoding import get_channel_importances
# dummy load a file to plot sensors with, do not use the data which is included
stim = imageio.imread('../data/Haus.jpg')
fixation = imageio.imread('../data/fixation.png')

plt.close('all')
vectorview = mne.channels.read_layout('Vectorview-all')


fig = plt.figure(figsize=[7,10],constrained_layout=False)
gs = fig.add_gridspec(nrows=3, ncols=3)

ax1 = fig.add_subplot(gs[:-1, :])
ax2 = fig.add_subplot(gs[-1, 0])
ax3 = fig.add_subplot(gs[-1, 1:])

for i, (subj) in enumerate(tqdm(layout.subjects, desc='subj')):

    data_x, data_y, _ = load_fast_images(subject=subj, acquisition='fast', verbose=False)

    tps = np.arange(0, data_x.shape[-1])
    res = Parallel(n_jobs=-3)(delayed(get_channel_importances)
                                      (data_x[:, :, tp], data_y, timepoint=tp) for tp in
                                      tqdm(tps, desc='evaluating timepoints'))

    accs = np.array([x[0] for x in res])
    df_acc_gif = pd.DataFrame({'accuracy': accs.ravel(),
                               'timepoint': np.repeat(np.arange(-200, 810, 10), 5)})
    importances = np.array([x[1] for x in res])

    vmin = np.min(importances)
    vmax = np.max(importances)

    png_files = []
    for tp, (acc, imp) in tqdm(enumerate(res), desc='creating plot'):
        sizes =  (15*(imp - np.min(imp))/vmax)**1.5
        ms = tp*10-200
        for ax in [ax1, ax2]:
            ax.clear()

        plot = ax1.scatter(*vectorview.pos[:,:2].T, s=sizes, c=imp, cmap='Reds', vmin=vmin/2, vmax=vmax)
        ax1.set_title(f'Sensor importance for classification at {ms=}')
        ax1.add_patch(plt.Circle((0.475, 0.45), 0.47, color='black', fill=False))
        ax1.add_patch(plt.Circle((0.25, 0.85), 0.03, color='black', fill=False))
        ax1.add_patch(plt.Circle((0.7, 0.85), 0.03, color='black', fill=False))
        ax1.add_patch(plt.Polygon([[0.425, 0.9], [0.475, 0.95], [0.525, 0.9]],fill=False, color='black'))
        ax1.set_axis_off()

        ax2.set_axis_off()
        ax2.imshow(np.zeros([16,16]))
        ax2.set_title('Participant screen')
        if tp<20:
            ax2.imshow(fixation)
        else:
            ax2.imshow(stim, cmap='gray')

        if tp==0:
            ax3.clear()
            ax3.hlines(0.2, -200, 800, linestyle='--', color='gray', alpha=0.5)
            sns.lineplot(data=df_acc_gif, x='timepoint', y='accuracy', ax=ax3, seed=0)
            ax3.set_xlabel('timepoint (ms)')
            tp_indicator = ax3.vlines(ms, *ax3.get_ylim(), color='red')
            ax3.set_title('Decodability after item presentation')
            ax3.set_ylim(0,0.8)
        else:
            tp_indicator.remove()
            tp_indicator = ax3.vlines(ms, *ax3.get_ylim(), color='red')


        plt.tight_layout()
        plt.pause(0.01)
        png = f'{settings.plot_dir}/TMP_importances_{tp:03}.png'
        plt.savefig(png)
        png_files.append(png)

    with imageio.get_writer(f'{settings.plot_dir}/electrodes_{subj}.gif',
                            mode='I', fps=4) as writer:
        for filename in png_files:
            image = imageio.imread(filename)
            writer.append_data(image)
