#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:38:18 2024

@author: simon.kern
"""
import mne
import sklearn
from tqdm import tqdm
import pandas as pd
from meg_utils import plotting, decoding
from bids import BIDSLayout
from bids_utils import layout, load_localizer, make_bids_fname
from meg_utils.decoding import cross_validation_across_time
import settings
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

#%% Settings
tmin = -0.2
tmax = 1
ex_per_fold = 8
best_C = 4.640625  # previously computed
tsel = 'best'  # either take the best individually timepoint or the mean peak across participants
pkl_localizer_Cs = settings.cache_dir + '/localizer_cs.pkl.gz'

# for subj in ['01', '23', '03', '04', '09', '13', '26']:
for subj in ['01', '03', '04', '09', '13', '23', '26']:
    # somehow, these give errors when loading, so leave them out for now
    # should be included in the final calculation
    # if you see this in a PR, please let me knowm ;-)
    if subj in layout.subjects:
        layout.subjects.remove(subj)

stop
#%% Calculate the best regularization parameter

df_all = pd.DataFrame()
fig, ax = plt.subplots(figsize=[8, 6])

Cs = np.logspace(-4, 4, 25, dtype=np.float16)
Cs = [best_C]  # comment this here to not having to loop over everything
for C in tqdm(Cs, desc='running through C'):
    df_c = pd.DataFrame()
    for i, subject in enumerate(layout.subjects):

        clf = LogisticRegression(penalty='l1', C=C, solver='liblinear')

        # load MEG data from the localizer
        data_x, data_y, _ = load_localizer(subject=subject, verbose=False, tmax=tmax)

        # do cross validation decoding in sensor space across time
        df_subj = cross_validation_across_time(data_x, data_y, subj=subject,
                                               n_jobs=-1, tmin=tmin, tmax=tmax,
                                               ex_per_fold=ex_per_fold, clf=clf,
                                               verbose=False)
        df_subj = df_subj.groupby('timepoint').mean(True).reset_index()
        df_subj['subject'] = subject
        df_subj['C'] = C
        df_c = pd.concat([df_c, df_subj])

    df_all = pd.concat([df_all, df_c], ignore_index=True)
    ax.clear()
    sns.lineplot(data=df_c, x='timepoint', y='accuracy', ax=ax)
    ax.set_title(f'{C=} all participants {len(layout.subjects)=}')
    fig.savefig(settings.plot_dir + f'localizer_C{C:011.6f}.png')
    plt.pause(0.1)

df_all['C'] = df_all['C'].astype(float)
joblib.dump(df_all, pkl_localizer_Cs)

#%% Train a final decoder with the best C and the best T per participant

# load precomputed results from disk
df_all = joblib.load(pkl_localizer_Cs)
df_all['C'] = df_all['C'].astype(float)

# calculate the best C for training a classifier
df_mean = df_all.groupby('timepoint').mean(True).reset_index()
peak_t_idx = df_mean.accuracy.argmax()
peak_t = df_mean.timepoint[peak_t_idx]

df_all_mean = df_all[df_all.timepoint==peak_t].groupby(['C']).mean(True).reset_index()
plt.figure()
ax = sns.lineplot(data=df_all_mean, x='C', y='accuracy')
ax.set(xscale='log')

best_C = df_all_mean.C[df_all_mean.accuracy.argmax()]

df_proba = pd.DataFrame()

for i, subject in enumerate(tqdm(layout.subjects, desc='training classifier')):
    # first calculate the peak decoding timepoint for this participant
    # load data
    data_x, data_y, _ = load_localizer(subject=subject, verbose=False)

    # train classifier and save
    if tsel=='best':
        t = df_all[df_all.subject==subject].groupby('timepoint').mean(True).reset_index().accuracy.argmax()
    elif tsel=='mean':
        t = peak_t_idx

    clf = LogisticRegression(penalty='l1', C=best_C, solver='liblinear')
    clf = decoding.LogisticRegressionOvaNegX(clf, C=best_C)

    clf.fit(data_x[:, :, t], data_y)
    clf_pkl_latest = make_bids_fname('latest', modality='clf', subject=f'sub-{subject}', suffix='clf')
    clf_pkl = make_bids_fname('latest', modality='clf', subject=f'sub-{subject}', suffix='clf',
                              tsel=tsel, C=np.round(best_C, 2), t=t)

    # save once as latest
    decoding.save_clf(clf, clf_pkl_latest, metadata=dict(data_y=data_y, t=t, tsel=tsel))
    # save another time with filename for archiving
    decoding.save_clf(clf, clf_pkl, metadata=dict(data_y=data_y, t=t, tsel=tsel))

    df_subj, probas = cross_validation_across_time(data_x, data_y, subj=subject,
                                                   n_jobs=-1, tmin=tmin, tmax=tmax,
                                                   ex_per_fold=ex_per_fold, clf=clf,
                                                   return_probas=True, verbose=False)

    df_proba_subj = pd.DataFrame()
    timepoint = np.repeat(df_subj.timepoint.unique(), probas.shape[-1])
    for i, (proba, y) in enumerate(zip(probas, data_y, strict=True)):
        label = ['other']*5
        label[y] = 'target'
        df_tmp = pd.DataFrame({'label': np.hstack([label]* probas.shape[1]),
                                      'timepoint': timepoint,
                                      'proba': proba.ravel(),
                                      'stimulus': settings.img_trigger[y]})
        df_proba_subj = pd.concat([df_proba_subj, df_tmp], ignore_index=True)

    df_proba_subj = df_proba_subj.groupby(['label', 'timepoint', 'stimulus']).mean().reset_index()
    df_proba_subj['subject'] = subject
    df_proba = pd.concat([df_proba, df_proba_subj], ignore_index=True)


#%% plot probabilities
fix, axs = plt.subplots(2, 3, figsize=[14, 8])
axs = axs.flatten()
ax_b = axs[-1]
# fig, axs, ax_b = plotting.make_fig(5, bottom_plots=[0, 0, 1], figsize=[12, 8],
                                   # xlabel='Timepoint', ylabel='Probability')

for stim in np.unique(data_y):
    ax = axs[stim]
    stimulus =settings.img_trigger[stim]
    sns.lineplot(df_proba[df_proba.stimulus==stimulus], x='timepoint', style='subject',
                 y='proba', hue='label', ax=ax, alpha=0.1, legend=False)
    sns.lineplot(df_proba[df_proba.stimulus==stimulus], x='timepoint',
                 y='proba', hue='label', ax=ax)
    ax.set_title(stimulus)
    plt.pause(0.1)



df_proba_mean = df_proba.groupby(['timepoint', 'label', 'subject']).mean(True).reset_index()

sns.lineplot(df_proba_mean, x='timepoint', y='proba', hue='label', style='subject',
             ax=ax_b, alpha=0.1, legend=False)
sns.lineplot(df_proba_mean, x='timepoint', y='proba', hue='label', ax=ax_b)
ax_b.set_title('All stimuli')
plotting.normalize_lims(list(axs))
fig.tight_layout()
plt.pause(0.1)
fig.savefig(settings.plot_dir + f'localizer_perclass.png')

#%% plot per participant

df = df_all[df_all.C==best_C]

fix, ax = plt.subplots(1, 1, figsize=[8, 6])


df_mean = df.groupby(['timepoint', 'subject']).mean(True).reset_index()
sns.lineplot(df_mean, x='timepoint', y='accuracy', hue='subject',
             legend=False, palette=sns.dark_palette("#69d", reverse=True),
             alpha=0.2)

df_mean = df.groupby(['timepoint']).mean(True).reset_index()
sns.lineplot(df_mean, x='timepoint', y='accuracy', linewidth=4, color=sns.color_palette()[0])
sns.despine()

ax.hlines(0.2, -200, 2000, linestyle='--', color='gray')


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

    data_x, data_y, _ = load_localizer(subject=subj, verbose=False)

    tps = np.arange(0, data_x.shape[-1])
    res = Parallel(n_jobs=-3)(delayed(get_channel_importances)
                                      (data_x[:, :, tp], data_y, timepoint=tp) for tp in
                                      tqdm(tps, desc='evaluating timepoints'))

    accs = np.array([x[0] for x in res])
    df_acc = pd.DataFrame({'accuracy': accs.ravel(),
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
        # plt.colorbar(plot, ax=ax1)
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
            sns.lineplot(data=df_acc, x='timepoint', y='accuracy', ax=ax3, seed=0)
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
        # if tp==2: asd

    with imageio.get_writer(f'{settings.plot_dir}/electrodes_{subj}.gif',
                            mode='I', fps=4) as writer:
        for filename in png_files:
            image = imageio.imread(filename)
            writer.append_data(image)
