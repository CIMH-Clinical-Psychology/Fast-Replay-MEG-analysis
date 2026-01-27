#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:38:18 2024

run localizer cross validation of decoding accuracy on MEG data.

We will first run the localizer as has been done in previous studies.
That means we will use global parameters for the classifiers, i.e. train each
classifier on the same timepoint and with the same regularization parameter.

then later, we will look at individually optimized parameters that are
subject-specific, additionally combine with PCA and/or time embeddings or more
sophisticated stuff.

Input: preprocessed MEG data
Output: - Final Classifier per participant
        - a csv file with cross-validated localizer trials

@author: simon.kern
"""

import os
import sys; sys.path.append('..')
import mne
import logging
import sklearn
from tqdm import tqdm
import pandas as pd
from meg_utils import plotting, decoding
from bids import BIDSLayout
from bids_utils import layout_MEG as layout
from bids_utils import load_localizer, make_bids_fname
from meg_utils.decoding import cross_validation_across_time, LogisticRegressionOvaNegX
import settings
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

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
# Ignore only the specific convergence warning from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#%% Settings
tmin = -0.2
tmax = 0.8
ex_per_fold = 8
best_C = 4.640625  # previously computed
n_subjects = len(layout.subjects)
pkl_localizer_Cs = settings.cache_dir + '/localizer_cs.pkl.gz'
pkl_probas = settings.cache_dir + '/localizer_probas.pkl.gz'
pkl_accs = settings.cache_dir + '/localizer_accs.pkl.gz'

stop
#%% Calculate the best regularization parameter

# we have a grid of values and evaluate them. Then we choose the peak
# and run an evaluation of the values around the peak.
# repeat n times and get an optimal value, similar to iterative halving
n_iterations = 5       # zoom in this many times
values_per_iter = 7    # evaluate 7 values per iteration
bounds = [0.1, 100]  # start with these bounds
ex_per_fold = 5

df = pd.DataFrame()
fig, ax = plt.subplots(figsize=[8, 6])

tqdm_loop = tqdm(total=n_iterations*n_subjects, desc='cross-validating')

def _run_one(subject, C, data_x, data_y):
    np.random.seed(int(subject))
    clf = LogisticRegressionOvaNegX(l1_ratio=1, C=C, solver='liblinear', max_iter=1000)

    df = cross_validation_across_time(
        data_x, data_y, subj=subject, n_jobs=2, tmin=tmin, tmax=tmax,
        ex_per_fold=ex_per_fold, clf=clf, verbose=False
    ).groupby('timepoint').mean(True).reset_index()

    df['subject'] = subject
    df['C'] = np.round(C, 3)
    df['iteration'] = iteration
    return df

for iteration in range(n_iterations):

    params = np.linspace(*bounds, values_per_iter)
    df_iter = pd.DataFrame()
    for subject in layout.subjects:
        # Parallelize over C for each subject
        data_x, data_y, _ = load_localizer(subject=subject, verbose=False)
        data_x, data_y = decoding.stratify(data_x, data_y)
        res =  Parallel(n_jobs=-1)(delayed(_run_one)(subject, C, data_x, data_y) for C in params)
        df_iter = pd.concat([df_iter, pd.concat(res)], ignore_index=True)
        tqdm_loop.update()

    ax.clear()
    df_mean = df_iter.groupby(['timepoint', 'subject', 'C']).mean(True).reset_index()
    sns.lineplot(data=df_mean, x='timepoint', y='accuracy', hue='C', ax=ax, legend='full')
    ax.set_title(f'{bounds=} all participants {len(layout.subjects)=}')
    plotting.savefig(fig, settings.plot_dir + f'/regularization/localizer_iteration-{iteration}_bounds-{bounds[0]:0.3f}-{bounds[1]:0.3f}.png')

    # calculate bounds for next iteration
    df_mean = df_iter.groupby(['timepoint', 'C']).mean(True).reset_index()
    best_c = df_mean.C[df_mean.accuracy.argmax()]
    bound_min = params[max(np.argmin(abs(params-best_c))-1, 0)]
    bound_max = params[min(np.argmin(abs(params-best_c))+1, len(params)-1)]
    bounds = [bound_min, bound_max]
    print(f'{best_c=}, new {bounds=}')
    telegram_send.send(messages=[f'{iteration=} {best_c=}, new {bounds=}'])
    df = pd.concat([df, df_iter], ignore_index=True)

# save for later use
joblib.dump(df, pkl_localizer_Cs)

# plot final result
df_mean = df.groupby(['timepoint', 'C']).mean(True).reset_index()
best_c = df_mean.C[df_mean.accuracy.argmax()]

pdf_mean = df[df.C==best_c].groupby(['timepoint', 'subject']).mean(True).reset_index()

ax.clear()
sns.lineplot(data=df_mean, x='timepoint', y='accuracy', hue='C', ax=ax)
ax.set_title(f'{best_c=} all participants {len(layout.subjects)=}')
plotting.savefig(fig, settings.plot_dir + f'/regularization/localizer_best.png')
plt.pause(0.1)

#%% plot peak C regularization
df_all = joblib.load(pkl_localizer_Cs)
best_c = df_mean.C[df_mean.accuracy.argmax()]

fig, ax = plt.subplots(figsize=[8, 6])
sns.lineplot(df_all, x='C', y='accuracy', hue='subject', ax=ax)

plotting.savefig(fig, settings.plot_dir + f'/regularization/localizer_regularization_overview.png')
plotting.savefig(fig, settings.plot_dir + f'/supplement/localizer_regularization_gridsearch.png')
#%% Train final decoders with the overall best C and timepoint

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

best_C = df_all_mean.C[df_all_mean.accuracy.argmax()].round(1)

df_proba = pd.DataFrame()

for i, subject in enumerate(tqdm(layout.subjects, desc='training classifier')):
    # first calculate the peak decoding timepoint for this participant
    # load data
    data_x, data_y, _ = load_localizer(subject=subject, verbose=False)

    if len(set(np.bincount(data_y)))!=1:
        logging.info(f'stratifying for {subject=} as unequal classes found')
        data_x, data_y = decoding.stratify(data_x, data_y)

    clf_base = LogisticRegression(l1_ratio=1, C=best_C, solver='liblinear')
    clf = decoding.LogisticRegressionOvaNegX(clf_base, C=best_C)

    clf.fit(data_x[:, :, peak_t_idx], data_y)
    clf_pkl_latest = make_bids_fname('latest', modality='clf', subject=f'sub-{subject}', suffix='clf')
    clf_pkl = make_bids_fname('latest', modality='clf', subject=f'sub-{subject}', suffix='clf',
                              C=best_C, t=peak_t_idx)
    # save once as latest
    decoding.save_clf(clf, clf_pkl_latest, metadata=dict(data_y=data_y, t=peak_t_idx))
    # save another time with filename for archiving
    decoding.save_clf(clf, clf_pkl, metadata=dict(data_y=data_y, t=peak_t_idx))

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

joblib.dump(df_proba, pkl_probas)

#%% FIGURE plot probabilities
fig, axs = plt.subplots(2, 3, figsize=[14, 6])
sns.despine(fig)
axs = axs.flatten()
# fig, axs, ax_b = plotting.make_fig(5, bottom_plots=[0, 0, 1], figsize=[12, 8],
                                   # xlabel='Timepoint', ylabel='Probability')

for stim in np.unique(data_y):
    ax = axs[stim]
    stimulus = settings.img_trigger[stim]
    stim_name = settings.trigger_translation[stimulus]
    ax.hlines(0.2, -200, 500, color='gray', linestyle='--')
    sns.lineplot(df_proba[df_proba.stimulus==stimulus], x='timepoint',  y='proba', hue='label', ax=ax)
    ax.set(title=stim_name, ylabel='probability', xlabel='ms after stim onset', xlim=[-100, 500])
    plt.pause(0.1)

# last, mean of all
ax = axs[-1]
df_proba_mean = df_proba.groupby(['timepoint', 'label', 'subject']).mean(True).reset_index()
ax.hlines(0.2, -200, 500, color='gray', linestyle='---', linewidths=1)
sns.lineplot(df_proba_mean, x='timepoint',  y='proba', hue='label', ax=ax)
ax.set(title=stim_name, ylabel='probability', xlabel='ms after stim onset', xlim=[-100, 500])

plotting.normalize_lims(list(axs))
plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_MEG_probabilities.png')

#%% FIGURE plot accuracy

fig, ax = plt.subplots(1, 1, figsize=[8, 6])

clf_base = LogisticRegression(penalty='l1', C=4.6, solver='liblinear')
clf = LogisticRegressionOvaNegX(clf_base, C=4.6)
df = pd.DataFrame()

for i, subject in enumerate(tqdm(layout.subjects, desc='training classifier')):
    # first calculate the peak decoding timepoint for this participant
    # load data

    data_x, data_y, _ = load_localizer(subject=subject, verbose=False)

    if len(set(np.bincount(data_y)))!=1:
        logging.info(f'stratifying for {subject=} as unequal classes found')
        data_x, data_y = decoding.stratify(data_x, data_y)

    df_subj = cross_validation_across_time(data_x, data_y, clf=clf,
                                           ex_per_fold=1, n_jobs=-1, subj=subject)
    df_subj = df_subj.groupby('timepoint').mean(True).reset_index()
    df_subj
    df = pd.concat([df, df_subj], ignore_index=True)

joblib.dump(df, settings.plot_dir + '/pkl/localizer_meg_accuracy.pkl.gz')

sns.lineplot(df, x='timepoint', y='accuracy', ax=ax)
ax.hlines(0.2, -100, 500, color='black', alpha=0.5, linestyle='--')
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
ax.set(xlabel='ms after stim onset', xticks=[-100, 0, 100, 200, 300, 400, 500], title='Decoding Accuracy')
sns.despine()
plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_MEG_accuracy.png')

#%% SUPPL: plot accuracy per participant


clf_base = LogisticRegression(penalty='l1', C=4.6, solver='liblinear')
clf = LogisticRegressionOvaNegX(clf_base, C=4.6)

# calculate accuracies over time via cross validation
df_acc = pd.DataFrame()
for subject in tqdm(layout.subjects):
    data_x, data_y, _ = load_localizer(subject=subject, verbose=False)

    if len(set(np.bincount(data_y)))!=1:
        logging.info(f'stratifying for {subject=} as unequal classes found')
        data_x, data_y = decoding.stratify(data_x, data_y)

    df_subj = cross_validation_across_time(data_x, data_y, clf=clf,
                                           ex_per_fold=1, n_jobs=-1,
                                           subj=subject)
    df_acc = pd.concat([df_acc, df_subj], ignore_index=True)

joblib.dump(df_acc, pkl_accs)

# next plotting
df_acc = joblib.load(pkl_accs)
fig, axs = plt.subplots(8, 4, figsize=[8, 10], sharex=True, sharey=True, dpi=70)
fig.suptitle('Decoding accuracy per participant')
sns.despine(fig)

for i, (subject, df_subj) in enumerate(tqdm(df_acc.groupby('subject'))):
    ax = axs.flat[i]
    ax.vlines(0, 0, 0.75,  color='black')
    sns.lineplot(df_subj, x='timepoint', y='accuracy', legend=False, ax=ax)
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([-100, 0, 100, 200, 300, 400, 500, 600])
    # ax.hlines(0.2, *ax.get_xlim(), linestyle='--', color='black', alpha=0.5)
    ax.grid('on')
    ax.set(title=f'{subject}')

# for ax in axs.flat[30:]: ax.set_axis_off()
plotting.normalize_lims(axs.flat[:30])
plotting.savefig(fig, settings.plot_dir + '/supplement/localizer_accuracies.png')
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


#%% make combined plot for MEG and fMRI to compare timescales
import bids_utils


# first load the
df_3T = pd.DataFrame()
for subject in tqdm([f'{s:02d}' for s in np.arange(1, 40)]):

    df_odd = bids_utils.load_decoding_seq_3T(subject, test_set='test-odd_long', classifier='log_reg')
    df_odd.tr_onset = df_odd.tr_onset.round(1)
    df_odd['target'] = df_odd['class']==df_odd['stim']
    df_odd['accuracy'] = df_odd['pred_label']==df_odd['stim']
    df_mean = df_odd.groupby(['tr_onset']).mean(True).reset_index()
    df_mean['subject'] = subject
    df_3T = pd.concat([df_3T, df_mean], ignore_index=True)



df_meg = joblib.load(settings.plot_dir + '/pkl/localizer_meg_accuracy.pkl.gz')

sns.lineplot(df, x='tr_onset', y='accuracy', ax=ax)
ax.hlines(0.2, -0.6, 8, color='black', alpha=0.5, linestyle='--')
ax.set(xlim=[-0.6, 8], xlabel='seconds after stim onset', title='Decoding accuracy')
# ax.set_xticks(np.arange(1, 8), [f'TR {x}\n{x*1250}' for x in range (1, 8)])
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
sns.despine()

plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_3T_accuracy.png')
