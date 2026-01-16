# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run SODA (Slope Order Dynamic Analysis) on MEG data

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
import matplotlib.pyplot as plt
import meg_utils
from meg_utils.misc import to_long_df
from meg_utils import sigproc
from meg_utils.plotting import savefig
from meg_utils.decoding import LogisticRegressionOvaNegX
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import soda

#%% settings
normalization = 'lambda x: x/x.mean(0)'
intervals = [32, 64, 128, 512]


stop
#%% SODA on raw trials

fig, axs = plt.subplots(2, 2, figsize=[12, 8])

# zscore = lambda x, axis, nan_policy:x
df = pd.DataFrame()
for subject in tqdm(layout.subjects):
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    slopes = {interval: [] for interval in intervals}

    for proba, df_trial in zip(probas, beh, strict=True):
        interval = df_trial.interval_time.values[0]
        proba = eval(normalization)(proba)
        slopes[interval] += [soda.compute_slopes(proba)]

    for i, (interval, slope) in enumerate(slopes.items()):
        slope = np.array(slope)
        timepoint = np.arange(0, slope.shape[1]*10, 10)
        df_isi = to_long_df(slope, columns = ['trial', 'timepoint'],
                            value_name='slope', timepoint=timepoint)
        df_isi['subject'] = subject
        df_isi['interval'] = interval
        df = pd.concat([df, df_isi], ignore_index=True)

        # ax = axs.flat[i]
        # ax.clear()
        # ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(), linewidth=0.5,
        #           color='black', alpha=0.3, label='image onset')
        # sns.lineplot(df_isi, x='timepoint', y='slope', ax=ax, label='slope')
        # ax.set(title=f'{interval=} ms')

    # fig.suptitle(f'{subject=}')
    # savefig(fig, settings.plot_dir + f'/soda_meg/{subject}_{interval}.png')


for i, (interval, df_isi) in enumerate(df.groupby('interval')):
    ax = axs.flat[i]
    ax.clear()
    ax.set(title=f'{interval=} ms')
    df_isi = df_isi.groupby(['timepoint', 'subject']).mean()
    sns.lineplot(df_isi, x='timepoint', y='slope', ax=ax, label='slope')
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(),
              linewidth=1, color='black', alpha=0.3, label='image onset')

fig.suptitle(f'{len(layout.subjects)=}')
savefig(fig, settings.plot_dir + f'/soda_meg/slopes_{interval}.png')


#%% SODA with gaussian smooth
from scipy.ndimage import gaussian_filter, convolve
from meg_utils import misc

sigma =  1

kernel = np.ones([1, 5, 1])

fig, axs = plt.subplots(2, 2, figsize=[12, 8])

# zscore = lambda x, axis, nan_policy:x
df = pd.DataFrame()
df_proba = pd.DataFrame()

for subject in tqdm(layout.subjects):
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    # add raw and smoothed to dataframe
    df_subj = pd.DataFrame()
    for p, x in zip(probas, beh, strict=True):
        for i, norm in enumerate([normalization, lambda x:x]):
            p_normed = eval(normalization)(p)
            pos_idx = [list(x.trigger).index(i) for i in  range(5)]
            df_trial = misc.to_long_df(p_normed, ['timepoint', 'cls'],
                                       cls={'class': settings.categories,
                                            'position': pos_idx},
                                       timepoint=timepoint,
                                       value_name='probability')
            df_trial['interval'] = x.interval_time.values[0]
            df_trial['trial'] = x.idx.values[0]
            df_trial['subject'] = subject
            df_trial['normalization'] = 'raw' if i else 'mean'
            df_subj = pd.concat([df_subj, df_trial], ignore_index=True)
    df_proba = pd.concat([df_proba, df_subj], ignore_index=True)


    # probas_smooth = gaussian_filter(probas, sigma = [0, sigma, 0], mode='nearest')
    probas_smooth = convolve(probas, kernel, mode='nearest')
    timepoint = np.arange(0, probas_smooth.shape[1]*10, 10)

    # add raw and smoothed to dataframe
    for name, prob in ({'raw': probas, 'smoothed': probas_smooth}).items():
        df_subj = pd.DataFrame()
        for p, x in zip(prob, beh, strict=True):
            p = eval(normalization)(p)
            pos_idx = [list(x.trigger).index(i) for i in  range(5)]
            df_trial = misc.to_long_df(p, ['timepoint', 'cls'],
                                       cls={'class': settings.categories,
                                            'position': pos_idx},
                                       timepoint=timepoint,
                                       value_name='probability')
            df_trial['interval'] = x.interval_time.values[0]
            df_trial['subject'] = subject
            df_trial['type'] = name
            df_subj = pd.concat([df_subj, df_trial], ignore_index=True)
        df_proba = pd.concat([df_proba, df_subj], ignore_index=True)

    slopes = {interval: [] for interval in intervals}

    for proba, df_trial in zip(probas_smooth, beh, strict=True):
        interval = df_trial.interval_time.values[0]
        proba = eval(normalization)(proba)
        slopes[interval] += [soda.compute_slopes(proba)]

    for i, (interval, slope) in enumerate(slopes.items()):
        slope = np.array(slope)
        df_isi = to_long_df(slope, columns = ['trial', 'timepoint'],
                            value_name='slope', timepoint=timepoint)
        df_isi['subject'] = subject
        df_isi['interval'] = interval
        df = pd.concat([df, df_isi], ignore_index=True)

        # ax = axs.flat[i]
        # ax.clear()
        # ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(), linewidth=0.5,
        #           color='black', alpha=0.3, label='image onset')
        # sns.lineplot(df_isi, x='timepoint', y='slope', ax=ax, label='slope')
        # ax.set(title=f'{interval=} ms')

    # fig.suptitle(f'{subject=}')
    # savefig(fig, settings.plot_dir + f'/soda_meg/{subject}_{interval}.png')


for i, (interval, df_isi) in enumerate(df.groupby('interval')):
    ax = axs.flat[i]
    ax.clear()
    ax.set(title=f'{interval=} ms')
    df_isi = df_isi.groupby(['timepoint', 'subject']).mean()
    sns.lineplot(df_isi, x='timepoint', y='slope', ax=ax, label='slope')
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(),
              linewidth=1, color='black', alpha=0.3, label='image onset')

fig.suptitle(f'{len(layout.subjects)=}, {sigma=}')
savefig(fig, settings.plot_dir + f'/soda_meg/slopes_smoothed_{interval}.png')


#%% plot raw probabilities before and after smoothing

plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend

fig, axs = plt.subplots(2, 1, figsize=[12, 8])

df_proba_32 = df_proba[df_proba.interval==32]
for i, (name, df) in enumerate(df_proba_32.groupby('type')):
    ax = axs[i]
    df = df.groupby(['subject', 'timepoint', 'position']).mean(True).reset_index()
    sns.lineplot(df, x='timepoint', y='probability', hue='position',
                 ax=ax, palette=settings.palette_wittkuhn1)
    ax.set_xlim(0, 1300)

axs[0].set_ylim([0.25, 4.3])
savefig(fig, settings.plot_dir + '/smoothed_probabilities.png')
