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
from meg_utils.plotting import savefig, normalize_lims
from meg_utils.decoding import LogisticRegressionOvaNegX
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import soda

#%% settings
normalization = 'lambda x: x/x.mean(0)'
intervals = [32, 64, 128, 512]


stop
#%% SODA on raw trials


# zscore = lambda x, axis, nan_policy:x
df = pd.DataFrame()
df_meg = pd.DataFrame()
lengths = []

for subject in tqdm(layout.subjects):
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])
    slopes = {interval: [] for interval in intervals}

    df_probas = probas

    for proba, df_trial in zip(probas, beh, strict=True):
        interval = df_trial.interval_time.values[0]
        proba = eval(normalization)(proba)
        slopes[interval] += [soda.compute_slopes(proba)]

    df_subj = pd.DataFrame()
    timepoint = np.hstack([np.arange(data_x.shape[-1])*10]*5)-200
    stimulus = np.repeat(np.arange(5), probas.shape[1])
    probas = eval(normalization)(probas)

    data_x, data_y, df_beh = bids_utils.load_fast_sequences(subject)

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

    for i, (interval, slope) in enumerate(slopes.items()):
        slope = np.array(slope)

        # truncate slope to length that we care about

        length = np.round(((interval+100)*5 + 200)/10).astype(int)
        lengths += [length]
        slope = slope[:, : length]
        timepoint = np.arange(0, slope.shape[1]*10, 10)[:length] - 200
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

#%% plotting
lengths = np.unique(lengths)
sizes = [np.round(int(((iv/10) + 10))/4).astype(int) for iv in lengths]
mosaic = ''.join([f'{i}'*l for i, l in enumerate (sizes)])
mosaic += '\n' + ''.join([f'{i+5}'*l for i, l in enumerate (sizes)])

fig, axs = plt.subplot_mosaic(mosaic, figsize=[22, 7], dpi=80)

for i, (interval, df_isi) in enumerate(df_meg.groupby('interval')):
    ax = list(axs.values())[i]
    ax.clear()
    ax.set(title=f'{interval} ms')
    length = lengths[i]
    df_isi = df_isi.groupby(['timepoint', 'subject', 'position']).mean().reset_index()
    df_isi = df_isi[df_isi.timepoint<length*10]
    sns.lineplot(df_isi, x='timepoint', y='proba', hue='position', ax=ax,
                 palette=settings.palette_wittkuhn1)
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(),
              linewidth=1, color='black', alpha=0.3, label='image onset')
    ax.get_legend().remove()
    ax.set_ylabel('' if i>0 else 'probability (normalized)')

for i, (interval, df_isi) in enumerate(df.groupby('interval')):
    ax = list(axs.values())[i+4]
    ax.clear()
    ax.axhline(0, linestyle='--', alpha=0.3, c='black')
    ax.set(title='')#f'{interval} ms')
    df_isi = df_isi.groupby(['timepoint', 'subject']).mean()
    sns.lineplot(df_isi, x='timepoint', y='slope', ax=ax, label='slope')
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim(),
              linewidth=1, color='black', alpha=0.3, label='image onset')
    ax.get_legend().remove()
    if i>0:
        ax.set_ylabel('')

normalize_lims(list(axs.values())[:4], which='y')
normalize_lims(list(axs.values())[4:], which='y')

for i in range(4):
    normalize_lims([axs[f'{i}'], axs[f'{i+5}'] ], which='x')

# hide y-tick labels for non-first axes per row (like sharey)
for ax in list(axs.values())[1:4]:
    ax.set_yticklabels([])
for ax in list(axs.values())[5:8]:
    ax.set_yticklabels([])


# figure-level legends at the top, next to title
upper_ax = list(axs.values())[0]
upper_handles, upper_labels = upper_ax.get_legend_handles_labels()
leg1 = fig.legend(upper_handles, upper_labels, title='position', ncol=3,
                   loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)

lower_ax = list(axs.values())[4]
lower_handles, lower_labels = lower_ax.get_legend_handles_labels()
# place second legend below the first, using the first legend's bbox
fig.canvas.draw()
leg1_bbox = leg1.get_window_extent().transformed(fig.transFigure.inverted())
fig.legend(lower_handles, lower_labels, ncol=2,
           loc='upper right', bbox_to_anchor=(0.99, leg1_bbox.y0), frameon=True)

fig.suptitle(f'SODA on MEG', x=0.5, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.96])
savefig(fig, settings.plot_dir + f'/soda_meg/soda_on_meg.png', tight=False)


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
