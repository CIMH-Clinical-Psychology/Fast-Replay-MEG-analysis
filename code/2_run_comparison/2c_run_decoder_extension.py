#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:16:00 2025

try out how we can increase the AUC of the decoding of fast images, so that
the decoding time courses become overlapping

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
from joblib import Parallel, delayed
import bids_utils
from meg_utils.decoding import LogisticRegressionOvaNegX
from mne.decoding import SlidingEstimator
from sklearn.ensemble import RandomForestClassifier
from settings import layout_MEG as layout

for subj in ['01', '23', '03', '04', '09', '13', '26']:
    # somehow, these give errors when loading, so leave them out for now
    # should be included in the final calculation
    # if you see this in a PR, please let me knowm ;-)
    if subj in layout.subjects:
        layout.subjects.remove(subj)

intervals = [32, 64, 128]
clf = LogisticRegressionOvaNegX(penalty='l1', C=4.6, solver='liblinear')
localizer = {}
fast_images = {}

for subject in tqdm(layout.subjects, desc='preloading data'):
    localizer[subject] = bids_utils.load_localizer(subject)
    fast_images[subject] = bids_utils.load_fast_images(subject)

def train_stacked(clf, train_x, train_y):
    if train_x.ndim==3:
        train_x = np.hstack(train_x).T
    return clf.fit(train_x, train_y)

def predict_across_time(clf, test_x):
    return np.transpose([clf.predict(test_x[:, :, t]) for t in range(test_x.shape[-1])])

stop
#%% training across various time ranges


df_ranges = pd.DataFrame()

times = np.arange(71) *10 -200
ranges = ([35, 36], [30, 40], [45, 50], [45, 55], [50, 65], [30, 60])
clf = RandomForestClassifier(2000)

p = Parallel(-1)
for t1, t2 in tqdm(ranges, 'calculating ranges'):
    clfs = p(delayed(clf.fit)(localizer[subj][0][:, :, t1:t2].transpose(0, 2, 1).reshape(-1, 306, order='F'),
                              np.hstack([localizer[subj][1]]*(t2-t1))) for subj in layout.subjects)

    preds = p(delayed(predict_across_time)(clf, fast_images[subj][0]) for clf, subj in zip(clfs, layout.subjects))
    accs = [(pred.T==fast_images[subj][1]).mean(1) for pred, subj in zip(preds, layout.subjects)]
    df_tmp = pd.DataFrame({'accuracy': np.ravel(accs),
                           'time': np.hstack([times]*len(accs)),
                           'subject': np.repeat(layout.subjects, len(times)),
                           'range': f'{10*(t1-20)}-{10*(t2-20)}'})
    df_ranges = pd.concat([df_ranges, df_tmp])


fig, ax = plt.subplots(1, 1)
sns.lineplot(df_ranges, x='time', y='accuracy', hue='range', ax=ax)
ax.set_title(f'{clf=} time ensemble training on fast images')
ax.vlines([(i+100)+150 for i in  intervals], *ax.get_ylim(), linestyle='--', color='gray')

#%% train across all time ranges, evaluate at 32/64/128ms image peak
clf = RandomForestClassifier(2000)
heatmaps = {interval:np.zeros([51, 51])+0.2 for interval in intervals}
loop = tqdm(total = 50*25)

for t1 in range(20, 70):
    for t2 in range(t1+1, 71):
        clfs = p(delayed(clf.fit)(localizer[subj][0][:, :, t1:t2].transpose(0, 2, 1).reshape(-1, 306, order='F'),
                                  np.hstack([localizer[subj][1]]*(t2-t1))) for subj in layout.subjects)
    preds = p(delayed(predict_across_time)(clf, fast_images[subj][0]) for clf, subj in zip(clfs, layout.subjects))
    accs = [(pred.T==fast_images[subj][1]).mean(1) for pred, subj in zip(preds, layout.subjects)]

    for interval in intervals:
        heatmaps[interval][t1-20, t2-20] = np.mean(accs, 0)[20 + (10+interval//10) + 15 ]
    loop.update()
joblib.dump(heatmaps, settings.cache_dir + '/heatmaps-fastimages.pkl.zip')


heatmaps = joblib.load(settings.cache_dir + '/heatmaps-fastimages.pkl.zip')
fig, axs  = plt.subplots(1, 3)

for i, (interval, heatmap) in enumerate(heatmaps.items()):
    ax = axs[i]
    ax.set_title(f'{interval=}')
    ax.imshow(heatmap, origin='lower')
    ax.set_xlabel('t2')
    ax.set_ylabel('t1')
    ax.set_xticks(np.arange(0, 50, 10), np.arange(0, 500, 100))
    ax.set_yticks(np.arange(0, 50, 10), np.arange(0, 500, 100))

#%% train with temporal voting
from meg_utils.decoding import TimeEnsembleVoting

df_ranges = pd.DataFrame()

times = np.arange(71) *10 -200
ranges = ([35, 36], [30, 40], [45, 50], [45, 55], [50, 65], [30, 60])

clf_vote = TimeEnsembleVoting(clf, voting='hard')

p = Parallel(-1)
for t1, t2 in tqdm(ranges, 'calculating ranges'):
    clfs = p(delayed(clf_vote.fit)(localizer[subj][0][:, :, t1:t2],
                                   localizer[subj][1]) for subj in layout.subjects)

    preds = p(delayed(predict_across_time)(clf, fast_images[subj][0]) for clf, subj in zip(clfs, layout.subjects))
    accs = [(pred.T==fast_images[subj][1]).mean(1) for pred, subj in zip(preds, layout.subjects)]
    df_tmp = pd.DataFrame({'accuracy': np.ravel(accs),
                           'time': np.hstack([times]*len(accs)),
                           'subject': np.repeat(layout.subjects, len(times)),
                           'range': f'{10*(t1-20)}-{10*(t2-20)}'})
    df_ranges = pd.concat([df_ranges, df_tmp])


fig, ax = plt.subplots(1, 1)
sns.lineplot(df_ranges, x='time', y='accuracy', hue='range', ax=ax)
ax.set_title(f'{clf_vote=} time ensemble training on fast images')
ax.vlines([(i+100)+150 for i in  intervals], *ax.get_ylim(), linestyle='--', color='gray')

#%% train with downsampled data
import meg_utils

df = pd.DataFrame()
t_sfreq = 10
for t_sfreq in tqdm([5, 10, 50, 99, 100]):
    for subject in tqdm(layout.subjects, desc='downsample solution'):
        train_x, train_y, _ = localizer[subject]
        test_x, test_y, _ = fast_images[subject]
        train_x = meg_utils.preprocessing.resample(train_x, o_sfreq=100, t_sfreq=t_sfreq)
        train_x = meg_utils.preprocessing.resample(train_x, o_sfreq=t_sfreq, t_sfreq=100)
        # test_x = meg_utils.preprocessing.resample(test_x, o_sfreq=100, t_sfreq=t_sfreq)
        # test_x = meg_utils.preprocessing.resample(test_x, o_sfreq=t_sfreq, t_sfreq=100)

        # t_test = int(np.round(35/(100/t_sfreq)))
        t_test = 35
        clf.fit(train_x[:, :, t_test], train_y)
        pred = predict_across_time(clf, test_x)
        acc = (pred.T==test_y).mean(1)
        df_subj = pd.DataFrame({'accuracy': acc,
                                'time': np.linspace(-200, 500, len(acc)),
                                'sfreq': t_sfreq})
        df = pd.concat([df, df_subj])

fig, ax = plt.subplots(1,1)
ax.hlines(0.2, -200, 500, linestyle='--', color='gray')
sns.lineplot(df, x='time', y='accuracy', hue='sfreq', ax=ax, palette='crest')
ax.set_title('only localizer down&uppsampled')
ax.vlines([(i+100)+150 for i in  intervals], *ax.get_ylim(), linestyle='--', color='gray')

#%%
asd
for subject in tqdm(settings.layout.subjects):

    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, df_beh = bids_utils.load_fast_images(subject)

    # run classifier across trial
    probas = np.swapaxes([clf.predict_proba(data_x[:, :, t]) for t in range(data_x.shape[-1])], 0, 1)
    # probas /= probas.mean(0).mean(0)
    timepoint = np.repeat(times, probas.shape[-1])

    df_subj = pd.DataFrame()
    for i, (proba, y, interval) in enumerate(zip(probas, data_y, df_beh.interval, strict=True)):
        label = ['other']*5
        label[y] = 'target'
        df_tmp = pd.DataFrame({'label': np.hstack([label]* probas.shape[1]),
                               'timepoint': timepoint,
                               'proba': proba.ravel(),
                               'stimulus': settings.img_trigger[y],
                               'interval': interval,
                               'subject':subject})
        df_subj = pd.concat([df_subj, df_tmp])
    df_subj = df_subj.groupby(['label', 'timepoint', 'stimulus', 'interval', 'subject']).mean().reset_index()
    df = pd.concat([df, df_subj], ignore_index=True)


fig, ax = plt.subplots(figsize=[8, 5])
ax.clear()
sns.lineplot(df, x='timepoint', y='proba',style='label',
             palette='muted', ax=ax)
ax.set_title(f'Fast images decoding, n={len(df.subject.unique())}')

#%% gauss smooth probability time series


from scipy.ndimage import gaussian_filter
from meg_utils import misc

sigma = [0, 5, 0]

fig, axs = plt.subplots(2, 2, figsize=[12, 8])

# zscore = lambda x, axis, nan_policy:x
df = pd.DataFrame()
for subject in tqdm(layout.subjects):
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_images(subject)
    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    probas_smooth = gaussian_filter(probas, sigma = sigma)

    timepoint = np.arange(-200, 510, 10)
    df_subj = misc.to_long_df(probas_smooth, ['target', 'timepoint', 'cls'],
                              target =beh.trigger,
                              cls= np.arange(5),
                              timepoint = timepoint,
                              value_name='probability')
    df_subj['target'] = df_subj.target==df_subj.cls

    sns.lineplot(df_subj, x='timepoint', y='probability', hue='target')

    slopes = {interval: [] for interval in intervals}

#%% gauss smooth data before predicting
from scipy.ndimage import gaussian_filter
from meg_utils import misc

sigma = [0, 5, 0]

fig, axs = plt.subplots(2, 2, figsize=[12, 8])

# zscore = lambda x, axis, nan_policy:x
df = pd.DataFrame()
for subject in tqdm(layout.subjects):
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_images(subject)
    data_x = gaussian_filter(data_x, sigma = [0, 0, 5])

    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])


    timepoint = np.arange(-200, 510, 10)
    df_subj = misc.to_long_df(probas, ['target', 'timepoint', 'cls'],
                              target =beh.trigger,
                              cls= np.arange(5),
                              timepoint = timepoint,
                              value_name='probability')
    df_subj['target'] = df_subj.target==df_subj.cls

    sns.lineplot(df_subj, x='timepoint', y='probability', hue='target')

    slopes = {interval: [] for interval in intervals}


#%% constant convolution
from scipy.ndimage import convolve

kernel = np.ones([1, 25, 1])

fig, axs = plt.subplots(2, 2, figsize=[12, 8])

# zscore = lambda x, axis, nan_policy:x
df = pd.DataFrame()
for subject in tqdm(layout.subjects):
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_images(subject)

    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    probas = convolve(probas, kernel/kernel.sum(), mode='constant', cval=0.2)

    timepoint = np.arange(-200, 510, 10)
    df_subj = misc.to_long_df(probas, ['target', 'timepoint', 'cls'],
                              target =beh.trigger,
                              cls= np.arange(5),
                              timepoint = timepoint,
                              value_name='probability')
    df_subj['target'] = df_subj.target==df_subj.cls

    sns.lineplot(df_subj, x='timepoint', y='probability', hue='target')

    slopes = {interval: [] for interval in intervals}
