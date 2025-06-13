#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:42:56 2025

run Wittkuhn&Schuck method on the MEG data with various paradigms

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
import meg_utils
from meg_utils.plotting import savefig
from meg_utils.decoding import LogisticRegressionOvaNegX
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wsdet

for subj in ['01', '23', '03', '04', '09', '13', '26']:
    # somehow, these give errors when loading, so leave them out for now
    # should be included in the final calculation
    # if you see this in a PR, please let me knowm ;-)
    if subj in layout.subjects:
        layout.subjects.remove(subj)

intervals = [32, 64, 128]
clf = LogisticRegressionOvaNegX(penalty='l1', C=4.6, solver='liblinear')


#%% run WS with up and downsampled data
normalization = 'lambda x: x/x.mean(0)'

t_sfreq = 10

df = pd.DataFrame()
for subject in tqdm(settings.layout.subjects):
    train_x, train_y, _ = bids_utils.load_localizer(subject)
    test_x, test_y, df_beh = bids_utils.load_fast_sequences(subject)

    train_x = meg_utils.preprocessing.resample(train_x, o_sfreq=100, t_sfreq=t_sfreq)
    train_x = meg_utils.preprocessing.resample(train_x, o_sfreq=t_sfreq, t_sfreq=100)
    test_x = meg_utils.preprocessing.resample(test_x, o_sfreq=100, t_sfreq=t_sfreq)
    test_x = meg_utils.preprocessing.resample(test_x, o_sfreq=t_sfreq, t_sfreq=100)

    t_test = 45
    clf.fit(train_x[:, :, t_test], train_y)

    probas = np.swapaxes([clf.predict_proba(test_x[:, :, t]) for t in range(test_x.shape[-1])], 0, 1)
    timepoint = np.hstack([np.arange(test_x.shape[-1])*10])-200
    stimulus = np.repeat(np.arange(5), probas.shape[1])
    probas = eval(normalization)(probas)

    df_subj = pd.DataFrame()
    for proba, df_trial in zip(probas, df_beh):
        rs = wsdet.compute_ws(proba.T, order=df_trial.sequence.values)

        pos_idx = [list(df_trial.trigger).index(i) for i in  range(5)]
        position = np.repeat(pos_idx, probas.shape[1])
        df_tmp = pd.DataFrame({'r': rs,
                               'timepoint': timepoint,
                               'interval': df_trial.interval_time.iloc[0],
                               'subject': subject})
        df_subj = pd.concat([df_subj, df_tmp])
    df_subj = df_subj.groupby(['timepoint', 'interval', 'r']).mean(True).reset_index()
    # sns.lineplot(df_subj, x='timepoint', y='proba', hue='interval')
    df = pd.concat([df, df_subj], ignore_index=True)


fig, axs = plt.subplots(2, 2, figsize=[18, 8])
axs = axs.flatten()

for i, interval in enumerate(df.interval.unique()):
    df_sel = df[df.interval==interval]
    ax = axs[i]
    sns.lineplot(df_sel, x='timepoint', y='r', hue='interval', palette='muted', ax=ax)
    ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim())
    # ax.set_xlim(-200, interval*5+750)
    ax.set_title(f'{interval=}ms\n{normalization=}')
    plt.pause(0.1)

fig.tight_layout()
plt.pause(0.1)
fig.savefig(settings.plot_dir + f'wsdet-fastimages.png')
