# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run SODA (Slope Order Dynamic Analysis) on fMRI data

@author: Simon.Kern
"""

import os
import settings
from settings import layout_3T as layout
import bids_utils

import numpy as np
import pandas as pd
import mne
from meg_utils import decoding, plotting, sigproc


#%% settings

for subj in ['01', '23', '03', '04', '09', '13', '26', 'emptyroom']:
    # somehow, these give errors when loading, so leave them out for now
    # should be included in the final calculation
    # if you see this in a PR, please let me knowm ;-)
    if subj in layout.subjects:
        layout.subjects.remove(subj)

intervals = [32, 64, 128]
clf = decoding.LogisticRegressionOvaNegX(penalty='l1', C=4.6, solver='liblinear')


#%% run WS with up and downsampled data
normalization = 'lambda x: x/x.mean(0)'

t_sfreq = 10

df = pd.DataFrame()
for subject in tqdm(settings.layout.subjects):
    train_x, train_y, _ = bids_utils.load_localizer(subject)
    test_x, test_y, df_beh = bids_utils.load_fast_sequences(subject)

    train_x = sigproc.resample(train_x, o_sfreq=100, t_sfreq=t_sfreq)
    train_x = sigproc.resample(train_x, o_sfreq=t_sfreq, t_sfreq=100)
    test_x = sigproc.resample(test_x, o_sfreq=100, t_sfreq=t_sfreq)
    test_x = sigproc.resample(test_x, o_sfreq=t_sfreq, t_sfreq=100)

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

plt.rcParams.update({'font.size':14})

fig, ax = plt.subplots(1, 1)

for i, interval in enumerate(df.interval.unique()):
    df_sel = df[df.interval==interval]
    sns.lineplot(df_sel, x='timepoint', y='r',
                 color=settings.palette_wittkuhn2[i], ax=ax,
                 label=int(interval))
    # ax.vlines(np.arange(5)* (100+interval), *ax.get_ylim())
    # ax.set_xlim(-200, interval*5+750)
    plt.pause(0.1)

ax.set_ylabel('Regression slope')
ax.set_xlabel('timepoint (ms)')
plt.legend()

fig.tight_layout()
plt.pause(0.1)
fig.savefig(settings.plot_dir + f'wsdet-fastimages.png')
