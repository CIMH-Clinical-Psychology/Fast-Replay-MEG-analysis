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
from meg_utils import plotting
from bids import BIDSLayout
from load_funcs import layout, load_localizer
from meg_utils.decoding import cross_validation_across_time
import settings
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%% decode best timepoint
data_localizer = {}

tmin=-0.2
tmax=0.8
ex_per_fold=8

df_all = pd.DataFrame()

# fig, axs, ax_b = plotting.make_fig(n_axs=len(layout.subjects), bottom_plots=[0,0,1])

for C in tqdm(np.logspace(-5, 4, 100)):
    df_c = pd.DataFrame()
    for i, subject in enumerate(layout.subjects):
        if subject in ['01', '23', '03', '04']:
            # somehow, these give errors when loading, so leave them out for now
            # should be included in the final calculation
            # if you see this in a PR, please let me knowm ;-)
            continue
        clf = LogisticRegression(penalty='l1', C=C, solver='liblinear')

        # load MEG data from the localizer
        data_x, data_y = load_localizer(subject=subject, verbose=False)

        # do cross validation decoding in sensor space across time
        df_subj = cross_validation_across_time(data_x, data_y, subj=subject, n_jobs=-1,
                                               tmin=tmin, tmax=tmax, ex_per_fold=ex_per_fold, clf=clf)
        df_subj = df_subj.groupby('timepoint').mean(True).reset_index()
        df_subj['C'] = C
        df_c = pd.concat([df_c, df_subj])

    df_all = pd.concat([df_all, df_c], ignore_index=True)
    fig, ax = plt.subplots(figsize=[8, 6])
    sns.lineplot(data=df_c, x='timepoint', y='accuracy', ax=ax)
    ax.set_title(f'{C=} all participants {len(layout.subjects)=}')
    fig.savefig(settings.plot_dir + f'localizer_C{C:011.6f}.png')
    plt.pause(0.1)


    # ax = axs[i]
    # sns.lineplot(data=df_subj, x='timepoint', y='accuracy', ax=ax)
    # ax.set_title(f'{subject=}')
    # plt.pause(0.1)


axs = [plt.figure(i).axes[0] for i in range(1, 101)]
plotting.normalize_lims(axs)


Cs = np.logspace(-5, 4, 100)
for i in tqdm(range(1, 101)):
    fig = plt.figure(i)
    fig.savefig(settings.plot_dir + f'localizer_C{Cs[i]:011.6f}.png')
