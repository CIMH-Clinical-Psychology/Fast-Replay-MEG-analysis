# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 08:48:42 2025

@author: Simon.Kern
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tdlm

import settings
from joblib import Memory
from tqdm import tqdm
from meg_utils import plotting, sigproc
import bids_utils
from bids_utils import load_decoding_3T
from bids import BIDSLayout
#%% BIDS layout for 3T
plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend
# ---- load data -----
subjects = [f'{i:02d}' for i in range(1, 41)]

layout = BIDSLayout(settings.bids_dir_3T)
layout_decoding = BIDSLayout(settings.bids_dir_3T_decoding, validate=False)
stop

#%% FIGURE: slow trial class probability
df_proba = pd.DataFrame()
for subject in tqdm(subjects):
    # read file
    df_odd = bids_utils.load_decoding_seq_3T(subject, test_set='test-odd_long', classifier='log_reg')
    df_odd['label'] = (df_odd['class']==df_odd['stim']).astype(object)
    df_odd.loc[df_odd['label'], 'label'] = 'target'
    df_odd.loc[df_odd['label']==0, 'label'] = 'other' #df_odd['stim']

    df_odd['subject'] = subject
    # round real wall time after stim onset of TR
    df_odd.tr_onset = df_odd.tr_onset.round(decimals=1)
    df_proba = pd.concat([df_proba, df_odd], ignore_index=True)


fig, axs = plt.subplots(2, 3, figsize=[14, 6])
sns.despine(fig)

for stim_idx, stim_name in enumerate(settings.trigger_translation.values()):
    ax = axs.flat[stim_idx]
    ax.hlines(0.2, -0.6, 8, color='gray', linestyle='--')
    df_sel = df_proba[df_proba.stim==stim_name].sort_values('label', ascending=False, key=lambda x:x=='other')
    sns.lineplot(df_sel, x='tr_onset',  y='probability', hue='label', ax=ax)
    ax.set(title=stim_name, ylabel='probability', xlabel='seconds after stim onset', xlim=[-0.6, 8])
    plt.pause(0.1)

ax = axs.flat[-1]
df_proba_mean = df_proba.groupby(['tr_onset', 'label', 'subject']).mean(True).reset_index()
ax.hlines(0.2, -0.6, 8, color='gray', linestyle='--')
sns.lineplot(df_proba_mean, x='tr_onset',  y='probability', hue='label', ax=ax)
ax.set(title='mean', ylabel='probability', xlabel='seconds after stim onset', xlim=[-0.6, 8])

plotting.normalize_lims(list(axs.flat))
plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_3T_probabilities.png')

#%% slow trials: accuracy

df = pd.DataFrame()
for subject in tqdm(subjects):

    df_odd = bids_utils.load_decoding_seq_3T(subject, test_set='test-odd_long', classifier='log_reg')

    df_odd.tr_onset = df_odd.tr_onset.round(1)

    df_odd['target'] = df_odd['class']==df_odd['stim']
    df_odd['accuracy'] = df_odd['pred_label']==df_odd['stim']
    df_mean = df_odd.groupby(['tr_onset']).mean(True).reset_index()
    df_mean['subject'] = subject
    df = pd.concat([df, df_mean], ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=[8, 6])
sns.lineplot(df, x='tr_onset', y='accuracy', ax=ax)
ax.hlines(0.2, -0.6, 8, color='black', alpha=0.5, linestyle='--')
ax.set(xlim=[-0.6, 8], xlabel='seconds after stim onset', title='Decoding accuracy')
# ax.set_xticks(np.arange(1, 8), [f'TR {x}\n{x*1250}' for x in range (1, 8)])
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
sns.despine()

plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_3T_accuracy.png')



#%% fast trials: probability
df = pd.DataFrame()
for subject in tqdm(subjects):
    # read file
    df_proba = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long',
                                               classifier='log_reg')
    df_seq = bids_utils.load_trial_data_3T(subject, condition='sequence')

    tmp = []
    for i, df_trial in df_proba.groupby('trial'):
        order = df_seq.loc[i-1].stim_label.tolist()
        df_trial['serial_position'] = df_trial['class'].apply(lambda x: order.index(x)+1)
        tmp += [df_trial]
    df_proba = pd.concat(tmp)
    df_proba.tr_onset = df_proba.tr_onset.round(decimals=1)
    df_proba['subject'] = subject

    # round real wall time after stim onset of TR
    df = pd.concat([df, df_proba], ignore_index=True)

    fig, axs = plt.subplots(1, 5, figsize=[20, 4], sharey=True)
    for i, (iti, df_iti) in enumerate(df_proba.groupby('tITI')):
        ax = axs[i]
        plot = sns.lineplot(df_iti, x='tr_onset', y='probability', hue='serial_position',
                     palette=settings.palette_wittkuhn1,
                     ax=ax, legend=False)
        ax.set_xticks(np.arange(1, 14, 2))
        ax.set_xlabel('ms after stim onset')
        ax.set_title(f'{int((float(iti)*1000))} ms')
        ax.hlines(0.2, 1, 7, color='black', alpha=0.5, linestyle='--')
    fig.suptitle(f'probabilities aligned to TR walltime after seq start {subject=}')
    fig.legend('1_2_3_4_5', title='Ser. Pos', ncols=5, fontsize=10)
    plotting.normalize_lims(axs)
    sns.despine()
    plotting.savefig(fig, settings.plot_dir + f'/tr_onset/sequence_{subject}.png')
    plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=[20, 4], sharey=True)
for i, (iti, df_iti) in enumerate(df.groupby('tITI')):
    ax = axs[i]
    plot = sns.lineplot(df_iti, x='tr_onset', y='probability', hue='serial_position',
                 palette=settings.palette_wittkuhn1,
                 ax=ax, legend=False)
    ax.set_xticks(np.arange(1, 14, 2))
    ax.set_xlabel('ms after stim onset')
    ax.set_title(f'{int((float(iti)*1000))} ms')
    ax.hlines(0.2, 1, 7, color='black', alpha=0.5, linestyle='--')
fig.legend('1_2_3_4_5', title='Ser. Pos', ncols=5, fontsize=10)

fig.suptitle(f'probabilities aligned to TR walltime after seq start n={len(subjects)}')
plotting.normalize_lims(axs)
sns.despine()
plotting.savefig(fig, settings.plot_dir + f'/tr_onset/sequence_all.png')
