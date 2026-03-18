# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 08:48:42 2025

this script creates a two plots, decoding of the localizer with MEG & fMRI


Input:
    MEG:
     - the corresponding values have been created in 1a_run_best_l1_meg
    fMRI:
     - values are provided by Wittkuhn et al (2021)

@author: Simon.Kern
"""
import sys; sys.path.append('..')
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tdlm

import settings
from bids_utils import layout_MEG, layout_3T

from joblib import Memory
from tqdm import tqdm
from meg_utils import plotting, sigproc
import bids_utils
from bids_utils import load_decoding_3T
from bids import BIDSLayout
from mne_bids import BIDSPath
import joblib

#%% plot settings
plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend

subjects_fmri = [f'{i:02d}' for i in range(1, 41)]
subjects_meg = [f'{i:02d}' for i in range(1, 31)]

#%% MEG data loading - load probability estimates from gridsearch results

df_meg_proba = pd.DataFrame()
df_meg_acc = pd.DataFrame()

for subject in tqdm(subjects_meg, desc='Loading MEG data'):
    # Build paths to gridsearch results
    path_proba = BIDSPath(
        root=layout_MEG.derivatives['derivatives'].root,
        datatype='results',
        subject=subject,
        task='main',
        acquisition='slow',
        processing='gridsearch',
        extension='.csv.pkl.gz',
        suffix='probas',
        check=False
    )
    path_acc = path_proba.copy().update(suffix='accuracy')

    if not path_proba.fpath.exists():
        print(f'Warning: {path_proba.fpath} not found, skipping subject {subject}')
        continue

    # Load data
    df_proba_subj = pd.read_pickle(path_proba.fpath)
    df_acc_subj = pd.read_pickle(path_acc.fpath)

    # Find best C value for this subject and filter
    best_C = df_acc_subj.loc[df_acc_subj.accuracy.idxmax(), 'C']
    df_proba_subj = df_proba_subj[df_proba_subj.C == best_C]
    df_acc_subj = df_acc_subj[df_acc_subj.C == best_C].copy()

    # Convert class indices to class names
    df_proba_subj['stim'] = df_proba_subj['trial_label'].map(settings.img_trigger)
    df_proba_subj['stim'] = df_proba_subj['stim'].map(settings.trigger_translation)
    df_proba_subj['class'] = df_proba_subj['class_idx'].map(settings.img_trigger)
    df_proba_subj['class'] = df_proba_subj['class'].map(settings.trigger_translation)

    # Add label column: 'target' if class matches trial label, else 'other'
    df_proba_subj['label'] = np.where(
        df_proba_subj['class_idx'] == df_proba_subj['trial_label'],
        'target', 'other'
    )

    # Rename proba to probability for consistency with fMRI
    df_proba_subj = df_proba_subj.rename(columns={'proba': 'probability'})

    df_meg_proba = pd.concat([df_meg_proba, df_proba_subj], ignore_index=True)
    df_meg_acc = pd.concat([df_meg_acc, df_acc_subj], ignore_index=True)

#%% fMRI data loading - load probability estimates for slow/oddball trials

df_fmri_proba = pd.DataFrame()
df_fmri_acc = pd.DataFrame()

for subject in tqdm(subjects_fmri, desc='Loading fMRI data'):
    # Load probability data
    df_odd = bids_utils.load_decoding_seq_3T(
        subject, test_set='test-odd_long', classifier='log_reg'
    )

    # Add label: 'target' if class matches stim, else 'other'
    df_odd['label'] = np.where(df_odd['class'] == df_odd['stim'], 'target', 'other')
    df_odd['subject'] = subject
    df_odd['tr_onset'] = df_odd['tr_onset'].round(decimals=1)

    df_fmri_proba = pd.concat([df_fmri_proba, df_odd], ignore_index=True)

    # Calculate accuracy per TR
    df_odd['target'] = df_odd['class'] == df_odd['stim']
    df_odd['accuracy'] = df_odd['pred_label'] == df_odd['stim']
    df_mean = df_odd.groupby(['tr_onset']).mean(numeric_only=True).reset_index()
    df_mean['subject'] = subject

    df_fmri_acc = pd.concat([df_fmri_acc, df_mean], ignore_index=True)


#%% Accuracy - MEG/fMRI combined
# Use pre-loaded df_fmri_acc and df_meg_acc

fig, axs = plt.subplots(1, 2, figsize=[14, 5])

# MEG plot (ms-based timepoints)
ax = axs[0]
df_meg_acc['time'] = df_meg_acc['timepoint']/1000  # convert to seconds
sns.lineplot(df_meg_acc, x='time', y='accuracy', ax=ax)
ax.axhline(0.2, color='black', alpha=0.5, linestyle='--')
ax.set(xlabel='seconds after stim onset', ylabel='accuracy',
       title=f'MEG\ndecoding accuracy (n={len(subjects_meg)})')
ax.axvline(0, color='k', label='image onset', linestyle='--')
ax.legend(['decoding acc.', 'SE', 'chance level', 'image onset'], fontsize=10, loc='upper right')

# fMRI plot (TR-based, seconds)
ax = axs[1]
sns.lineplot(df_fmri_acc, x='tr_onset', y='accuracy', ax=ax)
ax.hlines(0.2, -0.6, 8, color='black', alpha=0.5, linestyle='--')
ax.set(xlim=[-0.6, 8], xlabel='seconds after stim onset', ylabel='accuracy',
       title=f'fMRI\nlocalizer decoding accuracy (n={len(subjects_fmri)})')
ax.axvline(0, color='k', label='image onset', linestyle='--')
ax.legend(['decoding acc.', 'SE', 'chance level', 'image onset'], fontsize=10, loc='upper right')

plotting.normalize_lims(axs, which='y')

sns.despine()
plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_accuracy.png')


#%% FIGURE: slow trial class probability (fMRI)
df_proba = df_fmri_proba  # use loaded fMRI data

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


#%% fast trials: probability (fMRI)
df = pd.DataFrame()
for subject in tqdm(subjects_fmri):
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

fig.suptitle(f'probabilities aligned to TR walltime after seq start n={len(subjects_fmri)}')
plotting.normalize_lims(axs)
sns.despine()
plotting.savefig(fig, settings.plot_dir + f'/tr_onset/sequence_all.png')
