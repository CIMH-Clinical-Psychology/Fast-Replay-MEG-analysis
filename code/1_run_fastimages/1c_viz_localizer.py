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
from scipy.stats import ttest_1samp, ttest_ind
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

sns.set_context('paper', font_scale=1.5)

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
    df_proba_subj = df_proba_subj[df_proba_subj.C == best_C].copy()
    df_acc_subj = df_acc_subj[df_acc_subj.C == best_C].copy()
    df_proba_subj['subject'] = subject
    df_acc_subj['subject'] = subject

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

#%% statistics
df_fmri_mean = df_fmri_acc.groupby(['tr_onset']).mean(True).reset_index()
df_meg_mean = df_meg_acc.groupby(['timepoint']).mean(True).reset_index()

max_t_fmri = df_fmri_mean.tr_onset[df_fmri_mean.accuracy.argmax()]
max_t_meg = df_meg_mean.timepoint[df_meg_mean.accuracy.argmax()]

df_fmri_peak = df_fmri_acc[df_fmri_acc.tr_onset==max_t_fmri]
df_meg_peak = df_meg_acc[df_meg_acc.timepoint==max_t_meg]

t_fmri = ttest_1samp(df_fmri_peak.accuracy, popmean=0.2)
t_meg = ttest_1samp(df_meg_peak.accuracy, popmean=0.2)
t_diff = ttest_ind(df_fmri_peak.accuracy, df_meg_peak.accuracy)

def _stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

print('=' * 70)
print('Localizer peak decoding accuracy — statistics')
print('=' * 70)
print(f'{"":<12}{"peak time":>12}{"n":>6}{"mean":>10}{"std":>8}{"t":>10}{"p":>12}')
print('-' * 70)
print(f'{"fMRI":<12}{max_t_fmri:>10.2f} s{len(df_fmri_peak):>6d}'
      f'{df_fmri_peak.accuracy.mean():>10.3f}{df_fmri_peak.accuracy.std():>8.3f}'
      f'{t_fmri.statistic:>10.2f}{t_fmri.pvalue:>10.3e} {_stars(t_fmri.pvalue)}')
print(f'{"MEG":<12}{max_t_meg:>9.0f} ms{len(df_meg_peak):>6d}'
      f'{df_meg_peak.accuracy.mean():>10.3f}{df_meg_peak.accuracy.std():>8.3f}'
      f'{t_meg.statistic:>10.2f}{t_meg.pvalue:>10.3e} {_stars(t_meg.pvalue)}')
print('-' * 70)
print(f'One-sample t-test vs chance (0.2):')
print(f'  fMRI: t({len(df_fmri_peak)-1}) = {t_fmri.statistic:+.2f},  p = {t_fmri.pvalue:.3e}  {_stars(t_fmri.pvalue)}')
print(f'  MEG : t({len(df_meg_peak)-1}) = {t_meg.statistic:+.2f},  p = {t_meg.pvalue:.3e}  {_stars(t_meg.pvalue)}')
print()
print(f'Independent t-test (fMRI vs MEG peak accuracy):')
print(f'  t({len(df_fmri_peak)+len(df_meg_peak)-2}) = {t_diff.statistic:+.2f},  p = {t_diff.pvalue:.3e}  {_stars(t_diff.pvalue)}')
print('=' * 70)

#%% Accuracy - MEG/fMRI combined
# Use pre-loaded df_fmri_acc and df_meg_acc

fig, axs = plt.subplots(2, 1, figsize=[6, 8])

# MEG plot (ms-based timepoints)
ax = axs[0]
df_meg_acc['time'] = df_meg_acc['timepoint']/1000  # convert to seconds
sns.lineplot(df_meg_acc, x='time', y='accuracy', ax=ax)
ax.axhline(0.2, color='black', alpha=0.5, linestyle='--')
ax.set(xlabel='seconds after stim onset', ylabel='accuracy',
       title=f'Localizer decoding accuracy (MEG)')
ax.axvline(0, color='r', label='image onset', linewidth=5, alpha=0.2)
ax.set_xticks(np.arange(-2, 9)/10, minor=True)
ax.set_yticks(np.arange(0, 9)/10, minor=True)
ax.legend(['decoding acc.', f'SE (n={len(subjects_meg)})', 'chance level', 'image onset'], fontsize=11, loc='upper right')

#
# fMRI plot (TR-based, seconds)
ax = axs[1]
sns.lineplot(df_fmri_acc, x='tr_onset', y='accuracy', ax=ax)
ax.hlines(0.2, -0.6, 8, color='black', alpha=0.5, linestyle='--')
ax.set(xlim=[-0.6, 8], xlabel='seconds after stim onset', ylabel='accuracy',
       title=f'Localizer decoding accuracy (fMRI)')
# ax.axvline(0, color='k', label='image onset', linestyle='--')
ax.axvline(0, color='r', label='image onset', linewidth=5, alpha=0.2)
ax.set_xticks(np.arange(0, 9))
ax.set_yticks(np.arange(0, 9)/10, minor=True)
ax.legend([f'decoding acc.', f'SE (n={len(subjects_fmri)})', 'chance level', 'image onset'], fontsize=11,loc='upper right')

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
plotting.savefig(fig, settings.plot_dir + f'/supplement/localizer_3T_probabilities.png')

#%% FIGURE: slow trial class probability (MEG)
df_proba = df_meg_proba.copy()
df_proba['time'] = df_proba['timepoint'] / 1000  # convert ms → seconds

fig, axs = plt.subplots(2, 3, figsize=[14, 6])
sns.despine(fig)

tmin, tmax = df_proba['time'].min(), df_proba['time'].max()

for stim_idx, stim_name in enumerate(settings.trigger_translation.values()):
    ax = axs.flat[stim_idx]
    ax.hlines(0.2, tmin, tmax, color='gray', linestyle='--')
    df_sel = df_proba[df_proba.stim==stim_name].sort_values('label', ascending=False, key=lambda x:x=='other')
    sns.lineplot(df_sel, x='time', y='probability', hue='label', ax=ax)
    ax.set(title=stim_name, ylabel='probability', xlabel='seconds after stim onset', xlim=[tmin, tmax])
    plt.pause(0.1)

ax = axs.flat[-1]
df_proba_mean = df_proba.groupby(['time', 'label', 'subject']).mean(True).reset_index()
ax.hlines(0.2, tmin, tmax, color='gray', linestyle='--')
sns.lineplot(df_proba_mean, x='time', y='probability', hue='label', ax=ax)
ax.set(title='mean', ylabel='probability', xlabel='seconds after stim onset', xlim=[tmin, tmax])

plotting.normalize_lims(list(axs.flat))
plotting.savefig(fig, settings.plot_dir + f'/supplement/localizer_MEG_probabilities.png')
