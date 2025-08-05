# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 08:48:42 2025

@author: Simon.Kern
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import settings
from joblib import Memory
import numpy as np
from tqdm import tqdm
from meg_utils import plotting, sigproc
import bids_utils
from bids_utils import load_decoding_3T
from bids import BIDSLayout
import tdlm
#%% BIDS layout for 3T

mem = Memory(settings.cache_dir)

read_csv = mem.cache(lambda *x, **y: pd.read_csv(*x, **y))
plt.rcParams.update({'font.size':14})

# ---- load data -----
subjects = [f'{i:02d}' for i in range(1, 41)]

layout = BIDSLayout(settings.bids_dir_3T)
layout_decoding = BIDSLayout(settings.bids_dir_3T_decoding, validate=False)
stop
#%% slow trials: accuracy
df = pd.DataFrame()
for subject in tqdm(subjects):

    df_odd = bids_utils.load_decoding_seq_3T(subject, test_set='test-odd_long')

    # df_odd = df_subj[(df_subj.test_set=='test-odd_long')
    #                  & (df_subj.classifier=='log_reg')
    #                  & (df_subj['mask']=='cv')]

    df_odd['target'] = df_odd['class']==df_odd['stim']
    df_odd['accuracy'] = df_odd['pred_label']==df_odd['stim']
    df_mean = df_odd.groupby(['seq_tr']).mean(True).reset_index()
    df_mean['subject'] = subject
    df = pd.concat([df, df_mean], ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=[4, 3])
sns.lineplot(df, x='seq_tr', y='accuracy', ax=ax)
ax.hlines(0.2, 1, 7, color='black', alpha=0.5, linestyle='--')
ax.set_ylim(0, 0.8)
ax.set_xticks(np.arange(1, 8), [f'TR {x}\n{x*1250}' for x in range (1, 8)])
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
ax.set_xlabel('ms after stim onset')
sns.despine()
plt.pause(0.1)
plt.tight_layout()

#%% slow trials: probability
df = pd.DataFrame()
for subject in tqdm(subjects):
    # read file
    df_odd = bids_utils.load_decoding_seq_3T(subject, test_set='test-odd_long', classifier='log_reg')
    df_odd['target'] = df_odd['class']==df_odd['stim']
    df_odd['subject'] = subject
    # round real wall time after stim onset of TR
    df_odd.tr_onset = df_odd.tr_onset.round(decimals=1)
    df = pd.concat([df, df_odd], ignore_index=True)


fig, ax = plt.subplots(1, 1, figsize=[4, 3])
sns.lineplot(df, x='tr_onset', y='probability', hue='target', ax=ax)
ax.hlines(0.2, 1, 7, color='black', alpha=0.5, linestyle='--')
ax.set_ylim(0, 0.8)
ax.set_xticks(np.arange(1, 8), [f'TR {x}\n{x*1250}' for x in range (1, 8)])
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
ax.set_xlabel('ms after stim onset')
sns.despine()
plt.pause(0.1)
plt.tight_layout()


#%% fast trials: sequences
categories = ['cat', 'chair', 'face', 'house', 'shoe']
df = pd.DataFrame()
for subject in tqdm(subjects):
    df_probas = load_decoding_3T(subject,
                                test_set='test-seq_long',
                                mask='cv')
    df_probas = df_probas[df_probas.classifier.isin(categories)]
    df_probas = df_probas[df_probas['class'].isin(categories)]
    df_seq = bids_utils.load_sequences_3T(subject)


    # extract the sequences
    dfs = []
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial')):
        assert t1==t2
        seq_labels = list(df_s.stim_label.values[0])
        df_p['serial_position'] = df_p.classifier.apply(lambda x: seq_labels.index(x))
        dfs += [df_p]

    df_subj = pd.concat(dfs)
    df_subj = df_subj.groupby(['tITI', 'serial_position', 'seq_tr']).mean(True).reset_index()
    # for iti, df_iti in df_subj.groupby('tITI'):
    #     plt.figure()
    #     sns.lineplot(df_iti, x='seq_tr', y='probability', hue='serial_position')

    df = pd.concat([df, df_subj], ignore_index=True)


fig, axs = plt.subplots(1, 5, figsize=[20, 4])
for i, (iti, df_iti) in enumerate(df.groupby('tITI')):
    ax = axs[i]
    plot = sns.lineplot(df_iti, x='seq_tr', y='probability', hue='serial_position',
                 palette=settings.palette_wittkuhn1,
                 ax=ax, legend=None)
    ax.set_xticks(np.arange(1, 14, 2))
    if i > 0:
        ax.set_ylabel('')
    ax.set_xlabel('TR')
    ax.set_title(f'{int((float(iti)*1000))} ms')

# Draw the legend outside the rightmost axis
fig.legend(
    ['1', '_', '2', '_', '3', '_', '4', '_', '5'],
    title='item position',
    loc='center right',
    bbox_to_anchor=(0.96, 0.6),  # 1.02 puts it just outside the right edge, 0.5 is vertical center
    bbox_transform=fig.transFigure
)

plotting.normalize_lims(axs)
fig.tight_layout(rect=[0, 0, 0.88, 1])  # leave space on the right for the legend


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
    plotting.savefig(fig, f'tr_onset/sequence_{subject}.png')
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
plotting.savefig(fig, f'tr_onset/sequence_all.png')

#%% TR: TDLM on super trials
intervals = ['0.032', '0.128', '0.064', '0.512', '2.048']
categories = ['cat', 'chair', 'face', 'house', 'shoe']
df = pd.DataFrame()
sf = {interval: [] for interval in intervals}
sb = {interval: [] for interval in intervals}

for subject in tqdm(subjects):
    df_probas = load_decoding_seq_3T(subject)
    df_probas = df_probas[df_probas.classifier.isin(categories)]
    df_probas = df_probas[df_probas['class'].isin(categories)]
    df_seq = bids_utils.load_sequences_3T(subject)

    intervals = df_probas.tITI.unique()
    sf_subj = {interval: [] for interval in intervals}
    sb_subj = {interval: [] for interval in intervals}

    tf = tdlm.seq2tf('ABCDE')
    # extract the sequences
    probas = {interval:[] for interval in intervals}
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial')):
        assert t1==t2
        interval = df_s.tITI.values[0]
        seq_labels = list(df_s.stim_label.values[0])
        proba = [df_p[df_p.classifier==label].probability.values for label in seq_labels]
        probas[interval] += [proba]

    for i, interval in enumerate(intervals):
        sf_subj, sb_subj = tdlm.compute_1step(np.mean(probas[interval], axis=0).T, tf=tf, max_lag=3)
        sf[interval] += [sf_subj]
        sb[interval] += [sb_subj]


fig, ax = plt.subplots(1, 1)
for i, interval in enumerate(intervals):
    tdlm.plot_sequenceness(sf[interval], sb[interval],
                           maxlag=3,
                           which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           ax=ax,
                           clear=False,
                           plot95=False,
                           plotmax=i==0)
ax.set_title('raw TRs')
ax.legend([32, '_', '_', 64, 128, 512, 2048], title='interval')
ax.set_xticks(np.arange(0, 40, 10), [0, 1.25, 2.5, 3.75])
ax.set_xlabel('time lag (seconds)')
plt.ylim(-3, 3)


#%% TR: interpolated TDLM on super trials
intervals = ['0.032', '0.128', '0.064', '0.512', '2.048']
categories = ['cat', 'chair', 'face', 'house', 'shoe']
df = pd.DataFrame()
sf = {interval: [] for interval in intervals}
sb = {interval: [] for interval in intervals}



for subject in tqdm(subjects):
    df_probas = load_decoding_3T(subject,
                                test_set='test-seq_long',
                                mask='cv')
    df_probas = df_probas[df_probas.classifier.isin(categories)]
    df_probas = df_probas[df_probas['class'].isin(categories)]
    df_seq = bids_utils.load_sequences_3T(subject)

    intervals = df_probas.tITI.unique()
    sf_subj = {interval: [] for interval in intervals}
    sb_subj = {interval: [] for interval in intervals}

    tf = tdlm.seq2tf('ABCDE')

    # extract the sequences
    probas = {interval:[] for interval in intervals}
    for (t1, df_p), (t2, df_s) in zip(df_probas.groupby('trial'), df_seq.groupby('trial')):
        assert t1==t2
        interval = df_s.tITI.values[0]
        seq_labels = list(df_s.stim_label.values[0])
        proba = [df_p[df_p.classifier==label].probability.values for label in seq_labels]
        probas[interval] += [proba]

    for i, interval in enumerate(intervals):
        mean_proba = np.mean(probas[interval], axis=0).T
        probas_fitted = []
        plt.figure()
        for proba in mean_proba.T:
            amp_guess   = proba.max()             # peak height
            shift_guess = (np.arange(13)*1.25)[np.argmax(proba)]  # peak location
            initial_params = {
                "frequency": [(0.01, 0.2), 0.1],                 # fixed at 10 Hz
                "amplitude": [(amp_guess, 2 * amp_guess), amp_guess],      # start at peak
                "loc":     [(0, 13*1.25), shift_guess],
                "baseline":  [(0, 0), 0.0],                        # pinned to 0
            }
            fitted_t, fitted, _= sigproc.fit_curve(proba, model=sigproc.curves.sine_truncated,
                                            curve_params=initial_params,
                                            plot_fit=True)
            probas_fitted += [fitted]

        probas_fitted = np.transpose(probas_fitted)
        max_lag = int(float(interval)*1000/10 + 10)
        sf_subj, sb_subj = tdlm.compute_1step(probas_fitted, tf=tf, max_lag=max_lag)
        sf[interval] += [sf_subj]
        sb[interval] += [sb_subj]



fig, ax = plt.subplots(1, 1)
for i, interval in enumerate(intervals):
    # plt.figure()
    tdlm.plot_sequenceness(sf[interval], sb[interval],
                           maxlag=5,
                           which=['fwd'],
                           color=settings.palette_wittkuhn2[i],
                           # ax=ax,
                           clear=False,
                           plot95=False,
                           plotmax=False)
    plt.title(interval)
ax.set_title('TDLM on model fit')
ax.legend([32, '_', '_', 64, 128, 512, 2048], title='interval')
ax.set_xticks(np.arange(0, 50, 10), [0, 1, 2, 3, 4])
ax.set_xlabel('time lag (seconds)')
plt.ylim(-2, 2)

#%% Walltime: TDLM on merged trial data of wall-time

tf = tdlm.seq2tf('ABCDE')
intervals = ['0.032', '0.064', '0.128', '0.512', '2.048']

df = pd.DataFrame()
degree = 8
sfreq = 50
sf1 = {interval: [] for interval in intervals}
sf2 = {interval: [] for interval in intervals}
probas_all = {iti:[] for iti in intervals}

for subject in tqdm(subjects):
    # read file
    df_proba = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long',
                                               classifier='log_reg')
    df_seq = bids_utils.load_trial_data_3T(subject, condition='sequence')

    # extract serial position of class
    tmp = []
    for i, df_trial in df_proba.groupby('trial'):
        order = df_seq.loc[i-1].stim_label.tolist()
        df_trial['serial_position'] = df_trial['class'].apply(lambda x: order.index(x)+1)
        tmp += [df_trial]
    df_proba = pd.concat(tmp)

    # round real wall time after stim onset of TR
    df_proba.tr_onset = df_proba.tr_onset.round(decimals=1)
    df_proba['subject'] = subject

    # merge trials into one by tr_onset
    fig, axs = plt.subplots(2, 3, figsize=[14, 6])

    for i, (iti, df_iti) in enumerate(df_proba.groupby('tITI')):
        df_iti = df_iti.sort_values(['serial_position', 'tr_onset'])
        probas = np.reshape(df_iti.probability, [5, -1]).T
        times = np.reshape(df_iti.tr_onset, [5, -1])[0]

        samples = int(np.ptp(times) * sfreq)
        t, proba_inter = sigproc.interpolate(times, probas, kind='cubic', n_samples=samples)
        t, proba_poly = sigproc.polyfit(times, probas, degree=degree, n_samples=samples)

        proba_inter = np.nan_to_num(proba_inter)
        max_lag = {'0.032': 20, '0.064': 25, '0.128': 30,  '0.512': 100, '2.048': 250}[str(iti)]
        sf1_subj, _ = tdlm.compute_1step(proba_inter.T, tf, max_lag=max_lag)
        # sf2_subj, _ = tdlm.compute_1step(proba_poly, tf, max_lag=max_lag)

        probas_all[str(iti)] += [proba_inter]
        ax = axs.flat[i]
        tdlm.plot_sequenceness(sf1_subj, _, which=['fwd'], ax=ax, sfreq=sfreq)
        # tdlm.plot_sequenceness(sf2_subj, _, which=['fwd'], ax=ax, clear=False, color='peru')
        # ax.legend(['interpolated'] + list('____') + [f'polyfit_{degree}'])
        ax.set_title(iti)
        sf1[str(iti)] += [sf1_subj]
        # sf2[str(iti)] += [sf2_subj]


fig, axs = plt.subplots(2, 3, figsize=[14, 6])
for i, iti in enumerate(intervals):
    ax = axs.flat[i]
    tdlm.plot_sequenceness(sf1[iti], _, which=['fwd'], ax=ax, sfreq=sfreq)
    # tdlm.plot_sequenceness(sf2[iti], _, which=['fwd'], ax=ax, clear=False, color='peru')
    ax.legend(['interpolated'] + list('____') + [f'polyfit_{degree}'])
    ax.set_title(iti)
    if i==4:
        ax.set_ylim([-3, 3])
#%% Walltime: over all participants
