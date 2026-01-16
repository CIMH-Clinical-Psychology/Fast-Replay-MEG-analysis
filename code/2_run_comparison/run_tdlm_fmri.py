# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run TDLM (Temporally Delayed Linear Modelling) on fMRI data

@author: Simon.Kern
"""
import os
import settings
import bids_utils
from tqdm import tqdm
from bids import BIDSLayout
import numpy as np
import pandas as pd
import mne
from meg_utils import decoding, plotting, sigproc


#%% settings
subjects = [f'{i:02d}' for i in range(1, 41)]
normalization = 'lambda x: x/x.mean(0)'

layout = BIDSLayout(settings.bids_dir_3T)

#%% TR: TDLM on super trials
intervals = ['0.032', '0.128', '0.064', '0.512', '2.048']
categories = ['cat', 'chair', 'face', 'house', 'shoe']


df = pd.DataFrame()
sf = {interval: [] for interval in intervals}
sb = {interval: [] for interval in intervals}

for subject in tqdm(subjects):
    df_probas = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long',
                                                classifier=categories)
    df_seq = bids_utils.load_trial_data_3T(subject)

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
