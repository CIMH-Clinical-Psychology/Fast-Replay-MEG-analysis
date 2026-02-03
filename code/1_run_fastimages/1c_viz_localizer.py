#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 22:12:08 2026

@author: simon.kern
"""

1c
#%% make combined plot for MEG and fMRI to compare timescales
import bids_utils


# first load the
df_3T = pd.DataFrame()
for subject in tqdm([f'{s:02d}' for s in np.arange(1, 40)]):

    df_odd = bids_utils.load_decoding_seq_3T(subject, test_set='test-odd_long', classifier='log_reg')
    df_odd.tr_onset = df_odd.tr_onset.round(1)
    df_odd['target'] = df_odd['class']==df_odd['stim']
    df_odd['accuracy'] = df_odd['pred_label']==df_odd['stim']
    df_mean = df_odd.groupby(['tr_onset']).mean(True).reset_index()
    df_mean['subject'] = subject
    df_3T = pd.concat([df_3T, df_mean], ignore_index=True)


df_meg = joblib.load(settings.plot_dir + '/pkl/localizer_meg_accuracy.pkl.gz')

sns.lineplot(df, x='tr_onset', y='accuracy', ax=ax)
ax.hlines(0.2, -0.6, 8, color='black', alpha=0.5, linestyle='--')
ax.set(xlim=[-0.6, 8], xlabel='seconds after stim onset', title='Decoding accuracy')
# ax.set_xticks(np.arange(1, 8), [f'TR {x}\n{x*1250}' for x in range (1, 8)])
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
sns.despine()

plotting.savefig(fig, settings.plot_dir + f'/figures/localizer_3T_accuracy.png')
