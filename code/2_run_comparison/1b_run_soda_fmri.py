# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run SODA (Slope Order Dynamic Analysis) on fMRI data

@author: Simon.Kern
"""
import os
import sys; sys.path.append('..')
import warnings
import joblib
import mne
import soda  # SODA-python
import tdlm
from scipy.stats import ttest_1samp

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bids import BIDSLayout
from joblib import Parallel, delayed
from scipy.stats import pearsonr, zscore
from mne_bids import BIDSPath
from mne.stats import permutation_t_test, permutation_cluster_1samp_test

from meg_utils import misc
from meg_utils import decoding, plotting, sigproc
from meg_utils.plotting import savefig

import bids_utils
import settings
from settings import normalization
from settings import layout_3T as layout
from settings import bids_dir_3T_decoding as layout_decoding
from settings import subjects_3T as subjects
from settings import intervals_3T as intervals

#%% settings


bids_base = BIDSPath(
    root=layout.derivatives['derivatives'].root,
    datatype='results',
    subject='group',
    task='main',
    check=False
)
bids_base.mkdir(True)
#%% run SODA on sequence trials

df_slopes = pd.DataFrame()

for subject in tqdm(subjects, desc='loading decoding results'):

    # read file
    df_seq = bids_utils.load_decoding_seq_3T(subject, test_set='test-seq_long')
    df_seq = df_seq[df_seq['class'].isin(settings.categories)]

    df_beh = bids_utils.load_trial_data_3T(subject, condition='sequence')

    df_subj = pd.DataFrame()
    for (i, df_trial), (j, beh) in zip(df_seq.groupby('trial'), df_beh.groupby('trial'), strict=True):
        assert i == j

        # get the sequence order of this
        seq = [settings.categories.index(x) for x in beh.stim_label.values[0]]

        # make sure sorting is same as the category number
        df_trial = df_trial.sort_values(['class', 'tr_onset'], key=lambda x:
                                        [settings.categories.index(y) for y in x]
                                        if 'str' in str(x.dtype) else x)
        probas = np.array(df_trial.probability.values.reshape([13, 5], order='F'))

        probas = eval(normalization)(probas)
        slopes = soda.compute_slopes(probas, order=seq)

        df_tmp = pd.DataFrame({'slope': slopes,
                               'timepoint': sorted(df_trial.tr_onset.unique()),
                               'tr': np.arange(1, 14),
                               'interval': int(df_trial.tITI.unique()[0]*1000),
                               'trial': df_trial.trial.values[0],
                               'subject': subject})

        df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    df_slopes = pd.concat([df_slopes, df_subj], ignore_index=True)

# annotate slope forward and backward period
df_slopes['period'] = None
for interval in intervals:
    for period in ['fwd', 'bkw']:
        trs = settings.exp_tr[interval][period]
        tr_range = list(range(trs[0], trs[1]+1))
        sel = (df_slopes.interval==interval) & df_slopes.tr.isin(tr_range)
        df_slopes.loc[sel, 'period'] = period
pkl_slopes = str(bids_base.copy().update(processing='soda', suffix='slopes', extension='.pkl.gz'))
joblib.dump(df_slopes, pkl_slopes)

#%% overview of mean slopes across intervals
df_slopes = joblib.load(pkl_slopes)
df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

fig, ax = plt.subplots(1, 1, figsize=[8, 6])

sns.lineplot(df_mean[df_mean.interval!=2048], x='tr', y='slope', hue='interval',
             palette=settings.palette_wittkuhn2, ax=ax)
ax.axhline(0, linestyle='--', c='black', alpha=0.3)
ax.set(title='Slopes at different interval speeds')
savefig(fig, settings.plot_dir + '/SODA_slopes_3T.png')
#%% group-level: slope curves with cluster permutation and signflip tests

df_slopes = joblib.load(pkl_slopes)

intervals_plot = [iv for iv in intervals if iv != 2048]
n_iv = len(intervals_plot)
fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4])
if n_iv == 1:
    axs = [axs]

df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

for i, interval in enumerate(intervals_plot):

    tr_fwd = settings.exp_tr[interval]['fwd']
    tr_bkw = settings.exp_tr[interval]['bkw']

    ax = axs[i]

    df_sel = df_mean[df_mean.interval==interval]
    sns.lineplot(df_sel, x='tr', y='slope', color=settings.palette_wittkuhn2[i],
                 ax=ax)
    ax.set_xticks(np.arange(1, 14))
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.axvspan(*tr_fwd, color=settings.color_fwd, linewidth=1, alpha=0.2,
               label='fwd period')
    ax.axvspan(*tr_bkw, color=settings.color_bkw, linewidth=1, alpha=0.2,
               label='bkw period')

    ax.set_title(f'{interval=} ms')
    if i > 0:
        ax.set_ylabel('')

    slopes = misc.long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])

    # test forward (slopes > 0) and backward (slopes < 0) significance
    # plot significance bands above the data ylim to avoid overlap with axvspan
    for d, (data, tail) in enumerate([(slopes, 1), (-slopes, 1)]):

        # signflip permutation t-test
        _, pvals, _ = permutation_t_test(data, tail=tail, seed=i, verbose=False)
        clusters = misc.get_clusters(pvals < 0.05, start=1)
        bounds = [b for significant, b in clusters if significant]
        for b1, b2 in bounds:
            ax.axvspan(b1 - 0.5, b2 + 0.5, alpha=0.4,
                       color='black', ymin=0.9, ymax=1,
                       label='signflip-perm p<0.05')

        # cluster-based permutation test
        _, cl, cl_pvals, _ = permutation_cluster_1samp_test(data, tail=tail, seed=i, verbose=False)
        sig_clusters = [c[0] for c, p in zip(cl, cl_pvals) if p < 0.05]
        for b1, *_, b2 in sig_clusters:
            ax.axvspan(b1 + 0.5, b2 + 1.5, alpha=0.4, hatch='///',
                       color='grey', ymin=0.8, ymax=0.9,
                       label='cluster-perm p<0.05')

plotting.normalize_lims(axs)

# extend ylim so significance bands don't overlap with data/axvspan
for ax in axs:
    ymin, ymax = ax.get_ylim()
    margin = (ymax - ymin) * 0.15
    ax.set_ylim(ymin, ymax + margin)

by_label = {}
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(l, h)

fig.legend(by_label.values(), by_label.keys(), loc='upper right',
           bbox_to_anchor=(0.99, 0.99), ncol=2, fontsize='small')

fig.suptitle('fMRI SODA: Slopes during fast sequence presentation')
savefig(fig, settings.plot_dir + '/figures/soda_slopes_all.png')

stop


#%% participant-level: heatmap of slopes

fig, axs = plt.subplots(2, 3, figsize=[14, 8])
axs.flat[-1].axis('off')

for i, (interval, df_subj) in enumerate(df_slopes.groupby('interval')):

    df_subj = df_subj.groupby(['tr', 'subject']).mean().reset_index()
    slopes = misc.long_df_to_array(df_subj, 'slope',  columns=['subject', 'tr'])

    ax = axs.flat[i]
    ax.clear()
    tr_fwd = settings.exp_tr[interval]['fwd']

    df_interval = pd.DataFrame(slopes, columns=np.arange(1, 14), index=subjects)

    # sort based on forward period
    fwd_cols = list(range(tr_fwd[0]-1, tr_fwd[1]))
    df_interval = df_interval.assign(_sort=df_interval[fwd_cols].mean(axis=1)).sort_values('_sort', ascending=False).drop(columns='_sort')
    sns.heatmap(df_interval, cmap='RdBu_r', ax=ax)
    ax.set(ylabel='subject', xlabel='tr', title=f'{interval=} ms')

    ax.set_xticks(np.arange(13), np.arange(1, 14))
    ax.set_yticks(np.arange(len(subjects))[::5], subjects[::5])

plotting.normalize_lims(axs, which='v')

fig.suptitle('Mean slopes for all participants')
savefig(fig, settings.plot_dir + '/figures/soda_slopes_heatmap.png')



#%% participant-level: p values across trials
# plot p values per participant
pkl_pval = str(bids_base.copy().update(processing='soda', suffix='pvalues', extension='.pkl.gz'))

df_pval = pd.DataFrame()

for (interval, subj, period), df_sel in tqdm(list(df_slopes.groupby(['interval', 'subject', 'period'])),
                                             'calculating p values'):
    slopes = misc.long_df_to_array(df_sel, 'slope', ['trial', 'tr'])
    # flip sign if backward so we test in same direction
    slopes = slopes * (-1 if period=='bkw' else 1)
    t_obs, p, t_perms = permutation_t_test(slopes, tail=1, seed=int(subj), verbose=0)
    df_cond = pd.DataFrame({'subject': subj,
                            'period': period,
                            'interval': interval,
                            'p-value': min(p)}, index=[0])
    df_pval = pd.concat([df_pval, df_cond], ignore_index=True)

joblib.dump(df_pval, pkl_pval)

fig, axs = plt.subplots(2, 5, figsize=[16, 10], sharex=True)
fig.suptitle('Significant slopes for individual participants\' trials')

for i, ((period, interval), df_sel) in enumerate(df_pval.groupby(['period', 'interval'])):
    ax = axs.flat[i]
    plotting.tornadoplot(df_sel, x='p-value', y='subject', center=0,
                         low_label='p < 0', high_label='p > 0',
                         sort=True, ax=ax)
    ax.axvline(0.05, linestyle='--', c='darkred', linewidth=1.5, label='p=0.05')
    pct = (df_sel['p-value'] < 0.05).mean() * 100
    ax.set_title(f'{interval}\n{pct:.0f}% significant')
    ax.set_xlabel('p-value')

    if i == 0:
        ax.set_ylabel('subject')
    else:
        ax.set_ylabel('')

sns.despine()
savefig(fig, settings.plot_dir + '/figures/soda_participant_pvalues.png')


#%% heatmap for trials of selected participants
# np.random.seed(0)
# subjects_rnd = sorted(np.random.choice(subjects, 6, replace=False))

# fig, axs = plt.subplots(2, 3, figsize=[12, 8])
# axs.flat[-1].axis('off')

# df_sel = df_slopes[(df_slopes.interval == 0.512) & df_slopes.subject.isin(subjects_rnd)]

# for i, (subject, df_subj) in enumerate(df_sel.groupby('subject')):

#     slopes = [x.slope.values for _, x in df_subj.sort_values(['tr', 'trial']).groupby('trial')]
#     slopes = np.squeeze(slopes)

#     ax = axs.flat[i]
#     ax.clear()
#     sns.heatmap(pd.DataFrame(slopes, columns=np.arange(1, 14), index=np.arange(1, 16)),
#                 cmap='RdBu_r', ax=ax)
#     ax.set(ylabel='trial', xlabel='tr', title=f'{subject}')

#     ax.set_xticks(np.arange(13), np.arange(1, 14))

# plotting.normalize_lims(axs, which='v')

# fig.suptitle('Slopes of all trials for selected participants')
# savefig(fig, settings.plot_dir + '/figures/slopes_heatmap_trials.png')
#%% trial-level: individual trial peak slopes

import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

fig, axs = plt.subplots(2, n_iv, figsize=[20, 10])

for i, interval in enumerate(intervals):
    for j, period in enumerate(['fwd', 'bkw']):
        trs = settings.exp_tr[interval][period]
        lo, hi = trs[0] - 1, trs[1]

        df_sel = df_mean[df_mean.interval == interval]
        slopes = misc.long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])
        mean_slope = slopes[:, lo:hi].mean(-1)  # [n_subj]

        # permutation test per participant using their trial-level slopes
        pvals_subj = []
        df_long = pd.DataFrame()
        for s_idx, subject in enumerate(subjects):
            df_s = df_slopes[(df_slopes.interval == interval) & (df_slopes.subject == subject)]
            trial_slopes = np.array([g.sort_values('tr')['slope'].values for _, g in df_s.groupby('trial')])
            peak = trial_slopes[:, lo:hi].mean(-1) if len(trial_slopes) > 0 else np.array([])

            if len(trial_slopes) < 2:
                pvals_subj.append(np.nan)
            else:
                st_peak = peak[:, np.newaxis]
                tail = -1 if period == 'bkw' else 1
                t_obs, p, t_perms = permutation_t_test(st_peak, tail=tail, seed=s_idx, verbose=0)
                pvals_subj.append(p[0])

            df_long = pd.concat([df_long, pd.DataFrame({
                'peak_slope': peak,
                'subject': subject,
            })], ignore_index=True)

        # sort subjects ascending by mean slope
        sort_idx = np.argsort(mean_slope)[::-1 if period=='bkw' else 1]
        subjects_sorted = [subjects[k] for k in sort_idx]
        mean_slope_sorted = mean_slope[sort_idx]
        pvals_sorted = [pvals_subj[k] for k in sort_idx]

        norm = plt.Normalize(vmin=-np.abs(mean_slope_sorted).max(),
                             vmax=np.abs(mean_slope_sorted).max())
        # fwd: positive=green, negative=red; bkw: negative=green, positive=red
        if period == 'bkw':
            cmap = LinearSegmentedColormap.from_list('rg', ['seagreen', '#d0d0d0', 'crimson'])
        else:
            cmap = LinearSegmentedColormap.from_list('rg', ['crimson', '#d0d0d0', 'seagreen'])
        palette = [cmap(norm(v)) for v in mean_slope_sorted]

        ax = axs[j, i]
        ax.clear()
        sns.boxenplot(df_long, x='subject', y='peak_slope', order=subjects_sorted,
                      showfliers=False, palette=palette, ax=ax)
        sns.scatterplot(df_long, x='subject', y='peak_slope', alpha=0.5, c='gray',
                        s=10, ax=ax)
        for k, (subj, mean_val) in enumerate(zip(subjects_sorted, mean_slope_sorted)):
            ax.scatter(k, mean_val, marker='D', s=40, color='black', zorder=5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.axhline(0, linestyle='--', c='black')
        ax.set(title=f'{interval=} ms, {period}', ylabel='mean slope at expected TR')

        # annotate significant participants with a star above their column
        y_bottom, y_top = ax.get_ylim()
        y_star = y_top * 0.95
        for k, p in enumerate(pvals_sorted):
            if p < 0.05:
                ax.text(k, y_star, '*', ha='center', va='bottom', fontsize=14,
                        fontweight='bold', color='black')
        ax.set_ylim(y_bottom, y_top * 1.1)

legend_handles = [
    mlines.Line2D([], [], marker='o', color='gray', alpha=0.5, linestyle='None',
                  markersize=6, label='individual trials'),
    mlines.Line2D([], [], marker='D', color='black', linestyle='None',
                  markersize=6, label='mean'),
]
fig.legend(handles=legend_handles, loc='center right', ncol=1,
           bbox_to_anchor=(1.0, 0.5), frameon=True)
fig.tight_layout(rect=[0, 0, 0.92, 1])
savefig(fig, settings.plot_dir + '/figures/soda_trial_level_peaks.png')


#%% trial-level: correlate with behaviour

df_responses = bids_utils.load_responses_sequence_3T(subjects)

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4])

for i, interval in enumerate(intervals):
    tr_fwd = settings.exp_tr[interval]['fwd']
    lo, hi = tr_fwd[0] - 1, tr_fwd[1]
    df_sel = df_mean[df_mean.interval == interval]
    slopes = misc.long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])
    mean_slope = slopes[:, lo:hi].mean(-1)  # [n_subj]

    beh_acc = np.array([
        df_responses[(df_responses.subject == s) &
                     (df_responses.interval_time_y == interval)].accuracy.mean()
        for s in subjects
    ])

    ax = axs.flat[i]
    mask = ~np.isnan(beh_acc) & ~np.isnan(mean_slope)
    df_beh = pd.DataFrame({'behavioral accuracy': beh_acc, 'mean slope': mean_slope})
    r, p = pearsonr(beh_acc[mask], mean_slope[mask])
    sns.regplot(data=df_beh[mask], x='behavioral accuracy', y='mean slope',
                color=sns.color_palette()[1], scatter_kws={'alpha': 0.7},
                line_kws={'alpha': 0.7}, ax=ax)
    ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=12)
    ax.set(title=f'{interval=} ms')

fig.suptitle('Slopes correlations with behavioral accuracy')
savefig(fig, settings.plot_dir + '/figures/soda_correlations.png')


#%% trial-level: slopes at expected TR vs individual trial response

df_responses = bids_utils.load_responses_sequence_3T(subjects)

df_trial_resp = pd.DataFrame()

for interval in intervals:
    tr_fwd = settings.exp_tr[interval]['fwd']
    lo, hi = tr_fwd[0] - 1, tr_fwd[1]

    for s_idx, subject in enumerate(subjects):
        df_s = df_slopes[(df_slopes.interval == interval) & (df_slopes.subject == subject)]
        trial_slopes = np.array([g.sort_values('tr')['slope'].values for _, g in df_s.groupby('trial')])
        peak_slope = trial_slopes[:, lo:hi].mean(-1) if len(trial_slopes) > 0 else np.array([])

        df_subj = df_responses[(df_responses.subject == subject) &
                               (df_responses.interval_time == interval)].reset_index(drop=True)
        n_trials = min(len(df_subj), len(peak_slope))
        df_tmp = pd.DataFrame({
            'slope': peak_slope[:n_trials],
            'correct': df_subj.accuracy.values[:n_trials].astype(bool),
            'subject': subject,
            'interval': interval,
        })
        df_trial_resp = pd.concat([df_trial_resp, df_tmp], ignore_index=True)

df_trial_resp['response'] = df_trial_resp['correct'].map({True: 'correct', False: 'incorrect'})

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4], sharey=True)

for i, interval in enumerate(intervals):
    df_sel = df_trial_resp[df_trial_resp.interval == interval]
    ax = axs[i]

    sns.violinplot(data=df_sel, x='response', y='slope', hue='response',
                   palette={'correct': 'seagreen', 'incorrect': 'crimson'},
                   order=['correct', 'incorrect'], inner='quart',
                   legend=False, ax=ax)

    df_subj_mean = df_sel.groupby(['subject', 'response'])['slope'].mean().reset_index()
    sns.stripplot(data=df_subj_mean, x='response', y='slope',
                  order=['correct', 'incorrect'],
                  color='black', alpha=0.5, size=4, jitter=True, ax=ax)

    for subj in df_subj_mean.subject.unique():
        vals = df_subj_mean[df_subj_mean.subject == subj].set_index('response')['slope']
        if 'correct' in vals and 'incorrect' in vals:
            ax.plot([0, 1], [vals['correct'], vals['incorrect']],
                    c='black', alpha=0.2, linewidth=0.8)

    r, p = pearsonr(df_sel['correct'].astype(int), df_sel['slope'])
    ax.text(0.5, 0.98, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=9)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set(title=f'{interval=} ms', xlabel='')

sns.despine()
fig.suptitle('Trial slope at expected TR: correct vs incorrect responses')
savefig(fig, settings.plot_dir + '/figures/soda_vs_response.png')

#%% decoding accuracy vs TDLM sequenceness (fwd / bkw, mean across speed conditions)

import scipy.stats as stats
import matplotlib.pyplot as plt

dec_acc = np.array([bids_utils.get_decoding_accuracy_3T(s) for s in subjects])

# sequenceness at lag 1 TR (index 0), mean across 32–512 ms conditions
seq_fwd = np.mean([np.array([sf[iv][i][0] for i in range(len(subjects))])
                   for iv in plot_intervals], axis=0)
seq_bkw = np.mean([np.array([sb[iv][i][0] for i in range(len(subjects))])
                   for iv in plot_intervals], axis=0)

fig, axs = plt.subplots(1, 2, figsize=[10, 4])
for ax, seq, direction in zip(axs, [seq_fwd, seq_bkw], ['fwd', 'bkw']):
    r, p = stats.pearsonr(dec_acc, seq)
    ax.scatter(dec_acc, seq, color='steelblue', alpha=0.6, edgecolors='white', linewidths=0.5)
    m, b = np.polyfit(dec_acc, seq, 1)
    x_line = np.linspace(dec_acc.min(), dec_acc.max(), 100)
    ax.plot(x_line, m * x_line + b, color='steelblue', linewidth=1.5,
            linestyle='--' if p > 0.05 else '-')
    ax.set_xlabel('decoding accuracy')
    ax.set_ylabel('sequenceness (lag 1 TR)')
    ax.set_title(f'{direction}  r={r:.2f}, p={p:.3f}')

fig.suptitle('Decoding accuracy vs TDLM sequenceness (mean across 32–512 ms)')
fig.tight_layout()



#%% supplement: inter-interval: subject consistency across ISI conditions

# build [n_subj, n_intervals] matrix of peak slopes
peak_cols = []
for iv in intervals:
    df_sel = df_mean[df_mean.interval == iv]
    slopes = misc.long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])
    tr_fwd = settings.exp_tr[iv]['fwd']
    peak_cols.append(slopes[:, tr_fwd[0]-1:tr_fwd[1]].mean(-1))
peak_matrix = np.column_stack(peak_cols)  # [n_subj, n_intervals]

corr_r = np.zeros((n_iv, n_iv))
corr_p = np.zeros((n_iv, n_iv))
for a in range(n_iv):
    for b in range(n_iv):
        mask = ~(np.isnan(peak_matrix[:, a]) | np.isnan(peak_matrix[:, b]))
        if mask.sum() > 2:
            r, p = pearsonr(peak_matrix[mask, a], peak_matrix[mask, b])
        else:
            r, p = np.nan, np.nan
        corr_r[a, b] = r
        corr_p[a, b] = p

df_corr = pd.DataFrame(corr_r,
                       index=[f'{iv} ms' for iv in intervals],
                       columns=[f'{iv} ms' for iv in intervals])

fig, axs = plt.subplots(1, 2, figsize=[12, 4])

ax = axs[0]
mask_tri = np.zeros_like(corr_r, dtype=bool)
mask_tri[np.triu_indices_from(mask_tri)] = True
sns.heatmap(df_corr, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            mask=mask_tri, square=True, linewidths=0.5, ax=ax)
for a in range(n_iv):
    for b in range(n_iv):
        if a > b:
            stars = ('***' if corr_p[a, b] < 0.001 else
                     '**'  if corr_p[a, b] < 0.01  else
                     '*'   if corr_p[a, b] < 0.05  else '')
            ax.text(b + 0.5, a + 0.5, f'r={corr_r[a, b]:.2f}\n{stars}',
                    ha='center', va='center', fontsize=9)
ax.set(title='Inter-interval correlation (peak slope per subject)')

ax = axs[1]
ax.axis('off')
inner_fig, inner_axs = plt.subplots(n_iv, n_iv, figsize=[10, 10])
for a in range(n_iv):
    for b in range(n_iv):
        iax = inner_axs[a, b]
        if a == b:
            iax.hist(peak_matrix[:, a], bins=10, color='steelblue', edgecolor='white')
            iax.set_title(f'{intervals[a]} ms', fontsize=8)
        elif a > b:
            mask = ~(np.isnan(peak_matrix[:, b]) | np.isnan(peak_matrix[:, a]))
            r, p = pearsonr(peak_matrix[mask, b], peak_matrix[mask, a])
            iax.scatter(peak_matrix[mask, b], peak_matrix[mask, a],
                        s=15, alpha=0.7, color='steelblue')
            m, c = np.polyfit(peak_matrix[mask, b], peak_matrix[mask, a], 1)
            xs = np.linspace(peak_matrix[mask, b].min(), peak_matrix[mask, b].max(), 50)
            iax.plot(xs, m * xs + c, c='red', alpha=0.7, linewidth=1)
            stars = ('***' if p < 0.001 else '**' if p < 0.01 else
                     '*' if p < 0.05 else '')
            iax.set_title(f'r={r:.2f}{stars}', fontsize=7,
                          color='darkred' if p < 0.05 else 'black')
        else:
            iax.axis('off')
        iax.set_xticks([])
        iax.set_yticks([])

inner_fig.suptitle('Pairwise subject slopes across ISI conditions')
inner_fig.tight_layout()
savefig(inner_fig, settings.plot_dir + '/figures/soda_inter_interval_scatter.png')
savefig(fig, settings.plot_dir + '/figures/soda_inter_interval_corr.png')


#%% supplement: temporal drift: slopes vs trial position

df_drift = pd.DataFrame()

for interval in intervals:
    tr_fwd = settings.exp_tr[interval]['fwd']
    lo, hi = tr_fwd[0] - 1, tr_fwd[1]

    for s_idx, subject in enumerate(subjects):
        df_s = df_slopes[(df_slopes.interval == interval) & (df_slopes.subject == subject)]
        trial_slopes = np.array([g.sort_values('tr')['slope'].values for _, g in df_s.groupby('trial')])
        peak_slope = trial_slopes[:, lo:hi].mean(-1) if len(trial_slopes) > 0 else np.array([])
        n_trials = len(peak_slope)
        df_tmp = pd.DataFrame({
            'slope': peak_slope,
            'trial_position': np.arange(n_trials) / max(n_trials - 1, 1),
            'trial_index': np.arange(n_trials),
            'subject': subject,
            'interval': interval,
        })
        df_drift = pd.concat([df_drift, df_tmp], ignore_index=True)

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4], sharey=True)

for i, interval in enumerate(intervals):
    df_sel = df_drift[df_drift.interval == interval]
    ax = axs[i]

    sns.regplot(data=df_sel, x='trial_position', y='slope',
                scatter=False, lowess=True,
                line_kws={'color': 'black', 'linewidth': 2}, ax=ax)

    for subj in subjects:
        df_s = df_sel[df_sel.subject == subj]
        if len(df_s) < 3:
            continue
        sns.regplot(data=df_s, x='trial_position', y='slope',
                    scatter=False, lowess=True, truncate=True,
                    line_kws={'color': 'steelblue', 'linewidth': 0.7, 'alpha': 0.3}, ax=ax)

    mask = df_sel[['trial_position', 'slope']].notna().all(axis=1)
    r, p = pearsonr(df_sel.loc[mask, 'trial_position'], df_sel.loc[mask, 'slope'])
    ax.text(0.05, 0.97, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='left', fontsize=9)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set(title=f'{interval=} ms', xlabel='trial position (normalised)', ylabel='peak slope')

sns.despine()
fig.suptitle('Temporal drift: slopes across trial position (blue=individual, black=group)')
savefig(fig, settings.plot_dir + '/figures/soda_temporal_drift.png')


#%% supplement: trial-level: slopes vs reaction time

df_responses = bids_utils.load_responses_sequence_3T(subjects)

df_rt = pd.DataFrame()

for interval in intervals:
    tr_fwd = settings.exp_tr[interval]['fwd']
    lo, hi = tr_fwd[0] - 1, tr_fwd[1]

    for s_idx, subject in enumerate(subjects):
        df_s = df_slopes[(df_slopes.interval == interval) & (df_slopes.subject == subject)]
        trial_slopes = np.array([g.sort_values('tr')['slope'].values for _, g in df_s.groupby('trial')])
        peak_slope = trial_slopes[:, lo:hi].mean(-1) if len(trial_slopes) > 0 else np.array([])

        df_subj = df_responses[(df_responses.subject == subject) &
                               (df_responses.interval_time == interval)].reset_index(drop=True)
        n_trials = min(len(df_subj), len(peak_slope))
        rt_vals = pd.to_numeric(df_subj.duration.values[:n_trials], errors='coerce')
        rt_vals = zscore(rt_vals, nan_policy='omit')
        df_tmp = pd.DataFrame({
            'slope': peak_slope[:n_trials],
            'rt': rt_vals,
            'correct': df_subj.accuracy.values[:n_trials].astype(bool),
            'subject': subject,
            'interval': interval,
        })
        df_rt = pd.concat([df_rt, df_tmp], ignore_index=True)

# only use correct trials — RT on incorrect trials reflects a different process
df_rt_correct = df_rt[df_rt.correct]

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4])

for i, interval in enumerate(intervals):
    df_sel = df_rt_correct[df_rt_correct.interval == interval].dropna(subset=['rt'])
    ax = axs[i]

    sns.regplot(data=df_sel, x='rt', y='slope',
                scatter_kws={'color': 'steelblue', 'alpha': 0.3, 's': 10},
                line_kws={'color': 'red', 'alpha': 0.8}, ax=ax)

    if len(df_sel) > 2:
        r, p = pearsonr(df_sel['rt'], df_sel['slope'])
        ax.text(0.05, 0.97, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
                va='top', ha='left', fontsize=9)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set(title=f'{interval=} ms', xlabel='reaction time (z-scored)', ylabel='peak slope')

sns.despine()
fig.suptitle('Slopes vs reaction time (correct trials only)')
savefig(fig, settings.plot_dir + '/figures/soda_vs_rt.png')

#%% effect size visualization at expected TRs

import pingouin as pg

df_eff = pd.DataFrame()

for interval in intervals:
    df_sel = df_mean[df_mean.interval == interval]
    slopes = misc.long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])

    for period in ['fwd', 'bkw']:
        tr_range = settings.exp_tr[interval][period]
        lo, hi = tr_range[0] - 1, tr_range[1]
        slope_sign = -1 if period == 'bkw' else 1

        slopes_peak = slopes[:, lo:hi].mean(-1) * slope_sign  # [n_subj]
        n = len(slopes_peak)
        d = pg.compute_effsize(slopes_peak, y=0)
        ci_lo, ci_hi = pg.compute_esci(d, nx=n, ny=n, paired=True)
        df_tmp = pd.DataFrame({'interval': interval,
                               'cohens d': d,
                               'ci_lo': ci_lo,
                               'ci_hi': ci_hi,
                               'period': period}, index=[0])
        df_eff = pd.concat([df_eff, df_tmp])

fig, axs = plt.subplots(1, 2, figsize=[10, 4])

for j, period in enumerate(['fwd', 'bkw']):
    ax = axs[j]
    df_sel = df_eff[df_eff.period == period].reset_index(drop=True)
    x_pos = np.arange(len(df_sel))
    ax.bar(x_pos, df_sel['cohens d'].values,
           yerr=[df_sel['cohens d'].values - df_sel['ci_lo'].values,
                 df_sel['ci_hi'].values - df_sel['cohens d'].values],
           capsize=4, color=[settings.palette_wittkuhn2[i] for i in range(len(df_sel))],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sel['interval'].values)
    ax.set_xlabel('interval')
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Group mean effect size ({period}, Cohen's d with 95% CI)")
    ax.axhline(0, linestyle='--', alpha=0.5, c='black')

plotting.normalize_lims(axs)
savefig(fig, settings.plot_dir + '/figures/soda_effect_sizes.png')
