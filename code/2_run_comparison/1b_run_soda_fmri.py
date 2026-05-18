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
from settings import subjects_3T as subjects
from settings import intervals_3T as intervals


#%% settings
sns.set_context('paper', font_scale=1.5)

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

# annotate slope onset and offset period
df_slopes['period'] = None
for interval in intervals:
    for period in ['onset', 'offset']:
        trs = settings.exp_tr[interval][period]
        tr_range = list(range(trs[0], trs[1]+1))
        sel = (df_slopes.interval==interval) & df_slopes.tr.isin(tr_range)
        df_slopes.loc[sel, 'period'] = period
pkl_slopes = str(bids_base.copy().update(processing='soda', suffix='slopes', extension='.pkl.gz'))

joblib.dump(df_slopes, pkl_slopes)

df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()
intervals_plot = [iv for iv in intervals if iv != 2048]

#%% overview of mean slopes across intervals
df_slopes = joblib.load(pkl_slopes)
df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

fig, ax = plt.subplots(1, 1, figsize=[8, 6.4])

sns.lineplot(df_mean[df_mean.interval!=2048], x='tr', y='slope', hue='interval',
             palette=settings.palette_wittkuhn2, ax=ax)

ax.axhline(0, linestyle='--', c='black', alpha=0.3)
ax.set_xticks(np.arange(1, 14))
ax.set(title='Slopes at different interval speeds', xlabel='TR')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [f'{settings.format_interval(l)} ms' for l in labels],
          title='interval')
savefig(fig, settings.plot_dir + '/figures/SODA_slopes_3T.png')
#%% group-level: slope curves with cluster permutation and signflip tests

df_slopes = joblib.load(pkl_slopes)
df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

n_iv = len(intervals_plot)
fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 3.3])
if n_iv == 1:
    axs = [axs]


ttest_stars = []
for i, interval in enumerate(intervals_plot):

    tr_onset = settings.exp_tr[interval]['onset']
    tr_offset = settings.exp_tr[interval]['offset']

    ax = axs[i]

    df_sel = df_mean[df_mean.interval==interval]
    sns.lineplot(df_sel, x='tr', y='slope', color=settings.palette_wittkuhn2[i],
                 ax=ax)
    ax.set_xticks(np.arange(1, 14, 2))
    ax.set_xticks(np.arange(1, 14), minor=True)
    ax.set_yticks(np.arange(-2, 5, 2)/10)
    ax.set_yticks(np.arange(-2, 5, 1)/10, minor=True)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.axvspan(*tr_onset, color='#7a41a6', linewidth=1, alpha=0.2,
               label='onset period')
    ax.axvspan(*tr_offset, color='#23917f', linewidth=1, alpha=0.2,
               label='offset period')

    ax.set_title(f'{settings.format_interval(interval)} ms')
    if i > 0:
        ax.set_ylabel('')

    slopes = misc.long_df_to_array(df_sel, 'slope', columns=['subject', 'tr'])

    # test onset (slopes > 0) and offset (slopes < 0) significance
    # plot significance bands above the data ylim to avoid overlap with axvspan
    for d, (data, tail) in enumerate([(slopes, 1), (-slopes, 1)]):

        # signflip permutation t-test
        _, pvals, _ = permutation_t_test(data, tail=tail, seed=i, verbose=False)
        clusters = misc.get_clusters(pvals < 0.05, start=1)
        bounds = [b for significant, b in clusters if significant]
        for b1, b2 in bounds:
            ax.axvspan(b1 - 0.5, b2 + 0.5, alpha=0.4,
                       color='black', ymin=0.95, ymax=1,
                       label='signflip-perm p<0.05')

        # cluster-based permutation test
        _, cl, cl_pvals, _ = permutation_cluster_1samp_test(data, tail=tail, seed=i, verbose=False)
        sig_clusters = [c[0] for c, p in zip(cl, cl_pvals) if p < 0.05]
        for cl in sig_clusters:
            b1, b2 = cl[0], cl[-1]
            ax.axvspan(b1 + 0.5, b2 + 1.5, alpha=0.4, hatch='///',
                       color='grey', ymin=0.9, ymax=0.95,
                       label='cluster-perm p<0.05')

        # t-test on mean slope within expected period TRs
        period_key = 'onset' if d == 0 else 'offset'
        tr_range = settings.exp_tr[interval][period_key]
        lo, hi = tr_range[0] - 1, tr_range[1]
        mean_slope_period = slopes[:, lo:hi].mean(axis=1)  # per subject
        sign = -1 if d == 1 else 1
        t, p = ttest_1samp(mean_slope_period * sign, 0)
        if p < 0.05:
            tr_center = (tr_range[0] + tr_range[1]) / 2
            ttest_stars.append((i, tr_center))

plotting.normalize_lims(axs)

# extend ylim so significance bands don't overlap with data/axvspan
for ax in axs:
    ymin, ymax = ax.get_ylim()
    margin = (ymax - ymin) * 0.15
    ax.set_ylim(ymin, ymax + margin)

# plot t-test stars at consistent y position across all panels
ylim = axs[0].get_ylim()
ypos = ylim[0] + (ylim[1] - ylim[0]) * 0.05
for ax_idx, tr_center in ttest_stars:
    axs[ax_idx].plot(tr_center, ypos, '*', color='black', markersize=8,
                     zorder=10, label='t-test p<0.05')

by_label = {}
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(l, h)

leg = fig.legend(by_label.values(), by_label.keys(), loc='upper right',
           bbox_to_anchor=(0.99, 0.99), ncol=2, fontsize='small')

# fig.suptitle('SODA on fMRI\nSlopes during fast sequence presentation')
savefig(fig, settings.plot_dir + '/figures/soda_slopes_all.png')



#%% participant-level: heatmap of slopes

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4])
axs.flat[-1].axis('off')

for i, (interval) in enumerate(intervals[:-1]):
    df_iv = df_slopes[df_slopes.interval==interval]
    df_iv = df_iv.groupby(['tr', 'subject']).mean(True).reset_index()
    df_iv = df_iv[df_iv.interval!=2048]
    slopes = misc.long_df_to_array(df_iv, 'slope',  columns=['subject', 'tr'])

    ax = axs.flat[i]
    ax.clear()
    tr_onset = settings.exp_tr[interval]['onset']

    df_interval = pd.DataFrame(slopes, columns=np.arange(1, 14), index=subjects)

    # sort based on onset period
    onset_cols = list(range(tr_onset[0]-1, tr_onset[1]))
    df_interval = df_interval.assign(_sort=df_interval[onset_cols].mean(axis=1)).sort_values('_sort', ascending=False).drop(columns='_sort')
    sns.heatmap(df_interval, cmap='RdBu_r', ax=ax, cbar=False)
    ax.set(ylabel='subject', xlabel='tr', title=f'{settings.format_interval(interval)} ms')
    ax.set_xticks(np.arange(13) + 0.5, np.arange(1, 14))

    # y-ticks: every 3rd subject labeled (major), in-between as minor
    subjects_y = list(df_interval.index)
    y_positions = np.arange(len(subjects_y)) + 0.5
    ax.set_yticks(y_positions[::3])
    ax.set_yticklabels(subjects_y[::3], rotation=0)
    minor_mask = np.ones(len(y_positions), dtype=bool)
    minor_mask[::3] = False
    ax.set_yticks(y_positions[minor_mask], minor=True)

plotting.normalize_lims(axs, which='v')

fig_cb, ax_cb = plt.subplots(figsize=[10, 0.8], layout='constrained')
fig_cb.colorbar(axs.flat[0].collections[0], cax=ax_cb, label='slope',
                orientation='horizontal')
savefig(fig_cb, settings.plot_dir + '/figures/soda_slopes_heatmap_cbar.png', tight=False)

# fig.suptitle('Mean slopes for all participants')
savefig(fig, settings.plot_dir + '/figures/soda_slopes_heatmap.png')


#%% trial-subj overview: the complete plot

import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

fig, axs = plt.subplots(2, n_iv, figsize=[20, 8], sharey=True)

for i, interval in enumerate(intervals[:-1]):
    for j, period in enumerate(['onset', 'offset']):
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
                tail = -1 if period == 'offset' else 1
                t_obs, p, t_perms = permutation_t_test(st_peak, tail=tail, seed=s_idx, verbose=0)
                pvals_subj.append(p[0])

            df_long = pd.concat([df_long, pd.DataFrame({
                'peak_slope': peak,
                'subject': subject,
            })], ignore_index=True)

        # sort subjects ascending by mean slope
        sort_idx = np.argsort(mean_slope)[::-1 if period=='offset' else 1]
        subjects_sorted = [subjects[k] for k in sort_idx]
        mean_slope_sorted = mean_slope[sort_idx]
        pvals_sorted = [pvals_subj[k] for k in sort_idx]

        norm = plt.Normalize(vmin=-np.abs(mean_slope_sorted).max(),
                             vmax=np.abs(mean_slope_sorted).max())
        # onset: positive=green, negative=red; offset: negative=green, positive=red
        if period == 'offset':
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
        # x-ticks: every 2nd subject labeled (major, rotated), in-between as minor
        x_positions = np.arange(len(subjects_sorted))
        ax.set_xticks(x_positions[::2])
        ax.set_xticklabels(subjects_sorted[::2], rotation=90, fontsize=8)
        minor_mask = np.ones(len(x_positions), dtype=bool)
        minor_mask[::2] = False
        ax.set_xticks(x_positions[minor_mask], minor=True)
        ax.axhline(0, linestyle='--', c='black')
        ax.set(title=f'{settings.format_interval(interval)} ms, {period}', ylabel=f'mean slope')

        # annotate significant participants with a star below their column
        y_bottom, y_top = ax.get_ylim()
        y_star = y_bottom + (y_top - y_bottom) * 0.02
        for k, p in enumerate(pvals_sorted):
            if p < 0.05:
                ax.text(k, y_star, '*', ha='center', va='bottom', fontsize=14,
                        fontweight='bold', color='black')

        # blue band at top showing decoding accuracy per participant
        dec_accs = np.array([bids_utils.get_decoding_accuracies_3T().set_index('subject').loc[s, 'decoding accuracy'] for s in subjects_sorted])
        acc_norm = plt.Normalize(vmin=np.nanmin(dec_accs), vmax=np.nanmax(dec_accs))
        acc_cmap = plt.cm.Blues
        band_h = (y_top - y_bottom) * 0.03
        for k, acc in enumerate(dec_accs):
            ax.add_patch(plt.Rectangle((k - 0.5, y_top - band_h), 1, band_h,
                                       color=acc_cmap(acc_norm(acc)), clip_on=False))
        ax.set_ylim(y_bottom, y_top)

import matplotlib.patches as mpatches
legend_handles = [
    mlines.Line2D([], [], marker='o', color='gray', alpha=0.5, linestyle='None',
                  markersize=6, label='individual trials'),
    mlines.Line2D([], [], marker='D', color='black', linestyle='None',
                  markersize=6, label='mean'),
    mpatches.Patch(facecolor=plt.cm.Blues(0.6), label='decoding accuracy'),
]
fig.legend(handles=legend_handles, loc='center right', ncol=1,
           bbox_to_anchor=(1.0, 0.49), frameon=True)
fig.tight_layout(rect=[0.03, 0, 0.90, 1])

# row labels
fig.text(0.01, 0.75, 'onset period', va='center', ha='center', fontsize=14,
          rotation=90)
fig.text(0.01, 0.25, 'offset period', va='center', ha='center', fontsize=14,
         rotation=90)

# colorbar for mean slope (next to upper row)
cbar_ax1 = fig.add_axes([0.91, 0.55, 0.0075, 0.35])
sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb1 = fig.colorbar(sm1, cax=cbar_ax1, label='mean slope')
cb1.ax.tick_params(rotation=90)

# colorbar for decoding accuracy (next to lower row)
cbar_ax2 = fig.add_axes([0.91, 0.08, 0.0075, 0.35])
sm2 = plt.cm.ScalarMappable(cmap=acc_cmap, norm=acc_norm)
cb2 = fig.colorbar(sm2, cax=cbar_ax2, label='decoding accuracy')
cb2.ax.tick_params(rotation=90)

savefig(fig, settings.plot_dir + '/figures/soda_trial_level_peaks.png', tight=False)





#%% correlation decoding/behaviour ~ slopes

df_slopes = joblib.load(pkl_slopes)

# some subjects have their decoding accuracy missing, skip those

# behavioural accuracy
df_responses = bids_utils.load_responses_sequence_3T(subjects)
df_mean = df_responses.groupby(['subject', 'interval_time_y']).mean(True).reset_index()
df_beh = df_mean.groupby(['subject']).mean(True).reset_index()

# decoding accuracies
df_dec = bids_utils.get_decoding_accuracies_3T()
df_acc = df_beh.merge(df_dec, on='subject', how='inner')

df_slopes_mean = df_slopes.groupby(['interval', 'subject', 'period']).mean(True).reset_index()
df_slopes_mean = df_slopes_mean[df_slopes_mean.interval!=2048]
df_slopes_mean = df_slopes_mean.groupby(['subject', 'period']).mean(True).reset_index()

df_merged = df_slopes_mean.merge(df_acc, on='subject', how='right')

fig, axs = plt.subplots(2, 2, figsize=[8, 6.4])

for j, (period) in enumerate(['onset','offset']):
    # top row: decoding accuracy
    ax = axs[j, 0]
    df_sel = df_merged[df_merged.period==period]
    r, p = pearsonr(df_sel['decoding accuracy'], df_sel['slope'])
    sns.regplot(data=df_sel, x='decoding accuracy', y='slope',
                color=sns.color_palette()[0], scatter_kws={'alpha': 0.7},
                line_kws={'alpha': 0.7}, ax=ax)
    ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=12)
    ax.set(title=f'{period.capitalize()} slope ~ decoding acc.',
           ylabel=f'mean {period} slope')
    ax.set_xticks(np.arange(3, 10)/10, minor=True)
    if j==0:
        ax.set_xlabel('')

    # bottom row: behavioural accuracy
    ax = axs[j, 1]
    df_sel = df_merged[df_merged.period==period]
    r, p = pearsonr(df_sel['accuracy'], df_sel['slope'])
    sns.regplot(data=df_sel, x='accuracy', y='slope',
                color=sns.color_palette()[1], scatter_kws={'alpha': 0.7},
                line_kws={'alpha': 0.7}, ax=ax)
    ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=12)
    ax.set(title=f'{period.capitalize()} slope ~ behavioural acc.',
           xlabel='behavioural accuracy',
           ylabel=f'')
    ax.set_xticks(np.arange(5, 11)/10, minor=True)
    if j==0:
        ax.set_xlabel('')

# fig.suptitle('Mean slope across conditions vs decoding and behavioral accuracy')
fig.tight_layout()
savefig(fig, settings.plot_dir + '/figures/soda_correlations_mean.png')

#%% 2x4: behavioural accuracy vs slope per speed condition (onset / offset)
n_iv = 4
df_slopes_iv = df_slopes.groupby(['interval', 'subject', 'period']).mean(True).reset_index()
df_slopes_iv = df_slopes_iv[df_slopes_iv.interval != 2048]
df_merged_iv = df_slopes_iv.merge(df_acc, on='subject', how='inner')

fig, axs = plt.subplots(2, n_iv, figsize=[4 * n_iv, 8])
for i, interval in enumerate([32, 64, 128, 512]):
    for j, period in enumerate(['onset', 'offset']):
        ax = axs[j, i]
        df_sel = df_merged_iv[(df_merged_iv.interval == interval) &
                              (df_merged_iv.period == period)]
        r, p = pearsonr(df_sel['accuracy'], df_sel['slope'])
        sns.regplot(data=df_sel, x='accuracy', y='slope',
                    color=sns.color_palette()[1], scatter_kws={'alpha': 0.7},
                    line_kws={'alpha': 0.7}, ax=ax)
        ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
                va='top', ha='center', fontsize=12)
        ax.set(title=f'{settings.format_interval(interval)} ms')
        ax.set_ylabel(f'{period}\nmean slope' if i == 0 else '')
        ax.set_xlabel(f'behavioural performance' if j == 1 else '')

plotting.normalize_lims(axs.flatten())

fig.suptitle('Behavioural accuracy vs slope per speed condition')
fig.tight_layout()
savefig(fig, settings.plot_dir + '/supplement/soda_correlations_beh_per_interval.png')

#%% 2x4: decoding accuracy vs slope per speed condition (onset / offset)
n_iv = 4
fig, axs = plt.subplots(2, n_iv, figsize=[4 * n_iv, 8])
for i, interval in enumerate([32, 64, 128, 512]):
    for j, period in enumerate(['onset', 'offset']):
        ax = axs[j, i]
        df_sel = df_merged_iv[(df_merged_iv.interval == interval) &
                              (df_merged_iv.period == period)]
        r, p = pearsonr(df_sel['decoding accuracy'], df_sel['slope'])
        sns.regplot(data=df_sel, x='decoding accuracy', y='slope',
                    color=sns.color_palette()[0], scatter_kws={'alpha': 0.7},
                    line_kws={'alpha': 0.7}, ax=ax)
        ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
                va='top', ha='center', fontsize=12)
        ax.set(title=f'{settings.format_interval(interval)} ms')
        ax.set_ylabel(f'{period}\nmean slope' if i == 0 else '')
        ax.set_xlabel(f'decoding accuracy' if j == 1 else '')

plotting.normalize_lims(axs.flatten())
fig.suptitle('Decoding accuracy vs slope per speed condition')
fig.tight_layout()
savefig(fig, settings.plot_dir + '/supplement/soda_correlations_dec_per_interval.png')

#%% supplement: trial-level: slopes vs reaction time

df_responses = bids_utils.load_responses_sequence_3T(subjects)

df_rt = pd.DataFrame()

for interval in intervals[:-1]:
    for period in ['onset', 'offset']:
        tr_range = settings.exp_tr[interval][period]
        lo, hi = tr_range[0] - 1, tr_range[1]

        for s_idx, subject in enumerate(subjects):
            df_s = df_slopes[(df_slopes.interval == interval) & (df_slopes.subject == subject)]
            trial_slopes = np.array([g.sort_values('tr')['slope'].values for _, g in df_s.groupby('trial')])
            if len(trial_slopes) == 0:
                continue
            peak_slope = trial_slopes[:, lo:hi].mean(-1)

            df_subj = df_responses[(df_responses.subject == subject) &
                                   (df_responses.interval_time_y == interval/1000)].reset_index(drop=True)
            n_trials = min(len(df_subj), len(peak_slope))
            if n_trials == 0:
                continue
            rt_vals = pd.to_numeric(df_subj.response_time.values[:n_trials], errors='coerce')
            rt_vals = zscore(rt_vals, nan_policy='omit')
            df_tmp = pd.DataFrame({
                'slope': peak_slope[:n_trials],
                'rt': rt_vals,
                'correct': df_subj.accuracy.values[:n_trials].astype(bool),
                'subject': subject,
                'interval': interval,
                'period': period,
            })
            df_rt = pd.concat([df_rt, df_tmp], ignore_index=True)

# only use correct trials — RT on incorrect trials reflects a different process
df_rt_correct = df_rt[df_rt.correct]

n_iv = 4
fig, axs = plt.subplots(2, n_iv, figsize=[4 * n_iv, 8], sharey='row')

for j, period in enumerate(['onset', 'offset']):
    for i, interval in enumerate(intervals[:-1]):
        df_sel = df_rt_correct[(df_rt_correct.interval == interval) &
                               (df_rt_correct.period == period)].dropna(subset=['rt'])
        ax = axs[j, i]

        sns.regplot(data=df_sel, x='rt', y='slope',
                    scatter_kws={'color': 'steelblue', 'alpha': 0.3, 's': 10},
                    line_kws={'color': 'red', 'alpha': 0.8}, ax=ax)

        if len(df_sel) > 2:
            r, p = pearsonr(df_sel['rt'], df_sel['slope'])
            ax.text(0.05, 0.97, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
                    va='top', ha='left', fontsize=12)
        ax.axhline(0, linestyle='--', c='black', alpha=0.4)
        ax.set_title(f'{settings.format_interval(interval)} ms' if j == 0 else '')
        ax.set_ylabel(f'{period}\nmean slope' if i == 0 else '')
        ax.set_xlabel('reaction time (z-scored)' if j == 1 else '')

sns.despine()
fig.suptitle('Slopes vs reaction time (correct trials only)')
fig.tight_layout()
savefig(fig, settings.plot_dir + '/supplement/soda_vs_rt.png')
