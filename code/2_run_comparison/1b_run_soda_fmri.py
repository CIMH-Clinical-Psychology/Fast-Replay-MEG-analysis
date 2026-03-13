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

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bids import BIDSLayout
from joblib import Parallel, delayed
from scipy.stats import pearsonr, zscore
from mne_bids import BIDSPath

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

# bids_base = BIDSPath(
#     root=layout.derivatives['derivatives'].root,
#     datatype='results',
#     subject='group',
#     task='main',
#     check=False
# )

# pkl_seq = bids_base.copy().update(processing='trials', suffix='sequenceness', extension='.pkl.gz')

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

#%% group-level: slope curves and signflip permutations


# build slopes_mean: dict mapping interval -> [n_subj, n_trs] array
slopes_mean = {}
for interval in intervals:
    df_iv = df_slopes[df_slopes.interval == interval]
    subj_slopes = []
    for subject in subjects:
        df_s = df_iv[df_iv.subject == subject]
        mean_slope = df_s.groupby('tr')['slope'].mean().sort_index().values
        subj_slopes.append(mean_slope)
    slopes_mean[interval] = np.array(subj_slopes)

fig, axs = plt.subplots(3, 5, figsize=[18, 7])

df_mean = df_slopes.groupby(['tr', 'interval', 'subject']).mean(True).reset_index()

for i, interval in enumerate(intervals):

    tr_axis = np.arange(1, 14)

    tr_fwd = settings.exp_tr[interval]['fwd']
    tr_bkw= settings.exp_tr[interval]['bkw']

    # top: group mean ± SEM slope curve
    ax = axs[0, i]

    df_sel = df_mean[df_mean.interval==interval]
    sns.lineplot(df_sel, x='tr', y='slope', color=settings.palette_wittkuhn2[i], ax=ax)

    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.axvspan(*tr_fwd, color=sns.color_palette()[1], linewidth=1, alpha=0.1,
               label=f'fwd TR={tr_fwd}')
    ax.axvspan(*tr_bkw, color=sns.color_palette()[2], linewidth=1, alpha=0.1,
               label=f'bkw TR={tr_bkw}')

    ax.set_title(f'{interval=} s')
    ax.legend()
    if i>0:
        ax.set_ylabel('')


    # bottom: signflip permutation test for forward
    ax = axs[1, i]
    slopes = misc.long_df_to_array(df_sel, 'slope',  columns=['subject', 'tr'])
    slopes = slopes[:, tr_fwd[0]-1: tr_fwd[1]].mean(1)  # subselect forward period
    p, t_obs, t_perms = soda.signflip_test_slopes(slopes, n_perms=1000, rng=i, alternative='greater')
    warnings.warn('WRONG T_OBS TAKEN')
    tdlm.plot_tval_distribution(t_obs, t_perms, color=sns.color_palette()[1], thresholds=[0.05, 0.001], ax=ax)
    ax.set_title('forward period t-val dist')

    # bottom: signflip permutation test for bakcward
    ax = axs[2, i]
    slopes = misc.long_df_to_array(df_sel, 'slope',  columns=['subject', 'tr'])
    slopes = slopes[:, tr_bkw[0]-1: tr_bkw[1]+1].mean(1)  # subselect backward period
    p, t_obs, t_perms = soda.signflip_test_slopes(slopes, n_perms=1000, rng=i, alternative='greater')
    tdlm.plot_tval_distribution(t_obs*-1, t_perms*-1, color=sns.color_palette()[2], ax=ax)
    ax.set_title('backward period t-val dist')

plotting.normalize_lims(axs[0])
plotting.normalize_lims(axs[1:3])

fig.suptitle('fMRI SODA: Slopes during fast sequence presentation')
savefig(fig, settings.plot_dir + '/figures/soda_slopes_all.png')

#%% slopes of participants as heatmap

fig, axs = plt.subplots(2, 3, figsize=[14, 8])
axs.flat[-1].axis('off')

for i, (interval, df_subj) in enumerate(df_slopes.groupby('interval')):

    df = df_subj.groupby(['tr', 'subject']).mean().reset_index()

    slopes = np.squeeze([df_subj.sort_values('timepoint').slope for _, df_subj in df.groupby('subject')])

    ax = axs.flat[i]
    ax.clear()
    tr_fwd = settings.exp_tr[interval]['fwd']

    df_interval = pd.DataFrame(slopes, columns=np.arange(1, 14), index=subjects)
    fwd_cols = list(range(tr_fwd[0], tr_fwd[1] + 1))
    df_interval = df_interval.assign(_sort=df_interval[fwd_cols].mean(axis=1)).sort_values('_sort', ascending=False).drop(columns='_sort')
    sns.heatmap(df_interval, cmap='RdBu_r', ax=ax)
    ax.set(ylabel='subject', xlabel='tr', title=f'{interval=} ms')

    ax.set_xticks(np.arange(13), np.arange(1, 14))
    ax.set_yticks(np.arange(len(subjects))[::5], subjects[::5])

plotting.normalize_lims(axs, which='v')

fig.suptitle('Mean slopes for all participants')
savefig(fig, settings.plot_dir + '/figures/slopes_heatmap.png')

#%% heatmap for trials of selected participants
np.random.seed(0)
subjects_rnd = sorted(np.random.choice(subjects, 6, replace=False))

fig, axs = plt.subplots(2, 3, figsize=[12, 8])
axs.flat[-1].axis('off')

df_sel = df_slopes[(df_slopes.interval == 0.512) & df_slopes.subject.isin(subjects_rnd)]

for i, (subject, df_subj) in enumerate(df_sel.groupby('subject')):

    slopes = [x.slope.values for _, x in df_subj.sort_values(['tr', 'trial']).groupby('trial')]
    slopes = np.squeeze(slopes)

    ax = axs.flat[i]
    ax.clear()
    sns.heatmap(pd.DataFrame(slopes, columns=np.arange(1, 14), index=np.arange(1, 16)),
                cmap='RdBu_r', ax=ax)
    ax.set(ylabel='trial', xlabel='tr', title=f'{subject}')

    ax.set_xticks(np.arange(13), np.arange(1, 14))

plotting.normalize_lims(axs, which='v')

fig.suptitle('Slopes of all trials for selected participants')
savefig(fig, settings.plot_dir + '/figures/slopes_heatmap_trials.png')



#%% group-level: bootstrap participants

def bootstrap_group(slopes, n_samples, n_draws=1000, rng=None):
    n_subj = len(slopes)
    rng = np.random.default_rng(rng)
    all_idx = rng.integers(0, n_subj, (n_draws, n_samples))

    ps = []
    ts = []
    for i in range(n_draws):
        idx = all_idx[i]
        slopes_sampled = slopes[idx]
        p, t_obs, t_perms = signflip_test_slopes(slopes_sampled, rng=rng, n_perms=1000)
        ps.append(p)
        ts.append(t_obs)
    return ps, ts


df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
slopes_mean, _ = build_slopes_arrays(df_slope, subjects, intervals)

df_power = pd.DataFrame()
for i, interval in enumerate(tqdm(intervals, desc='bootstrapping')):
    sm = slopes_mean[interval]  # [n_subj, n_trs]

    # signflip_test is already using all cores so no improvement from Parallel here
    res = Parallel(n_jobs=-1)(delayed(bootstrap_group)(sm, n, rng=n) for n in range(2, 60))
    power = [(np.array(p) < 0.05).mean() for p, _ in res]

    df_tmp = pd.DataFrame({'power': power,
                           'interval': interval,
                           'n_samples': range(2, 60)})
    df_power = pd.concat([df_power, df_tmp], ignore_index=True)

csv_power_group = settings.plot_dir + '/data/soda_3T_power_group.csv.gz'
df_power.to_csv(csv_power_group)
df_power = pd.read_csv(csv_power_group)

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 3], sharey=True, sharex=True)
for i, interval in enumerate(intervals):
    df_sel = df_power[df_power.interval == interval]
    n_sign = (df_sel.n_samples.iloc[(df_sel.power > 0.8).argmax()]
              if (df_sel.power > 0.8).any() else np.nan)

    ax = axs.flat[i]
    sns.lineplot(df_sel, x='n_samples', y='power', ax=ax)
    ax.axhline(0.8, c='gray', alpha=0.7, linestyle='--')
    if not np.isnan(n_sign):
        ax.axvline(n_sign, c='darkred', alpha=0.7, linestyle='--')
        ax.text(n_sign + 1, 0.7, f'n={n_sign:.0f}', c='darkred')
    ax.set(title=f'{interval=} s',
           ylabel='power\n(% significant)',
           xlabel='bootstrapped sample size')

fig.suptitle('Bootstrapped power analysis, resampled participants')
savefig(fig, settings.plot_dir + '/figures/soda_bootstrapped_grouplevel.png')


#%% participant-level: p values across trials

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
_, slopes_trials = build_slopes_arrays(df_slope, subjects, intervals)

df_pval = pd.DataFrame()

for i, interval in enumerate(intervals):
    pvals = []
    for subj_idx, subject in enumerate(subjects):
        st = slopes_trials[interval][subj_idx]  # [n_trials, n_trs]
        if len(st) < 2:
            pvals.append(np.nan)
            continue
        p, _, _ = signflip_test_slopes(st, rng=subj_idx)
        pvals.append(p)
    df_interval = pd.DataFrame({'p-value': pvals,
                                'subject': subjects,
                                'interval': f'{interval} s'})
    df_pval = pd.concat([df_pval, df_interval])

fig, ax = plt.subplots(1, 1, figsize=[8, 4])
ax.axhline(0.05, linestyle='--', c='darkred')
ax.set_yticks(list(ax.get_yticks()) + [0.05])
fig.suptitle('Significant slopes for individual participants\' trials')

sns.violinplot(df_pval, y='p-value', x='interval', hue='interval', palette='tab10',
               inner='point', inner_kws={'alpha': 0.4, 'linewidth': 1, 's': 20}, cut=0,
               bw_adjust=0.5)

intervals_order = df_pval['interval'].unique()
for x_pos, interval_label in enumerate(intervals_order):
    vals = df_pval[df_pval['interval'] == interval_label]['p-value'].dropna()
    pct = (vals < 0.05).mean() * 100
    y_top = vals.max()
    ax.text(x_pos, y_top + 0.02, f'p < 0.05\n{pct:.0f}%',
            ha='center', va='bottom', fontsize=9, color='darkred')

sns.despine()
savefig(fig, settings.plot_dir + '/figures/soda_participant_pvalues.png')


#%% participant-level: bootstrap trials

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
_, slopes_trials = build_slopes_arrays(df_slope, subjects, intervals)


def bootstrap_participants(slopes_list, n_samples, n_draws=1000, rng=None):
    """Randomly draw a subset of n_trials per participant with repetition."""
    rng = np.random.default_rng(rng)
    ps = []
    for draw in range(n_draws):
        px = []
        for subj_idx in range(len(slopes_list)):
            st = slopes_list[subj_idx]  # [n_trials, n_trs]
            n_trials = len(st)
            if n_trials < 2:
                px.append(np.nan)
                continue
            idx = rng.integers(0, n_trials, n_samples)
            sampled = st[idx]
            p, _, _ = signflip_test_slopes(sampled, rng=rng, n_perms=1000)
            px.append(p)
        ps.append(px)
    return ps


df_power = pd.DataFrame()

for i, interval in enumerate(intervals):
    st = slopes_trials[interval]  # list of [n_trials, n_trs] per subject
    res = Parallel(n_jobs=-1)(delayed(bootstrap_participants)(st, n, rng=n)
                              for n in tqdm(range(2, 65)))
    power = (np.array(res) < 0.05).mean(1)
    df_tmp = misc.to_long_df(power, ['n_samples', 'subject'],
                             n_samples=range(2, 65),
                             value_name='power')
    df_tmp['interval'] = interval
    df_power = pd.concat([df_power, df_tmp], ignore_index=True)

csv_power_part = settings.plot_dir + '/data/soda_3T_power_participants.csv.gz'
df_power.to_csv(csv_power_part)
df_power = pd.read_csv(csv_power_part)

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 3], sharey=True, sharex=True)

for i, interval in enumerate(intervals):
    df_sel = df_power[df_power.interval == interval]
    ax = axs[i]
    ax.clear()
    sns.lineplot(df_sel, x='n_samples', y='power', hue='subject', ax=ax)

plotting.normalize_lims(axs)
sns.despine()
fig.suptitle('Bootstrapped power analysis, resampled trials')
savefig(fig, settings.plot_dir + '/figures/soda_bootstrapped_participantlevel.png')


#%% trial-level: individual trial peak slopes

import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
slopes_mean, slopes_trials = build_slopes_arrays(df_slope, subjects, intervals)

fig, axs = plt.subplots(2, int(np.ceil(n_iv / 2)), figsize=[18, 10])

for i, interval in enumerate(intervals):
    exp_tr = get_expected_tr_idx(float(interval))
    lo = max(0, exp_tr - 1)
    hi = min(12, exp_tr + 2)
    sm = slopes_mean[interval]  # [n_subj, n_trs]
    mean_slope = sm[:, lo:hi].mean(-1)  # [n_subj]

    # per-trial long-form for boxen + scatter
    df_long = pd.DataFrame()
    for s_idx, subject in enumerate(subjects):
        st = slopes_trials[interval][s_idx]  # [n_trials, n_trs]
        ps_trial = st[:, lo:hi].mean(-1)
        df_long = pd.concat([df_long, pd.DataFrame({
            'peak_slope': ps_trial,
            'subject': subject,
        })], ignore_index=True)

    norm = plt.Normalize(vmin=-np.abs(mean_slope).max(), vmax=np.abs(mean_slope).max())
    cmap = LinearSegmentedColormap.from_list('rg', ['crimson', '#d0d0d0', 'seagreen'])
    palette = [cmap(norm(v)) for v in mean_slope]

    ax = axs.flat[i]
    ax.clear()
    sns.boxenplot(df_long, x='subject', y='peak_slope', showfliers=False,
                  palette=palette, ax=ax)
    sns.scatterplot(df_long, x='subject', y='peak_slope', alpha=0.5, c='gray', ax=ax)
    for j, (subj, mean_val) in enumerate(zip(subjects, mean_slope)):
        ax.scatter(j, mean_val, marker='D', s=40, color='black', zorder=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(0, linestyle='--', c='black')
    ax.set(title=f'interval {interval} s', ylabel='slope at expected TR')

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

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
slopes_mean, _ = build_slopes_arrays(df_slope, subjects, intervals)

df_responses = load_responses_sequence_3T(subjects)

fig, axs = plt.subplots(1, n_iv, figsize=[4 * n_iv, 4])

for i, interval in enumerate(intervals):
    exp_tr = get_expected_tr_idx(float(interval))
    lo, hi = max(0, exp_tr - 1), min(12, exp_tr + 2)
    sm = slopes_mean[interval]
    mean_slope = sm[:, lo:hi].mean(-1)  # [n_subj]

    beh_acc = np.array([
        df_responses[(df_responses.subject == s) &
                     (df_responses.interval_time == interval)].accuracy.mean()
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
    ax.set(title=f'{interval=} s')

fig.suptitle('Slopes correlations with behavioral accuracy')
savefig(fig, settings.plot_dir + '/figures/soda_correlations.png')


#%% trial-level: slopes at expected TR vs individual trial response

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
_, slopes_trials = build_slopes_arrays(df_slope, subjects, intervals)

df_responses = load_responses_sequence_3T(subjects)

df_trial_resp = pd.DataFrame()

for interval in intervals:
    exp_tr = get_expected_tr_idx(float(interval))
    lo, hi = max(0, exp_tr - 1), min(12, exp_tr + 2)

    for s_idx, subject in enumerate(subjects):
        st = slopes_trials[interval][s_idx]  # [n_trials, n_trs]
        peak_slope = st[:, lo:hi].mean(-1)

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
    ax.set(title=f'{interval=} s', xlabel='')

sns.despine()
fig.suptitle('Trial slope at expected TR: correct vs incorrect responses')
savefig(fig, settings.plot_dir + '/figures/soda_vs_response.png')


#%% inter-interval: subject consistency across ISI conditions

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
slopes_mean, _ = build_slopes_arrays(df_slope, subjects, intervals)

# build [n_subj, n_intervals] matrix of peak slopes
peak_matrix = np.column_stack([
    slopes_mean[iv][:,
        max(0, get_expected_tr_idx(float(iv)) - 1):
        min(12, get_expected_tr_idx(float(iv)) + 2)
    ].mean(-1)
    for iv in intervals
])  # [n_subj, n_intervals]

n_iv = len(intervals)
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
                       index=[f'{iv} s' for iv in intervals],
                       columns=[f'{iv} s' for iv in intervals])

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
            iax.set_title(f'{intervals[a]} s', fontsize=8)
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


#%% temporal drift: slopes vs trial position

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
_, slopes_trials = build_slopes_arrays(df_slope, subjects, intervals)

df_drift = pd.DataFrame()

for interval in intervals:
    exp_tr = get_expected_tr_idx(float(interval))
    lo, hi = max(0, exp_tr - 1), min(12, exp_tr + 2)

    for s_idx, subject in enumerate(subjects):
        st = slopes_trials[interval][s_idx]  # [n_trials, n_trs]
        peak_slope = st[:, lo:hi].mean(-1)
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
    ax.set(title=f'{interval=} s', xlabel='trial position (normalised)', ylabel='peak slope')

sns.despine()
fig.suptitle('Temporal drift: slopes across trial position (blue=individual, black=group)')
savefig(fig, settings.plot_dir + '/figures/soda_temporal_drift.png')


#%% trial-level: slopes vs reaction time

df_slopes = joblib.load(pkl_slope)
intervals = sorted(df_slopes.interval.unique())
_, slopes_trials = build_slopes_arrays(df_slope, subjects, intervals)

df_responses = load_responses_sequence_3T(subjects)

df_rt = pd.DataFrame()

for interval in intervals:
    exp_tr = get_expected_tr_idx(float(interval))
    lo, hi = max(0, exp_tr - 1), min(12, exp_tr + 2)

    for s_idx, subject in enumerate(subjects):
        st = slopes_trials[interval][s_idx]
        peak_slope = st[:, lo:hi].mean(-1)

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
    ax.set(title=f'{interval=} s', xlabel='reaction time (z-scored)', ylabel='peak slope')

sns.despine()
fig.suptitle('Slopes vs reaction time (correct trials only)')
savefig(fig, settings.plot_dir + '/figures/soda_vs_rt.png')
