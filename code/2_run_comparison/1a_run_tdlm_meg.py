# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run TDLM (Temporally Delayed Linear Modelling) on MEG data

@author: Simon.Kern
"""
import sys; sys.path.append('..')
import mne
from tqdm import tqdm
import pandas as pd
import bids_utils
import settings
from settings import layout_MEG as layout
import numpy as np
import seaborn as sns
from meg_utils import plotting
import matplotlib.pyplot as plt
from meg_utils import misc
from meg_utils.decoding import predict_proba_along
from meg_utils.plotting import savefig, normalize_lims
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tdlm
from tdlm.utils import seq2tf, num2char
from scipy.stats import zscore, pearsonr, ttest_1samp
import contextprofiler
from joblib import Parallel, delayed
from mne_bids import BIDSPath
import joblib
import pingouin as pg
from mne.stats import permutation_cluster_1samp_test
from mne.stats import permutation_t_test
#%% settings

sns.set_context('paper', font_scale=1.5)
# zscore = lambda x, **kwargs: x

normalization = 'lambda x: x/x.mean(0)'
subjects = [f'sub-{i:02d}' for i in range(1, 31)]
intervals = [32, 64, 128, 512]
palette = sns.color_palette()

bids_base = BIDSPath(
    root=layout.derivatives['derivatives'].root,
    datatype='results',
    subject='group',
    task='main',
    check=False
)

pkl_seq = bids_base.copy().update(processing='trials', suffix='sequenceness', extension='.pkl.gz')


try:
    # tr loading precomputed results
    res = joblib.load(pkl_seq)
    sf_trials = res['sf_trials']
    sb_trials = res['sb_trials']
    sf_mean = res['sf_mean']
    sb_mean = res['sb_mean']
    df_seq = res['df_seq']
except FileNotFoundError:
    print('results not computed yet')

#%% run TDLM calculation, save individual trial's sequenceness


df = pd.DataFrame()

sf_trials = {interval: [] for interval in intervals}
sb_trials = {interval: [] for interval in intervals}
sf_mean = {interval: [] for interval in intervals}
sb_mean = {interval: [] for interval in intervals}

# also save in one homungus data frame
df_seq = pd.DataFrame()

for subject in tqdm(layout.subjects, 'calculating TDLM'):
    # load classifier that we previously computed
    # use mean classifier, not individually tuned
    clf = bids_utils.load_latest_classifier(subject, which='mean')

    # load fast sequence trials
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject)
    probas = predict_proba_along(clf, data_x, -1)

    # forward and backward sequenceness saved in here
    sf_subj = {interval: [] for interval in intervals}
    sb_subj = {interval: [] for interval in intervals}

    df_subj = pd.DataFrame()

    # loop through trials and their behaviour
    for proba, df_trial in zip(probas, beh, strict=True):

        # normalize probability within trial as in Wittkuhn et al (2021)
        # i.e. divide by mean probability per class
        proba = eval(normalization)(proba)

        # transition matrix for this specific trial
        tf = seq2tf(''.join(num2char(df_trial.trigger.values)))
        interval = df_trial.interval_time.values[0]

        # maximum lag we are interested in is interval + 100 ms plus 50%
        # ie.. prevent interactions of double time lag from img1 to img3
        # image onsets |-----|-----|--
        # mag lag      |........|
        max_lag = int(((interval/10) + 10)*1.5)
        # CHANGED, take same fixed max lag for all conditions
        # max_lag = 91

        # only analyse the time window up to the length that the images shown
        # else we are analysing the buffer period already.
        # length = (interval_ms + 100ms image duration) * 5 images + mag lag
        length = int((interval+100)*5)//10 + max_lag

        sf_trial, sb_trial = tdlm.compute_1step(proba[:length, :], tf, n_shuf=100,
                                                max_lag=max_lag, rng=int(subject))

        df_subj = pd.concat([df_subj, pd.DataFrame({'sequenceness': sf_trial[0],
                                                   'direction': 'forward',
                                                   'interval': int(interval),
                                                   'lag': np.arange(max_lag+1)*10}),
                                     pd.DataFrame({'sequenceness': sb_trial[0],
                                                   'direction': 'backward',
                                                   'interval': int(interval),
                                                   'lag': np.arange(max_lag+1)*10})])

        # zscore trial
        sf_subj[interval] += [zscore(sf_trial, axis=-1, nan_policy='omit')]
        sb_subj[interval] += [zscore(sb_trial, axis=-1, nan_policy='omit')]
    df_subj['subject'] = subject
    df_seq = pd.concat([df_seq, df_subj], ignore_index=True)

    for interval in intervals:
        sf_trials[interval] += [sf_subj[interval]]
        sb_trials[interval] += [sb_subj[interval]]


for interval in intervals:
    sf_trials[interval] = np.array(sf_trials[interval])
    sb_trials[interval] = np.array(sb_trials[interval])
    sf_mean[interval] = np.mean(sf_trials[interval], axis=1)
    sb_mean[interval] = np.mean(sb_trials[interval], axis=1)


joblib.dump({'sf_trials': sf_trials,
             'sb_trials': sb_trials,
             'sf_mean': sf_mean,
             'sb_mean': sb_mean,
             'df_seq': df_seq
             }, pkl_seq.fpath)

#%% overview of mean sequenceness across intervals
res = joblib.load(pkl_seq)
sf_mean = res['sf_mean']
df_seq = res['df_seq']
df_mean = df_seq.groupby(['subject', 'interval', 'lag', 'direction']).mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=[8, 4])
sns.lineplot(df_mean[df_mean.direction=='forward'], x='lag', y='sequenceness', hue='interval',
             palette=settings.palette_wittkuhn2, ax=ax)
ax.axhline(0, linestyle='--', c='black', alpha=0.3)
ax.set(title='Sequenceness at different interval speeds')
ax.set_ylabel('Sequenceness')
ax.set_xticks(np.arange(0, 900, 100))
ax.set_xticks(np.arange(0, 900, 50), minor=True)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [f'{settings.format_interval(l)} ms' for l in labels],
          title='sequence speed')
savefig(fig, settings.plot_dir + '/figures/TDLM_sequenceness_MEG.png')

#%% group-level: sequenceness
res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']
sb_trials = res['sb_trials']
sf_mean = res['sf_mean']
sb_mean = res['sb_mean']
df_seq = res['df_seq']

#### sequenceness curves and signflip permutations
max_lags = [np.round(int(((iv/10) + 10))/3).astype(int) for iv in intervals]
mosaic = ''.join([f'{i}'*l for i, l in enumerate (max_lags)])

fig, axs = plt.subplot_mosaic(mosaic, figsize=[18, 3], dpi=80, sharey=True)
# fig, axs = plt.subplots(2, 2, figsize=[14, 8], dpi=80)
axs = list(axs.values())

for i, interval in enumerate(intervals):

    ax = axs[ i]

    tdlm.plotting.plot_sequenceness(sf_mean[interval], sb_mean[interval],
                                    which=['fwd', 'bkw'],
                                    ax=ax, plot95=False, rescale=False)

    # ax.annotate(f'{settings.format_interval(interval)} ms', xy=(0, 0.5), xycoords='axes fraction',
    #             xytext=(-0.2, 0.5), textcoords='axes fraction',
    #             fontsize=12, fontweight='bold', rotation=90,
    #             va='center', ha='center', annotation_clip=False)

    ax.axvspan((interval+100)-10, (interval+100)+10, color='black', alpha=0.15,
               label='expected lag', ymax=0.9)
    # ax.set_ylim([-2, 2])

    ax.set_xticks(np.arange(0, max(ax.get_xticks()), 100))
    ax.set_xticks(np.arange(0, max(ax.get_xticks()), 50), minor=True)
    ticks = list(ax.get_xticks())
    if interval==512:
        ticks.remove(600)
    ax.set_xticks(ticks + [int(interval+100)])

    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.set_ylabel('' if i>0 else 'Sequenceness\n(z-scored)')

    ax.set_title(f'{settings.format_interval(interval)} ms')
    # fig.suptitle('TDLM on MEG\nSequenceness during fast sequence presentation')

plotting.normalize_lims(axs, which='y')

# statistical testing
for i, interval in enumerate(intervals):
    ax = axs[ i]
    # first check forward sequencenes
    for d, sx in enumerate([sf_mean, sb_mean]):
        direction = 'bkw' if d else 'fwd'
        sx = sx[interval][:, 0, 1:]
        _, pvals, _ = permutation_t_test(sx, seed=i, verbose=False)
        significant= pvals<0.05
        clusters = misc.get_clusters(pvals<0.05, start=1)
        bounds = [b for significant, b in clusters if significant]
        for b1, b2 in bounds:
            ax.axvspan(b1*10-5, b2*10+5, alpha=0.3, linestyle='-', linewidth=3,
                       color=settings.color_bkw if d else settings.color_fwd,
                       ymin=0.95, ymax=1, label=f'{direction} signlip-perm<0.05')

        _, clusters, pvals, _ = permutation_cluster_1samp_test(sx, seed=i, verbose=False)
        clusters = [c[0] for c, p in zip(clusters, pvals) if p<0.05]
        for cl in clusters:
            b1, b2 = cl[0], cl[-1]
            ax.axvspan(b1*10+5, b2*10+15, alpha=0.6, linestyle='--', linewidth=3,
                       color=settings.color_bkw if d else settings.color_fwd,
                       ymin=0.9, ymax=0.95, label=f'{direction} cluster-perm<0.05')

        # t-test at expected lag ±10ms against zero
        expected_lag_ms = interval + 100
        center_idx = round(expected_lag_ms / 10) - 1  # index after 1: slicing
        lag_slice = slice(max(0, center_idx - 1), center_idx + 2)  # ±10ms
        mean_at_lag = sx[:, lag_slice].mean(axis=1)  # per subject mean across window
        t, p = ttest_1samp(mean_at_lag, 0)
        if p < 0.05:
            ylim = ax.get_ylim()
            ypos = ylim[0] + (ylim[1] - ylim[0]) * 0.05
            ax.plot(expected_lag_ms, ypos, '*', color='black', markersize=8,
                    zorder=10, label=f't-test p<0.05' if direction=='bkw' else None)
    ax.get_legend().remove()

by_label = {}
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(l, h)

fig.legend(by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(1, 0.95),
           ncol=3)
# fig.tight_layout()
savefig(fig, f'{settings.plot_dir}/figures/fast_images_sequenceness_all.png')




#%% participant-level: heatmap

# load precomputed results
res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']
sb_trials = res['sb_trials']
sf_mean = res['sf_mean']
sb_mean = res['sb_mean']
df_seq = res['df_seq']

# heatmap

vmin = df_seq.sequenceness.quantile(0.001)
vmax = df_seq.sequenceness.quantile(0.999)

# mosaic heatmap: forward and backward in one row with proportional widths
max_lags = [np.round(int(((iv/10) + 10))/3).astype(int) for iv in intervals]
for direction, s_mean, dir_label in [('fwd', sf_mean, 'Forward'),
                                      ('bkw', sb_mean, 'Backward')]:
    mosaic = ''.join([f'{i}'*l for i, l in enumerate(max_lags)])
    fig, axs = plt.subplot_mosaic(mosaic, figsize=[18, 4], dpi=80, sharey=True)
    axs = list(axs.values())

    # vmin = df_seq.sequenceness.quantile(0.05)
    # vmax = df_seq.sequenceness.quantile(0.95)

    for i, interval in enumerate(intervals):
        s_isi = s_mean[interval][:, 0, :]
        ax = axs[i]
        max_lag = s_isi.shape[-1]
        df_heatmap = pd.DataFrame(s_isi,
                                  columns=np.arange(0, max_lag*10, 10),
                                  index=layout.subjects)
        exp_lag = settings.exp_lag[interval]*10
        df_heatmap['sort_index'] = np.mean([df_heatmap[exp_lag+j] for j in [-10, 0, 10]], axis=0)
        df_heatmap = df_heatmap.sort_values('sort_index', ascending=False).drop('sort_index', axis=1)

        sns.heatmap(df_heatmap, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, ax=ax,
                    cbar=False)
        ax.set(ylabel='subject' if i==0 else '', xlabel='time lag', title=f'{settings.format_interval(interval)} ms')

        # y-ticks: every 3rd subject labeled (major), in-between as minor
        subjects_y = list(df_heatmap.index)
        y_positions = np.arange(len(subjects_y)) + 0.5
        ax.set_yticks(y_positions[::3])
        ax.set_yticklabels(subjects_y[::3], rotation=0)
        minor_mask = np.ones(len(y_positions), dtype=bool)
        minor_mask[::3] = False
        ax.set_yticks(y_positions[minor_mask], minor=True)

        # x-ticks: every 50 ms
        lags = np.array(df_heatmap.columns)
        x_positions = np.arange(len(lags)) + 0.5
        major_mask = lags % 50 == 0
        ax.set_xticks(x_positions[major_mask])
        ax.set_xticklabels(lags[major_mask], rotation=0)

    fig.suptitle(f'{dir_label} sequenceness across participants')
    savefig(fig, f'{settings.plot_dir}/figures/sequenceness_heatmap_subjects_{direction}_mosaic.png')

    # separate colorbar figure
fig_cb, ax_cb = plt.subplots(figsize=[0.1, 8])
fig_cb.colorbar(axs[-1].collections[0], cax=ax_cb)
savefig(fig_cb, f'{settings.plot_dir}/figures/sequenceness_heatmap_subjects_cbar.png')




#%% trial-subj overview: the complete plot

res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']

df_responses = bids_utils.load_responses_sequence_MEG()


fig, axs = plt.subplots(2, 2, figsize=[18, 10], sharey=True)

# save information per trial
for i, (interval, sf) in enumerate(sf_trials.items()):

    # we expect sequenceness to peak around this time +- 10 ms
    expected_lag = settings.exp_lag[interval]
    peak_seq = sf[:, :, 0, expected_lag-1: expected_lag+2].mean(-1)
    df = misc.to_long_df(peak_seq, ['subject', 'trial'],
                         value_name='sequenceness',
                         subject=subjects)

    mean_seq = peak_seq.mean(axis=1)  # [n_subj]

    # signflip test per participant using their trial-level sequenceness
    pvals_subj = []
    for j in range(len(subjects)):
        sf_subj = sf[j, :, 0, :]  # [n_trials, n_lags]
        p, _, _ = tdlm.signflit_test(sf_subj, rng=j)
        pvals_subj.append(p)

    # sort subjects ascending by mean sequenceness
    sort_idx = np.argsort(mean_seq)
    subjects_sorted = [subjects[k] for k in sort_idx]
    mean_seq_sorted = mean_seq[sort_idx]
    pvals_sorted = [pvals_subj[k] for k in sort_idx]

    norm = plt.Normalize(vmin=-np.abs(mean_seq_sorted).max(), vmax=np.abs(mean_seq_sorted).max())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rg', ['crimson', '#d0d0d0', 'seagreen'])
    palette = [cmap(norm(v)) for v in mean_seq_sorted]

    ax = axs.flat[i]
    ax.clear()
    sns.boxenplot(df, x='subject', y='sequenceness', order=subjects_sorted,
                  showfliers=False, palette=palette, ax=ax)
    sns.scatterplot(df, x='subject', y='sequenceness', alpha=0.5, c='gray', ax=ax)
    for j, (subj, mean_val) in enumerate(zip(subjects_sorted, mean_seq_sorted)):
        ax.scatter(j, mean_val, marker='D', s=40, color='black', zorder=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticks(np.arange(-2, 2, 0.25), minor=True)
    ax.axhline(0, linestyle='--', c='black')
    ax.set(title=f'{settings.format_interval(interval)} ms\n', ylabel='mean sequenceness at expected lag')

    # annotate significant participants with a star below their column
    y_bottom, y_top = ax.get_ylim()
    y_star = y_bottom + (y_top - y_bottom) * 0.02
    for j, p in enumerate(pvals_sorted):
        if p < 0.05:
            ax.text(j, y_star, '*', ha='center', va='bottom', fontsize=14,
                   fontweight='bold', color='black')

    # blue band at top showing decoding accuracy per participant
    dec_accs = [bids_utils.get_decoding_accuracy_MEG(s) for s in subjects_sorted]
    dec_accs = np.array(dec_accs)
    acc_norm = plt.Normalize(vmin=dec_accs.min(), vmax=dec_accs.max())
    acc_cmap = plt.cm.Blues
    y_bottom, y_top = ax.get_ylim()
    band_h = (y_top - y_bottom) * 0.03
    for j, acc in enumerate(dec_accs):
        ax.add_patch(plt.Rectangle((j - 0.5, y_top - band_h), 1, band_h,
                                   color=acc_cmap(acc_norm(acc)), clip_on=False))
    ax.set_ylim(y_bottom, y_top)


import matplotlib.lines as mlines
import matplotlib.patches as mpatches
legend_handles = [
    mlines.Line2D([], [], marker='o', color='gray', alpha=0.5, linestyle='None', markersize=6, label='individual trials'),
    mlines.Line2D([], [], marker='D', color='black', linestyle='None', markersize=6, label='mean'),
    mpatches.Patch(facecolor=plt.cm.Blues(0.6), label='decoding accuracy'),
]
fig.legend(handles=legend_handles, loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.49),
           frameon=True)
fig.tight_layout(rect=[0, 0, 0.90, 1])

# colorbar for mean sequenceness (next to upper row)
cbar_ax1 = fig.add_axes([0.91, 0.55, 0.015, 0.35])
sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm1, cax=cbar_ax1, label='mean sequenceness')

# colorbar for decoding accuracy (next to lower row)
cbar_ax2 = fig.add_axes([0.91, 0.08, 0.015, 0.35])
sm2 = plt.cm.ScalarMappable(cmap=acc_cmap, norm=acc_norm)
fig.colorbar(sm2, cax=cbar_ax2, label='decoding accuracy')

savefig(fig, settings.plot_dir + '/figures/sequenceness_trial_level.png', tight=False)



#%% mean sequenceness across conditions: correlate with decoding and behaviour

mean_seq_across = np.mean(all_mean_seq, axis=0)  # [n_subj]

fig, axs = plt.subplots(1, 2, figsize=[10, 4])

# left: decoding accuracy vs mean sequenceness across conditions
ax = axs[0]
df_dec = pd.DataFrame({'decoding accuracy': dec_acc, 'mean sequenceness': mean_seq_across})
r, p = pearsonr(dec_acc, mean_seq_across)
sns.regplot(data=df_dec, x='decoding accuracy', y='mean sequenceness',
            color=sns.color_palette()[0], scatter_kws={'alpha': 0.7},
            line_kws={'alpha': 0.7}, ax=ax)
ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
        va='top', ha='center', fontsize=12)
ax.set_yticks(np.arange(0, 9, 1)/10, minor=True)
ax.set_xticks(np.arange(8, 20, 2)/20)
ax.set_xticks(np.arange(8, 20, 1)/20, minor=True)
ax.set(title='decoding accuracy', ylabel='mean sequenceness\nat expected time lag')

# right: behavioral accuracy (mean across conditions) vs mean sequenceness
beh_acc_mean = np.array([df_responses[df_responses.subject == s[-2:]].accuracy.mean()
                         for s in subjects])
ax = axs[1]
df_beh = pd.DataFrame({'behavioral accuracy': beh_acc_mean, 'mean sequenceness': mean_seq_across})
r, p = pearsonr(beh_acc_mean, mean_seq_across)
sns.regplot(data=df_beh, x='behavioral accuracy', y='mean sequenceness',
            color=sns.color_palette()[1], scatter_kws={'alpha': 0.7},
            line_kws={'alpha': 0.7}, ax=ax)
ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
        va='top', ha='center', fontsize=12)
ax.set_ylabel('')
ax.set_yticks(np.arange(0, 9, 1)/10, minor=True)
# ax.set_xticks(np.arange(16, 40, 2)/20)
# ax.set_xticks(np.arange(8, 20, 1)/20, minor=True)
ax.set(title='behavioural accuracy')

# fig.suptitle('Mean sequenceness across conditions vs decoding and behavioral accuracy')
fig.tight_layout()
savefig(fig, settings.plot_dir + '/figues/sequenceness_correlations_mean.png')


#%% trial-level: correlate with behaviour and decoding accuracy

res = joblib.load(pkl_seq)
sf_mean = res['sf_mean']

df_responses = bids_utils.load_responses_sequence_MEG()

# peak decoding accuracy per subject from localizer cross-validation
dec_acc = np.array([bids_utils.get_decoding_accuracy_MEG(s) for s in subjects])

fig, axs = plt.subplots(2, 4, figsize=[18, 8],  sharey=True)

from scipy.stats import pearsonr

all_mean_seq = []  # collect per-interval mean_seq to average across conditions

for i, (interval, sf) in enumerate(sf_mean.items()):
    expected_lag = settings.exp_lag[interval]
    mean_seq = sf[:, 0, expected_lag-1:expected_lag+2].mean(-1)  # [n_subj]
    all_mean_seq.append(mean_seq)

    # row 0: localizer decoding accuracy vs sequenceness
    ax = axs[0, i]
    df_dec = pd.DataFrame({'decoding accuracy': dec_acc, 'mean sequenceness': mean_seq})
    r, p = pearsonr(dec_acc, mean_seq)
    sns.regplot(data=df_dec, x='decoding accuracy', y='mean sequenceness',
                color=sns.color_palette()[0], scatter_kws={'alpha': 0.7},
                line_kws={'alpha': 0.7}, ax=ax)
    ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=12)
    ax.set(title=f'{settings.format_interval(interval)} ms')
    ax.set_ylabel('sequenceness at expected lag' if i ==0 else '')
    # row 1: behavioral response accuracy vs sequenceness
    beh_acc = np.array([df_responses[(df_responses.subject == s[-2:]) &
                                     (df_responses.interval_time == interval)].accuracy.mean()
                        for s in subjects])
    ax = axs[1, i]
    df_beh = pd.DataFrame({'behavioral accuracy': beh_acc, 'mean sequenceness': mean_seq})
    r, p = pearsonr(beh_acc, mean_seq)
    sns.regplot(data=df_beh, x='behavioral accuracy', y='mean sequenceness',
                color=sns.color_palette()[1], scatter_kws={'alpha': 0.7},
                line_kws={'alpha': 0.7}, ax=ax)
    ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=12)
    ax.set(title=f'{settings.format_interval(interval)} ms')
    ax.set_ylabel('sequenceness at expected lag' if i ==0 else '')

fig.suptitle('Sequenceness correlations with decoding and behavioral accuracy')
savefig(fig, settings.plot_dir + '/supplement/sequenceness_correlations.png')

#%% supplement: trial-level: sequenceness vs reaction time
from scipy.stats import zscore, pearsonr

res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']

df_responses = bids_utils.load_responses_sequence_MEG()

df_rt = pd.DataFrame()


for interval, sf in sf_trials.items():
    expected_lag = settings.exp_lag[interval]
    peak_seq = sf[:, :, 0, expected_lag-1:expected_lag+2].mean(-1)  # [n_subj, n_trials]

    for s_idx, subject in enumerate(subjects):
        df_subj = df_responses[(df_responses.subject == subject[-2:]) &
                               (df_responses.interval_time == interval)].reset_index(drop=True)
        n_trials = min(len(df_subj), peak_seq.shape[1])
        df_subj.rt = zscore(df_subj.rt)
        df_tmp = pd.DataFrame({
            'sequenceness': peak_seq[s_idx, :n_trials],
            'rt': df_subj.rt.values[:n_trials],
            'correct': df_subj.accuracy.values[:n_trials].astype(bool),
            'subject': subject,
            'interval': interval,
        })
        df_rt = pd.concat([df_rt, df_tmp], ignore_index=True)

# only use correct trials — RT on incorrect trials reflects a different process
df_rt_correct = df_rt[df_rt.correct]

fig, axs = plt.subplots(1, 4, figsize=[16, 4])

for i, interval in enumerate(sf_trials):
    df_sel = df_rt_correct[df_rt_correct.interval == interval].dropna(subset=['rt'])
    ax = axs[i]

    sns.regplot(data=df_sel, x='rt', y='sequenceness',
                scatter_kws={'color': 'steelblue', 'alpha': 0.3, 's': 10},
                line_kws={'color': 'red', 'alpha': 0.8}, ax=ax)

    r, p = pearsonr(df_sel['rt'], df_sel['sequenceness'])
    ax.text(0.05, 0.97, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='left', fontsize=9)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set(title=f'{settings.format_interval(interval)} ms', xlabel='reaction time (s)', ylabel='peak sequenceness')

sns.despine()
fig.suptitle('Sequenceness vs reaction time (correct trials only)')
savefig(fig, settings.plot_dir + '/supplement/sequenceness_vs_rt.png')
