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
from scipy.stats import zscore, pearsonr
import contextprofiler
from joblib import Parallel, delayed
from mne_bids import BIDSPath
import joblib
import pingouin as pg
from mne.stats import permutation_cluster_1samp_test
from mne.stats import permutation_t_test
#%% settings

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
stop
#%% run TDLM calculation, save individual trial's sequenceness


df = pd.DataFrame()

sf_trials = {interval: [] for interval in intervals}
sb_trials = {interval: [] for interval in intervals}
sf_mean = {interval: [] for interval in intervals}
sb_mean = {interval: [] for interval in intervals}

# also save in one homungus data frame
df_seq = pd.DataFrame()

for subject in tqdm(layout.subjects):
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

        # only analyse the time window up to the length that the images shown
        # else we are analysing the buffer period already.
        # length = (interval_ms + 100ms image duration) * 5 images + mag lag
        length = int((interval+100)*5)//10 + max_lag

        sf_trial, sb_trial = tdlm.compute_1step(proba[:length, :], tf, n_shuf=100,
                                                max_lag=max_lag, rng=int(subject))

        df_subj = pd.concat([df_subj, pd.DataFrame({'sequenceness': sf_trial[0],
                                                   'direction': 'forward',
                                                   'interval': interval,
                                                   'lag': np.arange(max_lag+1)*10}),
                                     pd.DataFrame({'sequenceness': sb_trial[0],
                                                   'direction': 'backward',
                                                   'interval': interval,
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


#%% group-level: sequenceness
res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']
sb_trials = res['sb_trials']
sf_mean = res['sf_mean']
sb_mean = res['sb_mean']
df_seq = res['df_seq']

#### sequenceness curves and signflip permutations
fig, axs = plt.subplots(2, 2, figsize=[14, 8], dpi=80)
axs = axs.flatten()

for i, interval in enumerate(intervals):

    ax = axs[ i]

    tdlm.plotting.plot_sequenceness(sf_mean[interval], sb_mean[interval],
                                    which=['fwd', 'bkw'],
                                    ax=ax, plot95=False, rescale=False)

    ax.annotate(f'{interval} ms', xy=(0, 0.5), xycoords='axes fraction',
                xytext=(-0.2, 0.5), textcoords='axes fraction',
                fontsize=12, fontweight='bold', rotation=90,
                va='center', ha='center', annotation_clip=False)

    ax.axvspan((interval+100)-10, (interval+100)+10, color='black', alpha=0.3,
               label='expected lag', ymax=0.8)
    # ax.set_ylim([-2, 2])
    # if interval==512:
    ax.set_xticks(np.arange(0, max(ax.get_xticks()), 100))

    ax.set_xticks(list(ax.get_xticks()) + [int(interval+100)])

    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.set_title(f'{interval=} ms')
    # Apply updated ticks
    fig.suptitle('MEG: Sequenceness during fast sequence presentation')

    # signflip permutation
plotting.normalize_lims(axs)

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
                       ymin=0.9, ymax=1, label=f'{direction} signlip-perm<0.05')

        _, clusters, pvals, _ = permutation_cluster_1samp_test(sx, seed=i, verbose=False)
        clusters = [c[0] for c, p in zip(clusters, pvals) if p<0.05]
        for b1, *_, b2 in clusters:
            ax.axvspan(b1*10+5, b2*10+15, alpha=0.6, linestyle='--', linewidth=3,
                       color=settings.color_bkw if d else settings.color_fwd,
                       ymin=0.8, ymax=0.9, label=f'{direction} cluster-perm<0.05')
    ax.get_legend().remove()

by_label = {}
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        by_label.setdefault(l, h)

fig.legend(by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(0.95, 0.85))

savefig(fig, f'{settings.plot_dir}/figures/fast_images_sequenceness_all.png')


#%% DEPRECATED group-level: bootstrap participants

def bootstrap_group(sf, n_samples, n_draws=1000, rng=None):
    n_subj = len(sf)
    rng = np.random.default_rng(rng)
    all_idx = rng.integers(0, n_subj, (n_draws, n_samples))

    ps = []
    ts = []
    for i in range(n_draws):
        idx = all_idx[i]
        sf_sampled = sf[idx]
        p, t_obs, t_perms = tdlm.signflit_test(sf_sampled, rng=rng, n_perms=1000)
        ps += [p]
        ts += [t_obs]
    return ps, ts

df_power = pd.DataFrame()
for i, interval in enumerate(tqdm(intervals, desc='bootstrapping')):
    sf = sf_mean[interval][:, 0, :]

    # signflip_test is already using all cores, so no improvements here.
    res = Parallel(n_jobs=-1)(delayed(bootstrap_group)(sf, n, rng=n) for n in range(2, 60))
    power = [(np.array(p)<0.05).mean() for p, _ in res]

    df_tmp = pd.DataFrame({'power': power,
                           'interval': interval,
                           'n_samples': range(2, 60)})
    df_power = pd.concat([df_power, df_tmp], ignore_index=True)

csv_file = bids_base.copy().update(processing='group', suffix='power', extension='csv.gz')
csv_file.mkdir()
df_power.to_csv(csv_file)

df_power = pd.read_csv(csv_file)

fig, axs = plt.subplots(1, 4, figsize=[12, 3], sharey=True, sharex=True)
for i, interval in enumerate(intervals):
    df_sel = df_power[df_power.interval==interval]
    # sample size that reaches 80% power
    n_sign = df_sel.n_samples.iloc[(df_sel.power>0.8).argmax()]

    ax = axs.flat[i]
    sns.lineplot(df_sel, x='n_samples', y='power', ax=ax)
    ax.axhline(0.8, c='gray', alpha=0.7, linestyle='--')
    ax.axvline(n_sign, c='darkred', alpha=0.7, linestyle='--')
    ax.text(n_sign + 1, 0.7, f'n={n_sign}', c='darkred')
    # ax.set_xticks(list(ax.get_xticks()) + [n_sign])
    ax.set_title(f'{interval=} ms')
    ax.set_ylabel('power\n(% significant)')
    ax.set_xlabel('bootstrapped sample size')

fig.suptitle('Bootstrapped power analysis, resampled participants')
savefig(fig, settings.plot_dir + '/figures/boostrapped_grouplevel.png')


#%% participant-level: heatmap

# load precomputed results
res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']
sb_trials = res['sb_trials']
sf_mean = res['sf_mean']
sb_mean = res['sb_mean']
df_seq = res['df_seq']

# heatmap
fig, axs = plt.subplots(2, 2, figsize=[12, 8])

vmin = df_seq.sequenceness.quantile(0.001)
vmax = df_seq.sequenceness.quantile(0.999)

for i, interval in enumerate(df_seq.interval.unique()):
    sf_isi = sf_mean[interval][:, 0, :]
    ax = axs.flat[i]
    ax.clear()
    max_lag = sf_isi.shape[-1]
    df_heatmap = pd.DataFrame(sf_isi,
                              columns=np.arange(0, max_lag*10, 10),
                              index=layout.subjects,)

    df_heatmap

    sns.heatmap(df_heatmap, cmap='RdBu_r', center=0,  vmin=vmin, vmax=vmax, ax=ax)
    # ax.set_xticks(np.arange(0, max_lag)[::5 if interval<500 else 10], np.arange(0, max_lag*10, 10)[::5 if interval<500 else 10])
    # ax.set_yticks(np.arange(len(layout.subjects))[::2], layout.subjects[::2])
    ax.set(ylabel='subject', xlabel='time lag', title=f'{interval=} ms')

# plotting.normalize_lims(axs, which='v')


fig.suptitle('Forward sequenceness across participants')
savefig(fig, settings.plot_dir + '/figures/sequenceness_heatmap_subjects.png')

#%% participant-level: p values across trials

df_pval = pd.DataFrame()

for i, (interval, sf_isi) in enumerate(sf_trials.items()):
    pvals = []
    for subj, sf in enumerate(sf_isi[:, :, 0, :]):
        p, t_obs, t_perms = tdlm.signflit_test(sf, rng=subj)
        pvals += [p]
    df_interval = pd.DataFrame({'p-value': pvals,
                  'subject': range(1, 31),
                  'interval': f'{interval} ms'})
    df_pval = pd.concat([df_pval, df_interval])

fig, axs = plt.subplots(1, 4, figsize=[16, 6], sharey=False)
fig.suptitle('Significant sequenceness for individual participants\' trials')

for i, interval_label in enumerate(df_pval['interval'].unique()):
    ax = axs[i]
    df_iv = df_pval[df_pval['interval'] == interval_label].copy()
    plotting.tornadoplot(df_iv, x='p-value', y='subject', center=0,
                         low_label='p < 0', high_label='p > 0',
                         sort=True, ax=ax)
    ax.axvline(0.05, linestyle='--', c='darkred', linewidth=1.5, label='p=0.05')
    pct = (df_iv['p-value'] < 0.05).mean() * 100
    ax.set_title(f'{interval_label}\n{pct:.0f}% significant')
    ax.set_xlabel('p-value')
    if i == 0:
        ax.set_ylabel('subject')
    else:
        ax.set_ylabel('')

sns.despine()
fig.tight_layout()
savefig(fig, settings.plot_dir + '/figures/sequenceness_participant_pvalues.png')

#%% DEPRECATED participant level: bootstrap one example participant
res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']
df_seq = res['df_seq']

def bootstrap_one(sf, n_samples, n_draws=10000, rng=None):
    """Construct one participant by drawing from trials of all participants """
    n_subj, n_trials, n_features = sf.shape
    n_trials_all = n_subj*n_trials

    all_trials = sf.reshape([-1, n_features])

    rng = np.random.default_rng(rng)
    all_idx = rng.integers(0, n_trials_all, [n_draws, n_samples])

    ps = []
    for draw in range(n_draws):
        idx = all_idx[draw]
        sf_sampled = all_trials[idx]
        p, t_obs, t_perms = tdlm.signflit_test(sf_sampled, rng=rng, n_perms=1000)
        ps += [p]
    return ps

df_power = pd.DataFrame()

for i, (interval, sf_interval) in enumerate(sf_trials.items()):

    sf = sf_interval[:, :, 0, 1:]
    # signflip_test is already using all cores, so no improvements here.
    res = Parallel(n_jobs=-1)(delayed(bootstrap_one)(sf, n, rng=n) for n in tqdm(range(2, 16*8)))
    df_tmp = misc.to_long_df(res, ['sample_size', 'shuffle'], value_name='p',
                            sample_size=range(2, 16*8))
    df_tmp['power'] = df_tmp.p<0.05
    df_tmp['interval'] = interval
    df_power = pd.concat([df_power, df_tmp], ignore_index=True)

fig, axs = plt.subplots(1, 2, figsize=[10, 4])
ax = axs[0]
sns.lineplot(df_power, x='sample_size', y='p', hue='interval', ax=ax)
ax.axhline(0.05, linestyle='--', c='gray', alpha=0.5, label='p=0.05')
ax.legend()
ax = axs[1]
sns.lineplot(df_power, x='sample_size', y='power', hue='interval', ax=ax)
ax.axhline(0.05, linestyle='--', c='gray', alpha=0.5, label='p=0.05')
ax.legend()
fig.suptitle('Bootstrapped virtual participant by sampling from all trials')
savefig(fig, settings.plot_dir + '/supplement/bootstrap_participant_from_all_trials.png')

#%% DEPRECATED participant-level: bootstrap trials
res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']
df_seq = res['df_seq']

def bootstrap_participants(sf, n_samples, n_draws=1000, rng=None):
    """  Randomly draw a subset of n_trials per participant with repetition. """
    n_subj, n_trials, n_features = sf.shape
    rng = np.random.default_rng(rng)
    all_idx = rng.integers(0, n_trials, (n_draws, n_subj, n_samples))

    ps = []
    for draw in range(n_draws):
        px = []
        for subj in range(n_subj):
            idx = all_idx[draw, subj]
            sf_sampled = sf[subj, idx]
            p, t_obs, t_perms = tdlm.signflit_test(sf_sampled, rng=rng, n_perms=1000)
            px += [p]
        ps += [px]
    return ps

df_power = pd.DataFrame()

for i, (interval, sf_interval) in enumerate(sf_trials.items()):

    sf = sf_interval[:, :, 0, 1:]
    # signflip_test is already using all cores, so no improvements here.
    res = Parallel(n_jobs=-1)(delayed(bootstrap_participants)(sf, n, rng=n) for n in tqdm(range(2, 65)))
    power = (np.array(res)<0.05).mean(1)
    df_tmp = misc.to_long_df(power, [ 'n_samples', 'subject',],
                             n_samples= range(2, 65),
                             value_name='power')
    df_tmp['interval'] = interval
    df_power = pd.concat([df_power, df_tmp], ignore_index=True)


csv_file = bids_base.copy().update(processing='participants', suffix='power', extension='csv.gz')
csv_file.mkdir()
df_power.to_csv(csv_file)
df_power = pd.read_csv(csv_file)

fig, axs = plt.subplots(1, 4, figsize=[12, 3], sharey=True, sharex=True)

for i, interval in enumerate(sf_trials):
    df_sel = df_power[df_power.interval==interval]
    # sample size that reaches 80% power
    ax = axs[i]
    ax.clear()
    # sns.scatterplot(df_sel, x='n_samples', y='power', hue='subject', ax=ax)
    sns.lineplot(df_sel, x='n_samples', y='power', hue='subject', ax=ax)

plotting.normalize_lims(axs)
sns.despine()
fig.suptitle('Bootstrapped power analysis, resampled trials')
savefig(fig, settings.plot_dir + '/figures/boostrapped_participantlevel.png')


#%% trial-level: heatmap
np.random.seed(0)
subjects_rnd = sorted(np.random.choice(layout.subjects, 6, replace=False))

fig, axs = plt.subplots(2, 3, figsize=[10, 6])

for i, subject in enumerate(tqdm(subjects_rnd)):
    subject = str(subject)
    # load classifier that we previously computed
    clf = bids_utils.load_latest_classifier(subject)
    data_x, data_y, beh = bids_utils.load_fast_sequences(subject, intervals=[32])

    probas = clf.predict_proba(data_x.transpose(0, 2, 1).reshape([-1, data_x.shape[1]])).reshape([data_x.shape[0], data_x.shape[-1], -1])

    tf_trials = [seq2tf(''.join(num2char(df_trial.trigger.values))) for df_trial in beh]

    sf_subj = []
    sb_subj = []

    for proba, df_trial in zip(tqdm(probas), beh, strict=True):
        proba = eval(normalization)(proba)

        # transition matrix for this specific trial
        seq_trial = num2char(df_trial.trigger.values)
        tf = seq2tf(''.join(seq_trial))
        interval = df_trial.interval_time.values[0]

        # only calculate up to the length that they have actually been shown!
        # else we are analysing the buffer period already +200ms for safety
        length = int((interval+100)*5)//10 + 20  # assuming 100 Hz sfreq
        max_lag = int(((interval/10) + 10)*1.5)

        sf_trial, sb_trial = tdlm.compute_1step(proba[:length, :], tf, n_shuf=0, max_lag=max_lag)
        sf_subj += [zscore(sf_trial, axis=-1, nan_policy='omit').squeeze()]
        sb_subj += [zscore(sb_trial, axis=-1, nan_policy='omit').squeeze()]

    sf_subj = np.array(sf_subj)
    sb_subj = np.array(sb_subj)

    df = pd.DataFrame(sf_subj, columns=np.arange(0, max_lag*10+10, 10))
    ax = axs.flat[i]
    sns.heatmap(df, cmap='RdBu_r', ax=ax)
    ax.set(xlabel='time lag', ylabel='trial', title=f'{subject=}')

plotting.normalize_lims(axs, 'v')
fig.suptitle('Forward sequenceness across participants for 32 ms condition')
savefig(fig, settings.plot_dir + '/figures/sequenceness_trial_level.png')

#%% trial-level: individual trial expected lag sequenceness

res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']

df_responses = bids_utils.load_responses_sequence()


fig, axs = plt.subplots(2, 2, figsize=[18, 10])

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
    ax.axhline(0, linestyle='--', c='black')
    ax.set(title=f'interval {interval} ms\n', ylabel='trial sequenceness at expected lag (expected_lag*10) ms +-10ms')

    # annotate significant participants with a star above their column
    y_bottom, y_top = ax.get_ylim()
    y_star = y_top * 0.95
    for j, p in enumerate(pvals_sorted):
        if p < 0.05:
            ax.text(j, y_star, '*', ha='center', va='bottom', fontsize=14,
                   fontweight='bold', color='black')
    ax.set_ylim(y_bottom, y_top*1.1)


import matplotlib.lines as mlines
import matplotlib.patches as mpatches
legend_handles = [
    mlines.Line2D([], [], marker='o', color='gray', alpha=0.5, linestyle='None', markersize=6, label='individual trials'),
    mlines.Line2D([], [], marker='D', color='black', linestyle='None', markersize=6, label='mean'),
]
fig.legend(handles=legend_handles, loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.5),
           frameon=True)
fig.tight_layout(rect=[0, 0, 0.92, 1])

savefig(fig, settings.plot_dir + '/figures/sequenceness_trial_level.png')


#%% trial-level: correlate with behaviour and decoding accuracy

res = joblib.load(pkl_seq)
sf_mean = res['sf_mean']

df_responses = bids_utils.load_responses_sequence()

# peak decoding accuracy per subject from localizer cross-validation
dec_acc = np.array([bids_utils.get_decoding_accuracy(s) for s in subjects])

fig, axs = plt.subplots(2, 4, figsize=[18, 8])

for i, (interval, sf) in enumerate(sf_mean.items()):
    expected_lag = settings.exp_lag[interval]
    mean_seq = sf[:, 0, expected_lag-1:expected_lag+2].mean(-1)  # [n_subj]

    from scipy.stats import pearsonr

    # row 0: localizer decoding accuracy vs sequenceness
    ax = axs[0, i]
    df_dec = pd.DataFrame({'decoding accuracy': dec_acc, 'mean sequenceness': mean_seq})
    r, p = pearsonr(dec_acc, mean_seq)
    sns.regplot(data=df_dec, x='decoding accuracy', y='mean sequenceness',
                color=sns.color_palette()[0], scatter_kws={'alpha': 0.7},
                line_kws={'alpha': 0.7}, ax=ax)
    ax.text(0.5, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=12)
    ax.set(title=f'{interval=} ms')

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
    ax.set(title=f'{interval=} ms')

fig.suptitle('Sequenceness correlations with decoding and behavioral accuracy')
savefig(fig, settings.plot_dir + '/figures/sequenceness_correlations.png')


#%% trial-level: sequenceness at expected lag vs individual trial response

res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']

df_responses = bids_utils.load_responses_sequence()

df_trial_resp = pd.DataFrame()

for interval, sf in sf_trials.items():
    expected_lag = settings.exp_lag[interval]
    # peak sequenceness per subject and trial: [n_subj, n_trials]
    peak_seq = sf[:, :, 0, expected_lag-1:expected_lag+2].mean(-1)

    for s_idx, subject in enumerate(subjects):
        df_subj = df_responses[(df_responses.subject == subject[-2:]) &
                               (df_responses.interval_time == interval)].reset_index(drop=True)
        n_trials = min(len(df_subj), peak_seq.shape[1])
        df_tmp = pd.DataFrame({
            'sequenceness': peak_seq[s_idx, :n_trials],
            'correct': df_subj.accuracy.values[:n_trials].astype(bool),
            'subject': subject,
            'interval': interval,
        })
        df_trial_resp = pd.concat([df_trial_resp, df_tmp], ignore_index=True)

df_trial_resp['response'] = df_trial_resp['correct'].map({True: 'correct', False: 'incorrect'})

fig, axs = plt.subplots(1, 4, figsize=[14, 4], sharey=True)

for i, interval in enumerate(sf_trials):
    df_sel = df_trial_resp[df_trial_resp.interval == interval]
    ax = axs[i]

    sns.violinplot(data=df_sel, x='response', y='sequenceness', hue='response',
                   palette={'correct': 'seagreen', 'incorrect': 'crimson'},
                   order=['correct', 'incorrect'], inner='quart',
                   legend=False, split=True,  ax=ax)

    # overlay individual subject means
    df_subj_mean = df_sel.groupby(['subject', 'response'])['sequenceness'].mean().reset_index()
    sns.stripplot(data=df_subj_mean, x='response', y='sequenceness',
                  order=['correct', 'incorrect'],
                  color='black', alpha=0.5, size=4, jitter=True, ax=ax)

    # connect paired subject means
    for subj in df_subj_mean.subject.unique():
        vals = df_subj_mean[df_subj_mean.subject == subj].set_index('response')['sequenceness']
        if 'correct' in vals and 'incorrect' in vals:
            ax.plot([0, 1], [vals['correct'], vals['incorrect']], c='black', alpha=0.2, linewidth=0.8)

    r, p = pearsonr(df_sel['correct'].astype(int), df_sel['sequenceness'])
    ax.text(0.5, 0.98, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='center', fontsize=9)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set(title=f'{interval=} ms', xlabel='')

sns.despine()
fig.suptitle('Trial sequenceness at expected lag: correct vs incorrect responses')
savefig(fig, settings.plot_dir + '/figures/sequenceness_vs_response.png')

#%% effect size visualization at expected time lags

df = pd.DataFrame()

for interval in intervals:
    sf = sf_mean[interval][:, 0, :]
    expected_lag = settings.exp_lag[interval]
    n_lags = sf.shape[-1]
    sf_peak = sf[:, expected_lag-1: expected_lag+2].mean(-1)
    d = pg.compute_effsize(sf[:, expected_lag], y=0)
    df_tmp = pd.DataFrame({'interval': interval,
                           'cohens d': d,
                           'type': 'mean'}, index=[0])
    df = pd.concat([df, df_tmp])


for interval in intervals:
    sf = sf_trials[interval][:, :, 0, 1:]
    expected_lag = settings.exp_lag[interval]
    n_lags = sf.shape[-1]
    for i, subj in enumerate(subjects):
        sf_peak = sf[i, :, expected_lag-1: expected_lag+2].mean(-1)
        d = pg.compute_effsize(sf_peak, y=0)
        df_tmp = pd.DataFrame({'interval': interval,
                               'cohens d': d,
                               'subject': subj,
                               'type': 'trials',
                               }, index=[0])
        df = pd.concat([df, df_tmp])

fig, axs = plt.subplots(1, 2, figsize=[8, 4])

ax = axs[0]
df_sel = df[df.type=='trials']
sns.boxplot(df_sel, x='interval', y='cohens d', fliersize=0, ax=ax)
sns.stripplot(data=df_sel, x="interval", y="cohens d", color="black", alpha=0.5, ax=ax)
ax.set_title('Participant\'s effect sizes')
ax.axhline(0, linestyle='--', alpha=0.5, c='black')

ax = axs[1]
df_sel = df[df.type=='mean']
sns.boxplot(df_sel, x='interval', y='cohens d',  fliersize=0, ax=ax)
ax.set_title('Group mean\'s effect size')
ax.axhline(0, linestyle='--', alpha=0.5, c='black')

plotting.normalize_lims(axs)
savefig(fig, settings.plot_dir + '/figures/tdlm_effect_sizes.png')


#%% supplement: inter-interval: subject consistency across ISI conditions

res = joblib.load(pkl_seq)
sf_mean = res['sf_mean']

# build [n_subj x n_intervals] matrix of peak sequenceness
interval_labels = list(sf_mean.keys())
peak_matrix = np.column_stack([
    sf_mean[iv][:, 0, (iv + 100) // 10 - 1:(iv + 100) // 10 + 2].mean(-1)
    for iv in interval_labels
])  # [n_subj, n_intervals]

# pairwise pearson r and p between intervals
n_iv = len(interval_labels)
corr_r = np.zeros((n_iv, n_iv))
corr_p = np.zeros((n_iv, n_iv))
for a in range(n_iv):
    for b in range(n_iv):
        r, p = pearsonr(peak_matrix[:, a], peak_matrix[:, b])
        corr_r[a, b] = r
        corr_p[a, b] = p

df_corr = pd.DataFrame(corr_r,
                        index=[f'{iv} ms' for iv in interval_labels],
                        columns=[f'{iv} ms' for iv in interval_labels])

fig, axs = plt.subplots(1, 2, figsize=[12, 4])

# left: correlation heatmap with r values and significance stars
ax = axs[0]
mask = np.zeros_like(corr_r, dtype=bool)
mask[np.triu_indices_from(mask)] = True  # upper triangle incl. diagonal
sns.heatmap(df_corr, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            mask=mask, square=True, linewidths=0.5, ax=ax)
# annotate lower triangle with r and stars
for a in range(n_iv):
    for b in range(n_iv):
        if a > b:
            stars = '***' if corr_p[a, b] < 0.001 else '**' if corr_p[a, b] < 0.01 else '*' if corr_p[a, b] < 0.05 else ''
            ax.text(b + 0.5, a + 0.5, f'r={corr_r[a, b]:.2f}\n{stars}',
                    ha='center', va='center', fontsize=9)
ax.set(title='Inter-interval correlation (peak sequenceness per subject)')

# right: scatter matrix — peak sequenceness for each pair of intervals
ax = axs[1]
ax.axis('off')
inner_fig, inner_axs = plt.subplots(n_iv, n_iv, figsize=[10, 10])
for a in range(n_iv):
    for b in range(n_iv):
        iax = inner_axs[a, b]
        if a == b:
            iax.hist(peak_matrix[:, a], bins=10, color='steelblue', edgecolor='white')
            iax.set_title(f'{interval_labels[a]} ms', fontsize=8)
        elif a > b:
            r, p = pearsonr(peak_matrix[:, b], peak_matrix[:, a])
            iax.scatter(peak_matrix[:, b], peak_matrix[:, a], s=15, alpha=0.7, color='steelblue')
            m, c = np.polyfit(peak_matrix[:, b], peak_matrix[:, a], 1)
            xs = np.linspace(peak_matrix[:, b].min(), peak_matrix[:, b].max(), 50)
            iax.plot(xs, m * xs + c, c='red', alpha=0.7, linewidth=1)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            iax.set_title(f'r={r:.2f}{stars}', fontsize=7, color='darkred' if p < 0.05 else 'black')
        else:
            iax.axis('off')
        iax.set_xticks([])
        iax.set_yticks([])

inner_fig.suptitle('Pairwise subject sequenceness across ISI conditions')
inner_fig.tight_layout()
savefig(inner_fig, settings.plot_dir + '/figures/sequenceness_inter_interval_scatter.png')
savefig(fig, settings.plot_dir + '/figures/sequenceness_inter_interval_corr.png')


#%% supplement: temporal drift: sequenceness vs trial position within experiment

res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']

df_drift = pd.DataFrame()

for interval, sf in sf_trials.items():
    expected_lag = settings.exp_lag[interval]
    # peak sequenceness per subject and trial: [n_subj, n_trials]
    peak_seq = sf[:, :, 0, expected_lag-1:expected_lag+2].mean(-1)

    for s_idx, subject in enumerate(subjects):
        n_trials = peak_seq.shape[1]
        df_tmp = pd.DataFrame({
            'sequenceness': peak_seq[s_idx],
            'trial_position': np.arange(n_trials) / (n_trials - 1),  # normalize 0–1
            'trial_index': np.arange(n_trials),
            'subject': subject,
            'interval': interval,
        })
        df_drift = pd.concat([df_drift, df_tmp], ignore_index=True)

fig, axs = plt.subplots(1, 4, figsize=[16, 4], sharey=True)

for i, interval in enumerate(sf_trials):
    df_sel = df_drift[df_drift.interval == interval]
    ax = axs[i]

    # smooth per-subject trend with lowess via seaborn, then group mean
    sns.regplot(data=df_sel, x='trial_position', y='sequenceness',
                scatter=False, lowess=True,
                line_kws={'color': 'black', 'linewidth': 2}, ax=ax)

    # individual subject trends (thin, transparent)
    for subj in subjects:
        df_s = df_sel[df_sel.subject == subj]
        if len(df_s) < 3:
            continue
        sns.regplot(data=df_s, x='trial_position', y='sequenceness',
                    scatter=False, lowess=True, truncate=True,
                    line_kws={'color': 'steelblue', 'linewidth': 0.7, 'alpha': 0.3}, ax=ax)

    r, p = pearsonr(df_sel['trial_position'], df_sel['sequenceness'])
    ax.text(0.05, 0.97, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes,
            va='top', ha='left', fontsize=9)
    ax.axhline(0, linestyle='--', c='black', alpha=0.4)
    ax.set(title=f'{interval=} ms', xlabel='trial position (normalised)', ylabel='peak sequenceness')

sns.despine()
fig.suptitle('Temporal drift: sequenceness across trial position (blue=individual, black=group)')
savefig(fig, settings.plot_dir + '/figures/sequenceness_temporal_drift.png')


#%% supplement: trial-level: sequenceness vs reaction time
from scipy.stats import zscore, pearsonr

res = joblib.load(pkl_seq)
sf_trials = res['sf_trials']

df_responses = bids_utils.load_responses_sequence()

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
    ax.set(title=f'{interval=} ms', xlabel='reaction time (s)', ylabel='peak sequenceness')

sns.despine()
fig.suptitle('Sequenceness vs reaction time (correct trials only)')
savefig(fig, settings.plot_dir + '/figures/sequenceness_vs_rt.png')
