#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:00 2026

Visualize the results of the classifier generalization analysis.

Plots mean heatmaps across participants and L1 values for slow2slow and slow2fast.

@author: simon.kern
"""
import os
import sys; sys.path.append('..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test
from bids_utils import layout_MEG as layout
from settings import plot_dir
from meg_utils import plotting
from scipy import stats
from joblib import Parallel, delayed

plt.rc('font', size=14)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('legend', fontsize=11)

#%% Settings

# L1 values from sbatch script
L1_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
             6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0, 18.0,
             20.0, 25.0, 30.0, 40.0, 50.0]

# Two conditions
CONDITIONS = ['slow2slow', 'slow2fast']

CONDITION_LABELS = {
    'slow2slow': 'Localizer → Localizer',
    'slow2fast': 'Localizer → Sequence',
}

#%% Load all heatmap CSVs

deriv = layout.derivatives['derivatives']

# Dictionary to store heatmaps: {condition: [list of all heatmaps across subjects and L1]}
heatmaps = {cond: [] for cond in CONDITIONS}

subjects = {cond: [] for cond in CONDITIONS}
l1_ratios = {cond: [] for cond in CONDITIONS}

times = {cond: [] for cond in CONDITIONS}

for cond in CONDITIONS:
    files = deriv.get(task='main',
                      proc=cond,
                      extension='.csv.gz',
                      suffix='heatmap',
                      )

    print(f'{cond}: found {len(files)} files')

    # sort by subject
    files = sorted(files, key=lambda x:int(x.entities['subject']))

    for f in tqdm(files, desc=f'Loading {cond}'):
        # Load heatmap
        l1 = float(f.entities['desc'][1:].replace('p', '.'))
        subj = f.entities['subject']
        df = pd.read_csv(f.path, index_col=0)

        subjects[cond] += [subj]
        l1_ratios[cond] += [l1]
        heatmaps[cond] += [df]

    heatmaps[cond] = np.array(heatmaps[cond])
    l1_ratios[cond] = np.array(l1_ratios[cond])
    subjects[cond] = np.array(subjects[cond])
    times[cond] = {'index': df.index, 'columns': df.columns}
    times[cond]['columns'].name = 'test_time ' + ('(slow)' if cond=='slow2slow' else '(fast)')

#%% Precompute all clusters with joblib.Parallel

def compute_cluster(l1, cond, heatmaps, l1_ratios):
    """Compute cluster permutation test for a given (l1, cond) combination."""
    sel = l1_ratios[cond] == l1
    data = np.stack(heatmaps[cond][sel], axis=0)

    t_thresh = stats.distributions.t.ppf(1 - 0.05, df=len(data) - 1)

    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        stats.zscore(data, axis=None), n_permutations=1000, tail=1,
        threshold=t_thresh, n_jobs=4, seed=42, out_type='mask', verbose='WARNING'
    )
    return (l1, cond), (clusters, cluster_pv)

# Build list of all (l1, cond) combinations
jobs = [(l1, cond) for l1 in np.unique(l1_ratios['slow2slow']) for cond in CONDITIONS]

# Run all cluster computations in parallel
print(f'Precomputing {len(jobs)} cluster permutation tests...')
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(compute_cluster)(l1, cond, heatmaps, l1_ratios) for l1, cond in tqdm(jobs)
)

# Store results in dict for fast lookup
cluster_results = dict(results)

#%% Plot 1x2 mean heatmap with precomputed clusters

fig, axs = plt.subplots(1, 2, figsize=[16, 6])

for j, l1 in enumerate(tqdm(np.unique(l1_ratios['slow2slow']))):
    for i, cond in enumerate(CONDITIONS):

        ax = axs[i]
        ax.clear()

        # select all heatmaps for that l1 ratio
        sel = l1_ratios[cond]==l1

        # Stack all heatmaps: shape (n_observations, train_time, test_time)
        data = np.stack(heatmaps[cond][sel], axis=0)
        n_obs = data.shape[0]

        # Compute mean heatmap
        mean_heatmap = pd.DataFrame(np.mean(data, axis=0),
                                    index=times[cond]['index'],
                                    columns=times[cond]['columns'],
                                    )

        # Plot heatmap
        sns.heatmap(mean_heatmap, ax=ax, cmap='viridis', center=0.2,
                        vmin=0, vmax=0.6, cbar=True if j==0 else False)
        ax.invert_yaxis()

        # Retrieve precomputed cluster results
        clusters, cluster_pv = cluster_results[(l1, cond)]

        # Create significance mask from significant clusters (p < 0.05)
        sig_mask = np.zeros(mean_heatmap.shape, dtype=bool)
        for cluster, pval in zip(clusters, cluster_pv):
            if pval < 0.05:
                sig_mask = sig_mask | cluster

        # Highlight significant clusters
        if sig_mask.any():
            plotting.highlight_cells(sig_mask, ax, color='darkred', linewidth=1)

        plt.suptitle(f'Generalization - {l1=:.1f}')
        ax.set_title(f'{CONDITION_LABELS[cond]}\n(n={n_obs})')

        # Add diagonal line

        ax.plot([0, min(mean_heatmap.shape)], [0, min(mean_heatmap.shape)],
                'k--', alpha=0.35, linewidth=0.5)
        plotting.savefig(fig, f'{plot_dir}/transfer/heatmap_transfer_{str(l1).replace(".", "p")}.png', dpi=150)
