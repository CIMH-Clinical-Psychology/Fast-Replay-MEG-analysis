#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:00 2026

Visualize the results of the classifier generalization analysis.

Plots mean heatmaps across participants for each C value and condition.

@author: simon.kern
"""
import os
import sys; sys.path.append('..')
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from bids_utils import layout_MEG as layout

plt.rc('font', size=14)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('legend', fontsize=11)

#%% Settings

# L1 values from sbatch script
L1_VALUES = [0.01, 0.05, 0.25, 1.0, 3, 5, 7, 10, 25, 100.0]

# Four conditions
CONDITIONS = ['slow2slow', 'fast2fast', 'slow2fast', 'fast2slow']

CONDITION_LABELS = {
    'slow2slow': 'Localizer → Localizer',
    'fast2fast': 'Sequence → Sequence',
    'slow2fast': 'Localizer → Sequence',
    'fast2slow': 'Sequence → Localizer',
}

#%% Load all heatmap CSVs

deriv = layout.derivatives['derivatives']

# Dictionary to store heatmaps: {condition: {C: [list of heatmaps per subject]}}
heatmaps = {cond: {c: [] for c in L1_VALUES} for cond in CONDITIONS}

for cond in CONDITIONS:
    files = deriv.get(task='main',
                      processing=cond,
                      extension='.csv',
                      suffix='heatmap',
                      invalid_filters='allow')

    print(f'{cond}: found {len(files)} files')

    for f in tqdm(files, desc=f'Loading {cond}'):
        # Extract C value from filename (description field)
        # Filename format: ..._desc-C{value}_heatmap.csv
        fname = os.path.basename(f.path)
        # Parse C value from description
        try:
            c_str = fname.split('desc-C')[1].split('_')[0]
            c_val = float(c_str)
        except (IndexError, ValueError):
            print(f'Could not parse C value from {fname}')
            continue

        # Round to match L1_VALUES
        c_val_rounded = min(L1_VALUES, key=lambda x: abs(x - c_val))

        # Load heatmap
        df = pd.read_csv(f.path, index_col=0)
        heatmaps[cond][c_val_rounded].append(df.values)

#%% Plot mean heatmap for each C value and condition

for cond in CONDITIONS:
    n_c = len(L1_VALUES)
    n_cols = 5
    n_rows = int(np.ceil(n_c / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=[16, 3.5*n_rows])
    axs = axs.flatten()

    for i, c_val in enumerate(L1_VALUES):
        ax = axs[i]

        if len(heatmaps[cond][c_val]) == 0:
            ax.set_title(f'C={c_val} (no data)')
            ax.axis('off')
            continue

        # Compute mean heatmap across participants
        mean_heatmap = np.mean(heatmaps[cond][c_val], axis=0)
        n_subj = len(heatmaps[cond][c_val])

        # Plot heatmap
        sns.heatmap(mean_heatmap, ax=ax, cmap='RdBu_r', center=0.2,
                    vmin=0, vmax=1, cbar=True,
                    xticklabels=False, yticklabels=False)
        ax.set_title(f'C={c_val} (n={n_subj})')
        ax.set_xlabel('Test time')
        ax.set_ylabel('Train time')

        # Add diagonal line
        ax.plot([0, mean_heatmap.shape[1]], [0, mean_heatmap.shape[0]],
                'k--', alpha=0.5, linewidth=1)

    # Hide unused subplots
    for i in range(n_c, len(axs)):
        axs[i].set_visible(False)

    fig.suptitle(f'{CONDITION_LABELS[cond]} - Mean heatmap per C value')
    plt.tight_layout()
    plt.savefig(f'heatmap_mean_{cond}.png', dpi=150)

#%% Summary plot: 4 conditions x best C value

# Find best C for each condition based on mean accuracy
best_c_per_cond = {}
for cond in CONDITIONS:
    best_acc = -1
    best_c = None
    for c_val in L1_VALUES:
        if len(heatmaps[cond][c_val]) > 0:
            mean_acc = np.mean([h.mean() for h in heatmaps[cond][c_val]])
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_c = c_val
    best_c_per_cond[cond] = best_c
    print(f'{cond}: best C={best_c} (mean acc={best_acc:.3f})')

# Plot 2x2 grid of best heatmaps
fig, axs = plt.subplots(2, 2, figsize=[12, 10])
axs = axs.flatten()

for i, cond in enumerate(CONDITIONS):
    ax = axs[i]
    c_val = best_c_per_cond[cond]

    if c_val is None or len(heatmaps[cond][c_val]) == 0:
        ax.set_title(f'{CONDITION_LABELS[cond]}\n(no data)')
        ax.axis('off')
        continue

    mean_heatmap = np.mean(heatmaps[cond][c_val], axis=0)
    n_subj = len(heatmaps[cond][c_val])

    sns.heatmap(mean_heatmap, ax=ax, cmap='RdBu_r', center=0.2,
                vmin=0, vmax=1, cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_title(f'{CONDITION_LABELS[cond]}\nBest C={c_val} (n={n_subj})')
    ax.set_xlabel('Test time')
    ax.set_ylabel('Train time')
    ax.plot([0, mean_heatmap.shape[1]], [0, mean_heatmap.shape[0]],
            'k--', alpha=0.5, linewidth=1)

fig.suptitle('Best heatmaps per condition')
plt.tight_layout()
plt.savefig('heatmap_best_per_condition.png', dpi=150)

print('\nDone!')
