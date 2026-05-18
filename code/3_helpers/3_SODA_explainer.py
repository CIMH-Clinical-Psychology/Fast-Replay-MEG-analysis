#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:08:36 2026

@author: simon
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soda


plt.rcParams['font.size'] = 14
colors = sns.color_palette()
fileid = os.path.basename(__file__)

plt.close('all')


def gaussian(t, mean, std):
    return np.exp(-0.5 * ((t - mean) / std) ** 2)


n_items = 5
n_trs = 5

tr_axis = np.arange(n_trs)
probas_tr = np.array([gaussian(tr_axis, mean=i, std=0.7) for i in range(n_items)]).T

t_smooth = np.linspace(-0.5, n_trs - 0.5, 400)
probas_smooth = np.array([gaussian(t_smooth, mean=i, std=0.7) for i in range(n_items)]).T


# 1x1 (A) | 2x2 (B1-B4) | 1x2 (C1, C2) horizontal mosaic
fig, axs = plt.subplot_mosaic(
    [['A', 'B1', 'B2', 'C1', 'C2'],
     ['A', 'B3', 'B4', 'C1', 'C2']],
    figsize=(22, 7),
)

# --- Panel A: decoded probabilities per item, sampled at each TR ---
ax = axs['A']
for i in range(n_items):
    ax.plot(t_smooth, probas_smooth[:, i], c=colors[i], label=f'Item {chr(65 + i)}')
    ax.scatter(tr_axis, probas_tr[:, i], c=[colors[i]], s=60, zorder=3)
for tr in tr_axis:
    ax.axvspan(tr - 0.05, tr + 0.05, color='gray', alpha=0.2, zorder=0)
ax.set_xlabel('TR')
ax.set_ylabel('decoded probability')
ax.set_xticks(tr_axis)
ax.set_ylim(-0.05, 1.1)
ax.set_title('Decoded probabilities')
ax.legend(loc='upper right', fontsize=10)


# --- Panels B1-B4: probability vs sequence position at each TR ---
positions = np.arange(1, n_items + 1)
for key, tr_idx in zip(['B1', 'B2', 'B3', 'B4'], [1, 2, 3, 4]):
    ax = axs[key]
    item_probs = probas_tr[tr_idx, :]
    for i in range(n_items):
        ax.scatter(positions[i], item_probs[i], c=[colors[i]], s=80, zorder=3)
    sns.regplot(x=positions, y=item_probs, ax=ax, scatter=False, ci=None,
                color='gray', line_kws={'linestyle': '--'})
    slope = np.polyfit(positions, item_probs, 1)[0]
    ax.set_xlabel('sequence position')
    ax.set_ylabel(f'P at TR {tr_idx}')
    ax.set_title(f'TR {tr_idx}: slope = {slope:+.3f}')
    ax.set_xticks(positions)
    ax.set_ylim(-0.05, 1.1)


# --- Panels C1, C2: continuous probabilities → SODA slope output ---
t_ms = np.arange(0, 1800)
probas_cont = np.array([gaussian(t_ms, mean=300 + i * 150, std=100)
                        for i in range(n_items)]).T

ax = axs['C1']
for i in range(n_items):
    ax.plot(t_ms, probas_cont[:, i], c=colors[i], label=f'Item {chr(65 + i)}')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('decoded probability')
ax.set_title('Continuous probabilities')
ax.legend(loc='upper right', fontsize=10)

slopes = soda.compute_slopes(probas_cont)
ax = axs['C2']
ax.plot(t_ms, slopes, color='darkred')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('slope (forward sequenceness)')
ax.set_title('SODA output')


plt.tight_layout()
plt.savefig(f'./{fileid}_mosaic.png')
plt.show()
