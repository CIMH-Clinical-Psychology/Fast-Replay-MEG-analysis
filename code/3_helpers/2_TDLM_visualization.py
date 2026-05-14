#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:10:37 2026

@author: simon
"""

import sys; sys.path.append('..')
import os
os.environ['NO_PRELOADING'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tdlm
import settings
from meg_utils import plotting

color_fwd = sns.color_palette()[1]
color_bkw = sns.color_palette()[2]


# tdlm.compute_1step assumes sfreq = 100 Hz, i.e. 1 sample = 10 ms.
sfreq = 100
duration_ms = 1800
n_samples = int(duration_ms * sfreq / 1000)  # 180 samples
t = np.arange(n_samples)
t_ms = t * (1000 / sfreq)
noise_scale = 0.025

def gaussian(t, shift, std=5):
    g = np.exp(-0.5 * ((t - shift) / std) ** 2)
    g[g<0.00001] = 0  # truncate zeros
    return g


# All time-related parameters expressed in samples at 100 Hz.
shift0 = 10      # 200 ms initial offset
item_lag = 6     # 60 ms inter-item lag
roll_amt = 80    # 600 ms gap between reactivation cycles
gauss_std = 0.75    # ~10 ms gaussian width

n_states = 5

# five probability spikes in a sequence row, repeated once
probas1 = np.array([gaussian(t, shift=shift0 + i * item_lag, std=gauss_std)
                    for i in range(n_states)]).T
probas1 += np.roll(probas1, roll_amt, axis=0)
probas1 += np.random.rand(*probas1.shape)*noise_scale  # add little bit of noise

# five probabilities, unequal inter-item time lag (samples = ms / 10)
shifts1 = np.cumsum([6, 10, 4, 8, 5])
shifts2 = np.cumsum([5, 9, 7, 4, 6])
probas2a = np.array([gaussian(t, shift=shift0 + shifts1[i], std=gauss_std) for i in range(n_states)]).T
probas2b = np.array([gaussian(t, shift=shift0 + shifts2[i], std=gauss_std) for i in range(n_states)]).T
probas2 = probas2a + np.roll(probas2b, roll_amt, axis=0)
probas2 += np.random.rand(*probas1.shape)*noise_scale  # add little bit noise

# backward replay: temporal order E -> D -> C -> B -> A
probas_back = np.array([gaussian(t, shift=shift0 + (n_states - 1 - i) * item_lag, std=gauss_std)
                        for i in range(n_states)]).T
probas_back += np.roll(probas_back, roll_amt, axis=0)
probas_back += np.random.rand(*probas_back.shape) * noise_scale

# missing steps: only states A, C, E reactivate (B & D missing), repeated once
probas_missing = np.array([gaussian(t, shift=shift0 + i * item_lag, std=gauss_std)
                           for i in range(n_states)]).T
probas_missing[:, [1, 3]] = 0
probas_missing += np.roll(probas_missing, roll_amt, axis=0)
# probas_missing += np.random.rand(*probas_missing.shape) * noise_scale

# wrong order: max-distance shuffled 5-step replay (D -> B -> E -> A -> C)
# adjacent original-index distances: 2, 3, 4, 2 (max possible sum = 11)
# no triplet matches ABC/BCD/CDE -> no second-order sequenceness
shuffled_order = [3, 1, 4, 0, 2]  # temporal position -> item index
probas_shuffled = np.zeros((n_samples, n_states))
for pos, item_idx in enumerate(shuffled_order):
    probas_shuffled[:, item_idx] = gaussian(t, shift=shift0 + pos * item_lag, std=gauss_std)
probas_shuffled += np.roll(probas_shuffled, roll_amt, axis=0)
probas_shuffled += np.random.rand(*probas_shuffled.shape) * noise_scale

# five probability spikes in a sequence row, repeated once
probas3 = np.array([gaussian(t, shift=shift0 + i * 2.5, std=5)
                    for i in range(n_states)]).T
probas3 += np.roll(probas3, roll_amt, axis=0)
probas3 += np.random.rand(*probas3.shape)*noise_scale  # add little bit of noise


tf = tdlm.utils.seq2tf('ABCDE')

#%% sequenceness calculation + figure
cases = [
    (probas1, 'Full sequence replay'),
    (probas_back, 'Backward replay'),
    (probas2, 'Unequal time lag'),
    (probas_missing, 'Missing steps'),
    (probas_shuffled, 'Wrong order'),
    (probas3, 'Rapid & overlapping'),
]

fig, axs = plt.subplots(len(cases), 2, figsize=(8, 12))

for i, (proba, title) in enumerate(cases):
    axs[i, 0].set_prop_cycle(color=settings.palette_wittkuhn1)
    axs[i, 0].plot(t_ms, proba)
    axs[i, 0].set_title(title)
    sf, sb = tdlm.compute_1step(proba, tf, n_shuf=2, max_lag=30)
    axs[i, 1].set_prop_cycle(color=[color_fwd, color_bkw])
    tdlm.plot_sequenceness(sf, sb, ax=axs[i, 1], plot95=False, plotmax=False,
                           which=['fwd', 'bkw'], rescale=False)

item_labels = [f'Item {chr(ord("A") + i)}' for i in range(n_states)]

#%%
plotting.normalize_lims(axs[:, 0])
plotting.normalize_lims(axs[:, 1])

for ax in axs[:, 0]:
    ax.set(xlabel='Time (ms)', ylabel='Probability')

for ax in axs[:, 1]:
    ax.set(xlabel='Time lag (ms)', ylabel='Sequenceness')

for ax, label in zip(axs[:, 0], 'ABCDEF'):
    ax.annotate(label, xy=(0, 0.5), xycoords='axes fraction',
                xytext=(-0.25, 0.5), textcoords='axes fraction',
                fontsize=14, fontweight='bold', va='center', ha='right')

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.legend(axs[0, 0].get_lines(), item_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.99), ncols=n_states)
plt.show()
plotting.savefig(fig, settings.plot_dir + '/figures/TDLM_explainer.png', tight=False)
