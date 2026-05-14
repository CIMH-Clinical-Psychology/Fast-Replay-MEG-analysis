#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:44:46 2026

@author: simon
"""
import sys; sys.path.append('..')
import os
os.environ['NO_PRELOADING'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import soda
import settings
from meg_utils import plotting


t = np.arange(0, 1800)
n_states = 5

def gaussian(t, shift, std=100):
    return np.exp(-0.5 * ((t - shift) / std) ** 2)

probas1 = np.array([gaussian(t, shift=300 + i*100) for i in range(n_states)]).T
probas2 = np.array([gaussian(t, shift=300 + i*300) for i in range(n_states)]).T

probas3 = probas2.copy()
probas3[:, 1:] = [0.01, 0.015, 0.02, 0.025]
probas3b = probas2.copy()
probas3b[:, :-1] = [0.01, 0.015, 0.02, 0.025]

probas4 = probas1.copy()
probas4[:, 1:-1] = [0.01, 0.015, 0.02]

# mangled order with short (overlapping) distances: A, D, B, C, E
shuffled_order_soda = [0, 2, 1, 4, 3]  # temporal position -> item index
probas6 = np.zeros_like(probas1)
for pos, item_idx in enumerate(shuffled_order_soda):
    probas6[:, item_idx] = gaussian(t, shift=300 + pos*100)


slopes1 = soda.compute_slopes(probas1)
slopes2 = soda.compute_slopes(probas2)
slopes3 = soda.compute_slopes(probas3)
slopes3b = soda.compute_slopes(probas3b)
slopes4 = soda.compute_slopes(probas4)
slopes6 = soda.compute_slopes(probas6)

#%%
fig, axs = plt.subplots(6, 2, figsize=(8, 12))
for ax in axs[:, 0]:
    ax.set_prop_cycle(color=settings.palette_wittkuhn1)
default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
for ax in axs[:, 1]:
    ax.set_prop_cycle(color=[default_blue])

ax = axs[0, 0]
ax.plot(t, probas1)
ax.set_title('Mostly overlapping')

ax = axs[1, 0]
ax.plot(t, probas4)
ax.set_title('Only sequence boundary replayed')

ax = axs[0, 1]
ax.plot(slopes1)

ax = axs[1, 1]
ax.plot(slopes4)

#%%

#%%
axs[2, 0].plot(t, probas3)
axs[2, 1].plot(slopes3)
axs[2, 0].set_title('Single reactivation of first item')
axs[3, 0].plot(t, probas3b)
axs[3, 1].plot(slopes3b)
axs[3, 0].set_title('Single reactivation of last item')

axs[4, 0].plot(t, probas6)
axs[4, 1].plot(slopes6)
axs[4, 0].set_title('Wrong order')

axs[5, 0].plot(t, probas2)
axs[5, 1].plot(slopes2)
axs[5, 0].set_title('Replay with little overlap')

# axs[0, 0].legend([f'Item {i}' for i in range(1, 5)], ncols=2)
#%%
plotting.normalize_lims(axs[:, 0])
plotting.normalize_lims(axs[:, 1])

for ax in axs[:, 0]:
    ax.set(xlabel='Time (ms)', ylabel='Probability')

for ax in axs[:, 1]:
    ax.set(xlabel='Time (ms)', ylabel='Slope')


for i, (ax, label) in enumerate(zip(axs[:, 0], 'ABCDEF')):
    ax.annotate(label, xy=(0, 0.5), xycoords='axes fraction',
                xytext=(-0.25, 0.5), textcoords='axes fraction',
                fontsize=14, fontweight='bold', va='center', ha='right')

item_labels = [f'Item {chr(ord("A") + i)}' for i in range(n_states)]

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.legend(axs[0, 0].get_lines(), item_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.99), ncols=n_states)
plt.show()
plotting.savefig(fig, settings.plot_dir + '/figures/SODA_explainer.png', tight=False)
