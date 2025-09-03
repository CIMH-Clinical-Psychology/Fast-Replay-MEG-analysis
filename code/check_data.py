#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:42:38 2025

sanity check the data, all files there? preprocessing ran successfully?

@author: simon.kern
"""
import sys
import re
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import mne
import numpy as np
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

import settings
import bids_utils
from meg_utils import preprocessing, plotting
from settings import layout_MEG as layout
bids_utils.mem.verbose=0

#%% first check that all files are present
for subject in tqdm(layout.subjects, desc='checking participants'):
    for task in ['rest1', 'rest2', 'main']:
        files = layout.get(subject=subject, suffix='raw',
                    scope='derivatives', proc='clean',
                    task=task, return_type='filenames')
        assert len(files)==1, f'missing file! {subject=} {task=}'

    files = layout.get(subject=subject, return_type='filenames', datatype='beh',
                          extension='tsv')
    assert len(files)==1, f'missing behaviour file! {subject=} {task=}'


for subject in tqdm(layout.subjects, desc='checking participants'):
    data_x, data_y, _ = bids_utils.load_localizer(subject=subject, verbose=False)
    if subject not in ['13', '26']:
        assert set(np.bincount(data_y))=={32}

    data_x, data_y, _ = bids_utils.load_fast_images(subject)
    assert set(np.bincount(data_y))=={64}

    try:
        df_beh = bids_utils.load_behaviour(subject)
        assert len(df_beh), '{subject} df_beh is empty!'
    except KeyError as e:
        print(f'{subject}: {e}')

#%% next check the data themselve. there should be no zero elements
sfreq = 100
axs = []
raws={}
for subject in tqdm(layout.subjects, desc='checking values'):
    files = layout.get(subject=subject, suffix='raw',
                       scope='derivatives', proc='clean',
                       task='main', return_type='filenames')
    raw = mne.io.read_raw(files[0], verbose='WARNING')
    raw.pick('meg')
    raw.plot(block=True)
    input()
    # raw.resample(100, n_jobs=-1, verbose='WARNING')
    # raw.filter(0.1, 40, n_jobs=-1, verbose='WARNING')
    raws[subject] = raw
    data = raw.get_data('meg')
    data = preprocessing.rescale_meg_transform_outlier(data)

    epoch_len = 5
    data = data[:, :int(len(raw)//(30*sfreq)*epoch_len*sfreq)]
    data = data.reshape([len(data), -1, int(epoch_len*sfreq)])
    stds = np.log10(1+np.std(data, axis=-1)) # get std per epoch as noise marker

    fig, ax = plt.subplots(1, 1, figsize=[14, 8])
    axs += [ax]
    ax.imshow(stds, aspect='auto', cmap='Spectral', interpolation='none',
              vmin=0, vmax=0.1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*epoch_len}s'))
    ax.set_yticks(np.arange(len(raw.ch_names))[::4], raw.ch_names[::4], fontsize=5)
    ax.set(title=f'noise std of {epoch_len=} for {subject=}', xlabel='time', ylabel='sensor')
    plotting.savefig(ax.figure, settings.plot_dir + f'/noise_std/noise_std_{subject}.png')

# plotting.normalize_lims(axs, which='v')
# for ax, subject in zip(axs, layout.subjects):
    # plotting.savefig(ax.figure, settings.plot_dir + f'/noise_std/noise_std_{subject}.png')
