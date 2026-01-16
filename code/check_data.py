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
from mne_bids import get_entities_from_fname
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
        assert len(files)==1, f'missing or too many files! {subject=} {task=} {files=}'

    files = layout.get(subject=subject, return_type='filenames', datatype='beh',
                          extension='tsv')
    assert len(files)==1, f'missing or too many behaviour files! {subject=} {task=} {files=}'
    print(f'{subject=} ok. all files present')

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
    print(f'{subject=} ok. all triggers present')


#%% next check the data themselve. there should be no zero elements
import matplotlib.colors as mcolors
sfreq = 100

spectral_r = plt.get_cmap('Spectral_r')
colors = spectral_r(np.linspace(0, 1, 32))
colors[0] = [0.8, 0.8, 0.8, 1.0]
# colors[1] = [0.8, 0.8, 0.8, 1.0]
cmap = mcolors.LinearSegmentedColormap.from_list('Spectral_gray', colors, N=32)

epoch_len = 5

fig, ax = plt.subplots(1, 1, figsize=[14, 8])

for subject in tqdm(layout.subjects, desc='checking values'):
    files = layout.get(subject=subject, suffix='raw',
                       scope='derivatives', proc='clean',
                       return_type='filenames')

    for file in files:

        # don't process the noise files
        if 'noise' in file:
            continue

        task = get_entities_from_fname(file)['task']

        raw = mne.io.read_raw(file, verbose='WARNING')
        raw.pick('meg')

        data = raw.get_data('meg')
        data = preprocessing.rescale_meg_transform_outlier(data[:, :-1])


        # truncate to multiple of epoch_len
        data = data[:, :int(data.shape[1]//(epoch_len*sfreq)*epoch_len*sfreq)]
        data = data.reshape([len(data), -1, int(epoch_len*sfreq)])
        stds = np.log10(1+np.std(data, axis=-1)) # get std per epoch as noise marker

        print(f'{stds.min()=:.3f} {stds.max()=:.3f}')

        ax.clear()
        ax.imshow(stds, aspect='auto', cmap=cmap, interpolation='none',
                  vmin=0, vmax=0.15)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*epoch_len}s'))
        ax.set_yticks(np.arange(len(raw.ch_names))[::4], raw.ch_names[::4], fontsize=5)
        ax.set(xlabel='time', ylabel='sensor')
        ax.set_title(f'{subject=} {task}\n noise std of {epoch_len=}')
        plotting.savefig(ax.figure, settings.plot_dir + f'/noise_std/noise_std_{subject}-{task}.png')
