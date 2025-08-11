#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:42:38 2025

sanity check the data

@author: simon.kern
"""
import numpy as np
from tqdm import tqdm

import settings
import bids_utils

for subject in tqdm(settings.layout_MEG.subjects, desc='checking participants'):
    data_x, data_y, _ = bids_utils.load_localizer(subject=subject, verbose=False)
    assert set(np.bincount(data_y))=={32}
    print(subject)
