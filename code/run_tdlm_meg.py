# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:34:57 2025

Code to run TDLM (Temporally Delayed Linear Modelling) on MEG data

@author: Simon.Kern
"""

import os
import settings
import bids_utils

import numpy as np
import pandas as pd
import mne
from meg_utils import decoding, plotting, sigproc


#%% settings
