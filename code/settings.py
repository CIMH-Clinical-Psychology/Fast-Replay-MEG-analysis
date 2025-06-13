#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:52:06 2024

this file contains user specific configuration such as the location of data
directories, caching dirs, output dirs etc, as well as other constants

@author: simon.kern
"""
import os
import sys
import warnings
import getpass
import platform
from bids import BIDSLayout  # pip install pybids

###############################
#%%userconf
# USER-SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

# overwrite this variable to set a custom data dir
bids_dir = "../data/"  # enter directory here where the data has been stored

# machine specific configuration overwrites general directory structure
if username == 'simon.kern' and host=='zislrds0035.zi.local':  # simons VM
    cache_dir = '/data/fastreplay/cache/'
    bids_dir = '/data/fastreplay/Fast-Replay-MEG-bids/'
    plot_dir = f'{home}/Nextcloud/ZI/2024.10 FastReplayAnalysis/plots/'

elif username == 'simon.kern' and host=='5cd320lfh8':
    cache_dir = f'{home}/Desktop/joblib-fasterplay/'
    bids_dir = "W:/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/"
    plot_dir = f'{home}/Nextcloud/ZI/2024.10 FastReplayAnalysis/plots/'

else:
    warnings.warn('No user specific settings found in settings.py')

#####################################
# END OF USER-SPECIFIC CONFIGURATION
#####################################

#%% initializations, should need no change
# set environment variable for `meg_utils`, where to cache function calls
os.environ['JOBLIB_CACHE_DIR'] = cache_dir

#%% initialize BIDS dir
layout = BIDSLayout(bids_dir, derivatives=True)
layout.subjects = layout.get_subjects()
if not layout.subjects:
        warnings.warn('No subjects in layout, are you sure it exists?')

#%% checks for stuff
if 'cache_dir' not in locals():
    cache_dir = f"{bids_dir}/cache/"  # used for caching
if 'plot_dir' not in locals():
    plot_dir = f"{bids_dir}/plots/"  # plots will be stored here


if not os.path.isdir(plot_dir):
    warnings.warn(f"plot_dir does not exist at {plot_dir}, create")
    os.makedirs(plot_dir, exist_ok=True)


def get_free_space_gb(path):
    """return the current free space in the cache dir in GB"""
    import shutil

    os.makedirs(path, exist_ok=True)
    total, used, free = shutil.disk_usage(path)
    total //= 1024**3
    used //= 1024**3
    free //= 1024**3
    return free

if get_free_space_gb(cache_dir) < 20:
    raise RuntimeError(f"Free space for {cache_dir} is below 20GB. Cannot safely run.")

#%% constants
img_trigger = {}  # here, offset of 1 is already removed!
img_trigger[0] = 'Gesicht'
img_trigger[1]   = 'Haus'
img_trigger[2]   = 'Katze'
img_trigger[3]   = 'Schuh'
img_trigger[4]   = 'Stuhl'

trigger_img = {}
trigger_img['Gesicht'] = 1
trigger_img['Haus']   = 2
trigger_img['Katze']   = 3
trigger_img['Schuh']   = 4
trigger_img['Stuhl']   = 5

trigger_break_start = 91
trigger_break_stop = 92
trigger_buf_start = 61
trigger_buf_stop = 62
trigger_sequence_sound0 = 70
trigger_sequence_sound1 = 71
trigger_localizer_sound0 = 30
trigger_localizer_sound1 = 31
trigger_fixation_pre1 = 81
trigger_fixation_pre2 = 82

# these get added depending on the trial
# ie. Gesicht in localizer = 1, as cue = 11, as sequence=21
trigger_base_val_localizer = 0
trigger_base_val_localizer_distractor = 100
trigger_base_val_cue = 10
trigger_base_val_sequence = 20
