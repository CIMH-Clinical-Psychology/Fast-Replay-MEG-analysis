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
import time
import psutil
import logging
from bids import BIDSLayout  # pip install pybids

###############################
#%%userconf
# USER-SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')
cwd = os.path.abspath(os.getcwd())


# the following directories are needed:
# cache_dir - temporary directory for caching and joblib
# plot_dir  - where plots will be dumped
# bids_dir_meg - directory of the MEG bids data
# bids_dir_3T  - https://gin.g-node.org/lnnrtwttkhn/highspeed-bids
# bids_dir_3T_decoding - https://gin.g-node.org/lnnrtwttkhn/highspeed-decoding

# machine specific configuration overwrites general directory structure
if username == 'simon.kern' and 'zislrd' in host:  # VM
    cache_dir = '/data/fastreplay/cache/'
    plot_dir = '../plots/'
    bids_dir_meg = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/'
    bids_dir_3T = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-3T-bids/'
    bids_dir_3T_decoding = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-3T-decoding/'

elif username == 'simon.kern' and host=='5cd320lfh8':
    cache_dir = f'{home}/Desktop/joblib-fasterplay/'
    plot_dir = '../plots/'
    bids_dir_meg = "W:/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids"
    bids_dir_3T = 'W:/group_klips/data/data/Simon/highspeed/highspeed-3T-bids/'
    bids_dir_3T_decoding = 'w:/group_klips/data/data/Simon/highspeed/highspeed-3T-decoding/'

elif username=='simon' and host=='kubuntu':
    cache_dir = f'{home}/Desktop/joblib-fasterplay/'
    plot_dir = '../plots/'
    # bids_dir_meg = "W:/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/"
    bids_dir_3T = bids_dir = '/home/simon/Desktop/highspeed-bids/'
    bids_dir_3T_decoding = '/home/simon/Desktop/highspeed-decoding/'

else:
    raise Exception('No user specific settings found in settings.py')

#####################################
# END OF USER-SPECIFIC CONFIGURATION
#####################################

#%% convert to abspaths

cache_dir = os.path.abspath(cache_dir)
plot_dir = os.path.abspath(plot_dir)
bids_dir_3T_decoding = os.path.abspath(bids_dir_3T_decoding)

#%% initialize MEG BIDS dir

# ignore_subjects = ['16',  # massive data loss during recording
#                    '12',  # massive data loss during recording
#                    '21',  # moderate data loss
#                    '22',  # massive data loss during recording
#                    '27',  # moderate data loss
#                    ]

if 'bids_dir_meg' in locals():
    bids_dir_meg = os.path.abspath(bids_dir_meg)
    # use database for faster loading. but recreate on each python startup
    db_path = cache_dir + '/bids_meg.db'
    python_start = psutil.Process(os.getpid()).create_time()
    reset_database = (os.path.getmtime(db_path) if os.path.exists(db_path) else 0) < python_start
    if reset_database:
        warnings.warn('resetting MEG BIDS database')
    layout_MEG = BIDSLayout(bids_dir_meg, derivatives=True, database_path=db_path, reset_database=reset_database)
    layout_MEG.subjects_all = [x for x in layout_MEG.get_subjects() if (not 'emptyroom' in x)]
    layout_MEG.subjects = [x for x in layout_MEG.subjects_all]

    if not layout_MEG.subjects:
        warnings.warn('No subjects in layout_MEG, are you sure it exists?')
else:
    layout_MEG = None
    warnings.warn('bids_dir_MEG has not been defined in settings.py')

#%% BIDS layout for 3T
if 'bids_dir_3T' in locals():
    bids_dir_3T = os.path.abspath(bids_dir_3T)
    # use database for faster loading. but recreate on each python startup
    db_path = cache_dir + '/bids_3T.db'
    python_start = psutil.Process(os.getpid()).create_time()
    reset_database = (os.path.getmtime(db_path) if os.path.exists(db_path) else 0) < python_start
    if reset_database:
        warnings.warn('resetting 3T BIDS database')
    layout_3T = BIDSLayout(bids_dir_3T, database_path=db_path, reset_database=reset_database)
    layout_3T.subjects = layout_3T.get_subjects()
    if not layout_3T.subjects:
        warnings.warn('No subjects in layout_3T, are you sure it exists?')
else:
    layout_3T = None
    warnings.warn('bids_dir_3T has not been defined in settings.py')

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

#%% initializations, should need no change
# set environment variable for `meg_utils`, where to cache function calls
os.environ['JOBLIB_CACHE_DIR'] = cache_dir

#%% constants

trigger_translation = {'Gesicht': 'face',
                       'Haus': 'house',
                       'Katze': 'cat',
                       'Schuh': 'shoe',
                       'Stuhl': 'chair',
                       }

img_trigger = {}  # here, offset of 1 is already removed!
img_trigger[0] = 'Gesicht'
img_trigger[1]   = 'Haus'
img_trigger[2]   = 'Katze'
img_trigger[3]   = 'Schuh'
img_trigger[4]   = 'Stuhl'

categories = ['face', 'house', 'cat', 'shoe', 'chair']

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


palette_wittkuhn1 = [
    "#f5191c",  # soft red
    "#e78f0a",  # soft orange
    "#eacb2b",  # soft yellow
    "#7cba96",  # soft green
    "#3b99b1"   # soft blue
]

palette_wittkuhn2 = [
    "#00204d",  # muted blue
    "#414d6b",  # light blue-gray
    "#7b7b7b",  # medium gray
    "#bcaf6f",  # beige
    "#ffea46",  # pale yellow
]
