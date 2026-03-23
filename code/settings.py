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
import seaborn as sns

# this variable can be set to suppress preloading of the BIDS data
NO_PRELOADING = os.environ.get('NO_PRELOADING')

###############################
#%%userconf
# USER-SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')
cwd = os.path.abspath(os.getcwd())
script_dir = os.path.dirname(__file__)

# the following directories are needed:
# cache_dir - temporary directory for caching and joblib
# plot_dir  - where plots will be dumped
# bids_dir_meg - directory of the MEG bids data
# bids_dir_3T  - https://gin.g-node.org/lnnrtwttkhn/highspeed-bids
# bids_dir_3T_decoding - https://gin.g-node.org/lnnrtwttkhn/highspeed-decoding

# machine specific configuration overwrites general directory structure
if username == 'simon.kern' and '.zi.local' in host:  # VM or cluster
    cache_dir = f'{home}/Desktop/highspeed-joblib/'
    bids_dir_meg = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/'
    bids_dir_3T = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-3T-bids/'
    bids_dir_3T_decoding = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-3T-decoding/'
    plot_dir = f'{script_dir}/../plots/'

elif username == 'simon.kern' and host=='5cd320lfh8':
    cache_dir = f'{home}/Desktop/joblib-fasterplay/'
    plot_dir = '../plots/'
    bids_dir_meg = "W:/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids"
    bids_dir_3T = 'W:/group_klips/data/data/Simon/highspeed/highspeed-3T-bids/'
    bids_dir_3T_decoding = 'w:/group_klips/data/data/Simon/highspeed/highspeed-3T-decoding/'
    plot_dir = f'{script_dir}/../plots/'

elif username=='simon' and host=='kubuntu':
    cache_dir = f'{home}/Desktop/joblib-fasterplay/'
    # bids_dir_meg = "W:/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/"
    bids_dir_3T = bids_dir = '/home/simon/Desktop/highspeed-bids/'
    bids_dir_3T_decoding = '/home/simon/Desktop/highspeed-decoding/'
    plot_dir = f'{script_dir}/../plots/'
else:
    raise Exception(f'No user specific settings found in settings.py with {username=} and {host=}')

#####################################
# END OF USER-SPECIFIC CONFIGURATION
#####################################

#%% convert to abspaths

cache_dir = os.path.abspath(cache_dir)
plot_dir = os.path.abspath(plot_dir)
bids_dir_3T_decoding = os.path.abspath(bids_dir_3T_decoding)

os.makedirs(cache_dir, exist_ok=True)

#%% initialize MEG BIDS dir

if NO_PRELOADING:
    print('Not preloading MEG BIDS as NO_PRELOADING env var was set')
elif 'bids_dir_meg' in locals():
    bids_dir_meg = os.path.abspath(bids_dir_meg)
    # use database for faster loading. but recreate on each python startup
    # db_path = cache_dir + '/bids_meg.db'
    # python_start = psutil.Process(os.getpid()).create_time()
    # reset_database = (os.path.getmtime(db_path) if os.path.exists(db_path) else 0) < python_start
    # if reset_database:
    #     warnings.warn('resetting MEG BIDS database')
    layout_MEG = BIDSLayout(bids_dir_meg, derivatives=True)
    layout_MEG.subjects_all = [x for x in layout_MEG.get_subjects() if (not x in ['emptyroom', 'group'])]
    layout_MEG.subjects = [x for x in layout_MEG.subjects_all]

    if not layout_MEG.subjects:
        warnings.warn('No subjects in layout_MEG, are you sure it exists?')
else:
    layout_MEG = None
    warnings.warn('bids_dir_MEG has not been defined in settings.py')

#%% BIDS layout for 3T
if NO_PRELOADING:
    print('Not preloading fMRI BIDS as NO_PRELOADING env var was set')
elif 'bids_dir_3T' in locals():
    bids_dir_3T = os.path.abspath(bids_dir_3T)
    derivatives_dir_3T = os.path.join(bids_dir_3T, 'derivatives')
    os.makedirs(derivatives_dir_3T, exist_ok=True)
    # create dataset_description.json if missing (required by pybids for derivatives)
    _desc_file = os.path.join(derivatives_dir_3T, 'dataset_description.json')
    if not os.path.isfile(_desc_file):
        import json
        with open(_desc_file, 'w') as f:
            json.dump({"Name": "highspeed-3T-derivatives",
                        "BIDSVersion": "1.6.0",
                        "GeneratedBy": [{"Name": "highspeed-MEG-analysis"}]}, f, indent=4)
    # use database for faster loading. but recreate on each python startup
    # db_path = cache_dir + '/bids_3T.db'
    # python_start = psutil.Process(os.getpid()).create_time()
    # reset_database = (os.path.getmtime(db_path) if os.path.exists(db_path) else 0) < python_start
    # if reset_database:
        # warnings.warn('resetting 3T BIDS database')
    layout_3T = BIDSLayout(bids_dir_3T, derivatives=True)
    layout_3T.subjects = layout_3T.get_subjects()
    if not layout_3T.subjects:
        warnings.warn('No subjects in layout_3T, are you sure it exists?')
else:
    layout_3T = None
    warnings.warn('bids_dir_3T has not been defined in settings.py')

#%% BIDS decoding layout for 3T

# is this a valid BIDS directory? does not seem to be recognized

# if 'bids_dir_3T_decoding' in locals():
#     layout_3T_decoding = BIDSLayout(bids_dir_3T_decoding + '/decoding')
#     layout_3T_decoding.subjects = layout_3T_decoding.get_subjects()
#     if not layout_3T_decoding.subjects:
#         warnings.warn('No subjects in layout_3T, are you sure it exists?')
# else:
#     layout_3T_decoding = None
#     warnings.warn('bids_dir_3T has not been defined in settings.py')


#%% initializations, should need no change
# set environment variable for `meg_utils`, where to cache function calls
os.environ['JOBLIB_CACHE_DIR'] = cache_dir

#%% constants

intervals_MEG = [32, 64, 128, 512]
intervals_3T = [32, 64, 128, 512, 2048]

subjects_3T = [f'{i:02d}' for i in range(1, 41)]   # 40 participants
subjects_MEG = [f'{i:02d}' for i in range(1, 31)]  # 30 participants

normalization = 'lambda x: x/x.mean(0)'  # normalization to be used on the probability vectors
tr_duration = 1.25  # TR duration in seconds

# these are the expected TRs that the forward an backward phases are occurring
# for, taken from Wittkuhn et al (2021).
# be aware: the indexing is using MATLAB style starting at 1!
# the slope arrays are usually starting at TR 1 as well.
exp_tr = {32:   {'fwd': [2, 4], 'bkw': [5, 7]},
          64:   {'fwd': [2, 4], 'bkw': [5, 7]},
          128:  {'fwd': [2, 4], 'bkw': [5, 8]},
          512:  {'fwd': [2, 5], 'bkw': [6, 9]},
          2048: {'fwd': [2, 7], 'bkw': [8, 13]}
          }

# time lags at which we expect the peak to be
# the resulting index is 0 for the 0th time lag.
# usually the indexing of sequenceness results
# includes the zero-time lag with np.nan, so indexing is zero-based.
exp_lag = {interval: round((interval+100)/10) for interval in intervals_MEG}

# translate trigger values from German to English
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

# the beautiful color palette used in the Wittkuhn et al. 2021 publication
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

color_fwd = sns.color_palette()[1]
color_bkw = sns.color_palette()[2]




#%% plotting settings
import matplotlib.pyplot as plt

plt.rc('font', size=14)          # default text
plt.rc('axes', titlesize=16)     # axes title
plt.rc('axes', labelsize=12)     # x and y labels
plt.rc('xtick', labelsize=11)    # x tick labels
plt.rc('ytick', labelsize=11)    # y tick labels
plt.rc('legend', fontsize=11)    # legend


#%% checks and safety measures

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
