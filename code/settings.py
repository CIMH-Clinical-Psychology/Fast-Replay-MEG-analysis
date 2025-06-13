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


###############################
#%%userconf
# USER SPECIFIC CONFIGURATION
###############################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

# overwrite this variable to set a custom data dir
bids_dir = "../data/"  # enter directory here where the data has been stored

# machine specific configuration overwrites general directory structure
if username == 'simon.kern' and host=='zislrds0035.zi.local':  # simons VM
    cache_dir = '/data/fastreplay/cache'
    bids_dir = '/data/fastreplay/Fast-Replay-MEG-bids/'
    plot_dir = f'{home}/Nextcloud/ZI/2024.10 FastReplayAnalysis//plots'

elif username == 'simon.kern' and host=='5cd320lfh8':
    cache_dir = f'{home}/Desktop/joblib-resting-state/'
    # bids_dir = "W:/group_klips/data/data/Simon/DeSMRRest/upload/"

elif username == 'simon' and host in ('thinkpad-simon', 'desktop-dakomj2'):
    cache_dir = 'z:/joblib-fastreplay/'
    bids_dir = "z:/fastreplay-MEG-bids/"
else:
    warnings.warn('No user specific settings found in settings.py')

os.environ['JOBLIB_CACHE_DIR'] = cache_dir

#%% checks for stuff
# if 'cache_dir' not in locals():
#     cache_dir = f"{bids_dir}/cache/"  # used for caching
# if 'plot_dir' not in locals():
#     plot_dir = f"{bids_dir}/plots/"  # plots will be stored here
# if 'log_dir' not in locals():
#     log_dir = f"{bids_dir}/plots/logs/"  # log files will be created here


# results_dir = os.path.expanduser(f"{bids_dir}/results/")  # final results here

# if bids_dir == "":
#     raise Exception(f"please set bids_dir configuration in settings.py")


# if not os.path.isdir(plot_dir):
#     warnings.warn(f"plot_dir does not exist at {plot_dir}, create")
#     os.makedirs(plot_dir, exist_ok=True)
# if not os.path.isdir(log_dir):
#     warnings.warn(f"log_dir does not exist at {log_dir}, create")
#     os.makedirs(log_dir, exist_ok=True)




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
