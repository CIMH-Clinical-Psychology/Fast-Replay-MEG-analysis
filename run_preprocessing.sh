#!/bin/bash

BIDS_ROOT=/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/

if [ ! -d "$BIDS_ROOT" ]; then
    echo "ERROR: BIDS_ROOT directory not found, edit in run_preprocessing.sh: $BIDS_ROOT" >&2
    exit 1
fi

 mne_bids_pipeline --task rest1 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf.py
 mne_bids_pipeline --task rest2 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf.py
 mne_bids_pipeline --task main --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf.py

 # next run tasks with missing EOGs
mne_bids_pipeline --task rest1 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf_eog-missing.py
mne_bids_pipeline --task rest2 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf_eog-missing.py
mne_bids_pipeline --task main --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf_eog-missing.py
