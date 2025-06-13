# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:58:25 2024

@author: Simon
"""
import os
import mne
import warnings
import numpy as np
from bids import BIDSLayout
import settings
from meg_utils.pipeline import DataPipeline, LoadRawStep, EpochingStep
from meg_utils.pipeline import ResampleStep, NormalizationStep, CustomStep
from meg_utils.pipeline import ToArrayStep, StratifyStep
from meg_utils.preprocessing import rescale_meg_transform_outlier

try:
    layout = BIDSLayout(settings.bids_dir, derivatives=True)
    layout.subjects = layout.get_subjects()
    if not layout.subjects:
        warnings.warn('No subjects in layout, are you sure it exists?')

except Exception as e:
    warnings.warn(f'Could not load BIDSLayout at {settings.bids_dir=}: {e} {repr(e)}')


def check_derivatives(layout=layout):
    """check if derivatives folder exists and preprocessing did run"""
    warnings.warn('NOTIMPLEMENTED')


def load_localizer(subject, tmin=-0.2, tmax=0.8, sfreq=100, verbose=False):
    """load all localizer trials ('slow trials')"""
    main_files = layout.get(subject=subject, suffix='raw',
                            scope='derivatives', proc='clean',
                            task='main', return_type='filenames')
    assert len(main_files)==1, f'[sub-{subject}] {len(main_files)=} more or less than 1, did preprocessing run?'

    pipe = DataPipeline([
        ('load raw', LoadRawStep()),
        (f'resampling to {sfreq}Hz (optional)', ResampleStep(sfreq=sfreq)),
        ('epoching', EpochingStep(event_id=np.arange(1, 6), tmin=tmin, tmax=tmax)),
        ('pick meg', CustomStep(lambda x: x.pick('meg'))),
        ('normalization', NormalizationStep(rescale_meg_transform_outlier, picks='meg')),
        ('stratify', StratifyStep()),
        ('to array', ToArrayStep(X=True, y=True))
        ], verbose=True)

    if not verbose:
        pipe.set_params_all(overwrite_param=True, verbose='WARNING')

    data_x, data_y = pipe.transform(main_files[0])
    data_y -= 1  # python indexing starts at 1
    assert min(data_y)==0
    # assert len(set(np.bincount(data_y)))==1
    warnings.warn('comment this back')
    return data_x, data_y

check_derivatives()

#%%
if __name__=='__main__':
    sfreq = 100
    tmin = -0.2
    tmax = 0.8
    pipe = DataPipeline([
        ('load raw', LoadRawStep()),
        (f'resampling to {sfreq}Hz', ResampleStep(sfreq=sfreq)),
        ('epoching', EpochingStep(event_id=np.arange(1, 6), tmin=tmin, tmax=tmax)),
        ('pick meg', CustomStep(lambda x: x.pick('meg'))),
        ('normalization', NormalizationStep(rescale_meg_transform_outlier, picks='meg')),
        ('to array', ToArrayStep(X=True, y=True))
        ], verbose=True)

    fif_file = 'Z:/fastreplay-MEG-bids/derivatives/sub-16/meg/sub-16_task-main_proc-clean_raw.fif'
    raw = pipe.transform(fif_file)
    # data_x, data_y = data.load_epochs(main_files[0], tmin=tmin, tmax=tmax,)
    # return data_x, data_y
