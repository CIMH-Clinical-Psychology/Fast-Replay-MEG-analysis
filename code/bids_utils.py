# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:58:25 2024

these functions create a connection between BIDS and the meg_utils, loading the
data given a specific participant

@author: Simon
"""
import os
import mne
import warnings
import joblib
import numpy as np
import pandas as pd
import settings
from settings import layout_MEG, layout_3T
from meg_utils.pipeline import DataPipeline, LoadRawStep, EpochingStep
from meg_utils.pipeline import ResampleStep, NormalizationStep, CustomStep
from meg_utils.pipeline import ToArrayStep, StratifyStep
from meg_utils.preprocessing import rescale_meg_transform_outlier


mem = joblib.Memory(settings.cache_dir)

def load_latest_classifier(subject):
    clfs = layout_MEG.get(subject=subject, suffix='clf', extension='pkl.gz')
    clfs = [clf for clf in clfs if 'latest_clf' in clf.filename]
    assert len(clfs)==1, f'{len(clfs)=} classifier found for {subject=}'
    clf = joblib.load(clfs[0])
    return clf


def make_bids_fname(filename, scope='derivatives', subject='group',
                    modality='', suffix='', **entities):
    # Initialize a BIDSLayout with a dummy path since we're only using it to build paths
    # In practice, you would use your actual BIDS dataset path
    # Add the subject to the entities dictionary

    # Define a template for the filename
    # This template should be adjusted based on the specific BIDS structure you are using
    folder = f'{layout_MEG.root}/{scope}/{subject}/{modality}/'
    os.makedirs(folder, exist_ok=True)

    # Add additional entities to the template
    # This example assumes a simple structure; adjust as needed for your dataset
    name, ext = os.path.splitext(filename)
    file = f'{name}'
    for key in entities:
        file += f'_{key}-{entities[key]}'
    file += f'_{suffix}' if suffix else ''

    # Add the file extension if provided
    file += ext
    return os.path.join(folder, file)

def tsv2events(df_events, sfreq=100):
    """convert behavioural TSV trigger events to mne events format

    MNE BIDS pipeline resamples the trigger channel and therefore, some
    triggers are missing or shifted by a bit. thereofre take the original
    from the behavioural file and convert to events format of MNE"""
    events = np.zeros([len(df_events), 3])
    factor = 1000/100  # factor to downsample the sample number
    events[:, 0] = np.round(df_events['sample'] / factor)
    events[:, 2] = df_events.value
    return events.astype(int)

@mem.cache
def load_behaviour(subject, **filters):
    """
    Load and filter a behavioral BIDS file for a given subject.

    This function retrieves a behavioral data file in BIDS format for a specified subject,
    loads it into a pandas DataFrame, and applies column-based filters to the data.

    these column are known
    'onset', 'duration', 'subject', 'session', 'condition', 'trial_type',
           'stim_label', 'key_down', 'orientation', 'interval_time',
           'response_time', 'accuracy', 'stim_index', 'serial_position',
           'key_expected', 'key_pressed', 'choice_left', 'choice_right',
           'choice_correct'

    Parameters
    ----------
    subject : str or int
        The subject identifier used to locate the behavioral data file.
    **filters : dict
        Key-value pairs where the key is the column name in the DataFrame and the value
        is the criterion used to filter the DataFrame. Each filter is applied sequentially.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered behavioral data.

    Raises
    ------
    AssertionError
        If no behavioral file is found, if more than one file is found, if a specified
        filter column does not exist in the DataFrame, or if a filter results in an empty DataFrame.

    Notes
    -----
    - The function assumes that the BIDS layout object `layout` is already defined and accessible.
    - The function uses caching to store results for faster subsequent access.
    - The function expects exactly one behavioral file per subject; otherwise, an assertion error is raised.

    Examples
    --------
    >>> df = load_behaviour(subject='01', task='main', run=1)
    >>> print(df.head())
    """
    beh_tsv = layout_MEG.get(subject=subject, return_type='filenames', datatype='beh',
                         extension='tsv')
    assert len(beh_tsv)==1, \
         f'[sub-{subject}] {len(beh_tsv)=} more or less than 1, did preprocessing run?'
    df_beh = pd.read_csv(beh_tsv[0], delimiter='\t')

    # add trigger values of stimuli
    df_beh['trigger'] = df_beh['stim_label'].apply(lambda x: settings.trigger_img.get(x, np.nan)-1)

    # apply filter on columns
    for filt in filters:
        assert filt in df_beh.columns, f'{filt=} not in {df_beh.columns=}'
        filter_val = filters[filt]
        if isinstance(filter_val, list):
            df_beh = df_beh[df_beh[filt].isin(filter_val)]
        else:
            df_beh = df_beh[df_beh[filt]==filter_val]
        assert len(df_beh)>0, f'{filt=}={filter_val} did not match any fields'
    return df_beh


@mem.cache
def load_decoding_3T(subject, **filters):
    """
    Load and filter a behavioral BIDS file for a given subject.

    This function retrieves a behavioral data file in BIDS format for a specified subject,
    loads it into a pandas DataFrame, and applies column-based filters to the data.

    these column are known
    'pred_label', 'stim', 'tr', 'seq_tr', 'stim_tr', 'trial', 'run_study',
           'session', 'tITI', 'id', 'test_set', 'classifier', 'mask', 'class',
           'probability'

    Parameters
    ----------
    subject : str or int
        The subject identifier used to locate the behavioral data file.
    **filters : dict
        Key-value pairs where the key is the column name in the DataFrame and the value
        is the criterion used to filter the DataFrame. Each filter is applied sequentially.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered behavioral data.
    """
    filename = f'{settings.bids_dir_3T_decoding}/decoding/sub-{subject}/data/sub-{subject}_decoding.csv'
    df_proba = pd.read_csv(filename, low_memory=False)

    # apply filter on columns
    for filt in filters:
        assert filt in df_proba.columns, f'{filt=} not in {df_proba.columns=}'
        if isinstance(filters[filt], list):
            df_proba = df_proba[df_proba[filt].isin(filters[filt])]
        else:
            df_proba = df_proba[df_proba[filt]==filters[filt]]

        assert len(df_proba), f'{filt=}={filters[filt]} did not match any fields'
    df_proba = df_proba.apply(pd.to_numeric, errors='ignore').convert_dtypes()
    return df_proba


def load_decoding_seq_3T(subject, classifier=settings.categories,
                         mask='cv', test_set='test-seq_long', **filters):
    """load the probability estimates of the sequence trials, wall-time aligned TR onsets"""
    if 'seq' in test_set:
        condition='sequence'
    elif 'odd' in test_set:
        condition='oddball'
    elif 'rep' in test_set:
        condition='repetition'
    else:
        raise Exception(f'unkonwn test set: {test_set}')
    df_proba = load_decoding_3T(subject,
                                test_set=test_set,
                                mask=mask,
                                classifier=classifier,
                                **filters)
    # df_proba = df_proba[df_proba['class'].isin(classifier)]
    # apparently there were 530 TRs for each run.
    # to get the exact number of TR per trial, reverse the calculation
    df_proba['run_tr'] = df_proba.tr - (df_proba.run_study-1)*530
    df_proba['run_tr_onset'] =  df_proba['run_tr'] * 1.25

    df_seq = load_trial_data_3T(subject, condition=condition)
    df_proba  = (
        df_proba
        .merge(df_seq[['trial', 'run_study', 'seq_onset']], on=['trial', 'run_study'], how='left')
    )
    df_proba['tr_onset'] = df_proba.run_tr_onset - df_proba.seq_onset

    with warnings.catch_warnings():
        # Tell Python to ignore *only* FutureWarning-class messages
        warnings.simplefilter(action="ignore", category=FutureWarning)
        df_proba = df_proba.apply(pd.to_numeric, errors='ignore').convert_dtypes()

    return df_proba


@mem.cache
def load_events_3T(subject, **filters):
    """load the task data of the 3T dataset"""
    events_tsv = layout_3T.get(subject=subject, return_type='filenames', datatype='func',
                         extension='tsv', suffix='events')
    assert len(events_tsv)==8, \
         f'[sub-{subject}] {len(events_tsv)=} more or less than 8'
    df_events = pd.concat([ pd.read_csv(f, delimiter='\t') for f in events_tsv], ignore_index=True)
    # apply filter on columns
    for filt in filters:
        assert filt in df_events.columns, f'{filt=} not in {df_events.columns=}'
        filter_val = filters[filt]
        if isinstance(filter_val, list):
            df_events = df_events[df_events[filt].isin(filter_val)]
        else:
            df_events = df_events[df_events[filt]==filter_val]
        assert len(df_events)>0, f'{filt=}={filter_val} did not match any fields'
    df_events = df_events.apply(pd.to_numeric, errors='ignore').convert_dtypes()
    return df_events


@mem.cache
def load_trial_data_3T(subject, condition='sequence'):
    """load the sequence or oddball data per trial"""
    df_events = load_events_3T(subject, condition=condition, trial_type='stimulus')

    seqs = []
    for trial, df in df_events.groupby('trial'):
        seqs += [{'seq_onset': df.iloc[0].onset,
                  'trial': df.trial.iloc[0],
                  'session': df.session.iloc[0],
                  'tITI': str(df.interval_time.iloc[0]),
                  'run_session': df.run_session.iloc[0],
                  'run_study': df.run_study.iloc[0],
                  'serial_position': df.serial_position.values,
                  'stim_index': df.stim_index.values,
                  'stim_label': df.stim_label.values,

                  }]
    df_seq = pd.concat([pd.Series(seq) for seq in seqs], axis=1).transpose()
    df_seq = df_seq.apply(pd.to_numeric, errors='ignore').convert_dtypes()
    return df_seq




@mem.cache
def load_fast_sequences(subject, intervals=[32, 64, 128, 512], sfreq=100,
                       tmin=3.1, tmax=3+(0.512+0.1)*5, verbose=False):
    """loads the fast sequences as one go, starting 3.1s after the text cue,
    i.e. 200ms before the first seq stimulus, ranging to maximum of sequences"""
    main_files = layout_MEG.get(subject=subject, suffix='raw',
                            scope='derivatives', proc='clean',
                            task='main', return_type='filenames')
    assert len(main_files)==1, \
        f'[sub-{subject}] {len(main_files)=} more or less than 1, did preprocessing run?'

    # actually, take the data_y from the behavioural file, because the triggers
    # sometimes get downsamples in a weird way in the bids pipeline, leading
    # to shifts in the value. Check if this is still halfway matching though!
    events_file = layout_MEG.get(subject=subject, task='main', suffix='events', extension='tsv')
    assert len(events_file)==1, \
        f'[sub-{subject}] {len(events_file)=} more or less than 1, did preprocessing run?'
    df_events = pd.read_csv(events_file[0], sep='\t')
    df_cue = df_events[(df_events.value<16) & (df_events.value>10)]
    events = tsv2events(df_cue, sfreq=sfreq)
    assert len(df_cue)==64

    pipe = DataPipeline([
        ('load raw', LoadRawStep()),
        (f'resampling to {sfreq}Hz (optional)', ResampleStep(sfreq=sfreq)),
        ('epoching', EpochingStep(events=events, tmin=tmin, tmax=tmax,
                                  reject_by_annotation=False, baseline=None)),
        ('pick meg', CustomStep(lambda x: x.pick('meg'))),
        ('normalization', NormalizationStep(rescale_meg_transform_outlier, picks='meg')),
        ('to array', ToArrayStep(X=True, y=True))
        ], verbose=True)

    if not verbose:
        pipe.set_params_all(overwrite_param=True, verbose='WARNING')

    data_x, data_y = pipe.transform(main_files[0])
    data_y -= settings.trigger_base_val_cue + 1  # subtract sequence baseval
    df_beh = load_behaviour(subject, condition='sequence', trial_type='stimulus')
    df_beh['idx'] = np.repeat(np.arange(64), 5)
    df_beh.drop(['duration', 'session', 'subject', 'onset', 'trial_type', 'condition'],
                axis=1, inplace=True)
    df_beh['sequence'] = df_beh['trigger'].apply(lambda x: chr(65 + int(x)))
    beh = [df for _, df in df_beh.groupby('idx')]

    assert len(beh)==len(data_y)
    return data_x, data_y, beh

@mem.cache
def load_fast_images(subject, intervals=[32, 64, 128, 512], tmin=-0.2, tmax=0.5,
                     sfreq=100, verbose=False, positions=[1,2,3,4, 5]):
    main_files = layout_MEG.get(subject=subject, suffix='raw',
                            scope='derivatives', proc='clean',
                            task='main', return_type='filenames')
    assert len(main_files)==1, \
        f'[sub-{subject}] {len(main_files)=} more or less than 1, did preprocessing run?'
    assert not 0 in positions, 'indexing starts at 1 here'

    # actually, take the data_y from the behavioural file, because the triggers
    # sometimes get downsamples in a weird way in the bids pipeline, leading
    # to shifts in the value. Check if this is still halfway matching though!
    events_file = layout_MEG.get(subject=subject, task='main', suffix='events', extension='tsv')
    assert len(events_file)==1, \
        f'[sub-{subject}] {len(events_file)=} more or less than 1, did preprocessing run?'
    df_events = pd.read_csv(events_file[0], sep='\t')
    df_events = df_events[(df_events.value<26) & (df_events.value>20)]
    events = tsv2events(df_events, sfreq=sfreq)
    assert len(df_events)==320

    pipe = DataPipeline([
        ('load raw', LoadRawStep()),
        (f'resampling to {sfreq}Hz (optional)', ResampleStep(sfreq=sfreq)),
        ('epoching', EpochingStep(events=events, tmin=tmin, tmax=tmax, reject_by_annotation=False)),
        ('pick meg', CustomStep(lambda x: x.pick('meg'))),
        ('normalization', NormalizationStep(rescale_meg_transform_outlier, picks='meg')),
        ('to array', ToArrayStep(X=True, y=True))
        ], verbose=True)

    if not verbose:
        pipe.set_params_all(overwrite_param=True, verbose='WARNING')

    data_x, data_y = pipe.transform(main_files[0])
    data_y -= settings.trigger_base_val_sequence + 1  # subtract sequence baseval

    df_beh = load_behaviour(subject, condition='sequence', trial_type='stimulus')
    df_beh.drop(['duration', 'session', 'subject', 'onset', 'trial_type',
                 'condition'],
                axis=1, inplace=True)
    idx_sel = (df_beh.serial_position.isin(np.array(positions)))
    df_beh = df_beh[idx_sel]
    data_x = data_x[idx_sel]
    data_y = data_y[idx_sel]

    assert len(df_beh)==len(data_y)
    assert all((df_beh.trigger).values==data_y)
    return data_x, data_y, df_beh

@mem.cache
def load_localizer(subject, tmin=-0.2, tmax=0.8, sfreq=100, verbose=False):
    """load all localizer trials ('slow trials')"""
    main_files = layout_MEG.get(subject=subject, suffix='raw',
                            scope='derivatives', proc='clean',
                            task='main', return_type='filenames')
    assert len(main_files)==1, f'[sub-{subject}] {len(main_files)=} more or less than 1, did preprocessing run?'

    # actually, take the data_y from the trigger events file, because the triggers
    # sometimes get downsamples in a weird way in the bids pipeline, leading
    # to shifts in the value.
    events_file = layout_MEG.get(subject=subject, task='main', suffix='events', extension='tsv')
    assert len(events_file)==1, \
        f'[sub-{subject}] {len(events_file)=} more or less than 1, did preprocessing run?'
    df_events = pd.read_csv(events_file[0], sep='\t')
    df_events = df_events[df_events.value<6]
    events = tsv2events(df_events, sfreq=sfreq)
    assert len(events)==160
    verbose_mne = 'ERROR' if not verbose else 'WARNING'
    # next define the pipeline for loading the raw data
    pipe = DataPipeline([
        ('load raw', LoadRawStep(verbose=verbose_mne)),
        (f'resampling to {sfreq}Hz (optional)', ResampleStep(sfreq=sfreq, verbose=verbose_mne)),
        ('epoching', EpochingStep(events=events, tmin=tmin, tmax=tmax, reject_by_annotation=False,
                                  verbose=verbose_mne)),
        ('pick meg', CustomStep(lambda x: x.pick('meg', verbose='ERROR'))),
        ('normalization', NormalizationStep(rescale_meg_transform_outlier, picks='meg')),
        # ('stratify', StratifyStep()),
        ('to array', ToArrayStep(X=True, y=True))
        ], verbose=verbose)


    if not verbose:
        pipe.set_params_all(overwrite_param=True, verbose=verbose_mne)

    data_x, data_y = pipe.transform(main_files[0])
    data_y -= 1

    df_beh = load_behaviour(subject, orientation=0, condition='localizer', trial_type='stimulus')
    # triggers = df_beh.trigger[df_beh.trigger<6]
    # assert sum(triggers!=data_y)<2, f'more than one trigger mismatch {subject=}, {sum(triggers!=data_y)=}?'
    # data_y = (triggers.values -1).astype(int)
    # assert len(data_y)==160
    assert all(events[:, 2]-1 == data_y)
    assert len(set(np.bincount(data_y)))==1
    assert min(data_y)==0
    return data_x, data_y, df_beh

#%% in-file testing
# if __name__=='__main__':
#     sfreq = 100
#     tmin = -0.2
#     tmax = 0.8
#     pipe = DataPipeline([
#         ('load raw', LoadRawStep()),
#         (f'resampling to {sfreq}Hz', ResampleStep(sfreq=sfreq)),
#         ('epoching', EpochingStep(event_id=np.arange(1, 6), tmin=tmin, tmax=tmax)),
#         ('pick meg', CustomStep(lambda x: x.pick('meg'))),
#         ('normalization', NormalizationStep(rescale_meg_transform_outlier, picks='meg')),
#         ('to array', ToArrayStep(X=True, y=True))
#         ], verbose=True)

#     fif_file = 'Z:/fastreplay-MEG-bids/derivatives/sub-16/meg/sub-16_task-main_proc-clean_raw.fif'
#     raw = pipe.transform(fif_file)
#     # data_x, data_y = data.load_epochs(main_files[0], tmin=tmin, tmax=tmax,)
#     # return data_x, data_y
