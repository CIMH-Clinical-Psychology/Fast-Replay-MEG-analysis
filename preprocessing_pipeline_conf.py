###############################################################################
### Preprocessing pipeline settings for mne-bids-pipeline |
###############################################################################
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

from annotated_types import Ge, Interval, Len, MinLen
from mne import Covariance
from mne_bids import BIDSPath

from mne_bids_pipeline.typing import (
    # ArbitraryContrast,
    # DigMontageType,
    # FloatArrayLike,
    PathLike,
)

# %%05
# # General settings
# Configuration file for mne-bids-pipeline

from typing import Sequence, Literal, Optional
from mne import Covariance
from mne_bids import BIDSPath

# %%
# General Settings

bids_root: PathLike | None = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/'
deriv_root: PathLike = f"{bids_root}/derivatives/"  # Save all processed data under /derivatives/
# subjects_dir: Optional[PathLike] = f"{deriv_root}/freesurfer/subjects/"  # Path to FreeSurfer subject reconstructions
interactive: bool = False  # Disable interactive elements
# sessions: Literal["all"] = "all"  # Process all sessions
# task: str = ""  # Process all tasks by leaving empty
task_is_rest: bool = True  # Treat data as resting-state, disable epoching
# runs: Literal["all"] = "all"  # Process all runs
exclude_runs: Optional[dict[str, list[str]]] = None  # No excluded runs

subjects: Sequence[str] | Literal["all"] = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']  # Analyze all subjects
# exclude_subjects: Sequence[str] = ['23']  # No excluded subjects
process_empty_room: bool = True  # Preprocess empty-room data
process_rest: bool = True  # Preprocess resting-state data
ch_types: Sequence[Literal["meg"]] = ["meg"]  # Include MEG and EEG channels
data_type: Literal["meg", "eeg"] = "meg"  # Data type is MEG
eog_channels: Sequence[str] = ["BIO002", "BIO003"]  # Specify EOG channels
ecg_channel: str = "BIO001"  # Specify ECG channel
spatial_filter: Literal["ica"] = "ica"  # Use ICA for artifact removal
ica_n_components: int = 50  # Number of ICA components
ica_algorithm: str = 'picard'
rest_epochs_duration = 2
rest_epochs_overlap = 0
epochs_tmin = 0

# on_error  = 'continue'
baseline = None

# %%
# Preprocessing

raw_resample_sfreq: float = 100.0  # Resample data to 100 Hz
l_freq: float = 0.1  # Apply high-pass filter at 0.1 Hz
h_freq: Optional[float] = None  # Disable low-pass filter
notch_freq: Sequence[float] = [50.0]  # Apply notch filter at 50 Hz
notch_trans_bandwidth: float = 1.0  # Set notch filter transition bandwidth to 1 Hz

# find_noisy_channels_meg = True
# find_flat_channels_meg = True
# use_maxwell_filter = False
# %%
# Artifact Removal via ICA

# The pipeline will automatically identify and remove ICA components related to EOG and ECG.

# %%
# Source-level Analysis

# run_source_estimation: bool = True  # Enable source-level analysis
# inverse_method: Literal["dSPM"] = "dSPM"  # Use dSPM as the inverse solution method
# loose: float = 0.2  # Weigh parallel dipole components by 0.2
# depth: float = 0.8  # Set depth weighting exponent to 0.8
# noise_cov = "emptyroom"  # Use resting-state recording for noise covariance

# # %%
# # FreeSurfer recon-all Settings

# recon_all: bool = True  # Enable FreeSurfer's recon-all
# freesurfer_verbose: bool = True  # print the complete recon-all pipeline

# %%
# Parallelization

n_jobs: int = 6  # don't use too many, memory overhead is enormous
parallel_backend: Literal["loky"] = "loky"  # Use 'loky' backend for parallel processing

# %%
# Logging

log_level: Literal["info"] = "info"  # Set pipeline logging verbosity to 'info'
mne_log_level: Literal["error"] = "error"  # Set MNE-Python logging verbosity to 'error'

# %%
# Error Handling

# on_error: Literal["abort"] = "continue"  # Abort processing on errors
config_validation: Literal["raise"] = "raise"  # Raise exceptions on config validation issues
