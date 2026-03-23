# Fast-Replay-MEG analysis

Analysis of the dataset [fastreplay-MEG-bids](https://github.com/CIMH-Clinical-Psychology/fastreplay-MEG-bids), replicating [Wittkuhn et al 2021](https://www.nature.com/articles/s41467-021-21970-2)

### Prequesists

TODO: xxx. explain datalad and git and python ect

### Prepare 3T fMRI data

We download a subsection of the data from Wittkhn et al 2021. We will not reproduce the entire results but instead use their precomputed results. For this we need the following commands.

1. Go to the directory where you want to store the data. You'll need X GB free space.

2. clone the highspeed-analysis dir `datalad clone https://gin.g-node.org/lnnrtwttkhn/highspeed-analysis`

### Preparation

Download



Assumption 1: You have downloaded the BIDS dataset to your local machine.

Assumption 2: you have installed the requirements via `pip install -r requirements.txt`

Run the preprocessing pipeline with the following commands within the root directory (`fastreplay-MEG-analysis/`), replacing `BIDS_ROOT` with the directory where you stored the [fastreplay-MEG-bids](https://github.com/CIMH-Clinical-Psychology/fastreplay-MEG-bids) dataset, e.g. `/data/fastreplay/Fast-Replay-MEG-bids/`

Each session/task must be run independently as `mne_bids_pipeline` can't process rest and task data together yet.

```bash
export BIDS_ROOT=/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/

mne_bids_pipeline --task rest1 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf.py --steps init,preprocessing
mne_bids_pipeline --task rest2 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/  --config=preprocessing_pipeline_conf.py --steps init,preprocessing
mne_bids_pipeline --task main --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/  --config=preprocessing_pipeline_conf.py --steps init,preprocessing

# next run tasks with missing EOGs
mne_bids_pipeline --task rest1 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/ --config=preprocessing_pipeline_conf_eog-missing.py --steps init,preprocessing
mne_bids_pipeline --task rest2 --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/  --config=preprocessing_pipeline_conf_eog-missing.py --steps init,preprocessing
mne_bids_pipeline --task main --root-dir $BIDS_ROOT --deriv_root $BIDS_ROOT/derivatives/  --config=preprocessing_pipeline_conf_eog-missing.py --steps init,preprocessing
```

```python
if username == 'user.name' and host=='hostname':  # VM
    cache_dir = '/data/fastreplay/cache/'  # used for caching results, needs some GB of space 
    bids_dir = '/data/fastreplay/Fast-Replay-MEG-bids/'  # BIDS_ROOT from above
    plot_dir = f'{home}/Nextcloud/ZI/2024.10 FastReplayAnalysis/plots/'  # final plots will be saved here
```

## Running the analysis

The analysis can be run with running the scripts in the following order:

```
run_decoding.py
run_analysis_fast_images.py
```

### 1. Create decoders on localizer data
