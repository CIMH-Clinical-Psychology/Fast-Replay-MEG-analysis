# Fast-Replay-MEG analysis

Analysis of the dataset [fastreplay-MEG-bids](https://github.com/CIMH-Clinical-Psychology/fastreplay-MEG-bids), replicating [Wittkuhn et al 2021](https://www.nature.com/articles/s41467-021-21970-2)

### Prequesists

TODO: xxx. explain datalad and git and python ect

### Prepare 3T fMRI data

We download a subsection of the data from Wittkhn et al 2021. We will not reproduce the entire results but instead use their precomputed results. For this we need the following commands.

1. Go to the directory where you want to store the data. You'll need X GB free space.

2. clone the highspeed-analysis dir `datalad clone https://gin.g-node.org/lnnrtwttkhn/highspeed-analysis`

### Preprocessing



Assumption: You have downloaded the BIDS dataset to your local machine.

In the `Makefile`, change the `BIDS_ROOT` to the directory where you cloned the BIDS directory to. Then call the following commands

```shell
make install       # install venv and mne_bids_pipeline
make init          # initialize the derivatives dir
make preprocessing # start the preprocessing, takes several hours

# alternatively, you can process participants in parallel 
# on a SLURM cluster. Edit run_preprocessing.sbatch and then 
# instead of the preprocessing command run 
make preprocessing_slurm
```

If there are no errors, you're preprocessed files should be ready and you can start the analysis.

## Analysis

The analysis can be run with running the scripts in the following order:

```
run_decoding.py
run_analysis_fast_images.py
```

### 1. Create decoders on localizer data
