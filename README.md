# FASTIMAGES - benchmark

FASTIMAGES - a benchmark for sequence replay detection methods in human neuroimaging using MEG and fMRI recordings. 

<video src="https://github.com/user-attachments/assets/b7301d15-ccd7-4406-918b-016d4c8b6894" controls="controls" muted="muted" class="d-block img-fluid" autoplay="autoplay" loop="loop" style="max-width: 100%;">
</video>

### Overview

We let 70 participants (30 MEG / 40 fMRI) view [fast sequences of five images](https://github.com/user-attachments/assets/b7301d15-ccd7-4406-918b-016d4c8b6894) in four different speeds (32/64/128/512 ms interval). Together with a functional localizer, this data can be used to test sequence detection algorithms under realistic conditions. 

The repository contains two things:

1) Probability time series and examples for benchmarking sequence detection algorithms (/examples and /sequence_predictions)

2) All code to reproduce the publication "FASTIMAGES: Validating replay detection methods in human neuroimaging using a combined MEG and fMRI dataset" (/code)

## Quickstart

If you're just interested to benchmark your sequence detection algorithm against real neural sequences of known order, we got you covered. 

1. Clone the repository via `git clone https://github.com/CIMH-Clinical-Psychology/FASTIMAGES-benchmark`

2. Install dependencies via `pip install -r requirements_noversions.txt`

3. Per participant, we provide probability time series per trial, decoding the five image categories stored at `/sequence_predictions`. Stand-alone scripts that work directly from the HDF5 files without importing are here:
   
   1. `examples/visualize_sequences.py` — recreates the per-position probability
      plot from `code/1_run_fastimages/3_run_visualize_sequences.py` for both
      modalities.
   
   2. `examples/run_tdlm_soda.py` — runs **TDLM** on the MEG data and **SODA** on
      the fMRI data and plots the resulting group-level curves. Requires the `tdlm` and `soda` Python packages.

If you want to train your own classifiers or tinker with the data some more, you need to fetch the BIDS dataset for fMRI and MEG (see below).

## Prequesists

To run the full pipeline to recreate the publication FASTIMAGES you need to

1. Clone the FASTIMAGES repository `git clone https://github.com/CIMH-Clinical-Psychology/FASTIMAGES-benchmark`

2. Install dependencies `pip install -r requirements.txt` (preferably in a new venv or conda env)

3. Download the MEG BIDS dataset from https://gin.g-node.org/skjerns/highspeed-MEG-bids, see instructions there how to do that.

4. Download the 3T fMRI BIDS dataset from xxx

5. Download the 3T fMRI decoding BIDS from xxx

6. Put the paths where you stored the files in `setting.py`
   
   

## Prepare 3T fMRI data

We download a subsection of the data from Wittkhn et al 2021. We will not reproduce the entire results but instead use their precomputed results. For this we need the following commands.

1. Go to the directory where you want to store the data. You'll need X GB free space.

2. clone the highspeed-analysis dir `datalad clone https://gin.g-node.org/lnnrtwttkhn/highspeed-analysis`

## Preprocessing

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

# after analysis is done, remove intermediate .fif files,
# the venv and the pipeline cache to free up disk space
make preprocessing-cleanup
```

If there are no errors, you're preprocessed files should be ready and you can start the analysis.

## Analysis

The analysis can be run with running the scripts in the following order:

```
run_decoding.py
run_analysis_fast_images.py
```

# 


