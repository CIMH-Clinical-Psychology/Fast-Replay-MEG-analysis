# FASTIMAGES - benchmark

FASTIMAGES - a benchmark for sequence replay detection methods in human neuroimaging using MEG and fMRI recordings.

### Overview

We let 70 participants (30 MEG / 40 fMRI) view [fast sequences of five images](https://github.com/user-attachments/assets/b7301d15-ccd7-4406-918b-016d4c8b6894) in four different speeds (32/64/128/512 ms interval). Together with a functional localizer, this data can be used to test sequence detection algorithms under realistic conditions.

The repository contains two things:

1. Simple per-trial probability time series and examples for benchmarking sequence-detection algorithms (`/examples` and `/sequence_predictions`).
2. The full code to reproduce the publication *"FASTIMAGES: Validating replay detection methods in human neuroimaging using a combined MEG and fMRI dataset"* (`/code`).

<video src="https://github.com/user-attachments/assets/b7301d15-ccd7-4406-918b-016d4c8b6894" controls="controls" width="200">
</video>

## Quickstart — benchmarking your method

If you just want to test your sequence-detection algorithm against real neural sequences of known order, you do **not** need to download any BIDS data. The repository already ships the per-trial classifier probabilities as HDF5 files under [`sequence_predictions/`](sequence_predictions/).

```bash
git clone https://github.com/CIMH-Clinical-Psychology/FASTIMAGES-benchmark
cd FASTIMAGES-benchmark
pip install h5py numpy pandas matplotlib seaborn  # core deps
pip install tdlm-python  # only if you run the TDLM example
pip install git+https://github.com/skjerns/SODA-Python/  # only if you run the SODA example
```

Then look at the two stand-alone scripts in [`examples/`](examples/):

- [`examples/visualize_sequences.py`](examples/visualize_sequences.py) — recreates the per-position probability plot for MEG and fMRI directly from the HDF5 files.
- [`examples/run_tdlm_soda.py`](examples/run_tdlm_soda.py) — runs **TDLM** on the MEG probabilities and **SODA** on the fMRI probabilities and plots the group-level curves.

The HDF5 schema (probability arrays, sequence labels, file attributes, plus load snippets for Python / R / MATLAB / Julia) is documented in [`sequence_predictions/README.md`](sequence_predictions/README.md).

## Reproduce FASTIMAGES

The steps below regenerate the figures and the published probabilities from raw data.

### Prerequisites

- **Python** ≥ 3.10 (Linux/macOS; Windows works for the analysis but not the preprocessing pipeline).
- **git** to clone this repository.
- A client to fetch the BIDS datasets: either [GIN](https://gin.g-node.org/G-Node/gin-cli-releases) (recommended) or [DataLad](https://www.datalad.org/) (works because GIN is git-annex under the hood).
- **Disk space**: ~205 GB for the MEG BIDS dataset and ~550 MB for the 3T fMRI BIDS dataset (much more if you also pull the original 4D BOLD NIfTIs from the upstream Wittkuhn repos — see step 4).
- **Optional**: SLURM cluster access for parallel power analyses (`code/2_run_comparison/3_submit_power_analysis.sh`).

### Setup

1. **Clone this repository and initialise the submodule** (`meg_utils` is a git submodule shared with other projects):

   ```bash
   git clone https://github.com/CIMH-Clinical-Psychology/FASTIMAGES-benchmark
   cd FASTIMAGES-benchmark
   git submodule update --init --recursive
   ```

   (Or in one shot: `git clone --recurse-submodules <url>`.)

2. **Install Python dependencies** into a fresh environment:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

   If you also want to run the example scripts that depend on third-party algorithms, install them too:

   ```bash
   pip install tdlm-python  # only if you run the TDLM example
   pip install git+https://github.com/skjerns/SODA-Python/ 
   ```

3. **Download the MEG BIDS dataset** (~205 GB; includes preprocessed `derivatives/`):

   ```bash
   gin get skjerns/FASTIMAGES-MEG-bids
   ```

   See the [FASTIMAGES-MEG-bids README](https://gin.g-node.org/skjerns/FASTIMAGES-MEG-bids) for alternatives (DataLad, selective download) and for re-running the preprocessing.

4. **Download the combined 3T fMRI BIDS + decoding dataset** ([`FASTIMAGES-3T-bids`](https://gin.g-node.org/skjerns/FASTIMAGES-3T-bids), ~550 MB):

   ```bash
   gin get skjerns/FASTIMAGES-3T-bids
   ```

   This single tree contains the BIDS events files together with the per-subject decoding CSVs (no separate `highspeed-decoding` repo is needed any more). See the [FASTIMAGES-3T-bids README](https://gin.g-node.org/skjerns/FASTIMAGES-3T-bids) for the DataLad alternative.

   > **Note:** `FASTIMAGES-3T-bids` is a *compressed* repackaging that drops the original 4D BOLD NIfTIs — the full analysis here only needs the events files plus the precomputed decoding probabilities, so the raw functional volumes are not shipped. If you want them (e.g. to retrain your own decoders or rerun fMRIPrep), grab the originals from Wittkuhn & Schuck 2021: [`lnnrtwttkhn/highspeed-bids`](https://gin.g-node.org/lnnrtwttkhn/highspeed-bids) for the raw BIDS data and [`lnnrtwttkhn/highspeed-decoding`](https://gin.g-node.org/lnnrtwttkhn/highspeed-decoding) for their decoding outputs. You'll need this to train your own decoders.

5. **Configure paths.** Open `code/settings.py` and set `bids_dir_meg` and `bids_dir_3T` to wherever you put the two datasets. Per-machine branches are supported; pick one that matches your `getpass.getuser()` / `platform.node()` or add your own.

### Run the analysis

The analysis is two phases, run in order from inside the relevant subfolder. Scripts named `*_viz_*` produce the figures for the script with the matching prefix; they read the saved results and can be re-run on their own.

**Phase 1 — train MEG decoders and apply them to the fast-sequence trials** (`code/1_run_fastimages/`):

```bash
python 1a_run_best_l1_meg.py             # L1 gridsearch on localizer trials
python 1a_viz_best_l1_meg.py             # plot the L1 gridsearch
python 1b_classifier_generalization.py   # temporal generalization matrices (TGMs)
python 1b_viz_classifier_generalization.py
python 1c_train_localizer_meg.py         # train final classifiers with best L1/timepoint
python 1c_viz_localizer.py               # MEG vs fMRI localizer decoding figure
python 2_run_analysis_fast_images.py     # aggregate MEG accuracies vs fMRI
python 3_run_visualize_sequences.py      # per-position probability traces
```

> **Note:** Phase 2 reads pickled intermediate outputs produced by Phase 1, so Phase 1 must have been run end-to-end (for the same `bids_dir_meg`) before Phase 2 will work.

**Phase 2 — run TDLM / SODA on both modalities and compare** (`code/2_run_comparison/`):

```bash
python ../scripts/0_extract_trial_probas.py   # (re)build sequence_predictions/*.h5

python 1a_run_tdlm_meg.py                     # TDLM on MEG
python 1a_suppl_2-step-tdlm.py                # supplementary: 2-step TDLM
python 1b_run_soda_fmri.py                    # SODA on fMRI

python 2a_run_soda_meg.py                     # SODA on MEG (cross-method)
python 2b_run_tdlm_fmri.py                    # TDLM on fMRI (cross-method)
python 2c_run_decoder_extension.py            # decoder extension experiments

python 4_compare_tdlm_soda.py                 # side-by-side effect-size comparison
```

**Power analyses** (`code/2_run_comparison/3a_*`, `3b_*`): these are heavier and need to be **run separately**, typically as SLURM array jobs — see `3_submit_power_analysis.sh`. Each method (TDLM / SODA) has three statistical-test variants (`signflip`, `cluster`, `ttest`) plus a corresponding `*_plot.py` to render the curves.

## Preprocessing

Preprocessed `.fif` files are already shipped inside the MEG BIDS dataset (`/derivatives`). If you want to regenerate them yourself, the pipeline lives next to the data — see the **Preprocessing** section of the [FASTIMAGES-MEG-bids README](https://gin.g-node.org/skjerns/FASTIMAGES-MEG-bids) (`make install-preprocessing && make preprocessing`, or `make preprocessing_slurm` on a SLURM cluster).

## Issues & contact

Bugs, questions or feature requests → please open an [issue on GitHub](https://github.com/CIMH-Clinical-Psychology/FASTIMAGES-benchmark/issues).

## License

This repository (code and documentation) is released under the [GNU GPL v3](LICENSE). The bundled probability files in [`sequence_predictions/`](sequence_predictions/) and the upstream BIDS datasets ([`FASTIMAGES-MEG-bids`](https://gin.g-node.org/skjerns/FASTIMAGES-MEG-bids), [`FASTIMAGES-3T-bids`](https://gin.g-node.org/skjerns/FASTIMAGES-3T-bids)) are distributed under Creative Commons **CC-BY-SA-4.0**, inherited from Wittkuhn & Schuck 2021.
