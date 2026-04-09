VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MNE_BIDS := $(VENV)/bin/mne_bids_pipeline
BIDS_ROOT := /zi/home/simon.kern/highspeed-MEG-bids

.PHONY: all init install preprocessing preprocessing_slurm check-bids

all: init install preprocessing

init: check-venv check-bids
	git submodule update --init --recursive
	$(MNE_BIDS) --task rest1 --root-dir $(BIDS_ROOT) --deriv_root $(BIDS_ROOT)/derivatives/ --config=preprocessing_pipeline_conf.py --steps init
	$(MNE_BIDS) --task rest2 --root-dir $(BIDS_ROOT) --deriv_root $(BIDS_ROOT)/derivatives/ --config=preprocessing_pipeline_conf.py --steps init
	$(MNE_BIDS) --task main  --root-dir $(BIDS_ROOT) --deriv_root $(BIDS_ROOT)/derivatives/ --config=preprocessing_pipeline_conf.py --steps init

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install mne_bids_pipeline==1.10.0

check-bids:
	@test -d "$(BIDS_ROOT)" || { echo "ERROR: BIDS_ROOT not found: $(BIDS_ROOT)"; exit 1; }

check-venv:
	@test -f "$(MNE_BIDS)" || { echo "ERROR: venv not initialized, run: make install"; exit 1; }

preprocessing: check-venv check-bids
	$(MNE_BIDS) --task rest1 --root-dir $(BIDS_ROOT) --deriv_root $(BIDS_ROOT)/derivatives/ --config=preprocessing_pipeline_conf.py --steps preprocessing
	$(MNE_BIDS) --task rest2 --root-dir $(BIDS_ROOT) --deriv_root $(BIDS_ROOT)/derivatives/ --config=preprocessing_pipeline_conf.py --steps preprocessing
	$(MNE_BIDS) --task main  --root-dir $(BIDS_ROOT) --deriv_root $(BIDS_ROOT)/derivatives/ --config=preprocessing_pipeline_conf.py --steps preprocessing

preprocessing_slurm: check-bids
	sbatch run_preprocessing.sbatch
