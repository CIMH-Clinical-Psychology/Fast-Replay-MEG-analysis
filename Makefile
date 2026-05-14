VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MNE_BIDS := $(VENV)/bin/mne_bids_pipeline
BIDS_ROOT := /zi/home/simon.kern/highspeed-MEG-bids

.PHONY: all init install preprocessing preprocessing_slurm preprocessing-cleanup check-bids

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

preprocessing-cleanup: check-bids
	@echo "=== Files that will be removed ==="
	@echo ""
	@echo "-- Intermediate .fif files (keeping *_proc-clean_raw.fif) --"
	@find $(BIDS_ROOT)/derivatives/ -name "*.fif" ! -name "*_proc-clean_raw.fif" -type f
	@echo ""
	@echo "-- Virtual environment: $(VENV) --"
	@if [ -d "$(VENV)" ]; then echo "  $(VENV)/ ($$(du -sh $(VENV) | cut -f1))"; else echo "  (not found)"; fi
	@echo ""
	@echo "-- Pipeline cache: $(BIDS_ROOT)/derivatives/_cache --"
	@if [ -d "$(BIDS_ROOT)/derivatives/_cache" ]; then echo "  $(BIDS_ROOT)/derivatives/_cache/ ($$(du -sh $(BIDS_ROOT)/derivatives/_cache | cut -f1))"; else echo "  (not found)"; fi
	@echo ""
	@read -p "Proceed with deletion? [y/N] " confirm && [ "$$confirm" = "y" ] || { echo "Aborted."; exit 1; }
	find $(BIDS_ROOT)/derivatives/ -name "*.fif" ! -name "*_proc-clean_raw.fif" -type f -delete
	@if [ -d "$(VENV)" ]; then rm -rf $(VENV); echo "Removed $(VENV)"; fi
	@if [ -d "$(BIDS_ROOT)/derivatives/_cache" ]; then rm -rf $(BIDS_ROOT)/derivatives/_cache; echo "Removed $(BIDS_ROOT)/derivatives/_cache"; fi
	@echo "Done."
