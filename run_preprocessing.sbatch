#!/bin/bash
#SBATCH --job-name=meg_maxfilter
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-30

source ~/Nextcloud/ZI/misc/setup_slurm.sh

SUBJECT=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
BIDS_ROOT="/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/"

# Removed invisible character and added standard spaces
mne_bids_pipeline --task rest1 --root-dir "$BIDS_ROOT" --deriv_root "$BIDS_ROOT/derivatives/" --config=preprocessing_pipeline_conf.py --steps init,preprocessing --subject "$SUBJECT"
mne_bids_pipeline --task rest2 --root-dir "$BIDS_ROOT" --deriv_root "$BIDS_ROOT/derivatives/" --config=preprocessing_pipeline_conf.py --steps init,preprocessing --subject "$SUBJECT"
mne_bids_pipeline --task main --root-dir "$BIDS_ROOT" --deriv_root "$BIDS_ROOT/derivatives/" --config=preprocessing_pipeline_conf.py --steps init,preprocessing --subject "$SUBJECT"
