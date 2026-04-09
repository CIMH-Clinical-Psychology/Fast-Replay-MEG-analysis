#!/bin/bash
#SBATCH --job-name=highspeed_preprocessing
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-90

# initialize institute specific variables.
# you likely have to remove this line on your machine
source ~/Nextcloud/ZI/misc/setup_slurm.sh

TASKS=(rest1 rest2 main)
TASK_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 30 ))
SUBJECT_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 30 + 1 ))
TASK=${TASKS[$TASK_IDX]}
SUBJECT=$(printf "%02d" $SUBJECT_IDX)

mne_bids_pipeline --task "$TASK" --root-dir "$BIDS_ROOT" --deriv_root "$BIDS_ROOT/derivatives/" --config=preprocessing_pipeline_conf.py --steps preprocessing --subject "$SUBJECT"
