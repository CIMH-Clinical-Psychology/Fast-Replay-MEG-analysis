#!/bin/bash
#SBATCH --job-name=highspeed-get-best-l1
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
python 1a_get_best_l1_meg.py --subject "$SUBJECT"
