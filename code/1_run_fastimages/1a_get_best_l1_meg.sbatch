#!/bin/bash
#SBATCH --job-name=highspeed-get-best-l1
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-60

source ~/Nextcloud/ZI/misc/setup_slurm.sh

# Calculate subject (1-30) and condition from array task ID
# Tasks 1-30: slow, Tasks 31-60: fast
SUBJECT_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 30 + 1 ))
SUBJECT=$(printf "%02d" "${SUBJECT_IDX}")
BIDS_ROOT="/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-bids/"

# Condition: 'slow' for localizer images or 'fast' for sequence images
if [ "${SLURM_ARRAY_TASK_ID}" -le 30 ]; then
    CONDITION="slow"
else
    CONDITION="fast"
fi

# Run L1 grid search
python 1a_get_best_l1_meg.py --subjects "$SUBJECT" --condition "$CONDITION"
