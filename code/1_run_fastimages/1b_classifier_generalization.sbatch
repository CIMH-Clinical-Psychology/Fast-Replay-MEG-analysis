#!/bin/bash
#SBATCH --job-name=generalization-highspeed
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-300

source ~/Nextcloud/ZI/misc/setup_slurm.sh

# 30 subjects x 10 L1 values = 300 jobs
L1_VALUES=(0.01 0.05 0.25 1.0 3 5 7 10 25 100.0)

if [ ${#L1_VALUES[@]} -ne 10 ]; then
    echo "ERROR: L1_VALUES must have exactly 10 entries, got ${#L1_VALUES[@]}. Update --array accordingly."
    exit 1
fi

L1_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 10 ))
SUBJECT_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 10 + 1 ))
SUBJECT=$(printf "%02d" "${SUBJECT_IDX}")
L1=${L1_VALUES[$L1_IDX]}

# PYTHON_SCRIPT is set in setup_slurm.sh by replacing .sbatch with .py
python "$PYTHON_SCRIPT" --subjects "$SUBJECT" --l1 "$L1"
