#!/bin/bash
#SBATCH --job-name=generalization-highspeed
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-900

source ~/Nextcloud/ZI/misc/setup_slurm.sh

# 30 subjects x 30 L1 values = 900 jobs
L1_VALUES=(0.1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 12.0 14.0 16.0 18.0 20.0 25.0 30.0 40.0 50.0)
N_L1=${#L1_VALUES[@]}

if [ "$N_L1" -ne 30 ]; then
    echo "ERROR: L1_VALUES must have exactly 30 entries, got $N_L1. Update --array accordingly."
    exit 1
fi

L1_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % N_L1 ))
SUBJECT_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / N_L1 + 1 ))
SUBJECT=$(printf "%02d" "${SUBJECT_IDX}")
L1=${L1_VALUES[$L1_IDX]}

echo "Running subject $SUBJECT with L1=$L1"

# Run slow2slow first, then slow2fast
python "$PYTHON_SCRIPT" --subjects "$SUBJECT" --l1 "$L1" --source slow --target slow
python "$PYTHON_SCRIPT" --subjects "$SUBJECT" --l1 "$L1" --source slow --target fast
