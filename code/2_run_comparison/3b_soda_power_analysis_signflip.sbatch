#!/bin/bash
#SBATCH --job-name=soda_signflip_power
#SBATCH --output=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.out
#SBATCH --error=/zi/home/simon.kern/slurmlogs/slurm_%x_%A-%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=2-80

source ~/Nextcloud/ZI/misc/setup_slurm.sh

N_SAMPLES=$SLURM_ARRAY_TASK_ID

python 3b_soda_power_analysis_signflip.py --n_samples $N_SAMPLES
