#!/bin/bash
#SBATCH --job-name=ciit_eval  # create a short name for your job
#SBATCH --nodes=1             # node count
#SBATCH --gres=gpu:4        # number of gpus per node
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -p ai_training
#SBATCH -o ./OUTPUT/slurm_log/%x_%j_output.txt
#SBATCH -e ./OUTPUT/slurm_log/%x_%j_error.txt

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE


srun run_mm_evaluate_torchrun.sh
