#!/bin/bash
#SBATCH --job-name=llava-pretrain  # create a short name for your job
#SBATCH --nodes=1             # node count
#SBATCH --ntasks-per-node=4     # total number of tasks per node
#SBATCH --cpus-per-task=8      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4         # number of gpus per node
#SBATCH -p ai_training_H800
###SBTACH --exclusive
###SBATCH -o slurm_log/%x_%j_output.txt
###SBATCH -e slurm_log/%x_%j_error.txt

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE

srun bash run_train_pretrain.sh
