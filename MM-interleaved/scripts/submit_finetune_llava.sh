#!/bin/bash
#SBATCH --job-name=chr-fine  # create a short name for your job
#SBATCH --nodes=1             # node count
#SBATCH --ntasks-per-node=8     # total number of tasks per node
#SBATCH --cpus-per-task=8      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8         # number of gpus per node
#SBATCH -p ai_training_H800
###SBTACH --nodelist=dx-ai-node69
###SBATCH --mem=1536G            # total memory per node (4 GB per cpu-core is default)1536G 
###SBATCH -o slurm_log/%x_%j_output.txt
###SBATCH -e slurm_log/%x_%j_error.txt

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE

srun bash run_finetune.sh
