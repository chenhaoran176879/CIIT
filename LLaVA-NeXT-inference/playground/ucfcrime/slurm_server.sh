#!/bin/bash
#SBATCH --job-name=summarize  # create a short name for your job
#SBATCH --nodes=1             # node count
#SBATCH --gres=gpu:1        # number of gpus per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p ai_training_H800
#SBATCH -o /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/%x_%j_output.txt
#SBATCH -e /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/%x_%j_error.txt

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE


python -u /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/LLM_summarization_classification_examination.py
