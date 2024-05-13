#!/bin/bash
#SBATCH --job-name=chrlust  # create a short name for your job
#SBATCH --nodes=1             # node count
#SBATCH --gres=gpu:1        # number of gpus per node
#SBATCH -p ai_training_H800
###SBATCH --ntasks-per-node=1     # total number of tasks per node
###SBATCH --cpus-per-task=1     # cpu-cores per task (>1 if multi-threaded tasks)
###SBTACH --exclusive
###SBATCH --mem=1536G            # total memory per node (4 GB per cpu-core is default)1536G 
###SBATCH -o slurm_log/%x_%j_output.txt
###SBATCH -e slurm_log/%x_%j_error.txt

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE

export CUDA_HOME=/public/software/cuda/cuda-11.8
export PATH=/public/software/cuda/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/software/cuda/cuda-11.8/lib64
srun /lustre/chenhaoran/anaconda3/bin/python /lustre/chenhaoran/llava-chr/llava/model/multimodal_encoder/vqgan.py
