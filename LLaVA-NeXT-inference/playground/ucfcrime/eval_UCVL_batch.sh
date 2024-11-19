#!/bin/bash
#SBATCH --job-name=eval_UCVL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p ai_training
#SBATCH --exclude=dx-ai-node2
#SBATCH -o /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/%x_%j_output.txt
#SBATCH -e /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/%x_%j_error.txt

echo "slurm_job_node_list: $SLURM_JOB_NODELIST"
echo "slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE"

# 获取传入的帧数和模型路径
NFRAMES=$1
MODEL_PATH=$2

if [ -z "$NFRAMES" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: sbatch run_eval.sh <nframes> <model_path>"
  exit 1
fi

# 提取模型名称
MODEL_NAME=$(basename "$MODEL_PATH")

# 生成 output_jsonl 的名称
OUTPUT_JSONL="/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_${MODEL_NAME}_${NFRAMES}frames_all_$(date +'%m%d').jsonl"

# 激活环境并运行
conda activate llava-interleave
python -u /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_UCVL.py \
       --jsonl_root /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_test_mcq.jsonl \
       --video_folder /mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/ \
       --nframes $NFRAMES \
       --model_path "$MODEL_PATH" \
       --data_split test \
       --questions anomaly_detection_question event_description_question crime_classification_question event_description_with_classification multiple_choice_question \
       --output_jsonl "$OUTPUT_JSONL"

