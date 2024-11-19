#!/bin/bash
#SBATCH --job-name=eval_UCVL  # create a short name for your job
#SBATCH --nodes=1             # node count
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p ai_training
#SBATCH --nodelist=dx-ai-node5
#SBATCH -o /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/%x_%j_output.txt
#SBATCH -e /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/slurm_log/%x_%j_error.txt

echo slurm_job_node_list: $SLURM_JOB_NODELIST
echo slurm_ntasks_per_node: $SLURM_NTASKS_PER_NODE

# 手动填写 model_path
NFRAMES=64
MODEL_PATH="/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-40B"
# current video models' collection
# "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-1B",# work on A800 only
#  /home/share/chenhaoran/model_zoo/qwen-vl2-2B
# "/home/share/chenhaoran/model_zoo/Qwen2-VL-7B-Instruct" # work; cuda out of memory on H800
# /home/share/chenhaoran/model_zoo/lmms-lab--llava-onevision-qwen2-7b-ov , # work
# "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-8B",# work
# "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVideo2_Chat_8B_InternLM2_5",# StaticCache / dimension mismatch
# "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-26B",# work
# "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-40B",# work
# /home/share/chenhaoran/model_zoo/qwen-vl2-72B
# "/home/share/chenhaoran/model_zoo/lmms-lab--llava-onevision-qwen2-72b-ov", # cpu offload problem




# 从 model_path 中提取模型名称
MODEL_NAME=$(basename "$MODEL_PATH")

# 生成 output_jsonl 的名称
OUTPUT_JSONL="/mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_results/eval_results_${MODEL_NAME}_${NFRAMES}frames_all_1023.jsonl"


conda activate llava-interleave
python -u /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_UCVL.py \
       --jsonl_root  /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/UCVL_gt_with_test_mcq.jsonl \
       --video_folder /mnt/lustre/chenhaoran/datasets/UCF-Crime-Train/ \
       --nframes $NFRAMES \
       --model_path "$MODEL_PATH" \
       --data_split test \
       --questions anomaly_detection_question event_description_question crime_classification_question event_description_with_classification multiple_choice_question \
       --output_jsonl "$OUTPUT_JSONL"

# eval all questions:
# --questions  anomaly_detection_question event_description_question crime_classification_question event_description_with_classification multiple_choice_question
# eval only subjective questions:
# --questions anomaly_detection_question crime_classification_question multiple_choice_question
