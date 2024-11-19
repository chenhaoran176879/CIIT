#!/bin/bash

# 模型列表
MODEL_LIST=(
    "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-1B"
    "/home/share/chenhaoran/model_zoo/qwen-vl2-2B"
    "/home/share/chenhaoran/model_zoo/Qwen2-VL-7B-Instruct"
    "/home/share/chenhaoran/model_zoo/lmms-lab--llava-onevision-qwen2-7b-ov"
    "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-8B"
    "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-26B"
    "/home/share/chenhaoran/model_zoo/OpenGVLab--InternVL2-40B"
    "/home/share/chenhaoran/model_zoo/qwen-vl2-72B"
)

# 从命令行传入的帧数
NFRAMES=8

if [ -z "$NFRAMES" ]; then
  echo "Usage: ./batch_eval.sh <nframes>"
  exit 1
fi

# 循环遍历模型列表，逐个提交作业
for MODEL_PATH in "${MODEL_LIST[@]}"; do
  sbatch /mnt/lustre/chenhaoran/CIIT/LLaVA-NeXT-inference/playground/ucfcrime/eval_UCVL_batch.sh $NFRAMES "$MODEL_PATH"
  sleep 1  # 为了避免作业提交过快，适当等待
done
