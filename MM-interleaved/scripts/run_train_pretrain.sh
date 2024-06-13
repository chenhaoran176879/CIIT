#!/bin/bash
resolution=768
patch_size=8
model_name=llava-vae-noclip-${resolution}-p${patch_size}-h2048-20240320
cd /lustre/chenhaoran/llava-chr/scripts/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
local_rank=${SLURM_LOCALID}
node_rank=${SLURM_NODEID}
node_num=${SLURM_NNODES}
gpu_per_node=${SLURM_NTASKS_PER_NODE}

export LD_LIBRARY_PATH="/lustre/chenhaoran/anaconda3/lib/":${LD_LIBRARY_PATH}

if [[ $local_rank == 0 ]]; then
    echo master_addr: $master_addr
    echo master_port: $master_port
    echo node_num: $node_num
    echo gpu_per_node: $gpu_per_node
    echo node_rank: $node_rank
    echo local_rank: $local_rank
    
    # ------------------------------- train_script select ------------------------------- #
    #train_script=train_finetue.py
    #train_script=pretrain_stream.py
    train_script=/lustre/chenhaoran/llava-chr/llava/train/train_mem.py

    #log_file=pretrain_stream
    log_file=/lustre/chenhaoran/llava-chr/scripts/log/llava_pretrain_${model_name}.log

    echo model_name_or_path: $model_name_or_path
    #echo fsdp_transformer_layer_cls_to_wrap: $fsdp_transformer_layer_cls_to_wrap
    echo data_path: $data_path
    echo output_dir: $output_dir
    echo micro_batch_size: $micro_batch_size
    echo batch_size: $batch_size

    /lustre/chenhaoran/anaconda3/bin/torchrun \
    --nnodes=$node_num --node_rank=${node_rank}\
    --master_addr=${master_addr}\
    --nproc_per_node=${gpu_per_node}\
    --master_port=${master_port} \
    ${train_script} \
    --deepspeed /lustre/chenhaoran/llava-chr/scripts/zero2.json \
    --model_name_or_path /home/share/chenhaoran/model_zoo/vicuna-7b-v1.3/ \
    --version plain \
    --data_path /home/share/chenhaoran/datasets/LLaVA-Pretrain-558K/blip_laion_cc_sbu_558k.json \
    --image_folder /home/share/chenhaoran/datasets/LLaVA-Pretrain-558K/ \
    --vision_tower /home/share/chenhaoran/model_zoo/clip-vit-large-patch14-336/ \
    --vae_vision_tower /lustre/chenhaoran/model_zoo/stabilityai_sdxl-vae/ \
    --use_vae_llava True \
    --use_clip_llava False \
    --interleaved_image_features False \
    --vae_image_size ${resolution} \
    --vae_patch_size ${patch_size} \
    --vae_projector_hidden_size 2048 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /home/share/chenhaoran/checkpoints/${model_name}-${node_num}/ \
    --num_train_epochs  3 \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --save_steps 10000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none 
fi
