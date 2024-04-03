#!/bin/bash

model_name='llava-vae-noclip-768-p8-h2048-20240320'
date='20240321'



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
    task_name=llava-${model_name}-finetune-${date}-${node_num}node-3epoch
    log_file=/lustre/chenhaoran/llava-chr/scripts/log/${task_name}.log

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
    --deepspeed /home/chenhaoran/llava-chr/scripts/zero3.json \
     > ${log_file} 2>&1 
fi
