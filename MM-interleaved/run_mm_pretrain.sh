#!/bin/bash
datetime_str=$(date '+%m%d%H%M')
cd /mnt/lustre/chenhaoran/CIIT/MM-interleaved

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
node_num=$SLURM_JOB_NUM_NODES
gpu_per_node=$SLURM_NTASKS_PER_NODE
node_rank=$SLURM_NODEID
local_rank=$SLURM_LOCALID

export LD_LIBRARY_PATH="/mnt/lustre/chenhaoran/anaconda3/envs/torch201cu118/lib":${LD_LIBRARY_PATH}

if [[ $local_rank == 0 ]]; then
    echo master_addr: $master_addr
    echo master_port: $master_port
    echo node_num: $node_num
    echo gpu_per_node: $gpu_per_node
    echo node_rank: $node_rank
    echo local_rank: $local_rank
    
    train_script=./train.py
    log_file=./OUTPUT/log/mmint-ciit_pretrain_${datetime_str}.log

    /mnt/lustre/chenhaoran/anaconda3/envs/torch201cu118/bin/accelerate launch \
    --debug \
    --main_process_ip $master_addr \
    --main_process_port $master_port \
    --num_processes $((node_num * gpu_per_node)) \
    --num_machines $node_num \
    --machine_rank $node_rank \
    ${train_script} \
    --config_file="./mm_interleaved/configs/release/mm_pretrain_cn.yaml" \
    --output_dir="/mnt/lustre/chenhaoran/CIIT/MM-interleaved/OUTPUT/mm_pretrain_ciit_only"
fi