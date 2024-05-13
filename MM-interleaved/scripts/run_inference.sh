#!/bin/bash
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
local_rank=${SLURM_LOCALID}
node_rank=${SLURM_NODEID}
node_num=${SLURM_NNODES}
gpu_per_node=${SLURM_NTASKS_PER_NODE}

export LD_LIBRARY_PATH="/mnt/lustre/chenhaoran/anaconda3/envs/torch201cu118/lib/":${LD_LIBRARY_PATH}
cd /mnt/lustre/chenhaoran/CIIT/MM-interleaved
if [[ $local_rank == 0 ]]; then
    echo master_addr: $master_addr
    echo master_port: $master_port
    echo node_num: $node_num
    echo gpu_per_node: $gpu_per_node
    echo node_rank: $node_rank
    echo local_rank: $local_rank

    
    /mnt/lustre/chenhaoran/anaconda3/envs/torch201cu118/bin/torchrun \
    --nnodes=${node_num}\
    --node_rank=${node_rank}\
    --master_addr=${master_addr}\
    --nproc_per_node=${gpu_per_node}\
    --master_port=${master_port}\
    ./inference.py\
    --config_file=./mm_interleaved/configs/release/mm_inference.yaml
fi
