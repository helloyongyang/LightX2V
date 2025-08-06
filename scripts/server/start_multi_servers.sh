#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start multiple servers
python -m lightx2v.api_multi_servers \
    --num_gpus $num_gpus \
    --start_port 8000 \
    --model_cls wan2.1_distill \
    --task i2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/distill/wan_i2v_distill_4step_cfg.json
