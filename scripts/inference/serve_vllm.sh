#! /bin/bash

cuda_visible_devices=0
card_num=1
model_name_or_path="/workspace/models/JarvisVLA-Qwen2-VL-7B" #"/path/to/your/model/directory"

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port 9020 \
    --max-model-len 8192 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --served_model_name "jarvisvla" \
    --limit-mm-per-prompt image=5 \

# CUDA_VISIBLE_DEVICES=0 vllm serve /workspace/models/JarvisVLA-Qwen2-VL-7B --port 9020