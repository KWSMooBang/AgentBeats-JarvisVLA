#!/bin/bash

cuda_visible_devices=1

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python examples/rollout_mcu_benchmark.py \
    --category "crafting" \
    --max-steps 600 \
    --model-path "/workspace/models/JarvisVLA-Qwen2-VL-7B" \
    --base-url "http://localhost:9020/v1" \
    --temperature 0.9 \
    --history-num 4 \
    --action-chunk-len 1 \
    --instruction-type "simple" \
    --verbose
