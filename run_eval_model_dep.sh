#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mode='prompt-response'
base_save_dir="results_guard"

# mode: prompt-response
# cls_path: meta-llama/LlamaGuard-7b meta-llama/Meta-Llama-Guard-2-8B meta-llama/Llama-Guard-3-8B nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0 nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0 OpenSafetyLab/MD-Judge-v0.1 cais/HarmBench-Llama-2-13b-cls cais/HarmBench-Mistral-7b-val-cls allenai/wildguard
# cal_method: origin ts cc bs

cls_path='meta-llama/LlamaGuard-7b'
dataset='harmbench-adv-model'

save_path="${base_save_dir}/${cls_path}/${dataset}/${mode}/model_dep_origin/result_"
python3 -u eval_model_dep.py \
    --cls_path $cls_path \
    --dataset $dataset \
    --mode $mode \
    --save_path $save_path \
    --cal_method origin


