#!/bin/bash
# 多卡在后台运行
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_sft.py > output_sft.log 2>&1 &