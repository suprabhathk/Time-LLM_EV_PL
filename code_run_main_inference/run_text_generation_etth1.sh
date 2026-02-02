#!/bin/bash

# Text Generation Script for TimeLLM_Text with ETTh1 dataset
# This script runs the model in text generation mode on electricity transformer data

# ============================
# DOWNLOAD ETTh1 DATASET (if not exists)
# ============================
DATA_DIR="./dataset/ETT-small"
if [ ! -f "$DATA_DIR/ETTh1.csv" ]; then
    echo "Downloading ETTh1 dataset..."
    mkdir -p $DATA_DIR
    wget -O $DATA_DIR/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
    echo "Download complete!"
fi

# ============================
# INFERENCE ONLY (Text Generation with ETTh1)
# ============================
python run_main_text.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id etth1_text_gen \
  --model_comment 'text_generation_etth1' \
  --model TimeLLM_Text \
  --data ETTh1 \
  --features M \
  --target 'OT' \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --llm_layers 6 \
  --train_epochs 1 \
  --percent 100 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --num_workers 0 \
  --prompt_domain 1 \
  --patch_len 16 \
  --stride 8 \
  --n_heads 8 \
  --output_mode text \
  --max_new_tokens 100 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k_sampling 50
