#!/bin/bash

# Text Generation Script for TimeLLM_Text with GEMMA
# This script runs the model in text generation mode using GEMMA LLM

# ============================
# INFERENCE ONLY (Text Generation with GEMMA)
# ============================
python run_main_text.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/synthetic/h1n1/ \
  --data_path ili_seasonal_sir_1980_2024_weekly.csv \
  --model_id ili_sir_text_gen_gemma \
  --model_comment 'text_generation_gemma' \
  --model TimeLLM_Text \
  --data Epi_SEIR \
  --features S \
  --target 'I' \
  --seq_len 28 \
  --label_len 14 \
  --pred_len 14 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --batch_size 4 \
  --learning_rate 0.01 \
  --llm_layers 6 \
  --train_epochs 1 \
  --percent 100 \
  --llm_model GEMMA \
  --llm_dim 640 \
  --num_workers 0 \
  --prompt_domain 1 \
  --patch_len 7 \
  --stride 4 \
  --n_heads 8 \
  --output_mode text \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k_sampling 50
