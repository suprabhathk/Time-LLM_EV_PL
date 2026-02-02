#!/bin/bash

# Text Generation Script for TimeLLM_Text with 30-year epidemic dataset
# Uses epidemics_30years_full.csv (weekly SEIR data 1990-2020)

# ============================
# INFERENCE ONLY (Text Generation)
# ============================
python run_main_text.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./ \
  --data_path epidemics_30years_full.csv \
  --model_id epidemic_30yr_text_gen \
  --model_comment 'text_generation_epidemic' \
  --model TimeLLM_Text \
  --data Epi_SEIR \
  --features S \
  --target 'I_total' \
  --seq_len 52 \
  --label_len 26 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --patch_len 7 \
  --stride 4 \
  --n_heads 8 \
  --output_mode text \
  --max_new_tokens 100 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k_sampling 50
