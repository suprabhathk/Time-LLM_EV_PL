#!/bin/bash

# Configuration
model_name=TimeLLM
data_path="epidemics_30years_full.csv"
model_id="Epi_LongTerm_52_4"

# Architecture (Gemma-270M Specifics)
llm_layers=6
llm_dim=640
d_model=128
d_ff=512

# Task: 52 weeks history -> 4 weeks forecast
seq_len=52
label_len=13
pred_len=4

# Execute Training
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data synthetic_Epi_WeinerProcess \
  --features S \
  --target I_child \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --train_epochs 10 \
  --llm_model GEMMA \
  --llm_dim $llm_dim \
  --llm_layers $llm_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len 13 \
  --stride 4 \
  --learning_rate 0.00001 \
  --prompt_domain 1 \
  --model_comment "Gemma270M-52to4-S-Features-v2"