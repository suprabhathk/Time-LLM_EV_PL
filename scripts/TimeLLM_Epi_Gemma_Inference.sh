#!/bin/bash

# 1. Configuration - Updated to match Long-Term ID
model_name=TimeLLM
data_path="epidemics_30years_full.csv"
model_id="Epi_LongTerm_52_4"

# 2. Architecture - MUST match your v2 training (d_model=32, d_ff=128)
llm_model=GEMMA
llm_layers=6
llm_dim=640
d_model=32
d_ff=128

# 3. Task Specifics - Updated to 156-week history
seq_len=52
label_len=13
pred_len=4

# 4. Checkpoint Path - Update this to the folder created by your 156-week run
# Usually, it's the folder with "sl156" and "dm32" in the name
checkpoint_path="./checkpoints/long_term_forecast_Epi_LongTerm_52_4_TimeLLM_synthetic_Epi_WeinerProcess_ftS_sl52_ll13_pl4_dm32_nh8_el2_dl1_df128_fc1_ebtimeF_test_0-Gemma270M-52to4-S-Features-v2/checkpoint"

# Execute Inference
python run_inference_short_term.py \
  --task_name long_term_forecast \
  --model $model_name \
  --checkpoint_path "$checkpoint_path" \
  --data synthetic_Epi_WeinerProcess \
  --root_path ./ \
  --data_path $data_path \
  --features S \
  --target I_child \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --llm_layers $llm_layers \
  --patch_len 13 \
  --stride 4 \
  --prompt_domain 1 \
  --batch_size 1 \
  --output_path ./results/champion_52wk_analysis