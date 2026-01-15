#!/bin/bash

echo "=========================================="
echo "Age-SIR Short-Term Forecasting"
echo "Input: All 6 variables (SA,IA,RA,SC,IC,RC)"
echo "Output: IA and IC predictions (7 days ahead)"
echo "=========================================="

python run_main.py \
--task_name short_term_forecast \
--is_training 1 \
--root_path ./dataset/synthetic/age_sir/ \
--data_path age_sir_all_vars.csv \
--model_id age_sir_all_vars_7day \
--model_comment 'all_vars_predict_IA_IC' \
--model TimeLLM \
--data Age_SIR \
--features MS \
--target 'IA' \
--seq_len 21 \
--label_len 7 \
--pred_len 7 \
--e_layers 2 \
--d_layers 1 \
--factor 3 \
--enc_in 6 \
--dec_in 6 \
--c_out 1 \
--d_model 16 \
--d_ff 32 \
--batch_size 8 \
--learning_rate 0.01 \
--llm_layers 1 \
--train_epochs 10 \
--llm_model GEMMA \
--llm_dim 640 \
--num_workers 0 \
--prompt_domain 1 \
--patch_len 7 \
--stride 3 \
--n_heads 8 \
--patience 5

echo ""
echo "=========================================="
echo "✓ TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Model configuration:"
echo "  - Input: 6 variables (all SIR compartments)"
echo "  - Output: 2 variables (IA, IC only)"
echo "  - Context: 21 days"
echo "  - Forecast: 7 days ahead"
echo ""
echo "Checkpoint saved to:"
echo "  ./checkpoints/short_term_forecast_age_sir_all_vars_7day_*/"
echo ""
echo "✓ Next step: Run inference script"
