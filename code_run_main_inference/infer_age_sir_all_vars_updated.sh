echo "=========================================="
echo "Age-SIR Forecasting - INFERENCE"
echo "=========================================="

# Find the checkpoint directory
CHECKPOINT_BASE="./checkpoints/short_term_forecast_age_sir_all_vars_7day_TimeLLM_Age_SIR_ftMS_sl21_ll7_pl7_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-all_vars_predict_IA_IC"

if [ ! -d "$CHECKPOINT_BASE" ]; then
    echo "⚠ Warning: Checkpoint not found at expected location"
    echo "Looking for checkpoints..."
    ls -d ./checkpoints/short_term_forecast_age_sir_all_vars_7day_* 2>/dev/null || echo "No checkpoints found!"
    echo ""
    echo "Please update CHECKPOINT_BASE in this script with the correct path"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_BASE"
echo ""

python run_inference_improved.py \
--checkpoint_path ./checkpoints/short_term_forecast_age_sir_all_vars_7day_TimeLLM_Age_SIR_ftMS_sl21_ll7_pl7_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-all_vars_predict_IA_IC/checkpoint \
--task_name short_term_forecast \
--root_path ./dataset/synthetic/age_sir/ \
--data_path age_sir_all_vars.csv \
--model TimeLLM \
--data Age_SIR \
--features MS \
--target 'IA' \
--seq_len 21 \
--label_len 7 \
--pred_len 7 \
--enc_in 6 \
--dec_in 6 \
--c_out 1 \
--d_model 16 \
--d_ff 32 \
--llm_model GEMMA \
--llm_dim 640 \
--llm_layers 1 \
--patch_len 7 \
--stride 3 \
--prompt_domain 1 \
--inference_mode around_peak \
--peak_window 7 \
--output_path ./results/age_sir_around_peak/

echo ""
echo "=========================================="
echo "✓ INFERENCE COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: ./results/age_sir_all_vars_forecast/"
echo ""
echo "Files generated:"
echo "  - predictions.npy     (shape: n_samples × 7 × 2)"
echo "  - true_values.npy     (shape: n_samples × 7 × 2)"
echo "  - metrics.txt         (MAE, MSE, RMSE, MAPE)"
echo "  - per_horizon_metrics.csv"
echo ""
echo "✓ Next step: Run visualization script"
