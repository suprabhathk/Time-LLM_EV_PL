#!/bin/bash

# =============================================================================
# MULTI-HORIZON EPIDEMIC FORECASTING TRAINING SCRIPT
# =============================================================================
# This script trains TimeLLM models for 1, 2, 3, and 4 week forecasts
# using the epidemics_30years_full.csv dataset
# =============================================================================

echo "=========================================="
echo "MULTI-HORIZON EPIDEMIC FORECASTING"
echo "Training models for 1, 2, 3, 4 week forecasts"
echo "=========================================="
echo ""

# Dataset configuration
DATA_PATH="epidemics_30years_full.csv"
ROOT_PATH="./"
DATA_NAME="synthetic_Epi_WeinerProcess"

# Model architecture (optimized for weekly epidemic data)
LLM_MODEL="GEMMA"
LLM_DIM=640
LLM_LAYERS=6
D_MODEL=16
D_FF=64
PATCH_LEN=1
STRIDE=1
N_HEADS=8
E_LAYERS=2
D_LAYERS=1

# Training hyperparameters
BATCH_SIZE=8
LEARNING_RATE=0.001
TRAIN_EPOCHS=20
PATIENCE=5

# Input sequence configuration
SEQ_LEN=8     # Look back 8 weeks
LABEL_LEN=4   # Decoder start tokens: 4 weeks

# Which variable to forecast
# Options: S_child, I_child, R_child, S_adult, I_adult, R_adult, S_total, I_total, R_total
TARGET="I_child"

# Feature mode: 'S' (single variable) or 'MS' (multivariate to single)
FEATURES="S"

# Encoder/decoder input dimensions
# For 'S' mode: enc_in=1, dec_in=1, c_out=1
# For 'MS' mode: enc_in=10 (all vars), dec_in=10, c_out=1
ENC_IN=1
DEC_IN=1
C_OUT=1

echo "Configuration:"
echo "  Dataset: $DATA_PATH"
echo "  Target Variable: $TARGET"
echo "  Feature Mode: $FEATURES"
echo "  Input Sequence: $SEQ_LEN weeks"
echo "  LLM: $LLM_MODEL (dim=$LLM_DIM, layers=$LLM_LAYERS)"
echo ""

# =============================================================================
# Train models for each prediction horizon
# =============================================================================

HORIZONS=(1 2 3 4)

for PRED_LEN in "${HORIZONS[@]}"; do
    echo "=========================================="
    echo "Training ${PRED_LEN}-week forecast model"
    echo "=========================================="

    MODEL_ID="epidemic_${TARGET}_${PRED_LEN}week"

    python run_main.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --model_id "$MODEL_ID" \
        --model_comment "Forecast $TARGET $PRED_LEN weeks ahead" \
        --model TimeLLM \
        --data "$DATA_NAME" \
        --features "$FEATURES" \
        --target "$TARGET" \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
        --pred_len "$PRED_LEN" \
        --e_layers "$E_LAYERS" \
        --d_layers "$D_LAYERS" \
        --factor 3 \
        --enc_in "$ENC_IN" \
        --dec_in "$DEC_IN" \
        --c_out "$C_OUT" \
        --d_model "$D_MODEL" \
        --d_ff "$D_FF" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --llm_layers "$LLM_LAYERS" \
        --train_epochs "$TRAIN_EPOCHS" \
        --llm_model "$LLM_MODEL" \
        --llm_dim "$LLM_DIM" \
        --num_workers 0 \
        --prompt_domain 1 \
        --patch_len "$PATCH_LEN" \
        --stride "$STRIDE" \
        --n_heads "$N_HEADS" \
        --patience "$PATIENCE" \
        --embed timeF \
        --freq w

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $PRED_LEN-week model training complete!"
        echo ""
    else
        echo ""
        echo "✗ $PRED_LEN-week model training failed!"
        exit 1
    fi
done

echo "=========================================="
echo "✓ ALL MODELS TRAINED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  ./checkpoints/long_term_forecast_epidemic_${TARGET}_1week_*/"
echo "  ./checkpoints/long_term_forecast_epidemic_${TARGET}_2week_*/"
echo "  ./checkpoints/long_term_forecast_epidemic_${TARGET}_3week_*/"
echo "  ./checkpoints/long_term_forecast_epidemic_${TARGET}_4week_*/"
echo ""
echo "Next steps:"
echo "  1. Run inference: bash run_inference_multi_horizon.sh"
echo "  2. Or use individual inference: python run_inference_short_term.py --checkpoint_path <path>"
echo ""
