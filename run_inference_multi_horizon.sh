#!/bin/bash

# =============================================================================
# MULTI-HORIZON EPIDEMIC FORECASTING INFERENCE SCRIPT
# =============================================================================
# This script runs inference for all trained models (1, 2, 3, 4 weeks)
# and generates comprehensive results
# =============================================================================

echo "=========================================="
echo "MULTI-HORIZON EPIDEMIC FORECASTING"
echo "Running inference for 1, 2, 3, 4 week forecasts"
echo "=========================================="
echo ""

# Dataset configuration
DATA_PATH="epidemics_30years_full.csv"
ROOT_PATH="./"
DATA_NAME="synthetic_Epi_WeinerProcess"

# Model architecture (must match training)
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

# Input sequence configuration (must match training)
SEQ_LEN=8
LABEL_LEN=4

# Which variable to forecast (must match training)
TARGET="I_child"
FEATURES="S"

# Input/output dimensions
ENC_IN=1
DEC_IN=1
C_OUT=1

echo "Configuration:"
echo "  Dataset: $DATA_PATH"
echo "  Target Variable: $TARGET"
echo "  Feature Mode: $FEATURES"
echo ""

# =============================================================================
# Run inference for each prediction horizon
# =============================================================================

HORIZONS=(1 2 3 4)

for PRED_LEN in "${HORIZONS[@]}"; do
    echo "=========================================="
    echo "Running ${PRED_LEN}-week forecast inference"
    echo "=========================================="

    MODEL_ID="epidemic_${TARGET}_${PRED_LEN}week"

    # Find the checkpoint directory
    CHECKPOINT_PATTERN="./checkpoints/long_term_forecast_${MODEL_ID}_TimeLLM_${DATA_NAME}_ft${FEATURES}_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_dm${D_MODEL}_nh${N_HEADS}_el${E_LAYERS}_dl${D_LAYERS}_df${D_FF}_fc3_ebtimeF_test_0-Forecast*"

    CHECKPOINT_DIR=$(ls -d $CHECKPOINT_PATTERN 2>/dev/null | head -1)

    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "⚠ Warning: Checkpoint not found for ${PRED_LEN}-week model"
        echo "Expected pattern: $CHECKPOINT_PATTERN"
        echo "Skipping..."
        echo ""
        continue
    fi

    CHECKPOINT_PATH="$CHECKPOINT_DIR/checkpoint"

    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "⚠ Warning: Checkpoint file not found at $CHECKPOINT_PATH"
        echo "Skipping..."
        echo ""
        continue
    fi

    echo "Using checkpoint: $CHECKPOINT_PATH"

    OUTPUT_PATH="./results/epidemic_${TARGET}_${PRED_LEN}week"

    python run_inference_short_term.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --task_name long_term_forecast \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --model TimeLLM \
        --data "$DATA_NAME" \
        --features "$FEATURES" \
        --target "$TARGET" \
        --seq_len "$SEQ_LEN" \
        --label_len "$LABEL_LEN" \
        --pred_len "$PRED_LEN" \
        --enc_in "$ENC_IN" \
        --dec_in "$DEC_IN" \
        --c_out "$C_OUT" \
        --d_model "$D_MODEL" \
        --d_ff "$D_FF" \
        --llm_model "$LLM_MODEL" \
        --llm_dim "$LLM_DIM" \
        --llm_layers "$LLM_LAYERS" \
        --patch_len "$PATCH_LEN" \
        --stride "$STRIDE" \
        --n_heads "$N_HEADS" \
        --e_layers "$E_LAYERS" \
        --d_layers "$D_LAYERS" \
        --prompt_domain 1 \
        --embed timeF \
        --freq w \
        --output_path "$OUTPUT_PATH" \
        --batch_size 1 \
        --num_workers 0

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $PRED_LEN-week inference complete!"
        echo "Results saved to: $OUTPUT_PATH"
        echo ""
    else
        echo ""
        echo "✗ $PRED_LEN-week inference failed!"
        echo ""
    fi
done

echo "=========================================="
echo "✓ INFERENCE COMPLETE FOR ALL HORIZONS"
echo "=========================================="
echo ""
echo "Results directories:"
echo "  ./results/epidemic_${TARGET}_1week/"
echo "  ./results/epidemic_${TARGET}_2week/"
echo "  ./results/epidemic_${TARGET}_3week/"
echo "  ./results/epidemic_${TARGET}_4week/"
echo ""
echo "Each directory contains:"
echo "  - preds_original.npy   (predictions in original scale)"
echo "  - trues_original.npy   (ground truth in original scale)"
echo "  - preds_scaled.npy     (predictions in scaled space)"
echo "  - trues_scaled.npy     (ground truth in scaled space)"
echo "  - metrics.txt          (detailed performance metrics)"
echo ""
