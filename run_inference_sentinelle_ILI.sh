#!/bin/bash

# =============================================================================
# FRENCH ILI FORECASTING INFERENCE SCRIPT
# =============================================================================
# Runs inference for all trained French ILI models
# Applies inverse log-transform to get predictions in original scale
# =============================================================================

echo "=========================================="
echo "FRENCH ILI FORECASTING - INFERENCE"
echo "Running inference for 1, 2, 4, 8 week forecasts"
echo "=========================================="
echo ""

# Dataset configuration
DATA_PATH="sentinelle_ILI_France_1984_2025.csv"
ROOT_PATH="./"
DATA_NAME="sentinelle_ILI_France"
TARGET="inc"

# Model architecture (must match training)
LLM_MODEL="GEMMA"
LLM_DIM=640
LLM_LAYERS=12
D_MODEL=32
D_FF=128
PATCH_LEN=2
STRIDE=2
N_HEADS=8
E_LAYERS=3
D_LAYERS=1

# Input sequence configuration (must match training)
SEQ_LEN=26
LABEL_LEN=13

# Feature mode
FEATURES="S"
ENC_IN=1
DEC_IN=1
C_OUT=1

echo "Configuration:"
echo "  Dataset: $DATA_PATH"
echo "  Target: $TARGET"
echo "  Scaling: LOG TRANSFORM (inverse applied)"
echo ""

# =============================================================================
# Run inference for each prediction horizon
# =============================================================================

HORIZONS=(1 2 4 8)

for PRED_LEN in "${HORIZONS[@]}"; do
    echo "=========================================="
    echo "Running ${PRED_LEN}-week forecast inference"
    echo "=========================================="

    MODEL_ID="sentinelle_ILI_${PRED_LEN}week"

    # Find the checkpoint directory
    CHECKPOINT_PATTERN="./checkpoints/long_term_forecast_${MODEL_ID}_*"

    CHECKPOINT_DIR=$(ls -d $CHECKPOINT_PATTERN 2>/dev/null | head -1)

    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "[WARNING] Checkpoint not found for ${PRED_LEN}-week model"
        echo "Expected pattern: $CHECKPOINT_PATTERN"
        echo "Skipping..."
        echo ""
        continue
    fi

    CHECKPOINT_PATH="$CHECKPOINT_DIR/checkpoint"

    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "[WARNING] Checkpoint file not found at $CHECKPOINT_PATH"
        echo "Skipping..."
        echo ""
        continue
    fi

    echo "Using checkpoint: $CHECKPOINT_PATH"

    OUTPUT_PATH="./results/sentinelle_ILI_${PRED_LEN}week"

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
        echo "[OK] $PRED_LEN-week inference complete!"
        echo "Results saved to: $OUTPUT_PATH"
        echo ""
    else
        echo ""
        echo "[ERROR] $PRED_LEN-week inference failed!"
        echo ""
    fi
done

echo "=========================================="
echo "[OK] INFERENCE COMPLETE FOR ALL HORIZONS"
echo "=========================================="
echo ""
echo "Results directories:"
echo "  ./results/sentinelle_ILI_1week/"
echo "  ./results/sentinelle_ILI_2week/"
echo "  ./results/sentinelle_ILI_4week/"
echo "  ./results/sentinelle_ILI_8week/"
echo ""
echo "Each directory contains:"
echo "  - preds_original.npy   (predictions in original scale - ILI cases)"
echo "  - trues_original.npy   (ground truth in original scale)"
echo "  - preds_scaled.npy     (predictions in log-scaled space)"
echo "  - trues_scaled.npy     (ground truth in log-scaled space)"
echo "  - metrics.txt          (detailed performance metrics)"
echo ""
