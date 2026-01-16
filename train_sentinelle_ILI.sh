#!/bin/bash

# =============================================================================
# FRENCH ILI FORECASTING TRAINING SCRIPT
# =============================================================================
# Trains TimeLLM models for French ILI data (Réseau Sentinelles 1984-2025)
# Uses log-transform scaling for high variance incidence data
# =============================================================================

echo "=========================================="
echo "FRENCH ILI FORECASTING - TRAINING"
echo "Dataset: sentinelle_ILI_France_1984_2025.csv"
echo "=========================================="
echo ""

# Dataset configuration
DATA_PATH="epidemics_30years_full.csv"
ROOT_PATH="./"
DATA_NAME="synthetic_Epi_WeinerProcess"
TARGET="I_child"

# Model architecture
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

# Training hyperparameters
BATCH_SIZE=8
LEARNING_RATE=0.0005
TRAIN_EPOCHS=50
PATIENCE=10

# Input sequence configuration
SEQ_LEN=26      # Look back 26 weeks (~6 months)
LABEL_LEN=13    # Decoder start tokens

# Feature mode
FEATURES="MS"    # Single variable forecasting
ENC_IN=10
DEC_IN=10
C_OUT=1

echo "Configuration:"
echo "  Dataset: $DATA_PATH"
echo "  Target: $TARGET (ILI incidence)"
echo "  Feature Mode: $FEATURES (univariate)"
echo "  Input Sequence: $SEQ_LEN weeks (~6 months)"
echo "  Scaling: LOG TRANSFORM + StandardScaler"
echo "  LLM: $LLM_MODEL (dim=$LLM_DIM, layers=$LLM_LAYERS)"
echo ""

# =============================================================================
# Train models for multiple prediction horizons
# =============================================================================

HORIZONS=(1 2 4 8)  # 1, 2, 4, 8 week forecasts

for PRED_LEN in "${HORIZONS[@]}"; do
    echo "=========================================="
    echo "Training ${PRED_LEN}-week forecast model"
    echo "=========================================="

    MODEL_ID="sentinelle_ILI_${PRED_LEN}week"

    python run_main.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --model_id "$MODEL_ID" \
        --model_comment "French ILI forecast $PRED_LEN weeks ahead" \
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
        echo "[OK] $PRED_LEN-week model training complete!"
        echo ""
    else
        echo ""
        echo "[ERROR] $PRED_LEN-week model training failed!"
        exit 1
    fi
done

echo "=========================================="
echo "[OK] ALL MODELS TRAINED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  ./checkpoints/long_term_forecast_sentinelle_ILI_1week_*/"
echo "  ./checkpoints/long_term_forecast_sentinelle_ILI_2week_*/"
echo "  ./checkpoints/long_term_forecast_sentinelle_ILI_4week_*/"
echo "  ./checkpoints/long_term_forecast_sentinelle_ILI_8week_*/"
echo ""
echo "Next steps:"
echo "  1. Run inference: bash run_inference_sentinelle_ILI.sh"
echo ""
