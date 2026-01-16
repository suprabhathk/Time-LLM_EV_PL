#!/bin/bash

# =============================================================================
# FRENCH ILI SETUP VALIDATION SCRIPT
# =============================================================================
# Validates that all necessary files are in place before training
# =============================================================================

echo "======================================================================"
echo "FRENCH ILI FORECASTING SETUP VALIDATION"
echo "======================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Data file exists
echo "[1/6] Checking data file..."
if [ -f "sentinelle_ILI_France_1984_2025.csv" ]; then
    echo "  [OK] sentinelle_ILI_France_1984_2025.csv found"
    ROWS=$(wc -l < sentinelle_ILI_France_1984_2025.csv)
    echo "    - File has $ROWS rows"
    if [ $ROWS -lt 100 ]; then
        echo "    [WARNING] File seems too small (< 100 rows)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  [ERROR] sentinelle_ILI_France_1984_2025.csv not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: Training script
echo "[2/6] Checking training script..."
if [ -f "train_sentinelle_ILI.sh" ]; then
    echo "  [OK] train_sentinelle_ILI.sh found"
    if [ -x "train_sentinelle_ILI.sh" ]; then
        echo "    - Executable permissions: OK"
    else
        echo "    [WARNING] Not executable, run: chmod +x train_sentinelle_ILI.sh"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  [ERROR] train_sentinelle_ILI.sh not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: Inference script
echo "[3/6] Checking inference script..."
if [ -f "run_inference_sentinelle_ILI.sh" ]; then
    echo "  [OK] run_inference_sentinelle_ILI.sh found"
    if [ -x "run_inference_sentinelle_ILI.sh" ]; then
        echo "    - Executable permissions: OK"
    else
        echo "    [WARNING] Not executable, run: chmod +x run_inference_sentinelle_ILI.sh"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  [ERROR] run_inference_sentinelle_ILI.sh not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: Log-transform data loader
echo "[4/6] Checking log-transform data loader..."
if [ -f "data_provider/data_loader_log.py" ]; then
    echo "  [OK] data_loader_log.py found"
    if grep -q "Dataset_Custom_Log" data_provider/data_loader_log.py; then
        echo "    - Dataset_Custom_Log class: FOUND"
    else
        echo "    [ERROR] Dataset_Custom_Log class not found!"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q "inverse_transform" data_provider/data_loader_log.py; then
        echo "    - inverse_transform method: FOUND"
    else
        echo "    [ERROR] inverse_transform method not found!"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  [ERROR] data_provider/data_loader_log.py not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: Data factory registration
echo "[5/6] Checking data_factory.py registration..."
if [ -f "data_provider/data_factory.py" ]; then
    echo "  [OK] data_factory.py found"
    if grep -q "sentinelle_ILI_France" data_provider/data_factory.py; then
        echo "    - sentinelle_ILI_France registered: YES"
    else
        echo "    [ERROR] sentinelle_ILI_France not registered in data_dict!"
        ERRORS=$((ERRORS + 1))
    fi
    if grep -q "Dataset_Custom_Log" data_provider/data_factory.py; then
        echo "    - Dataset_Custom_Log imported: YES"
    else
        echo "    [ERROR] Dataset_Custom_Log not imported!"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  [ERROR] data_provider/data_factory.py not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 6: Prompt file
echo "[6/6] Checking prompt file..."
if [ -f "dataset/prompt_bank/sentinelle_ILI_France.txt" ]; then
    echo "  [OK] Custom prompt file found"
    LINES=$(wc -l < dataset/prompt_bank/sentinelle_ILI_France.txt)
    echo "    - Prompt has $LINES lines"
else
    echo "  [WARNING] dataset/prompt_bank/sentinelle_ILI_France.txt not found"
    echo "    Model will use generic prompt (may reduce accuracy)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Summary
echo "======================================================================"
echo "VALIDATION SUMMARY"
echo "======================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "[OK] ALL CHECKS PASSED!"
    echo ""
    echo "You are ready to:"
    echo "  1. Train models:  bash train_sentinelle_ILI.sh"
    echo "  2. Run inference: bash run_inference_sentinelle_ILI.sh"
elif [ $ERRORS -eq 0 ]; then
    echo "[OK] VALIDATION PASSED WITH WARNINGS"
    echo ""
    echo "Warnings: $WARNINGS"
    echo ""
    echo "You can proceed, but review warnings above."
else
    echo "[ERROR] VALIDATION FAILED"
    echo ""
    echo "Errors:   $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please fix errors above before proceeding."
    exit 1
fi

echo ""
echo "======================================================================"
