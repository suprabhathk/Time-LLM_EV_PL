#!/bin/bash

# =============================================================================
# Setup Validation Script
# =============================================================================
# This script validates that all necessary files are in place and configured
# correctly before training.
# =============================================================================

echo "======================================================================"
echo "EPIDEMIC FORECASTING SETUP VALIDATION"
echo "======================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Data file exists
echo "[1/8] Checking data file..."
if [ -f "epidemics_30years_full.csv" ]; then
    echo "  ✓ epidemics_30years_full.csv found"
    ROWS=$(wc -l < epidemics_30years_full.csv)
    echo "    - File has $ROWS rows"
    if [ $ROWS -lt 100 ]; then
        echo "    ⚠ WARNING: File seems too small (< 100 rows)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: epidemics_30years_full.csv not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: Training script
echo "[2/8] Checking training script..."
if [ -f "train_epidemic_multi_horizon.sh" ]; then
    echo "  ✓ train_epidemic_multi_horizon.sh found"
    if [ -x "train_epidemic_multi_horizon.sh" ]; then
        echo "    - Executable permissions: OK"
    else
        echo "    ⚠ WARNING: Not executable, run: chmod +x train_epidemic_multi_horizon.sh"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: train_epidemic_multi_horizon.sh not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: Inference script
echo "[3/8] Checking inference script..."
if [ -f "run_inference_multi_horizon.sh" ]; then
    echo "  ✓ run_inference_multi_horizon.sh found"
    if [ -x "run_inference_multi_horizon.sh" ]; then
        echo "    - Executable permissions: OK"
    else
        echo "    ⚠ WARNING: Not executable, run: chmod +x run_inference_multi_horizon.sh"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: run_inference_multi_horizon.sh not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: Inference Python script
echo "[4/8] Checking inference Python script..."
if [ -f "run_inference_short_term.py" ]; then
    echo "  ✓ run_inference_short_term.py found"
    # Check if inverse scaling is implemented
    if grep -q "inverse_transform" run_inference_short_term.py; then
        echo "    - Inverse scaling: IMPLEMENTED ✓"
    else
        echo "    ✗ ERROR: Inverse scaling not found in script!"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ ERROR: run_inference_short_term.py not found!"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 5: Core dependencies
echo "[5/8] Checking core Python files..."
REQUIRED_FILES=("run_main.py" "models/TimeLLM.py" "data_provider/data_factory.py" "data_provider/data_loader.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file found"
    else
        echo "  ✗ ERROR: $file not found!"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check 6: Prompt file
echo "[6/8] Checking prompt file..."
if [ -f "dataset/prompt_bank/synthetic_Epi_WeinerProcess.txt" ]; then
    echo "  ✓ Prompt file found"
else
    echo "  ⚠ WARNING: dataset/prompt_bank/synthetic_Epi_WeinerProcess.txt not found"
    echo "    Model will use generic prompt (may reduce accuracy)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check 7: Diagnostic script
echo "[7/8] Checking diagnostic tools..."
if [ -f "diagnose_data.py" ]; then
    echo "  ✓ diagnose_data.py found"
    if [ -x "diagnose_data.py" ]; then
        echo "    - Executable permissions: OK"
    fi
else
    echo "  ⚠ WARNING: diagnose_data.py not found"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check 8: Python environment
echo "[8/8] Checking Python environment..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  ✓ Python found: $PYTHON_VERSION"
else
    echo "  ✗ ERROR: Python not found!"
    ERRORS=$((ERRORS + 1))
fi

# Check for key Python packages (non-fatal)
echo ""
echo "  Checking Python packages..."
PACKAGES=("torch" "transformers" "accelerate" "numpy" "pandas" "sklearn")
for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "    ✓ $pkg installed"
    else
        echo "    ✗ $pkg NOT installed"
        WARNINGS=$((WARNINGS + 1))
    fi
done
echo ""

# Summary
echo "======================================================================"
echo "VALIDATION SUMMARY"
echo "======================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED!"
    echo ""
    echo "You are ready to:"
    echo "  1. (Optional) Run diagnostics: python diagnose_data.py"
    echo "  2. Train models:               bash train_epidemic_multi_horizon.sh"
    echo "  3. Run inference:              bash run_inference_multi_horizon.sh"
elif [ $ERRORS -eq 0 ]; then
    echo "⚠ VALIDATION PASSED WITH WARNINGS"
    echo ""
    echo "Warnings: $WARNINGS"
    echo ""
    echo "You can proceed, but review warnings above."
    echo "Recommended: Install missing packages with: pip install -r requirements.txt"
else
    echo "✗ VALIDATION FAILED"
    echo ""
    echo "Errors:   $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please fix errors above before proceeding."
    exit 1
fi

echo ""
echo "======================================================================"
