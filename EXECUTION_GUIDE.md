# Epidemic Forecasting Execution Guide

## Overview
This guide explains how to train and run multi-horizon forecasting (1, 2, 3, 4 weeks) using the `epidemics_30years_full.csv` dataset with the Time-LLM framework.

## ✅ Quick Start (3 Steps)

### Step 1: Validate Your Data
```bash
python diagnose_data.py --data_path epidemics_30years_full.csv --target I_child
```

**What this does:**
- Checks data quality (missing values, outliers, etc.)
- Analyzes train/val/test split (70/20/10)
- Validates scaling behavior
- Reports distribution shifts
- Provides warnings if issues detected

**Expected output:** Summary statistics and recommendations

---

### Step 2: Train Models for All Horizons
```bash
bash train_epidemic_multi_horizon.sh
```

**What this does:**
- Trains 4 separate models (1-week, 2-week, 3-week, 4-week forecasts)
- Uses GEMMA LLM backbone with 6 layers
- Input: 8 weeks of historical data
- Default target: `I_child` (infected children)
- Saves checkpoints to `./checkpoints/`

**Time estimate:** ~10-30 minutes per model (depends on GPU)

**Configuration:** Edit `train_epidemic_multi_horizon.sh` to change:
- `TARGET="I_child"` → Choose which variable to forecast
  - Options: S_child, I_child, R_child, S_adult, I_adult, R_adult, S_total, I_total, R_total
- `FEATURES="S"` → Single variable mode (recommended)
  - `"S"`: Forecast only the target variable
  - `"MS"`: Use all variables to predict target (multivariate-to-single)
- `SEQ_LEN=8` → How many weeks of history to use
- `BATCH_SIZE=8` → Reduce if out of memory
- `TRAIN_EPOCHS=20` → Increase for better accuracy

---

### Step 3: Run Inference on Test Set
```bash
bash run_inference_multi_horizon.sh
```

**What this does:**
- Runs inference for all 4 trained models
- Automatically finds checkpoint files
- Applies **inverse scaling** to get predictions in original scale
- Computes metrics (MAE, MSE, RMSE, MAPE) per horizon
- Saves results to `./results/epidemic_<target>_<horizon>week/`

**Output files per horizon:**
- `preds_original.npy` - Predictions (actual counts, not z-scores)
- `trues_original.npy` - Ground truth (actual counts)
- `preds_scaled.npy` - Predictions (z-score normalized)
- `trues_scaled.npy` - Ground truth (z-score normalized)
- `metrics.txt` - Performance summary

---

## 📊 Understanding the Output

### Metrics File Example (`metrics.txt`)
```
==================================================
INFERENCE METRICS (Original Scale)
==================================================

Target Variable: I_child
Prediction Horizon: 4 weeks
Sequence Length: 8 weeks

Overall Metrics:
  MAE:       1234.56
  MSE:    5678901.23
  RMSE:      2383.67
  MAPE:        12.34%

Per-Horizon Metrics:
Horizon    | MAE          | MSE             | RMSE
------------------------------------------------------------
Week 1     |      987.65  |     1234567.89  |     1111.11
Week 2     |     1234.56  |     5678901.23  |     2383.67
Week 3     |     1456.78  |     8901234.56  |     2983.49
Week 4     |     1678.90  |    12345678.90  |     3513.64
```

### What the metrics mean:
- **MAE** (Mean Absolute Error): Average prediction error in actual counts
  - Example: MAE = 1000 means predictions are off by ~1000 infected individuals on average
- **MSE** (Mean Squared Error): Squared errors (penalizes large errors more)
- **RMSE** (Root Mean Squared Error): Square root of MSE, same units as data
- **MAPE** (Mean Absolute Percentage Error): Percentage error
  - Example: MAPE = 10% means predictions are off by 10% on average

### Interpreting results:
- **Good performance:** MAE < 10% of mean value, MAPE < 15%
- **Horizon degradation:** Later weeks (3, 4) typically have higher error
- **Scale matters:** Check if predictions are in original scale (thousands) not z-scores (-3 to 3)

---

## 🔧 Advanced Usage

### Train Single Horizon Model
If you only want to forecast 2 weeks ahead:

```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./ \
  --data_path epidemics_30years_full.csv \
  --model_id epidemic_I_child_2week \
  --model TimeLLM \
  --data synthetic_Epi_WeinerProcess \
  --features S \
  --target I_child \
  --seq_len 8 \
  --label_len 4 \
  --pred_len 2 \
  --llm_model GEMMA \
  --llm_dim 640 \
  --llm_layers 6 \
  --batch_size 8 \
  --train_epochs 20 \
  --prompt_domain 1
```

### Run Single Horizon Inference
```bash
# Find your checkpoint first
ls ./checkpoints/

# Run inference with specific checkpoint
python run_inference_short_term.py \
  --checkpoint_path ./checkpoints/<your_checkpoint_dir>/checkpoint \
  --data_path epidemics_30years_full.csv \
  --target I_child \
  --seq_len 8 \
  --label_len 4 \
  --pred_len 2 \
  --llm_model GEMMA \
  --llm_dim 640 \
  --llm_layers 6 \
  --output_path ./results/my_inference
```

### Forecast Different Variables
To forecast adult infections instead of child infections:

1. Edit `train_epidemic_multi_horizon.sh`:
   ```bash
   TARGET="I_adult"  # Change this line
   ```

2. Edit `run_inference_multi_horizon.sh`:
   ```bash
   TARGET="I_adult"  # Change this line
   ```

3. Run training and inference as usual

### Use Multivariate Mode (MS)
To use all variables to predict a single target:

1. Edit `train_epidemic_multi_horizon.sh`:
   ```bash
   FEATURES="MS"  # Instead of "S"
   ENC_IN=10      # Number of input variables (all columns except date)
   DEC_IN=10
   C_OUT=1        # Still predicting 1 variable
   ```

2. Edit `run_inference_multi_horizon.sh` with same changes

3. Retrain models

**Note:** MS mode may improve accuracy but increases complexity and training time.

---

## ⚠️ Troubleshooting

### Issue: "Checkpoint not found"
**Solution:**
- Check `./checkpoints/` directory exists
- Ensure training completed successfully
- Verify MODEL_ID matches between training and inference scripts

### Issue: "Out of memory (OOM)"
**Solutions:**
- Reduce `BATCH_SIZE` (try 4, 2, or 1)
- Reduce `LLM_LAYERS` (try 3 or 1)
- Reduce `SEQ_LEN` (try 6 or 4 weeks)

### Issue: "Predictions are all near zero or very large"
**Solution:**
- This indicates scaling issues
- Run `diagnose_data.py` to check data quality
- Verify inverse scaling is applied (check metrics are in original scale, not -3 to 3)

### Issue: "MAE is tiny (< 1.0) but predictions look wrong"
**Cause:** Metrics computed in scaled space instead of original scale
**Solution:** Use the updated `run_inference_short_term.py` which applies inverse scaling

### Issue: "MAPE is NaN or infinite"
**Cause:** True values contain zeros (common in epidemic data)
**Solution:** MAPE is skipped for zero values automatically. Check MAE/RMSE instead.

---

## 📂 File Structure
```
Time-LLM_EV_PL/
├── epidemics_30years_full.csv          # Your data
├── diagnose_data.py                    # Data validation tool
├── train_epidemic_multi_horizon.sh     # Training script
├── run_inference_multi_horizon.sh      # Inference script
├── run_inference_short_term.py         # Core inference code (fixed scaling)
├── checkpoints/                        # Trained models
│   ├── long_term_forecast_epidemic_I_child_1week_*/
│   ├── long_term_forecast_epidemic_I_child_2week_*/
│   ├── long_term_forecast_epidemic_I_child_3week_*/
│   └── long_term_forecast_epidemic_I_child_4week_*/
└── results/                            # Inference outputs
    ├── epidemic_I_child_1week/
    │   ├── preds_original.npy
    │   ├── trues_original.npy
    │   └── metrics.txt
    ├── epidemic_I_child_2week/
    ├── epidemic_I_child_3week/
    └── epidemic_I_child_4week/
```

---

## 🔍 Data Scaling Explained

### The Problem (Now Fixed!)
Previously, metrics were computed in **z-score space** (scaled data):
- Z-scores typically range from -3 to +3
- MAE of 0.5 in z-score space is meaningless for epidemic forecasting
- You want errors in actual case counts (e.g., 1000 people)

### The Solution
The updated `run_inference_short_term.py` now:
1. Runs model inference (outputs are in scaled space)
2. **Applies inverse scaling** using the same scaler fitted during training
3. Computes metrics in **original scale** (actual counts)
4. Clips predictions to non-negative (epidemic constraint: can't have negative infections)

### How it works:
```python
# Training (data_loader.py)
scaler.fit(train_data)           # Fit on train data only (70%)
scaled_data = scaler.transform(all_data)  # Transform all data

# Inference (run_inference_short_term.py)
predictions = model(input)       # Model outputs scaled predictions
predictions_original = scaler.inverse_transform(predictions)  # Convert back!
metrics = compute(predictions_original, true_values_original)
```

---

## 📈 Expected Performance Benchmarks

For `I_child` (infected children) forecasting:

| Horizon | Expected MAE | Expected RMSE |
|---------|--------------|---------------|
| 1 week  | 5,000-15,000 | 10,000-25,000 |
| 2 weeks | 8,000-20,000 | 15,000-35,000 |
| 3 weeks | 12,000-25,000| 20,000-45,000 |
| 4 weeks | 15,000-30,000| 25,000-55,000 |

*Note: These are rough estimates based on epidemic data variability. Your results may vary.*

**Factors affecting performance:**
- Data quality and variability
- Epidemic phase (exponential growth vs. endemic)
- Model hyperparameters
- Amount of training data

---

## 🎯 Best Practices

1. **Always run diagnostics first:**
   ```bash
   python diagnose_data.py --target I_child
   ```

2. **Start with single variable mode (`FEATURES="S"`)**
   - Simpler, faster, easier to debug
   - Upgrade to MS mode if needed

3. **Use shorter horizons for critical decisions**
   - 1-2 week forecasts are more reliable than 4-week

4. **Check both MAE and RMSE**
   - MAE: Average error
   - RMSE: Penalizes large errors more (useful for detecting outliers)

5. **Validate on multiple targets**
   - Train separate models for I_child, I_adult, S_total, etc.
   - Compare performance across variables

6. **Monitor training logs**
   - Validation loss should decrease
   - If validation loss increases, reduce learning rate or add regularization

---

## 📚 Next Steps

After successful inference:
1. **Visualize predictions:** Create plots comparing predictions vs. ground truth
2. **Error analysis:** Identify when/where model fails (epidemic peaks? endemic periods?)
3. **Hyperparameter tuning:** Experiment with SEQ_LEN, LLM_LAYERS, PATCH_LEN
4. **Ensemble models:** Combine predictions from multiple horizons
5. **Real-time forecasting:** Apply to new epidemic data as it arrives

---

## 📞 Support

If you encounter issues:
1. Check troubleshooting section above
2. Run `diagnose_data.py` to validate data
3. Review logs in `./logs/` (if created during training)
4. Verify all scripts have matching parameters (TARGET, FEATURES, SEQ_LEN, etc.)

---

**Happy Forecasting!** 🎉
