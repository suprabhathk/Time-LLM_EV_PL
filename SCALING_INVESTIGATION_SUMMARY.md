# Data Scaling Investigation Summary - 2026-01-16

## Key Findings

### Best Configuration
- **Mode**: MS (multivariate-to-single) with 10 compartments
- **Scaling**: StandardScaler only (NO log transform)
- **Dataset**: epidemics_30years_full.csv
- **Target**: I_child

### Performance Results (Training Metrics)

| Horizon | Best Test Loss | Best MAE (z-scores) | Est. Error (cases) | vs Mean |
|---------|---------------|---------------------|-------------------|---------|
| 1-week  | 27.42 (E7)    | 3.07                | ~153K             | 7.8x    |
| 2-week  | 14.53 (E7)    | 2.32                | ~116K             | 5.9x    |
| 4-week  | (stopped)     | -                   | -                 | -       |
| 8-week  | (stopped)     | -                   | -                 | -       |

**2-week model is 47% better than 1-week!**

## What Worked
1. ✅ MS mode - Using all 10 SIR compartments (S/I/R for child/adult/total)
2. ✅ StandardScaler - Avoids exponential explosion from log transform
3. ✅ Longer horizons - 2-week better than 1-week (captures trajectories not noise)
4. ✅ RevIN normalization - Built into TimeLLM, handles distribution shifts

## What Didn't Work
1. ❌ Log transform - Predictions exploded to 10^28 (expm1 of extreme z-scores)
2. ❌ S mode - Test loss stuck at 43-60, couldn't learn epidemic dynamics
3. ❌ SEQ_LEN=52 - Made performance worse (only 40 training examples)
4. ❌ French ILI data (S mode) - MAE 215K, unusable

## Critical Fixes Applied
1. run_inference_short_term.py: data_set.scaler.inverse_transform → data_set.inverse_transform
2. Unicode errors: Replaced ✓ with [OK]
3. Checkpoint patterns: Fixed wildcard matching in inference scripts

## Architecture Insights
- TimeLLM designed for: smooth, high-frequency data (electricity/weather)
- Epidemic data challenges: exponential spikes, 50% zeros, extreme variance
- MS mode helps by providing cross-compartment dynamics (S→I→R relationships)

## Next Steps
1. Complete 4-week and 8-week training
2. Run inference to get ACTUAL MAE in original scale
3. Compare trend-following vs absolute error
4. Consider: Prophet, ARIMA, or mechanistic SEIR if MAE > 20K

## Status: IN PROGRESS
- 1-week: ✅ Complete
- 2-week: ✅ Complete  
- 4-week: ⏸️ Stopped at epoch ~3
- 8-week: ⏸️ Not started
- Inference: ⏸️ Pending
