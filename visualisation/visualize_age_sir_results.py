"""
Visualize Age-SIR forecasting results
Shows predictions vs ground truth around the epidemic peak
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("AGE-SIR FORECASTING RESULTS VISUALIZATION")
print("=" * 70)

# Load results
results_dir = 'results/age_sir_all_vars_forecast'
predictions = np.load(os.path.join(results_dir, 'predictions.npy'))
true_values = np.load(os.path.join(results_dir, 'true_values.npy'))

print(f"\nPredictions shape: {predictions.shape}")
print(f"  → (n_samples={predictions.shape[0]}, pred_len={predictions.shape[1]}, n_vars={predictions.shape[2]})")
print(f"  → Variable 0: IA (Adults Infected)")
print(f"  → Variable 1: IC (Children Infected)")

# Load original data
df = pd.read_csv('dataset/synthetic/age_sir/age_sir_all_vars.csv')
print(f"\nOriginal data shape: {df.shape}")

# Load metrics
print(f"\n--- Overall Metrics ---")
with open(os.path.join(results_dir, 'metrics.txt'), 'r') as f:
    metrics_content = f.read()
    print(metrics_content)

# Calculate per-variable metrics
mae_ia = np.mean(np.abs(predictions[:, :, 0] - true_values[:, :, 0]))
mae_ic = np.mean(np.abs(predictions[:, :, 1] - true_values[:, :, 1]))
mape_ia = np.mean(np.abs((true_values[:, :, 0] - predictions[:, :, 0]) / (true_values[:, :, 0] + 1e-8))) * 100
mape_ic = np.mean(np.abs((true_values[:, :, 1] - predictions[:, :, 1]) / (true_values[:, :, 1] + 1e-8))) * 100

print(f"\n--- Per-Variable Metrics ---")
print(f"IA (Adults):")
print(f"  MAE:  {mae_ia:.2f}")
print(f"  MAPE: {mape_ia:.2f}%")
print(f"IC (Children):")
print(f"  MAE:  {mae_ic:.2f}")
print(f"  MAPE: {mape_ic:.2f}%")

# Find peak in original data
peak_ia_day = df['IA'].idxmax()
peak_ic_day = df['IC'].idxmax()

print(f"\n--- Peak Information ---")
print(f"Peak IA: Day {peak_ia_day}, Count = {df.loc[peak_ia_day, 'IA']:,.0f}")
print(f"Peak IC: Day {peak_ic_day}, Count = {df.loc[peak_ic_day, 'IC']:,.0f}")

# Visualization
n_samples = predictions.shape[0]
pred_len = predictions.shape[1]

# Select days to visualize (around peak and after)
sample_days = [peak_ia_day - 14, peak_ia_day - 7, peak_ia_day,
                peak_ia_day + 7, peak_ia_day + 14, peak_ia_day + 21]
sample_days = [d for d in sample_days if 0 <= d < n_samples]

n_plots = len(sample_days)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('Age-SIR 7-Day Ahead Forecasts: IA and IC Predictions',
            fontsize=14, fontweight='bold', y=0.995)

for idx, day in enumerate(sample_days):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    # Historical context (21 days)
    hist_start = max(0, day - 21)
    hist_days = np.arange(hist_start, day + 1)
    hist_ia = df.loc[hist_start:day, 'IA'].values
    hist_ic = df.loc[hist_start:day, 'IC'].values

    # Forecast (7 days ahead)
    pred_days = np.arange(day + 1, min(day + 1 + pred_len, len(df)))
    actual_pred_len = len(pred_days)

    pred_ia = predictions[day, :actual_pred_len, 0]
    pred_ic = predictions[day, :actual_pred_len, 1]
    true_ia = true_values[day, :actual_pred_len, 0]
    true_ic = true_values[day, :actual_pred_len, 1]

    # Plot historical
    ax.plot(hist_days, hist_ia, 'g-', alpha=0.4, linewidth=1.5, label='Historical IA')
    ax.plot(hist_days, hist_ic, 'b-', alpha=0.4, linewidth=1.5, label='Historical IC')

    # Plot ground truth
    ax.plot(pred_days, true_ia, 'g-', linewidth=2.5, label='True IA', marker='o', markersize=4)
    ax.plot(pred_days, true_ic, 'b-', linewidth=2.5, label='True IC', marker='o', markersize=4)

    # Plot predictions
    ax.plot(pred_days, pred_ia, 'g--', linewidth=2.5, label='Pred IA', marker='s', markersize=4)
    ax.plot(pred_days, pred_ic, 'b--', linewidth=2.5, label='Pred IC', marker='s', markersize=4)

    # Vertical line at forecast start
    ax.axvline(x=day, color='red', linestyle=':', alpha=0.6, linewidth=1.5)

    # Calculate forecast error for this sample
    mae_ia_sample = np.mean(np.abs(pred_ia - true_ia))
    mae_ic_sample = np.mean(np.abs(pred_ic - true_ic))

    title = f'Forecast from Day {day}'
    if day == peak_ia_day:
        title += ' (Peak IA)'
    elif day == peak_ic_day:
        title += ' (Peak IC)'

    ax.set_title(f'{title}\nMAE: IA={mae_ia_sample:.0f}, IC={mae_ic_sample:.0f}',
                fontsize=10)
    ax.set_xlabel('Day', fontsize=10)
    ax.set_ylabel('Infected Count', fontsize=10)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

# Hide empty subplots
for idx in range(n_plots, n_rows * n_cols):
    row = idx // n_cols
    col = idx % n_cols
    axes[row, col].axis('off')

plt.tight_layout()
output_path = os.path.join(results_dir, 'forecast_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")
plt.show()

# Create forecast error heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Errors for IA
errors_ia = np.abs(predictions[:, :, 0] - true_values[:, :, 0])
im1 = ax1.imshow(errors_ia.T, aspect='auto', cmap='YlOrRd', origin='lower')
ax1.set_xlabel('Forecast Start Day', fontsize=11)
ax1.set_ylabel('Forecast Horizon (days ahead)', fontsize=11)
ax1.set_title('Absolute Error: IA (Adults Infected)', fontsize=12, fontweight='bold')
ax1.set_yticks(range(pred_len))
ax1.set_yticklabels(range(1, pred_len + 1))
plt.colorbar(im1, ax=ax1, label='Absolute Error')

# Errors for IC
errors_ic = np.abs(predictions[:, :, 1] - true_values[:, :, 1])
im2 = ax2.imshow(errors_ic.T, aspect='auto', cmap='YlOrRd', origin='lower')
ax2.set_xlabel('Forecast Start Day', fontsize=11)
ax2.set_ylabel('Forecast Horizon (days ahead)', fontsize=11)
ax2.set_title('Absolute Error: IC (Children Infected)', fontsize=12, fontweight='bold')
ax2.set_yticks(range(pred_len))
ax2.set_yticklabels(range(1, pred_len + 1))
plt.colorbar(im2, ax=ax2, label='Absolute Error')

plt.tight_layout()
heatmap_path = os.path.join(results_dir, 'error_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Error heatmap saved to: {heatmap_path}")
plt.show()

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  1. {output_path}")
print(f"  2. {heatmap_path}")
