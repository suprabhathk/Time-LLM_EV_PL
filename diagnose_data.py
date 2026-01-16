#!/usr/bin/env python
"""
Data Diagnostics for Epidemic Forecasting
==========================================
This script helps you understand your epidemic data, check scaling behavior,
and validate the dataset before training.

Usage:
    python diagnose_data.py --data_path epidemics_30years_full.csv --target I_child
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
import os

def main():
    parser = argparse.ArgumentParser(description='Epidemic Data Diagnostics')
    parser.add_argument('--data_path', type=str, default='epidemics_30years_full.csv',
                        help='Path to CSV data file')
    parser.add_argument('--target', type=str, default='I_child',
                        help='Target variable to forecast')
    parser.add_argument('--root_path', type=str, default='./',
                        help='Root path to data directory')
    args = parser.parse_args()

    print("="*70)
    print("EPIDEMIC DATA DIAGNOSTICS")
    print("="*70)
    print()

    # 1. Load data
    data_path = os.path.join(args.root_path, args.data_path)
    print(f"Loading data from: {data_path}")

    if not os.path.exists(data_path):
        print(f"❌ ERROR: File not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"✓ Data loaded successfully")
    print()

    # 2. Basic info
    print("="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    print()

    # 3. Check target variable
    print("="*70)
    print(f"TARGET VARIABLE: {args.target}")
    print("="*70)

    if args.target not in df.columns:
        print(f"❌ ERROR: Target '{args.target}' not found in columns!")
        print(f"Available columns: {list(df.columns)}")
        return

    target_data = df[args.target].values
    print(f"✓ Target variable found")
    print(f"  Min:    {target_data.min():.2f}")
    print(f"  Max:    {target_data.max():.2f}")
    print(f"  Mean:   {target_data.mean():.2f}")
    print(f"  Median: {np.median(target_data):.2f}")
    print(f"  Std:    {target_data.std():.2f}")
    print()

    # 4. Check for missing values
    print("="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)
    missing = df[args.target].isna().sum()
    print(f"Missing values: {missing} ({missing/len(df)*100:.2f}%)")

    negatives = (target_data < 0).sum()
    print(f"Negative values: {negatives} ({negatives/len(df)*100:.2f}%)")

    zeros = (target_data == 0).sum()
    print(f"Zero values: {zeros} ({zeros/len(df)*100:.2f}%)")
    print()

    # 5. Train/Val/Test split analysis
    print("="*70)
    print("TRAIN/VAL/TEST SPLIT ANALYSIS (70/20/10)")
    print("="*70)

    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.9 * n)

    train_data = target_data[:train_end]
    val_data = target_data[train_end:val_end]
    test_data = target_data[val_end:]

    print(f"Train: {len(train_data)} samples ({len(train_data)/n*100:.1f}%)")
    print(f"  Range: [{train_data.min():.2f}, {train_data.max():.2f}]")
    print(f"  Mean:  {train_data.mean():.2f}")
    print(f"  Std:   {train_data.std():.2f}")
    print()

    print(f"Val:   {len(val_data)} samples ({len(val_data)/n*100:.1f}%)")
    print(f"  Range: [{val_data.min():.2f}, {val_data.max():.2f}]")
    print(f"  Mean:  {val_data.mean():.2f}")
    print(f"  Std:   {val_data.std():.2f}")
    print()

    print(f"Test:  {len(test_data)} samples ({len(test_data)/n*100:.1f}%)")
    print(f"  Range: [{test_data.min():.2f}, {test_data.max():.2f}]")
    print(f"  Mean:  {test_data.mean():.2f}")
    print(f"  Std:   {test_data.std():.2f}")
    print()

    # 6. Scaling analysis
    print("="*70)
    print("SCALING ANALYSIS (StandardScaler)")
    print("="*70)

    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, 1))

    print(f"Scaler statistics (fitted on TRAIN data only):")
    print(f"  Mean: {scaler.mean_[0]:.2f}")
    print(f"  Std:  {scaler.scale_[0]:.2f}")
    print()

    # Transform all data
    scaled_train = scaler.transform(train_data.reshape(-1, 1)).flatten()
    scaled_val = scaler.transform(val_data.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test_data.reshape(-1, 1)).flatten()

    print(f"Scaled data ranges (z-scores):")
    print(f"  Train: [{scaled_train.min():.2f}, {scaled_train.max():.2f}]")
    print(f"  Val:   [{scaled_val.min():.2f}, {scaled_val.max():.2f}]")
    print(f"  Test:  [{scaled_test.min():.2f}, {scaled_test.max():.2f}]")
    print()

    # Check for extreme values
    extreme_threshold = 5.0
    train_extreme = (np.abs(scaled_train) > extreme_threshold).sum()
    val_extreme = (np.abs(scaled_val) > extreme_threshold).sum()
    test_extreme = (np.abs(scaled_test) > extreme_threshold).sum()

    print(f"Extreme values (|z-score| > {extreme_threshold}):")
    print(f"  Train: {train_extreme} ({train_extreme/len(scaled_train)*100:.2f}%)")
    print(f"  Val:   {val_extreme} ({val_extreme/len(scaled_val)*100:.2f}%)")
    print(f"  Test:  {test_extreme} ({test_extreme/len(scaled_test)*100:.2f}%)")

    if max(train_extreme, val_extreme, test_extreme) > len(train_data) * 0.01:
        print("\n  ⚠ WARNING: More than 1% of data has extreme z-scores!")
        print("  This may indicate:")
        print("    - Outliers or data quality issues")
        print("    - Distribution shift between train and val/test")
        print("    - Need for different scaling approach (e.g., log transform)")
    print()

    # 7. Distribution shift check
    print("="*70)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*70)

    mean_shift_val = abs(val_data.mean() - train_data.mean()) / train_data.std()
    mean_shift_test = abs(test_data.mean() - train_data.mean()) / train_data.std()

    print(f"Mean shift (in train std units):")
    print(f"  Val vs Train:  {mean_shift_val:.2f} std")
    print(f"  Test vs Train: {mean_shift_test:.2f} std")

    if mean_shift_val > 0.5 or mean_shift_test > 0.5:
        print("\n  ⚠ WARNING: Significant distribution shift detected!")
        print("  Val/Test data has different mean than training data.")
        print("  This may affect model generalization.")
    else:
        print("\n  ✓ Distribution appears relatively stable")
    print()

    # 8. Temporal dynamics
    print("="*70)
    print("TEMPORAL DYNAMICS")
    print("="*70)

    # Compute differences
    diffs = np.diff(target_data)
    print(f"Week-to-week changes:")
    print(f"  Mean change: {diffs.mean():.2f}")
    print(f"  Std change:  {diffs.std():.2f}")
    print(f"  Max increase: {diffs.max():.2f}")
    print(f"  Max decrease: {diffs.min():.2f}")

    # Stationarity check (simple)
    first_half_mean = target_data[:len(target_data)//2].mean()
    second_half_mean = target_data[len(target_data)//2:].mean()
    print(f"\nStationarity check:")
    print(f"  First half mean:  {first_half_mean:.2f}")
    print(f"  Second half mean: {second_half_mean:.2f}")
    print(f"  Difference:       {abs(first_half_mean - second_half_mean):.2f}")
    print()

    # 9. Summary and recommendations
    print("="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print()

    if scaler.scale_[0] < 1.0:
        print("⚠ Low standard deviation detected:")
        print(f"  Std = {scaler.scale_[0]:.4f}")
        print("  Recommendation: Check if target variable has very small values")
        print()

    if scaler.scale_[0] > 10000:
        print("⚠ High standard deviation detected:")
        print(f"  Std = {scaler.scale_[0]:.2f}")
        print("  Recommendation: Consider log-transform for epidemic data with exponential growth")
        print()

    if (target_data == 0).sum() > len(target_data) * 0.1:
        print("⚠ Many zero values detected:")
        print(f"  {(target_data == 0).sum()} / {len(target_data)} samples are zero")
        print("  Recommendation: This may be expected for epidemic data (endemic periods)")
        print()

    print("✓ Diagnostics complete!")
    print()
    print("Next steps:")
    print("  1. If no warnings, proceed with training:")
    print("     bash train_epidemic_multi_horizon.sh")
    print("  2. If warnings exist, consider data preprocessing or model adjustments")
    print()

if __name__ == "__main__":
    main()
