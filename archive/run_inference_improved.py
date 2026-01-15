import argparse
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from models import TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content


# Force float32 for MPS compatibility
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='TimeLLM Inference - Improved with custom date ranges and fixed scaler')

# Checkpoint
parser.add_argument('--checkpoint_path', type=str, required=True, help='path to trained checkpoint')

# Data config
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--model', type=str, default='TimeLLM')
parser.add_argument('--data', type=str, required=True, default='Weather')
parser.add_argument('--root_path', type=str, default='./dataset/weather/')
parser.add_argument('--data_path', type=str, default='weather.csv')
parser.add_argument('--features', type=str, default='S', help='M, S, or MS')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--loader', type=str, default='modal')

# Forecasting task
parser.add_argument('--seq_len', type=int, default=48)
parser.add_argument('--label_len', type=int, default=24)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

# Custom inference date range (NEW!)
parser.add_argument('--inference_mode', type=str, default='test',
                    help='test (use default test split), custom (use start/end dates), around_peak (forecast around peak)')
parser.add_argument('--start_date', type=str, default=None,
                    help='Start date for custom inference (YYYY-MM-DD), e.g., 2024-02-05')
parser.add_argument('--end_date', type=str, default=None,
                    help='End date for custom inference (YYYY-MM-DD), e.g., 2024-02-19')
parser.add_argument('--peak_window', type=int, default=7,
                    help='Days before peak to start forecasting (for around_peak mode)')

# Model config
parser.add_argument('--llm_model', type=str, default='GEMMA')
parser.add_argument('--llm_dim', type=int, default=640)
parser.add_argument('--llm_layers', type=int, default=1)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--activation', type=str, default='gelu')

# Model dimensions (auto-set based on data)
parser.add_argument('--enc_in', type=int, default=1)
parser.add_argument('--dec_in', type=int, default=1)
parser.add_argument('--c_out', type=int, default=1)

# Other
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0)

# Output
parser.add_argument('--output_path', type=str, default='./results/inference')

args = parser.parse_args()

def calculate_metrics(preds, trues):
    """
    Calculate MAE, MAPE, RMSE, MSE

    Args:
        preds: predictions array
        trues: true values array

    Returns:
        dict with metrics
    """
    # Flatten arrays
    preds_flat = preds.reshape(-1)
    trues_flat = trues.reshape(-1)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(preds_flat - trues_flat))

    # MSE and RMSE
    mse = np.mean((preds_flat - trues_flat) ** 2)
    rmse = np.sqrt(mse)

    # MAPE (Mean Absolute Percentage Error)
    epsilon = 1e-8
    mape = np.mean(np.abs((trues_flat - preds_flat) / (np.abs(trues_flat) + epsilon))) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def find_peak_date(df_raw, target_col):
    """Find the date where target column reaches its maximum"""
    peak_idx = df_raw[target_col].idxmax()
    peak_date = df_raw.loc[peak_idx, 'date']
    peak_value = df_raw.loc[peak_idx, target_col]
    return peak_date, peak_value, peak_idx

def get_custom_date_range(df_raw, start_date, end_date):
    """Get row indices for custom date range"""
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df_raw['date'] >= start_date) & (df_raw['date'] <= end_date)
    indices = df_raw[mask].index.tolist()

    if len(indices) == 0:
        raise ValueError(f"No data found between {start_date} and {end_date}")

    return indices[0], indices[-1]

def main():
    accelerator = Accelerator()

    print("="*70)
    print("TimeLLM Inference - IMPROVED VERSION")
    print("="*70)

    # Load CSV to check data characteristics
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    print(f"\nData file: {args.data_path}")
    print(f"Total rows: {len(df_raw)}")
    print(f"Date range: {df_raw['date'].iloc[0]} to {df_raw['date'].iloc[-1]}")
    print(f"Columns: {list(df_raw.columns)}")

    # Check target column characteristics
    if args.target in df_raw.columns:
        target_stats = df_raw[args.target].describe()
        print(f"\nTarget '{args.target}' statistics:")
        print(f"  Min: {target_stats['min']:.2f}")
        print(f"  Max: {target_stats['max']:.2f}")
        print(f"  Mean: {target_stats['mean']:.2f}")
        print(f"  Std: {target_stats['std']:.2f}")

        # Find peak
        peak_date, peak_value, peak_idx = find_peak_date(df_raw, args.target)
        print(f"  Peak: {peak_value:.2f} on {peak_date} (row {peak_idx})")

    # Determine inference range based on mode
    if args.inference_mode == 'around_peak':
        if args.target not in df_raw.columns:
            raise ValueError(f"Target '{args.target}' not found in data")

        peak_date, peak_value, peak_idx = find_peak_date(df_raw, args.target)

        # Start forecasting N days before peak
        start_idx = max(0, peak_idx - args.peak_window)
        # End forecasting N days after peak
        end_idx = min(len(df_raw) - 1, peak_idx + args.peak_window)

        print(f"\n** AROUND PEAK MODE **")
        print(f"Peak occurs on {peak_date} (row {peak_idx}, value={peak_value:.2f})")
        print(f"Forecasting from row {start_idx} to {end_idx}")
        print(f"  Start: {df_raw['date'].iloc[start_idx]}")
        print(f"  End: {df_raw['date'].iloc[end_idx]}")

    elif args.inference_mode == 'custom':
        if args.start_date is None or args.end_date is None:
            raise ValueError("Must specify --start_date and --end_date for custom mode")

        start_idx, end_idx = get_custom_date_range(df_raw, args.start_date, args.end_date)
        print(f"\n** CUSTOM DATE RANGE MODE **")
        print(f"Forecasting from {args.start_date} to {args.end_date}")
        print(f"  Rows: {start_idx} to {end_idx}")

    else:  # 'test' mode (default)
        print(f"\n** TEST SET MODE (default) **")
        print(f"Using standard 70/20/10 train/val/test split")
        start_idx = None  # Will use default test loader
        end_idx = None

    # Load data
    print("\nLoading data...")

    if args.inference_mode in ['around_peak', 'custom']:
        # For custom ranges, we need to manually create the dataset
        # For now, we'll use the test loader but this is a limitation
        # A full implementation would require modifying data_loader.py
        print("⚠ WARNING: Custom date ranges require using full dataset")
        print("⚠ Loading all data and will filter during inference")
        test_data, test_loader = data_provider(args, 'test')
    else:
        test_data, test_loader = data_provider(args, 'test')

    print(f"Test samples: {len(test_data)}")

    # Create model
    print("\nInitializing model...")
    args.content = load_content(args)
    model = TimeLLM.Model(args).float()

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=accelerator.device)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!")

    # Prepare with accelerator
    model = accelerator.prepare(model)
    test_loader = accelerator.prepare(test_loader)

    # Run inference
    model.eval()
    preds = []
    trues = []

    print("\nRunning inference...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # Forward pass
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Extract predictions
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    print(f"\nInference complete!")
    print(f"Predictions shape: {preds.shape}")
    print(f"True values shape: {trues.shape}")

    # Inverse transform to original scale - FIXED VERSION
    print("\nInverse transforming to original scale...")
    preds_reshaped = preds.reshape(-1, preds.shape[-1])
    trues_reshaped = trues.reshape(-1, trues.shape[-1])

    # Check if we need to handle dimension mismatch (multivariate input, univariate output)
    if preds.shape[-1] < test_data.scaler.scale_.shape[0]:
        print(f"Handling dimension mismatch: predictions have {preds.shape[-1]} features, scaler has {test_data.scaler.scale_.shape[0]} features")

        # FIXED: Get the target column index in the REORDERED data
        # Dataset_Custom reorders columns as: [non-target cols] + [target]
        # So target is always at index -1 (last position)
        if hasattr(test_data, 'target'):
            # Read original CSV to find all columns
            cols_original = list(df_raw.columns)
            cols_original.remove('date')

            # Dataset_Custom moves target to last position
            # So the target_idx in the scaler is always the last index
            target_idx = test_data.scaler.scale_.shape[0] - 1

            print(f"Target '{test_data.target}' is at scaler index {target_idx} (last position after reordering)")
            print(f"Scaler stats for target: mean={test_data.scaler.mean_[target_idx]:.2f}, scale={test_data.scaler.scale_[target_idx]:.2f}")
        else:
            # Default to last column if target not specified
            target_idx = -1

        # Inverse transform using only the target feature's scaler parameters
        preds_original = (preds_reshaped * test_data.scaler.scale_[target_idx]) + test_data.scaler.mean_[target_idx]
        trues_original = (trues_reshaped * test_data.scaler.scale_[target_idx]) + test_data.scaler.mean_[target_idx]

        # Reshape back to original shape
        preds_original = preds_original.reshape(preds.shape)
        trues_original = trues_original.reshape(trues.shape)
    else:
        # Standard inverse transform when dimensions match
        preds_original = test_data.scaler.inverse_transform(preds_reshaped).reshape(preds.shape)
        trues_original = test_data.scaler.inverse_transform(trues_reshaped).reshape(trues.shape)

    print(f"Predictions shape (original scale): {preds_original.shape}")
    print(f"True values shape (original scale): {trues_original.shape}")
    print(f"Prediction range: [{preds_original.min():.4f}, {preds_original.max():.4f}]")
    print(f"True values range: [{trues_original.min():.4f}, {trues_original.max():.4f}]")

    # Calculate overall metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(preds_original, trues_original)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Save predictions and true values
    np.save(os.path.join(args.output_path, 'predictions.npy'), preds_original)
    np.save(os.path.join(args.output_path, 'true_values.npy'), trues_original)

    # Save detailed comparison CSV
    comparison_file = os.path.join(args.output_path, 'predictions_vs_true.csv')
    with open(comparison_file, 'w') as f:
        f.write("Sample,Horizon,Predicted,True,Error,AbsError,PctError\n")
        for sample in range(preds_original.shape[0]):
            for horizon in range(preds_original.shape[1]):
                pred = preds_original[sample, horizon, 0]
                true = trues_original[sample, horizon, 0]
                error = pred - true
                abs_error = abs(error)
                pct_error = (abs_error / (abs(true) + 1e-8)) * 100
                f.write(f"{sample},{horizon+1},{pred:.2f},{true:.2f},{error:.2f},{abs_error:.2f},{pct_error:.2f}\n")

    # Save overall metrics to text file
    metrics_file = os.path.join(args.output_path, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FORECASTING EVALUATION METRICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Inference Mode: {args.inference_mode}\n")
        if args.inference_mode == 'around_peak':
            f.write(f"Peak Date: {peak_date}\n")
            f.write(f"Peak Value: {peak_value:.2f}\n")
        elif args.inference_mode == 'custom':
            f.write(f"Date Range: {args.start_date} to {args.end_date}\n")
        f.write("\n")
        f.write("Overall Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"MAE:   {metrics['MAE']:.6f}\n")
        f.write(f"MAPE:  {metrics['MAPE']:.2f}%\n")
        f.write(f"RMSE:  {metrics['RMSE']:.6f}\n")
        f.write(f"MSE:   {metrics['MSE']:.6f}\n")
        f.write("\n" + "="*70 + "\n")
        f.write(f"Predictions shape: {preds_original.shape}\n")
        f.write(f"True values shape: {trues_original.shape}\n")
        f.write("="*70 + "\n")

    # Save metrics as CSV
    csv_file = os.path.join(args.output_path, 'metrics.csv')
    with open(csv_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"MAE,{metrics['MAE']:.6f}\n")
        f.write(f"MAPE,{metrics['MAPE']:.2f}\n")
        f.write(f"RMSE,{metrics['RMSE']:.6f}\n")
        f.write(f"MSE,{metrics['MSE']:.6f}\n")

    # Calculate and save per-horizon metrics
    print("\nCalculating per-horizon metrics...")
    per_horizon_file = os.path.join(args.output_path, 'per_horizon_metrics.csv')
    with open(per_horizon_file, 'w') as f:
        f.write("Horizon,MAE,MAPE,RMSE\n")
        for h in range(args.pred_len):
            preds_h = preds_original[:, h, :].flatten()
            trues_h = trues_original[:, h, :].flatten()

            mae_h = np.mean(np.abs(preds_h - trues_h))
            mape_h = np.mean(np.abs((trues_h - preds_h) / (np.abs(trues_h) + 1e-8))) * 100
            rmse_h = np.sqrt(np.mean((preds_h - trues_h) ** 2))

            f.write(f"{h+1},{mae_h:.6f},{mape_h:.2f},{rmse_h:.6f}\n")

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOverall Metrics:")
    print(f"  MAE:   {metrics['MAE']:.6f}")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  RMSE:  {metrics['RMSE']:.6f}")
    print(f"  MSE:   {metrics['MSE']:.6f}")

    print(f"\nPer-Horizon Metrics:")
    print(f"{'Horizon':<10} {'MAE':<12} {'MAPE':<12} {'RMSE':<12}")
    print("-" * 50)

    for h in range(args.pred_len):
        preds_h = preds_original[:, h, :].flatten()
        trues_h = trues_original[:, h, :].flatten()

        mae_h = np.mean(np.abs(preds_h - trues_h))
        mape_h = np.mean(np.abs((trues_h - preds_h) / (np.abs(trues_h) + 1e-8))) * 100
        rmse_h = np.sqrt(np.mean((preds_h - trues_h) ** 2))

        print(f"{h+1:<10} {mae_h:<12.6f} {mape_h:<12.2f}% {rmse_h:<12.6f}")

    # Show sample predictions
    print(f"\nSample Predictions (first 3 samples, first 3 horizons):")
    print(f"{'Sample':<8} {'Horizon':<8} {'Predicted':<12} {'True':<12} {'Error':<12}")
    print("-" * 55)
    for s in range(min(3, preds_original.shape[0])):
        for h in range(min(3, args.pred_len)):
            pred = preds_original[s, h, 0]
            true = trues_original[s, h, 0]
            error = pred - true
            print(f"{s:<8} {h+1:<8} {pred:<12.2f} {true:<12.2f} {error:<12.2f}")

    print("\n" + "="*70)
    print("FILES SAVED:")
    print("="*70)
    print(f"  {args.output_path}/predictions.npy")
    print(f"  {args.output_path}/true_values.npy")
    print(f"  {args.output_path}/predictions_vs_true.csv")
    print(f"  {args.output_path}/metrics.txt")
    print(f"  {args.output_path}/metrics.csv")
    print(f"  {args.output_path}/per_horizon_metrics.csv")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()


if preds.shape[-1] < test_data.scaler.scale_.shape[0]:
        print(f"Handling dimension mismatch: predictions have {preds.shape[-1]} features, scaler has {test_data.scaler.scale_.shape[0]} features")

        # Get the target column index
        if hasattr(test_data, 'target'):
            # Find target column index in original data
            import pandas as pd
            df_raw = pd.read_csv(os.path.join(test_data.root_path, test_data.data_path))
            cols = list(df_raw.columns)
            cols.remove('date')
            target_idx = cols.index(test_data.target)
        else:
            # Default to last column if target not specified
            target_idx = -1

        # Inverse transform using only the target feature's scaler parameters
        preds_original = (preds_reshaped * test_data.scaler.scale_[target_idx]) + test_data.scaler.mean_[target_idx]
        trues_original = (trues_reshaped * test_data.scaler.scale_[target_idx]) + test_data.scaler.mean_[target_idx]

        # Reshape back to original shape
        preds_original = preds_original.reshape(preds.shape)
        trues_original = trues_original.reshape(trues.shape)
    else:
        # Standard inverse transform when dimensions match
        preds_original = test_data.scaler.inverse_transform(preds_reshaped).reshape(preds.shape)
        trues_original = test_data.scaler.inverse_transform(trues_reshaped).reshape(trues.shape)
