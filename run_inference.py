import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from accelerate import Accelerator
from models import TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content



# Force float32 for MPS compatibility
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='TimeLLM Inference - Calculate MAE and MAPE')

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
    Calculate MAE and MAPE
    
    Args:
        preds: predictions array
        trues: true values array
    
    Returns:
        dict with MAE and MAPE
    """
    # Flatten arrays
    preds_flat = preds.reshape(-1)
    trues_flat = trues.reshape(-1)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(preds_flat - trues_flat))
    
    # MSE and RMSE (bonus metrics)
    mse = np.mean((preds_flat - trues_flat) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((trues_flat - preds_flat) / (np.abs(trues_flat) + epsilon))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def main():
    accelerator = Accelerator()
    
    print("="*60)
    print("TimeLLM Inference")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
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
    
    print("\nRunning inference on test set...")
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

    # ADD THESE LINES HERE:
    # Inverse transform to original scale using the dataset's scaler
    print("\nInverse transforming to original scale...")
    preds_reshaped = preds.reshape(-1, preds.shape[-1])
    trues_reshaped = trues.reshape(-1, trues.shape[-1])
    
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
    print(f"\nSaved predictions to: {args.output_path}")
    
    # Save overall metrics to text file
    metrics_file = os.path.join(args.output_path, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FORECASTING EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        f.write("Overall Metrics:\n")
        f.write("-"*60 + "\n")
        f.write(f"MAE:   {metrics['MAE']:.6f}\n")
        f.write(f"MAPE:  {metrics['MAPE']:.2f}%\n")
        f.write(f"RMSE:  {metrics['RMSE']:.6f}\n")
        f.write(f"MSE:   {metrics['MSE']:.6f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write(f"Predictions shape: {preds_original.shape}\n")
        f.write(f"True values shape: {trues_original.shape}\n")
        f.write("="*60 + "\n")
    
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
        f.write("Horizon,MAE,MAPE\n")
        for h in range(args.pred_len):
            preds_h = preds_original[:, h, :].flatten()
            trues_h = trues_original[:, h, :].flatten()
            
            mae_h = np.mean(np.abs(preds_h - trues_h))
            mape_h = np.mean(np.abs((trues_h - preds_h) / (np.abs(trues_h) + 1e-8))) * 100
            
            f.write(f"{h+1},{mae_h:.6f},{mape_h:.2f}\n")
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  MAE:   {metrics['MAE']:.6f}")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print(f"  RMSE:  {metrics['RMSE']:.6f}")
    print(f"  MSE:   {metrics['MSE']:.6f}")
    
    print(f"\nPer-Horizon Metrics:")
    print(f"{'Horizon':<10} {'MAE':<12} {'MAPE':<12}")
    print("-" * 35)
    
    for h in range(args.pred_len):
        preds_h = preds_original[:, h, :].flatten()
        trues_h = trues_original[:, h, :].flatten()

        mae_h = np.mean(np.abs(preds_h - trues_h))
        mape_h = np.mean(np.abs((trues_h - preds_h) / (np.abs(trues_h) + 1e-8))) * 100
        
        print(f"{h+1:<10} {mae_h:<12.6f} {mape_h:<12.2f}%")
    
    print("\n" + "="*60)
    print("FILES SAVED:")
    print("="*60)
    print(f"  {args.output_path}/predictions.npy")
    print(f"  {args.output_path}/true_values.npy")
    print(f"  {args.output_path}/metrics.txt")
    print(f"  {args.output_path}/metrics.csv")
    print(f"  {args.output_path}/per_horizon_metrics.csv")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
