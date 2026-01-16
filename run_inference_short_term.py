import torch
import numpy as np
import os
import argparse
from data_provider.data_factory import data_provider
from models import TimeLLM
from utils.tools import EarlyStopping, adjust_learning_rate
from torch import nn

parser = argparse.ArgumentParser(description='Time-LLM Short Term Inference')

# --- Basic Config ---
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--model', type=str, default='TimeLLM')
parser.add_argument('--model_id', type=str, default='inference_run')
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--data', type=str, default='synthetic_Epi_WeinerProcess')
parser.add_argument('--root_path', type=str, default='./')
parser.add_argument('--data_path', type=str, default='epidemics_30years_full.csv')
parser.add_argument('--features', type=str, default='S')
parser.add_argument('--target', type=str, default='I_child')
parser.add_argument('--seq_len', type=int, default=8)
parser.add_argument('--label_len', type=int, default=4)
parser.add_argument('--pred_len', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--output_path', type=str, default='./results/short_term_analysis')

# --- Architecture (Gemma-270M) ---
parser.add_argument('--llm_model', type=str, default='GEMMA')
parser.add_argument('--llm_dim', type=int, default=640)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--patch_len', type=int, default=1)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--enc_in', type=int, default=1)
parser.add_argument('--dec_in', type=int, default=1)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--freq', type=str, default='w')
parser.add_argument('--prompt_domain', type=int, default=1)

# --- Metadata & Missing Flags (The "Fixes") ---
parser.add_argument('--content', type=str, default='Epidemic child infection rates with Wiener process spikes')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--seasonal_patterns', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--inverse', action='store_true', default=False)
parser.add_argument('--cols', type=str, nargs='+', default=None)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--use_amp', action='store_true', default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

# 1. Load Data
print("--- Loading Test Data ---")
data_set, data_loader = data_provider(args, flag='test')

# 2. Build Model & Load Checkpoint
print("--- Initializing Model ---")
model = TimeLLM.Model(args).float()
model.load_state_dict(torch.load(args.checkpoint_path, map_location='cuda:0'))
model.to('cuda:0')
model.eval()

# 3. Run Inference
print(f"--- Running Inference (Horizon: {args.pred_len} weeks) ---")
preds = []
trues = []

with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        batch_x = batch_x.float().to('cuda:0')
        batch_y = batch_y.float().to('cuda:0')
        batch_x_mark = batch_x_mark.float().to('cuda:0')
        batch_y_mark = batch_y_mark.float().to('cuda:0')

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to('cuda:0')

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        outputs = outputs[:, -args.pred_len:, -args.c_out:]
        batch_y = batch_y[:, -args.pred_len:, -args.c_out:]

        preds.append(outputs.detach().cpu().numpy())
        trues.append(batch_y.detach().cpu().numpy())

preds = np.array(preds).squeeze()
trues = np.array(trues).squeeze()

print(f"\nPredictions shape (scaled): {preds.shape}")
print(f"True values shape (scaled): {trues.shape}")

# 4. Inverse transform to original scale
print("\n" + "="*40)
print("   INVERSE SCALING TO ORIGINAL SCALE")
print("="*40)

# Check scaler statistics
print(f"Scaler mean: {data_set.scaler.mean_}")
print(f"Scaler std (scale): {data_set.scaler.scale_}")

# Reshape for inverse transform
if preds.ndim == 1:
    preds_reshaped = preds.reshape(-1, 1)
    trues_reshaped = trues.reshape(-1, 1)
elif preds.ndim == 2:
    preds_reshaped = preds.reshape(-1, 1)
    trues_reshaped = trues.reshape(-1, 1)
else:
    raise ValueError(f"Unexpected prediction shape: {preds.shape}")

# Inverse transform
preds_original = data_set.inverse_transform(preds_reshaped)
trues_original = data_set.inverse_transform(trues_reshaped)

# Reshape back and clip to non-negative (epidemic data constraint)
preds_original = preds_original.reshape(preds.shape)
trues_original = trues_original.reshape(trues.shape)
preds_original = np.clip(preds_original, 0, None)
trues_original = np.clip(trues_original, 0, None)

print(f"Predictions shape (original): {preds_original.shape}")
print(f"Sample predictions (original): {preds_original[:3]}")

# 5. Weekly Breakdown (ORIGINAL SCALE)
print("\n" + "="*40)
print("   WEEKLY PERFORMANCE ANALYSIS")
print("   (Original Scale - Actual Counts)")
print("="*40)
print(f"{'Horizon':<15} | {'MAE':<12} | {'MSE':<15} | {'RMSE':<12}")
print("-" * 60)

for week in range(args.pred_len):
    if preds_original.ndim == 1:
        p = preds_original[week::args.pred_len]
        t = trues_original[week::args.pred_len]
    else:
        p = preds_original[:, week]
        t = trues_original[:, week]

    mae = np.mean(np.abs(p - t))
    mse = np.mean((p - t) ** 2)
    rmse = np.sqrt(mse)
    print(f"Week {week+1:<10} | {mae:<12.2f} | {mse:<15.2f} | {rmse:<12.2f}")

# 6. Overall Metrics
print("\n" + "="*40)
print("   OVERALL METRICS (Original Scale)")
print("="*40)
overall_mae = np.mean(np.abs(preds_original - trues_original))
overall_mse = np.mean((preds_original - trues_original) ** 2)
overall_rmse = np.sqrt(overall_mse)
# MAPE (avoiding division by zero)
mask = trues_original > 0
if mask.sum() > 0:
    overall_mape = np.mean(np.abs((preds_original[mask] - trues_original[mask]) / trues_original[mask])) * 100
else:
    overall_mape = float('nan')

print(f"MAE:  {overall_mae:12.2f}")
print(f"MSE:  {overall_mse:12.2f}")
print(f"RMSE: {overall_rmse:12.2f}")
print(f"MAPE: {overall_mape:12.2f}%")

# 7. Save results
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Save both scaled and original predictions
np.save(os.path.join(args.output_path, 'preds_scaled.npy'), preds)
np.save(os.path.join(args.output_path, 'trues_scaled.npy'), trues)
np.save(os.path.join(args.output_path, 'preds_original.npy'), preds_original)
np.save(os.path.join(args.output_path, 'trues_original.npy'), trues_original)

# Save metrics to text file
with open(os.path.join(args.output_path, 'metrics.txt'), 'w') as f:
    f.write("="*50 + "\n")
    f.write("INFERENCE METRICS (Original Scale)\n")
    f.write("="*50 + "\n\n")
    f.write(f"Target Variable: {args.target}\n")
    f.write(f"Prediction Horizon: {args.pred_len} weeks\n")
    f.write(f"Sequence Length: {args.seq_len} weeks\n\n")
    f.write("Overall Metrics:\n")
    f.write(f"  MAE:  {overall_mae:12.2f}\n")
    f.write(f"  MSE:  {overall_mse:12.2f}\n")
    f.write(f"  RMSE: {overall_rmse:12.2f}\n")
    f.write(f"  MAPE: {overall_mape:12.2f}%\n\n")
    f.write("Per-Horizon Metrics:\n")
    f.write(f"{'Horizon':<10} | {'MAE':<12} | {'MSE':<15} | {'RMSE':<12}\n")
    f.write("-" * 60 + "\n")
    for week in range(args.pred_len):
        if preds_original.ndim == 1:
            p = preds_original[week::args.pred_len]
            t = trues_original[week::args.pred_len]
        else:
            p = preds_original[:, week]
            t = trues_original[:, week]
        mae = np.mean(np.abs(p - t))
        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        f.write(f"Week {week+1:<5} | {mae:<12.2f} | {mse:<15.2f} | {rmse:<12.2f}\n")

print(f"\n{'='*60}")
print(f"[OK] Results saved to {args.output_path}")
print(f"{'='*60}")
print("\nFiles generated:")
print(f"  - preds_scaled.npy     (scaled predictions)")
print(f"  - trues_scaled.npy     (scaled ground truth)")
print(f"  - preds_original.npy   (original scale predictions)")
print(f"  - trues_original.npy   (original scale ground truth)")
print(f"  - metrics.txt          (detailed metrics report)")