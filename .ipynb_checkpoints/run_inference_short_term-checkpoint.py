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

# 4. Weekly Breakdown
print("\n" + "="*40)
print("   WEEKLY PERFORMANCE ANALYSIS")
print("="*40)
print(f"{'Horizon':<15} | {'MAE':<10} | {'MSE':<10}")
print("-" * 40)

for week in range(args.pred_len):
    p = preds[:, week]
    t = trues[:, week]
    mae = np.mean(np.abs(p - t))
    mse = np.mean((p - t) ** 2)
    print(f"Week {week+1:<10} | {mae:<10.4f} | {mse:<10.4f}")

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
np.save(os.path.join(args.output_path, 'preds.npy'), preds)
np.save(os.path.join(args.output_path, 'trues.npy'), trues)
print(f"\nResults saved to {args.output_path}")