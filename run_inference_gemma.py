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

# Force float32 for consistency
torch.set_default_dtype(torch.float32)

def main():
    parser = argparse.ArgumentParser(description='Time-LLM Short-Term Inference (8 to 4)')
    
    # Path Configuration
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--data_path', type=str, default='epidemics_30years_full.csv')
    parser.add_argument('--output_path', type=str, default='./results/short_term_analysis')

    # Short-Term Architecture (Must match .sh file)
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--data', type=str, default='synthetic_Epi_WeinerProcess')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='I_child')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--label_len', type=int, default=4)
    parser.add_argument('--pred_len', type=int, default=4)
    
    # Model Capacity (Must match .sh file)
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--patch_len', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)

    # Gemma Settings
    parser.add_argument('--llm_model', type=str, default='GEMMA')
    parser.add_argument('--llm_dim', type=int, default=640)
    parser.add_argument('--llm_layers', type=int, default=32)
    parser.add_argument('--prompt_domain', type=int, default=1)
    
    # Defaults
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    args = parser.parse_args()
    accelerator = Accelerator()

    # 1. Load Data
    test_data, test_loader = data_provider(args, flag='test')
    scaler = test_data.scaler

    # 2. Initialize Model
    args.content = load_content(args)
    model = TimeLLM.Model(args).float()
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=accelerator.device))
    model = accelerator.prepare(model)
    model.eval()

    # 3. Inference Loop
    preds, trues = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, -1:]
            batch_y = batch_y[:, -args.pred_len:, -1:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    # 4. Concatenate and Inverse Scale
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Back to original infection counts
    preds_orig = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_orig = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
    preds_orig = np.clip(preds_orig, 0, None) # No negative infections

    # 5. PER-WEEK ANALYSIS LOGIC
    print("\n" + "="*50)
    print(f"{'Forecast Week':<15} | {'MAE':<15} | {'MSE':<15}")
    print("-" * 50)
    
    weekly_results = []
    for w in range(args.pred_len):
        w_pred = preds_orig[:, w, 0]
        w_true = trues_orig[:, w, 0]
        
        mae = np.mean(np.abs(w_pred - w_true))
        mse = np.mean((w_pred - w_true)**2)
        
        print(f"Week {w+1:<10} | {mae:<15.2f} | {mse:<15.2f}")
        weekly_results.append({"Week": w+1, "MAE": mae, "MSE": mse})

    # Save Metrics to CSV
    os.makedirs(args.output_path, exist_ok=True)
    pd.DataFrame(weekly_results).to_csv(f"{args.output_path}/weekly_metrics.csv", index=False)
    np.save(f"{args.output_path}/preds.npy", preds_orig)
    np.save(f"{args.output_path}/trues.npy", trues_orig)
    print("="*50)

if __name__ == '__main__':
    main()