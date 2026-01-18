import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_methodology_data_full(output_file='epidemics_30years_full.csv'):
    # --- 1. BASE PARAMETERS ---
    np.random.seed(42)
    beta_0 = 0.022
    gamma_base = 1/5
    seasonal_amplitude = 0.3
    N_C, N_A = 1500000, 3500000  # Total populations
    M = np.array([[18, 9], [3, 12]]) # Contact Matrix
    
    num_years = 30
    days_per_epidemic = 365
    dt = 0.1  # Step for Euler-Maruyama solver
    sampling_interval = 7 # Weekly resolution
    start_year = 1990

    # --- 2. AGGRESSIVE VARIABILITY ---
    """
    OLD PARAMETER RANGES:
    beta_multipliers = np.random.uniform(0.6, 1.4, num_years)
    gamma_values = np.random.uniform(1/10, 1/3, num_years)
    start_date_offsets = np.random.randint(-60, 61, num_years)
    initial_I_C_vals = np.random.randint(1, 51, num_years)
    initial_I_A_vals = np.random.randint(1, 51, num_years)
    phase_shifts = np.random.randint(-45, 46, num_years)
    """
    # --- ADAPTED FOR REGULAR SEASONALITY ---
    # We keep the noise, but stop the 'macro' parameters from changing every year
    beta_multipliers = np.ones(num_years)      # Intensity is now consistent
    gamma_values = np.full(num_years, 1/5)     # Outbreak speed is consistent
    start_date_offsets = np.zeros(num_years)    # Peak timing is roughly fixed
    initial_I_C_vals = np.full(num_years, 20)  # Every year starts with 20 infected children
    initial_I_A_vals = np.full(num_years, 20)  # Every year starts with 20 infected adults
    phase_shifts = np.zeros(num_years)         # Seasonality center is fixed

    # Inside the Euler-Maruyama loop, the Wiener noise still acts:
    # dW = np.random.normal(0, 1, 4)  <-- KEEP THIS
    # dS_C = -dt * inf_C - np.sqrt(dt * max(0, inf_C)) * dW[0] <-- KEEP THIS

    all_epidemic_data = []
    overlaid_curves = [] 

    # --- 3. STOCHASTIC SIMULATION ---
    for year_idx in range(num_years):
        calendar_year = start_year + year_idx
        
        # Initial states for Children and Adults
        S_C, I_C, R_C = N_C - initial_I_C_vals[year_idx], float(initial_I_C_vals[year_idx]), 0.0
        S_A, I_A, R_A = N_A - initial_I_A_vals[year_idx], float(initial_I_A_vals[year_idx]), 0.0
        
        base_start = datetime(calendar_year, 1, 1)
        start_date = base_start + timedelta(days=int(start_date_offsets[year_idx]))
        
        num_steps = int(days_per_epidemic / dt)
        next_sample_day = 0
        year_curve = [] 

        for step in range(num_steps):
            current_day = step * dt
            
            # Seasonal Beta with Weiner Process noise
            base_beta = beta_0 * beta_multipliers[year_idx]
            seasonal_factor = 1 + seasonal_amplitude * np.cos(2 * np.pi * (current_day - phase_shifts[year_idx]) / days_per_epidemic)
            beta_t = max(0.001, (base_beta * seasonal_factor) + np.random.normal(0, 0.001))
            
            lambda_C = beta_t * (M[0,0] * I_C / N_C + M[0,1] * I_A / N_A)
            lambda_A = beta_t * (M[1,0] * I_C / N_C + M[1,1] * I_A / N_A)
            
            # Weekly sampling including all compartments
            if current_day >= next_sample_day:
                total_I = I_C + I_A
                all_epidemic_data.append({
                    'date': start_date + timedelta(days=int(current_day)),
                    'OT': total_I,         # Target for Time-LLM
                    'S_child': S_C, 'I_child': I_C, 'R_child': R_C,
                    'S_adult': S_A, 'I_adult': I_A, 'R_adult': R_A,
                    'S_total': S_C + S_A, 'I_total': total_I, 'R_total': R_C + R_A
                })
                year_curve.append(total_I)
                next_sample_day += sampling_interval

            # Euler-Maruyama SDE update
            dW = np.random.normal(0, 1, 4)
            
            # Children
            inf_C = lambda_C * S_C
            rec_C = gamma_values[year_idx] * I_C
            dS_C = -dt * inf_C - np.sqrt(dt * max(0, inf_C)) * dW[0]
            dI_C = (dt * inf_C + np.sqrt(dt * max(0, inf_C)) * dW[0]) - (dt * rec_C + np.sqrt(dt * max(0, rec_C)) * dW[1])
            dR_C = dt * rec_C + np.sqrt(dt * max(0, rec_C)) * dW[1]
            
            # Adults
            inf_A = lambda_A * S_A
            rec_A = gamma_values[year_idx] * I_A
            dS_A = -dt * inf_A - np.sqrt(dt * max(0, inf_A)) * dW[2]
            dI_A = (dt * inf_A + np.sqrt(dt * max(0, inf_A)) * dW[2]) - (dt * rec_A + np.sqrt(dt * max(0, rec_A)) * dW[3])
            dR_A = dt * rec_A + np.sqrt(dt * max(0, rec_A)) * dW[3]

            S_C, I_C, R_C = max(0, S_C + dS_C), max(0, I_C + dI_C), max(0, R_C + dR_C)
            S_A, I_A, R_A = max(0, S_A + dS_A), max(0, I_A + dI_A), max(0, R_A + dR_A)
            
        overlaid_curves.append(year_curve)

    df = pd.DataFrame(all_epidemic_data)
    df.to_csv(output_file, index=False)
    
    # --- 4. VISUALIZATION ---
    # Chart 1: Overlaid Epidemic Curves
    plt.figure(figsize=(12, 6))
    for curve in overlaid_curves: plt.plot(curve, alpha=0.5)
    plt.title("30 HIGHLY VARIABLE Independent Epidemic Curves (Overlaid)")
    plt.xlabel("Weeks into Epidemic")
    plt.ylabel("Total Infected")
    plt.savefig('overlaid_curves.png')

    # Chart 2: Continuous Timeline
    plt.figure(figsize=(15, 5))
    plt.plot(df['date'], df['OT'], color='firebrick')
    plt.title("30-Year Continuous Epidemic Timeline (1990-2019)")
    plt.savefig('continuous_timeline.png')

    plt.figure(figsize=(15, 5))
    plt.plot(df['date'], df['I_adult'], color='firebrick')
    plt.title("30-Year Continuous Epidemic Timeline (1990-2019) - I_Adult")
    plt.savefig('continuous_timeline_adult.png')


    plt.figure(figsize=(15, 5))
    plt.plot(df['date'], df['I_child'], color='firebrick')
    plt.title("30-Year Continuous Epidemic Timeline (1990-2019) - I_Child")
    plt.savefig('continuous_timeline_child.png')

    return output_file

if __name__ == "__main__":
    generate_methodology_data_full()
