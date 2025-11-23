import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from datetime import date, timedelta

# --- Configuration ---
# 45 years * ~365.25 days/year
SIMULATION_DURATION_DAYS = 365.25 * (2024 - 1980 + 1)
POPULATION = 100_000_000  # Total Population (N)
OUTPUT_FILENAME = 'ili_seasonal_sir_1980_2024_weekly.csv'

# Model Parameters
R0_AVG = 1.5           # Average Basic Reproduction Number (R0 = beta/gamma)
GAMMA = 1/7.0          # Recovery rate (1/Infectious Period in days, e.g., 7 days)
BETA_0 = R0_AVG * GAMMA  # Base Transmission Rate (beta_0)
ALPHA = 0.2            # Seasonality amplitude (20% variation)

# Initial Conditions (1980-01-01)
I0 = 1000             # Initial Infected
R0 = 100000           # Initial Recovered (Initial immunity)
S0 = POPULATION - I0 - R0 # Initial Susceptible

# --- SIR Model Definition ---

def seasonal_beta(t, beta_0, alpha, N_days_per_cycle=365.25):
    """
    Calculates the time-dependent transmission rate beta(t).
    Uses a cosine function to introduce seasonality, peaking around t=0 (start of year/winter).
    """
    # t is in days. The cosine argument ensures one cycle per year.
    return beta_0 * (1 + alpha * np.cos(2 * np.pi * t / N_days_per_cycle))

def sir_model(t, y, N, beta_0, gamma, alpha):
    """
    The ODE system for the Seasonal SIR model.
    y = [S, I, R]
    """
    S, I, R = y
    
    # Calculate the seasonal transmission rate
    beta_t = seasonal_beta(t, beta_0, alpha)
    
    # Differential equations
    dSdt = -beta_t * S * I / N
    dIdt = beta_t * S * I / N - gamma * I
    dRdt = gamma * I
    
    # Ensure S+I+R = N (conservation of mass, mainly for debugging/checks)
    # The sum dSdt + dIdt + dRdt should be close to zero.
    
    return [dSdt, dIdt, dRdt]

# --- Simulation and Output Generation ---

def run_simulation_and_save():
    """
    Runs the SIR model simulation and processes the results into weekly CSV output.
    """
    print("--- Starting Seasonal SIR Simulation ---")
    print(f"Total Duration: {SIMULATION_DURATION_DAYS:.0f} days (1980 to 2024)")
    print(f"Average R0: {R0_AVG:.2f} | Recovery Period: {1/GAMMA:.0f} days")
    print("-" * 40)
    
    # Initial state vector
    y0 = [S0, I0, R0]
    
    # Time points for ODE solution (daily step)
    t_span = (0, SIMULATION_DURATION_DAYS)
    
    # We use a daily step size (1.0) for high-fidelity integration
    t_points = np.arange(t_span[0], t_span[1], 1.0)

    # Solve the ODE system
    print("Solving ODE system (daily time points)...")
    sol = solve_ivp(
        fun=sir_model,
        t_span=t_span,
        y0=y0,
        t_eval=t_points,
        args=(POPULATION, BETA_0, GAMMA, ALPHA),
        method='RK45'
    )
    
    if not sol.success:
        print(f"ERROR: Solver failed: {sol.message}")
        return

    # Extract results and transpose to (Time, States)
    results_daily = sol.y.T
    time_daily = sol.t
    
    # --- Weekly Sampling and Date Generation ---
    
    # We want weekly data. Sample every 7th day (index 0, 7, 14, ...)
    WEEKLY_STEP = 7
    indices = np.arange(0, len(time_daily), WEEKLY_STEP)
    
    weekly_data = results_daily[indices]
    
    # Generate weekly dates starting from 1980-01-01
    start_date = date(1980, 1, 1)
    weekly_dates = [start_date + timedelta(days=int(t)) for t in time_daily[indices]]

    # Create DataFrame
    df = pd.DataFrame(weekly_data, columns=['S', 'I', 'R'])
    # RENAMED COLUMN TO 'date'
    df['date'] = weekly_dates
    
    # Reorder columns and ensure date is the first column
    # The default pandas to_csv format for date objects is YYYY-MM-DD
    df = df[['date', 'S', 'I', 'R']].round(0).astype({'S': 'int', 'I': 'int', 'R': 'int'})
    
    # Save to CSV
    df.to_csv(OUTPUT_FILENAME, index=False)
    
    print("-" * 40)
    print(f"✓ Simulation complete! Data saved to: {OUTPUT_FILENAME}")
    print(f"Generated {len(df)} weekly records.")
    print(f"First week: {df.iloc[0].to_dict()}")
    print(f"Last week: {df.iloc[-1].to_dict()}")


if __name__ == '__main__':
    # Check for required libraries
    try:
        import numpy as np
        import pandas as pd
        from scipy.integrate import solve_ivp
    except ImportError as e:
        print(f"Error: Required library missing. Please install it: {e}")
        print("Required libraries: numpy, pandas, scipy")
        print("You can install them using: pip install numpy pandas scipy")
        exit(1)
        
    run_simulation_and_save()
