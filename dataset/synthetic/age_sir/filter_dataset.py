import pandas as pd

# Read original CSV
df = pd.read_csv('./dataset/synthetic/age_sir/age_sir_all_vars.csv')

# Filter to cutoff date
cutoff_date = '2024-03-26'
df_filtered = df[df['date'] <= cutoff_date]

# Save filtered dataset
output_path = './dataset/synthetic/age_sir/age_sir_all_vars_filtered.csv'
df_filtered.to_csv(output_path, index=False)

# Print summary
print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(df_filtered)}")
print(f"Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
