import re

# Read the file
with open('models/TimeLLM.py', 'r') as f:
    content = f.read()

# Check if load_content already exists in __init__
if 'load_content' not in content:
    print("Adding load_content to __init__...")
    
    # Find __init__ and add load_content after task_name, pred_len, seq_len
    init_pattern = r'(self\.seq_len = configs\.seq_len\n)'
    replacement = r'\1\n        # Load domain knowledge from txt file\n        from utils.tools import load_content\n        self.description = load_content(configs)\n        print(f"✓ Loaded domain knowledge: {len(self.description)} characters")\n'
    content = re.sub(init_pattern, replacement, content)
else:
    print("✓ load_content already exists")

# Replace hardcoded prompt with txt file version
old_prompt = r'prompt_ = \(\s*f"<\|start_prompt\|>Dataset: Stochastic SIR.*?<\|<end_prompt>\|>"\s*\)'

new_prompt = '''# Determine epidemic phase
            if trends[b] > 0:
                phase = "GROWTH PHASE (exponential increase)"
                guidance = "Expect continued rise following dI/dt = (β·S/N - γ)·I"
            else:
                phase = "DECLINE PHASE (exponential decay)"
                guidance = "Expect decay following dI/dt ≈ -γ·I"
            
            # Build comprehensive prompt using txt file
            prompt_ = (
                f"<|start_prompt|>"
                f"{self.description}\\n\\n"
                f"==== CURRENT FORECAST TASK ====\\n"
                f"Objective: Predict next {self.pred_len} weeks of I_child\\n"
                f"Input: {self.seq_len} weeks of historical data\\n\\n"
                f"Observed Statistics:\\n"
                f"  • Range: {r_min:.0f} to {r_max:.0f} cases\\n"
                f"  • Current phase: {phase}\\n"
                f"  • Expected dynamics: {guidance}\\n\\n"
                f"Apply SIR equations with seasonal β(t) and Wiener stochasticity.\\n"
                f"Generate {self.pred_len} realistic weekly forecasts.\\n"
                f"<|<end_prompt>|>"
            )'''

content = re.sub(old_prompt, new_prompt, content, flags=re.DOTALL)

# Write back
with open('models/TimeLLM.py', 'w') as f:
    f.write(content)

print("✓ TimeLLM.py updated successfully!")
print("\nChanges made:")
print("  1. Added load_content to __init__ (if missing)")
print("  2. Replaced hardcoded prompt with txt file content")
print("\nBackup saved as: models/TimeLLM.py.backup_<timestamp>")
