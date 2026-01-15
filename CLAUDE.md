# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TimeLLM is a time series forecasting framework that reprograms Large Language Models for time series analysis. This repository extends TimeLLM with custom epidemic modeling capabilities, integrating SEIR and SIR epidemic simulators with LLM-based forecasting.

The project combines:
- **TimeLLM**: Patch-based time series forecasting using frozen LLM backbones
- **EpiSimulator**: Modular epidemic model simulator (SEIR, SIR variants)
- **Multi-LLM Support**: LLAMA, GPT-2, BERT, and GEMMA architectures

## Core Architecture

### TimeLLM Model (`models/TimeLLM.py`)

The model uses a reprogramming approach:
1. **Patch Embedding**: Time series divided into patches (configurable `patch_len` and `stride`)
2. **Reprogramming Layer**: Maps time series patches to LLM embedding space
3. **Frozen LLM**: Pretrained language model processes the reprogrammed inputs
4. **Output Projection**: FlattenHead projects LLM outputs back to time series predictions

**Supported LLM Backbones:**
- `LLAMA`: LlamaModel, dim: 4096, layers: 32
- `GPT2`: GPT2Model, dim: 768, layers: 12
- `BERT`: BertModel, dim: 768, layers: 12
- `GEMMA`: GemmaModel (google/gemma-3-270m), dim: 640, layers: 28

Each LLM is loaded with frozen parameters and configurable layer depth via `--llm_layers`.

### EpiSimulator (`EpiSimulator/`)

Modular epidemic model simulator with ODE-based implementations:

**Architecture:**
```
EpiSimulator/
├── models/              # Epidemic model implementations
│   ├── base_model.py           # Abstract base class
│   ├── h1n1_age_structured.py  # 2-age-group SEIR
│   └── ili_sir.py              # Simple SIR (1980-2024 ILI)
├── configs/             # YAML model configurations
├── solvers/             # ODE solver wrappers (scipy)
├── generators/          # CSV time series generation
└── run_simulator.py     # Main entry with MODEL_MAPPING
```

**Current Models:**
- **H1N1_SEIR**: Age-structured (0-18, 19-65+) SEIR with time-varying contact matrices
- **ILI_SIR**: Simple SIR for Influenza-Like Illness data

Output format: CSV with date column and compartment values (S, E, I, R per age group).

### Data Pipeline

1. EpiSimulator generates synthetic epidemic CSV data
2. `data_provider/` loads CSV with `Dataset_Custom` (70/20/10 train/val/test split)
3. Domain-specific prompts from `dataset/prompt_bank/` provide context to LLM
4. TimeLLM forecasts future epidemic states
5. Metrics (MAE, MAPE, RMSE) computed via `run_inference.py`

## Development Commands

### Environment Setup

```bash
pip install -r requirements.txt
# Python 3.11 recommended
# Key deps: torch==2.2.2, transformers==4.31.0, accelerate==0.28.0
```

### Training

**Single GPU:**
```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimeLLM \
  --data Weather \
  --llm_model GEMMA \
  --llm_dim 640 \
  --llm_layers 6 \
  --seq_len 48 --pred_len 12 \
  --batch_size 8 --train_epochs 10
```

**Multi-GPU with Accelerate:**
```bash
accelerate launch --multi_gpu --mixed_precision bf16 \
  --num_processes 8 run_main.py [args...]
```

**Using Provided Scripts:**
```bash
bash ./scripts/TimeLLM_Weather.sh
bash ./scripts/TimeLLM_ETTh1.sh
bash ./code_run_main_inference/sir_run_main.sh  # Epidemic models
```

### Inference

```bash
python run_inference.py \
  --checkpoint_path ./checkpoints/[model]/checkpoint \
  --data Weather \
  --llm_model GEMMA \
  --output_path ./results/inference
```

Output: `predictions.npy`, `true_values.npy`, `metrics.txt`, `per_horizon_metrics.csv`

### Epidemic Simulation

```bash
# Generate synthetic epidemic data
cd EpiSimulator
python run_simulator.py --config configs/ili_sir_1980_2024.yaml

# Train TimeLLM on epidemic data
cd ..
bash code_run_main_inference/sir_run_main.sh

# Run inference
bash code_run_main_inference/sir_run_inference.sh
```

## Key Configuration

### Model Parameters

- `--seq_len`: Input sequence length (e.g., 28 weeks for epidemic forecasting)
- `--label_len`: Start token length (e.g., 14 weeks)
- `--pred_len`: Prediction horizon (e.g., 14 weeks)
- `--patch_len`: Patch size (e.g., 7 for weekly data)
- `--stride`: Patch stride (e.g., 4)
- `--llm_model`: LLM backend (LLAMA, GPT2, BERT, GEMMA)
- `--llm_dim`: LLM hidden dimension (must match model: 4096, 768, 640)
- `--llm_layers`: Number of LLM layers to use (1-32, fewer for efficiency)
- `--d_model`: TimeLLM embedding dimension (16-32)
- `--d_ff`: Feed-forward dimension (32-128)
- `--prompt_domain`: 0=generic, 1=use dataset-specific prompt from prompt_bank

### Dataset Types

- **ETT**: Electricity Transformer Temperature (ETTh1, ETTh2, ETTm1, ETTm2)
- **Weather**: 21 meteorological indicators, 10-minute intervals
- **Traffic**: Road occupancy rates
- **ECL**: Electricity consuming load
- **M4**: M4 competition dataset
- **Epi_SEIR**: Custom epidemic SEIR/SIR models

Registered in `data_provider/data_factory.py:data_dict`.

### File Locations

- **Checkpoints**: `./checkpoints/[experiment_name]/checkpoint`
- **Results**: `./results/[experiment_name]/`
- **Epidemic Data**: `./EpiSimulator/dataset/synthetic/[model_name]/`
- **Prompts**: `./dataset/prompt_bank/[dataset_name].txt`
- **Scripts**: `./scripts/` (standard datasets), `./code_run_main_inference/` (epidemic models)

## Extending the Codebase

### Adding New Epidemic Models

To add a new epidemic model to EpiSimulator (e.g., SIRS, SEIRD, age-structured variants):

**1. Create Model Class** (`EpiSimulator/models/your_model.py`)

Inherit from `BaseEpidemicModel` and implement 5 required methods:

```python
from .base_model import BaseEpidemicModel
import numpy as np

class YourModel(BaseEpidemicModel):
    def __init__(self, config):
        super().__init__(config)  # Calls validate_config()
        self.beta = config['disease_params']['transmission_rate']
        # Extract other parameters...

    def validate_config(self):
        """Validate config has all required keys."""
        required = ['disease_params', 'population', 'initial_conditions']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing {key}")

    def get_derivatives(self, t, y):
        """
        Compute dy/dt for ODE system.
        Args: t (float), y (np.ndarray) - state vector
        Returns: dydt (np.ndarray)
        """
        # Implement your model equations
        pass

    def get_initial_conditions(self):
        """Return initial state vector y0."""
        return np.array([S0, E0, I0, R0, ...])

    def get_state_names(self):
        """Return list of state variable names."""
        return ['S_group1', 'E_group1', 'I_group1', 'R_group1', ...]

    def get_age_groups(self):
        """Return age group labels."""
        return self.age_groups  # From config
```

**2. Create YAML Configuration** (`EpiSimulator/configs/your_model.yaml`)

Required structure:
```yaml
model_name: "your_model"
disease_params:
  transmission_rate: 0.2
  recovery_rate: 0.1
  # Add custom parameters
population:
  age_groups: ['group1', 'group2']
  sizes: [5000, 5000]
initial_conditions:
  susceptible: [4990, 4990]
  infected: [10, 10]
  recovered: [0, 0]
simulation:
  duration: 365
  timestep: 0.1
  output_frequency: 1
  solver: {method: "RK45"}
output:
  path: "./dataset/synthetic/your_model/"
  filename: "output.csv"
  start_date: "2024-01-01"
```

**3. Register in MODEL_MAPPING** (`EpiSimulator/run_simulator.py`)

```python
from models.your_model import YourModel

MODEL_MAPPING = {
    'H1N1_SEIR': H1N1AgeStructuredModel,
    'ILI_SIR': ILI_SIR_Model,
    'YOUR_KEY': YourModel,  # Add this line
}
```

**4. Run Simulator**
```bash
python EpiSimulator/run_simulator.py --config EpiSimulator/configs/your_model.yaml
```

**Output Format**: CSV with columns `date,S_group1,E_group1,I_group1,R_group1,...` compatible with `Dataset_Custom`.

**Key Requirements:**
- State vector must match `len(get_state_names())`
- Population conservation: S + E + I + R = constant (checked by validator)
- Time-varying parameters: Use `t` parameter in `get_derivatives()`

### Adding New LLM Architectures

To add a new LLM backbone (e.g., T5, Falcon, Mistral) beyond the current LLAMA/GPT2/BERT/GEMMA:

**1. Import in TimeLLM.py** (`models/TimeLLM.py` lines 6-7)

```python
from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    GemmaConfig, GemmaModel, GemmaTokenizer,
    # Add your model:
    YourModelConfig, YourModel, YourModelTokenizer
)
```

**2. Add Conditional Block** (`models/TimeLLM.py` lines 43-191)

```python
elif configs.llm_model == 'YOUR_MODEL':
    # Load config
    self.your_config = YourModelConfig.from_pretrained('org/model-name')
    self.your_config.num_hidden_layers = configs.llm_layers
    self.your_config.output_attentions = True
    self.your_config.output_hidden_states = True

    # Load model (try local first, then download)
    try:
        self.llm_model = YourModel.from_pretrained(
            'org/model-name',
            trust_remote_code=True,
            local_files_only=True,
            config=self.your_config,
        )
    except EnvironmentError:
        print("Downloading model...")
        self.llm_model = YourModel.from_pretrained(
            'org/model-name',
            trust_remote_code=True,
            local_files_only=False,
            config=self.your_config,
        )

    # Load tokenizer
    self.tokenizer = YourModelTokenizer.from_pretrained('org/model-name')
```

**3. Update Parser** (`run_main.py` line 83-84)

```python
parser.add_argument('--llm_model', type=str, default='LLAMA',
                    help='LLAMA, GPT2, BERT, GEMMA, YOUR_MODEL')
parser.add_argument('--llm_dim', type=int, default='4096',
                    help='LLama:4096; GPT2:768; BERT:768; Gemma:640; YourModel:XXX')
```

**4. Determine Model Dimensions**

You need three key parameters:
- **llm_dim**: `model.config.hidden_size` (e.g., 1024, 2048, 4096)
- **llm_layers**: `model.config.num_hidden_layers` (use fewer for efficiency)
- **vocab_size**: Automatically extracted from embeddings

**5. Create Training Script**

```bash
python run_main.py \
  --llm_model YOUR_MODEL \
  --llm_dim 1024 \
  --llm_layers 8 \
  --data Epi_SEIR \
  --seq_len 512 --pred_len 96
```

**Critical**: `llm_dim` MUST match the model's actual hidden dimension or initialization will fail.

### Adding New Dataset Types

**1. Register in data_factory.py** (`data_provider/data_factory.py`)

```python
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'Weather': Dataset_Custom,
    'Epi_SEIR': Dataset_Custom,
    'YourDataset': Dataset_Custom,  # Add this
}
```

**2. Create Prompt File** (`dataset/prompt_bank/YourDataset.txt`)

Plain text description used as LLM context:
```
This is [description of your data]. The data contains [frequency]
measurements of [variables] with dynamics driven by [key features].
```

**3. Update load_content()** (`utils/tools.py` line 232)

```python
def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    elif args.data == 'Epi_SEIR':
        file = 'Epi_SEIR'
    elif args.data == 'YourDataset':  # Add this
        file = 'YourDataset'
    else:
        file = args.data
```

**CSV Format**: Must have `date` column for `Dataset_Custom`. Example:
```csv
date,variable1,variable2
2024-01-01,123.45,67.89
2024-01-02,124.56,68.01
```

## Project Structure

```
TimeLLM-forecasting/
├── models/                  # TimeLLM, Autoformer, DLinear implementations
├── layers/                  # Embedding, attention, normalization layers
├── data_provider/           # Dataset loaders and data factory
├── utils/                   # Tools, metrics, time features
├── scripts/                 # Training scripts for standard datasets
├── dataset/
│   ├── prompt_bank/        # Domain-specific LLM prompts
│   └── synthetic/          # Generated epidemic data
├── EpiSimulator/           # Epidemic model simulator
│   ├── models/             # BaseEpidemicModel implementations
│   ├── configs/            # YAML configurations
│   ├── solvers/            # ODE solver wrappers
│   └── run_simulator.py    # Main simulator entry
├── code_run_main_inference/ # Epidemic-specific training/inference scripts
├── checkpoints/            # Trained model checkpoints
├── results/                # Inference results and metrics
├── run_main.py             # Main training script
├── run_inference.py        # Inference with metrics
└── requirements.txt        # Dependencies
```

## Common Pitfalls

1. **Incorrect llm_dim**: Must exactly match the model's hidden dimension
2. **Missing validation in epidemic models**: Negative populations cause solver crashes
3. **Forgot MODEL_MAPPING registration**: Results in KeyError at runtime
4. **Wrong CSV format**: `Dataset_Custom` requires 'date' column
5. **Population not conserved**: Check ODE equations ensure S+E+I+R=constant
6. **Prompt file missing**: Falls back to generic prompt, reducing forecast quality
