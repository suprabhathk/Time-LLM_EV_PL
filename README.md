# Time-LLM: Stochastic Epidemic Forecasting (Age-Structured SIR)

This repository is a specialized implementation of **Time-LLM** (Large Language Models for Time Series Forecasting) applied to synthetic epidemic data. We use a frozen LLM backbone to predict the trajectory of infectious disease outbreaks based on a stochastic, age-structured SIR engine.

## 📊 Methodology: Stochastic Simulation
The training data is generated using an **Euler-Maruyama** numerical solver for Stochastic Differential Equations (SDEs).

### Key Simulation Features:
* **Temporal Resolution:** Simulated at a fine-grained step of $dt=0.1$ days to ensure numerical stability and captured at **weekly snapshots** for Time-LLM compatibility.
* **Age-Structure:** Population is divided into **Children (C)** and **Adults (A)** with a custom contact matrix $M$ defining inter-group interactions.
* **Seasonality:** Transmission rate $\beta(t)$ is modulated by a cosine-driven seasonal factor to simulate annual winter peaks.
* **Wiener Noise:** Demographic stochasticity is modeled using four independent Wiener process channels, ensuring realistic "jitter" and peak variability.

## 🛠️ Repository Structure
* `models/`: Contains the **Time-LLM** architecture and the LLM-reprogramming layers.
* `data_provider/`: Custom data loaders for the 30-year epidemic CSV.
* `generate_methodology_data.py`: The stochastic SIR engine used to create the training/testing sets.
* `run_main.py`: The entry point for training and evaluation.

## 🚀 Getting Started

### 1. Installation
Clone your fork and install dependencies:
```bash
git clone [https://github.com/suprabhathk/Time-LLM_EV_PL.git](https://github.com/suprabhathk/Time-LLM_EV_PL.git)
cd Time-LLM_EV_PL
pip install -r requirements.txt
