# EpiSimulator - H1N1 Age-Structured SEIR Epidemic Simulator

This simulator implements the age-structured SEIR model from:
**Eames et al. (2012)** - "Measured Dynamic Social Contact Patterns Explain the Spread of H1N1v Influenza"

## Model Overview

### SEIR Framework

Each person progresses through 4 disease states: Susceptible (S) → Exposed (E) → Infectious (I) → Recovered (R)

- **Susceptible (S)**: Healthy, can get infected
- **Exposed (E)**: Infected but not yet contagious (incubation period)
- **Infectious (I)**: Can transmit disease to others
- **Recovered (R)**: Immune, cannot be reinfected

### Age Structure

Population divided into **2 age groups**:
- **Group 0**: Ages 0-18 (children and students)
- **Group 1**: Ages 19-65+ (adults and elderly)

---

## Mathematical Model

For each age group **i** (i = 0, 1):
dSi/dt = -τ × Si × Σj [Bi,j(t) × Ij/nj]

dEi/dt = τ × Si × Σj [Bi,j(t) × Ij/nj] - ν × Ei

dIi/dt = ν × Ei - γ × Ii

dRi/dt = γ × Ii


### Parameters:

- **τ (tau)**: Transmission rate per contact
- **ν (nu)**: Rate from Exposed → Infectious (1/latent_period)
- **γ (gamma)**: Rate from Infectious → Recovered (1/infectious_period)
- **Bi,j(t)**: Time-varying contact matrix (changes with school calendar)
- **Ij/nj**: Proportion of age group j that is infectious

### Key Innovation: Time-Varying Contacts

**B(t) switches between two matrices:**
- **B^T**: Contact matrix during school term time
- **B^H**: Contact matrix during school holidays

This temporal variation captures epidemic waves driven by school schedules.

---

## Default Parameters

### Disease Dynamics

| Parameter | Value | Description |
|-----------|-------|-------------|
| **transmission_rate (τ)** | 0.04 | Probability of transmission per contact |
| **latent_period** | 1.0 days | Incubation period (E state duration) |
| **infectious_period** | 1.8 days | Infectious period (I state duration) |
| **nu (ν)** | 1.0 day⁻¹ | 1/latent_period |
| **gamma (γ)** | 0.556 day⁻¹ | 1/infectious_period |

### Population Structure

| Age Group | Population | Percentage |
|-----------|-----------|-----------|
| **0-18** | 2,000 | 20% |
| **19-65+** | 8,000 | 80% |
| **Total** | 10,000 | 100% |

### Contact Matrices (2x2)

**Term Time Contact Matrix (B^T):**

            0-18    19-65+
0-18     [ 25.0     11.0  ]
19-65+   [ 3.5      14.5  ]


**Key observation**: Children make **56% fewer contacts** with each other during holidays (25.0 → 11.0)!

### School Calendar (Example)

| Period | Days | Type |
|--------|------|------|
| Term 1 | 1-100 | Term time (B^T) |
| Holiday 1 | 101-129 | School holiday (B^H) |
| Term 2 | 130-250 | Term time (B^T) |
| Holiday 2 | 251-279 | School holiday (B^H) |
| Term 3 | 280-365 | Term time (B^T) |

### Initial Conditions (Day 0)

| Age Group | S | E | I | R | Total |
|-----------|---|---|---|---|-------|
| **0-18** | 1,800 | 0 | 10 | 190 | 2,000 |
| **19-65+** | 7,200 | 0 | 0 | 800 | 8,000 |
| **Total** | **9,000** | **0** | **10** | **1,000** | **10,000** |

- 90% of population susceptible
- 10% has prior immunity
- 10 initial infections seeded in children

### Simulation Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **duration** | 365 days | Simulation length (1 year) |
| **timestep** | 0.1 days | Integration step size |
| **output_frequency** | 1 day | Save data daily |
| **solver** | RK45 | ODE solver (scipy) |

---

## Output Format

Generated CSV files will be compatible with TimeLLM forecasting.

### CSV Structure:

```csv
date,S_0-18,E_0-18,I_0-18,R_0-18,S_19-65+,E_19-65+,I_19-65+,R_19-65+,total_I,total_R
2024-01-01,1800,0,10,190,7200,0,0,800,10,990
2024-01-02,1795,3,12,190,7198,1,1,800,13,990
2024-01-03,1788,8,15,189,7194,3,3,800,18,989
...
