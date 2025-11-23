import numpy as np
from .base_model import BaseEpidemicModel 

class ILI_SIR_Model(BaseEpidemicModel):
    """
    Implements the standard Susceptible-Infectious-Recovered (SIR) model,
    reading parameters from H1N1-style config keys (disease_params, etc.).
    """
    
    def __init__(self, config):
        
        # --- Load Parameters from new config structure ---
        dp = config['disease_params']
        pop = config['population']
        ic = config['initial_conditions']
        
        # SIR Rates
        self.beta = dp['transmission_rate']
        self.gamma = dp['recovery_rate']
        
        # Population and Age Groups
        self.age_groups = pop['age_groups']
        self.total_pop = pop['sizes'][0] 
        
        # Initial Conditions (stored as single values since SIR is a single group)
        self.initial_S = ic['susceptible'][0]
        self.initial_I = ic['infected'][0]
        self.initial_R = ic['recovered'][0]
        
        # The parent constructor (BaseEpidemicModel) calls validate_config()
        super().__init__(config) 
        
    def validate_config(self):
        """Validate that the configuration has all required SIR parameters."""
        # Check required keys based on the H1N1-style structure
        required_keys = ['disease_params', 'population', 'initial_conditions']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Config must include '{key}' key.")
        
        # Check parameter values
        if self.beta <= 0 or self.gamma <= 0:
            raise ValueError("transmission_rate and recovery_rate must be positive.")
        
        # Check that initial conditions sum up to the population
        total_initial = self.initial_S + self.initial_I + self.initial_R
        if abs(total_initial - self.total_pop) > 1e-6:
             raise ValueError(f"Initial conditions ({total_initial}) do not sum up exactly to total population ({self.total_pop}).")
        
    def get_derivatives(self, t, y):
        """
        Compute derivatives dy/dt for the SIR ODE system. 
        y = [S, I, R]
        """
        S, I, R = y
        N = self.total_pop
        
        # Differential equations for SIR
        dSdt = -self.beta * S * I / N
        dIdt = (self.beta * S * I / N) - (self.gamma * I)
        dRdt = self.gamma * I
        
        return np.array([dSdt, dIdt, dRdt])

    def get_initial_conditions(self):
        """
        Get initial conditions for all compartments: [S, I, R].
        """
        y0 = np.array([
            self.initial_S,
            self.initial_I,
            self.initial_R
        ])
        return y0
    
    def get_state_names(self):
        """
        Get names of all state variables: ['S_all_ages', 'I_all_ages', 'R_all_ages'].
        We append the age group to match the H1N1 output format.
        """
        # SIR model is not age-structured, but use the group name for compatibility
        group = self.age_groups[0] 
        return [f'S_{group}', f'I_{group}', f'R_{group}']
    
    def get_age_groups(self):
        """
        Get age group labels. (Required abstract method)
        """
        return self.age_groups
