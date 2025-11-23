"""
H1N1 Age-Structured SEIR Model with Time-Varying Contact Patterns

Based on: Eames et al. (2012) - "Measured Dynamic Social Contact Patterns 
Explain the Spread of H1N1v Influenza"

Model equations:
    dSi/dt = -τ × Si × Σj [Bi,j(t) × Ij/nj]
    dEi/dt = τ × Si × Σj [Bi,j(t) × Ij/nj] - ν × Ei
    dIi/dt = ν × Ei - γ × Ii
    dRi/dt = γ × Ii
"""

import numpy as np
from .base_model import BaseEpidemicModel


class H1N1AgeStructuredModel(BaseEpidemicModel):
    """
    Age-structured SEIR model with time-varying contact patterns.
    
    Contact patterns switch between term time and holidays based on
    school calendar.
    """
    
    def __init__(self, config):
        """
        Initialize H1N1 SEIR model.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing:
            - disease_params: transmission_rate, latent_period, infectious_period
            - population: age_groups, sizes
            - contact_patterns: term_time, holidays
            - school_calendar: term_periods, holiday_periods
            - initial_conditions: susceptible, exposed, infected, recovered
        """
        super().__init__(config)
        
        # Extract parameters
        self.transmission_rate = config['disease_params']['transmission_rate']
        self.latent_period = config['disease_params']['latent_period']
        self.infectious_period = config['disease_params']['infectious_period']
        
        # Calculate rates
        self.nu = 1.0 / self.latent_period  # E -> I rate
        self.gamma = 1.0 / self.infectious_period  # I -> R rate
        
        # Population structure
        self.age_groups = config['population']['age_groups']
        self.population_sizes = np.array(config['population']['sizes'])
        self.n_age_groups = len(self.age_groups)
        
        # Contact matrices
        self.contact_matrix_term = np.array(config['contact_patterns']['term_time'])
        self.contact_matrix_holiday = np.array(config['contact_patterns']['holidays'])
        
        # School calendar
        self.term_periods = config['school_calendar']['term_periods']
        self.holiday_periods = config['school_calendar']['holiday_periods']
        
        # Initial conditions
        self.initial_S = np.array(config['initial_conditions']['susceptible'])
        self.initial_E = np.array(config['initial_conditions']['exposed'])
        self.initial_I = np.array(config['initial_conditions']['infected'])
        self.initial_R = np.array(config['initial_conditions']['recovered'])
    
    def validate_config(self):
        """Validate configuration parameters."""
        required_keys = ['disease_params', 'population', 'contact_patterns', 
                        'school_calendar', 'initial_conditions']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate disease parameters are positive
        dp = self.config['disease_params']
        if dp['transmission_rate'] <= 0:
            raise ValueError("transmission_rate must be positive")
        if dp['latent_period'] <= 0:
            raise ValueError("latent_period must be positive")
        if dp['infectious_period'] <= 0:
            raise ValueError("infectious_period must be positive")
        
        # Validate population sizes
        pop_sizes = self.config['population']['sizes']
        if any(p <= 0 for p in pop_sizes):
            raise ValueError("All population sizes must be positive")
        
        # Validate contact matrices are square and match age groups
        n_groups = len(self.config['population']['age_groups'])
        term_matrix = np.array(self.config['contact_patterns']['term_time'])
        holiday_matrix = np.array(self.config['contact_patterns']['holidays'])
        
        if term_matrix.shape != (n_groups, n_groups):
            raise ValueError(f"Term contact matrix must be {n_groups}x{n_groups}")
        if holiday_matrix.shape != (n_groups, n_groups):
            raise ValueError(f"Holiday contact matrix must be {n_groups}x{n_groups}")
        
        # Validate initial conditions sum to population
        ic = self.config['initial_conditions']
        for i, age_group in enumerate(self.config['population']['age_groups']):
            total = (ic['susceptible'][i] + ic['exposed'][i] + 
                    ic['infected'][i] + ic['recovered'][i])
            expected = self.config['population']['sizes'][i]
            if abs(total - expected) > 1e-6:
                raise ValueError(
                    f"Initial conditions for {age_group} don't sum to population size: "
                    f"{total} != {expected}"
                )
    
    def get_contact_matrix(self, t):
        """
        Get the contact matrix for time t based on school calendar.
        
        Parameters:
        -----------
        t : float
            Current time (in days)
            
        Returns:
        --------
        B : np.ndarray
            Contact matrix (term_time or holidays)
        """
        # Check if current day is in term time or holidays
        day = int(t)
        
        # Check if in any term period
        for start, end in self.term_periods:
            if start <= day <= end:
                return self.contact_matrix_term
        
        # Check if in any holiday period
        for start, end in self.holiday_periods:
            if start <= day <= end:
                return self.contact_matrix_holiday
        
        # Default to term time if not specified
        return self.contact_matrix_term
    
    def get_derivatives(self, t, y):
        """
        Compute dy/dt for the SEIR model.
        
        State vector y is organized as:
        [S_0, S_1, ..., E_0, E_1, ..., I_0, I_1, ..., R_0, R_1, ...]
        
        Parameters:
        -----------
        t : float
            Current time
        y : np.ndarray
            State vector [S, E, I, R] for all age groups
            
        Returns:
        --------
        dydt : np.ndarray
            Derivatives
        """
        n = self.n_age_groups
        
        # Extract compartments
        S = y[0:n]
        E = y[n:2*n]
        I = y[2*n:3*n]
        R = y[3*n:4*n]
        
        # Get contact matrix for current time
        B = self.get_contact_matrix(t)
        
        # Initialize derivatives
        dS = np.zeros(n)
        dE = np.zeros(n)
        dI = np.zeros(n)
        dR = np.zeros(n)
        
        # Compute force of infection for each age group
        # lambda_i = τ × Σj [Bi,j × Ij/nj]
        force_of_infection = np.zeros(n)
        for i in range(n):
            for j in range(n):
                force_of_infection[i] += B[i, j] * I[j] / self.population_sizes[j]
            force_of_infection[i] *= self.transmission_rate
        
        # SEIR equations for each age group
        for i in range(n):
            # S -> E (infection)
            new_infections = force_of_infection[i] * S[i]
            
            # E -> I (becoming infectious)
            new_infectious = self.nu * E[i]
            
            # I -> R (recovery)
            new_recovered = self.gamma * I[i]
            
            # Update derivatives
            dS[i] = -new_infections
            dE[i] = new_infections - new_infectious
            dI[i] = new_infectious - new_recovered
            dR[i] = new_recovered
        
        # Concatenate all derivatives
        dydt = np.concatenate([dS, dE, dI, dR])
        
        return dydt
    
    def get_initial_conditions(self):
        """
        Get initial state vector.
        
        Returns:
        --------
        y0 : np.ndarray
            Initial conditions [S, E, I, R] for all age groups
        """
        y0 = np.concatenate([
            self.initial_S,
            self.initial_E,
            self.initial_I,
            self.initial_R
        ])
        return y0
    
    def get_state_names(self):
        """
        Get names of all state variables.
        
        Returns:
        --------
        names : list of str
            State variable names
        """
        names = []
        compartments = ['S', 'E', 'I', 'R']
        
        for comp in compartments:
            for age_group in self.age_groups:
                names.append(f"{comp}_{age_group}")
        
        return names
    
    def get_age_groups(self):
        """Get age group labels."""
        return self.age_groups
    
    def unpack_solution(self, y):
        """
        Unpack solution array into compartments.
        
        Parameters:
        -----------
        y : np.ndarray
            State vector or solution array (time x states)
            
        Returns:
        --------
        compartments : dict
            Dictionary with keys 'S', 'E', 'I', 'R'
            Each value is array of shape (time, n_age_groups) or (n_age_groups,)
        """
        n = self.n_age_groups
        
        if y.ndim == 1:
            # Single time point
            return {
                'S': y[0:n],
                'E': y[n:2*n],
                'I': y[2*n:3*n],
                'R': y[3*n:4*n]
            }
        else:
            # Multiple time points
            return {
                'S': y[:, 0:n],
                'E': y[:, n:2*n],
                'I': y[:, 2*n:3*n],
                'R': y[:, 3*n:4*n]
            }
