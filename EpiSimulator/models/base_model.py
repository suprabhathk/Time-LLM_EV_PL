"""
Base abstract class for epidemic models.
All epidemic models should inherit from this class.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEpidemicModel(ABC):
    """
    Abstract base class for epidemic models.
    
    All epidemic models must implement:
    - get_derivatives(): ODE system equations
    - get_initial_conditions(): Initial state
    - get_state_names(): Names of compartments
    """
    
    def __init__(self, config):
        """
        Initialize the epidemic model.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with model parameters
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self):
        """
        Validate that the configuration has all required parameters.
        Raise ValueError if configuration is invalid.
        """
        pass
    
    @abstractmethod
    def get_derivatives(self, t, y):
        """
        Compute derivatives dy/dt for the ODE system.
        
        Parameters:
        -----------
        t : float
            Current time
        y : array-like
            Current state vector
            
        Returns:
        --------
        dydt : np.ndarray
            Derivatives of state variables
        """
        pass
    
    @abstractmethod
    def get_initial_conditions(self):
        """
        Get initial conditions for all compartments.
        
        Returns:
        --------
        y0 : np.ndarray
            Initial state vector
        """
        pass
    
    @abstractmethod
    def get_state_names(self):
        """
        Get names of all state variables.
        
        Returns:
        --------
        names : list of str
            Names of compartments (e.g., ['S_0-18', 'E_0-18', ...])
        """
        pass
    
    @abstractmethod
    def get_age_groups(self):
        """
        Get age group labels.
        
        Returns:
        --------
        age_groups : list of str
            Age group names (e.g., ['0-18', '19-65+'])
        """
        pass
    
    def validate_state(self, y):
        """
        Validate that the state vector has valid values.
        
        Parameters:
        -----------
        y : np.ndarray
            State vector to validate
            
        Returns:
        --------
        valid : bool
            True if state is valid
        message : str
            Error message if invalid
        """
        # Check for negative values
        if np.any(y < 0):
            neg_indices = np.where(y < 0)[0]
            state_names = self.get_state_names()
            neg_names = [state_names[i] for i in neg_indices]
            return False, f"Negative values in compartments: {neg_names}"
        
        # Check for NaN or Inf
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return False, "NaN or Inf values detected in state"
        
        return True, ""
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
        --------
        info : dict
            Model information
        """
        return {
            'model_type': self.__class__.__name__,
            'n_compartments': len(self.get_state_names()),
            'n_age_groups': len(self.get_age_groups()),
            'state_names': self.get_state_names(),
            'age_groups': self.get_age_groups()
        }
