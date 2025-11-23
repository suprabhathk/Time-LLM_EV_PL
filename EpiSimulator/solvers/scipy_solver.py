"""
ODE Solver wrapper using scipy.integrate.solve_ivp
"""

import numpy as np
from scipy.integrate import solve_ivp


class ODESolver:
    """
    Wrapper for scipy ODE solvers to solve epidemic models.
    """
    
    def __init__(self, model, config):
        """
        Initialize ODE solver.
        
        Parameters:
        -----------
        model : BaseEpidemicModel
            Epidemic model instance
        config : dict
            Configuration dictionary with simulation parameters
        """
        self.model = model
        self.config = config
        
        # Extract simulation parameters
        sim_config = config['simulation']
        self.duration = sim_config['duration']
        self.output_frequency = sim_config.get('output_frequency', 1)
        
        # Solver settings
        solver_config = sim_config.get('solver', {})
        self.method = solver_config.get('method', 'RK45')
        self.rtol = solver_config.get('rtol', 1e-6)
        self.atol = solver_config.get('atol', 1e-8)
        
        # Validation settings
        val_config = config.get('validation', {})
        self.check_negative = val_config.get('check_negative_values', True)
        self.check_conservation = val_config.get('check_conservation', True)
    
    def solve(self):
        """
        Solve the ODE system.
        
        Returns:
        --------
        solution : dict
            Dictionary containing:
            - 't': time points
            - 'y': solution array (time x states)
            - 'success': whether solver succeeded
            - 'message': solver message
        """
        # Get initial conditions
        y0 = self.model.get_initial_conditions()
        
        # Validate initial conditions
        valid, msg = self.model.validate_state(y0)
        if not valid:
            raise ValueError(f"Invalid initial conditions: {msg}")
        
        # Check initial population conservation
        if self.check_conservation:
            self._check_population_conservation(y0, time=0)
        
        # Time span
        t_span = (0, self.duration)
        
        # Evaluation points (daily by default)
        t_eval = np.arange(0, self.duration + self.output_frequency, 
                          self.output_frequency)
        
        # Define ODE function with validation
        def ode_with_validation(t, y):
            # Check for negative values
            if self.check_negative and np.any(y < 0):
                # Set negative values to zero (prevents solver from crashing)
                y = np.maximum(y, 0)
            
            # Compute derivatives
            dydt = self.model.get_derivatives(t, y)
            
            return dydt
        
        print(f"Solving ODE system...")
        print(f"  Method: {self.method}")
        print(f"  Duration: {self.duration} days")
        print(f"  Output frequency: {self.output_frequency} day(s)")
        print(f"  Initial infected: {y0[2*self.model.n_age_groups:3*self.model.n_age_groups].sum():.0f}")
        
        # Solve ODE
        try:
            sol = solve_ivp(
                fun=ode_with_validation,
                t_span=t_span,
                y0=y0,
                method=self.method,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                vectorized=False
            )
            
            if not sol.success:
                print(f"WARNING: Solver did not complete successfully: {sol.message}")
            else:
                print(f"✓ Solver completed successfully")
                print(f"  Time points: {len(sol.t)}")
                print(f"  Final infected: {sol.y[2*self.model.n_age_groups:3*self.model.n_age_groups, -1].sum():.0f}")
            
            # Validate final state
            if self.check_negative:
                if np.any(sol.y < 0):
                    print("WARNING: Negative values detected in solution")
                    # Clip to zero
                    sol.y = np.maximum(sol.y, 0)
            
            # Check conservation at checkpoints
            if self.check_conservation:
                checkpoint_indices = [0, len(sol.t)//4, len(sol.t)//2, 
                                     3*len(sol.t)//4, -1]
                for idx in checkpoint_indices:
                    self._check_population_conservation(sol.y[:, idx], 
                                                        time=sol.t[idx])
            
            return {
                't': sol.t,
                'y': sol.y.T,  # Transpose to (time x states)
                'success': sol.success,
                'message': sol.message
            }
            
        except Exception as e:
            print(f"ERROR: Solver failed with exception: {str(e)}")
            raise
    
    def _check_population_conservation(self, y, time=0):
        """
        Check that total population is conserved (S + E + I + R = constant).
        
        Parameters:
        -----------
        y : np.ndarray
            State vector
        time : float
            Current time (for error message)
        """
        n = self.model.n_age_groups
        
        # Sum across all compartments for each age group
        for i in range(n):
            total = y[i] + y[n+i] + y[2*n+i] + y[3*n+i]  # S + E + I + R
            expected = self.model.population_sizes[i]
            
            relative_error = abs(total - expected) / expected
            
            if relative_error > 1e-4:  # 0.01% tolerance
                age_group = self.model.age_groups[i]
                print(f"WARNING: Population not conserved for {age_group} at t={time:.1f}")
                print(f"  Expected: {expected:.1f}, Got: {total:.1f}, Error: {relative_error:.2%}")
