"""
Time Series Generator - Converts ODE solutions to CSV format for TimeLLM
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from utils.visualization import SimulationVisualizer



class TimeSeriesGenerator:
    """
    Generates TimeLLM-compatible CSV files from epidemic model solutions.
    """
    
    def __init__(self, model, solution, config):
        """
        Initialize time series generator.
        
        Parameters:
        -----------
        model : BaseEpidemicModel
            Epidemic model instance
        solution : dict
            Solution from ODESolver with keys 't' and 'y'
        config : dict
            Configuration dictionary
        """
        self.model = model
        self.solution = solution
        self.config = config
        
        # Extract output settings
        output_config = config.get('output', {})
        self.output_path = output_config.get('path', './output/')
        self.output_filename = output_config.get('filename', 'epidemic_data.csv')
        self.save_metadata = output_config.get('save_metadata', True)
        self.metadata_filename = output_config.get('metadata_filename', 
                                                   'epidemic_metadata.json')
        
        # Date settings
        self.start_date = output_config.get('start_date', '2024-01-01')
        self.date_format = output_config.get('date_format', '%Y-%m-%d')
        
        # What to include in output
        self.save_compartments = output_config.get('save_compartments', True)
        self.save_totals = output_config.get('save_totals', True)
        self.save_incidence = output_config.get('save_incidence', True)
        self.save_prevalence = output_config.get('save_prevalence', True)
    
    def generate_csv(self):
        """
        Generate CSV file from solution.
        
        Returns:
        --------
        output_file : str
            Path to generated CSV file
        """
        print(f"\nGenerating time series CSV...")
        
        # Create output directory if needed
        os.makedirs(self.output_path, exist_ok=True)
        
        # Build dataframe
        df = self._build_dataframe()
        
        # Filter to remove flat tail after epidemic dies out
        cutoff_idx = self._find_epidemic_end(df, threshold=1.0)
        df_filtered = df.iloc[:cutoff_idx].copy()
        
        print(f"  Original data: {len(df)} days")
        print(f"  Filtered data: {len(df_filtered)} days ({len(df) - len(df_filtered)} days removed)")
        
        # Save to CSV
        output_file = os.path.join(self.output_path, self.output_filename)
        df_filtered.to_csv(output_file, index=False)
        
        print(f"✓ Saved CSV to: {output_file}")
        print(f"  Rows: {len(df_filtered)}")
        print(f"  Columns: {len(df_filtered.columns)}")
        
        # Save metadata
        if self.save_metadata:
            self._save_metadata(df_filtered)
        
        # CREATE VISUALIZATIONS
        visualizer = SimulationVisualizer(self.model, self.solution, 
                                         self.config, df_filtered)
        figure_files = visualizer.create_all_plots()
        
        return output_file

    
    def _build_dataframe(self):
        """
        Build pandas DataFrame from solution.
        
        Returns:
        --------
        df : pd.DataFrame
            Time series data
        """
        t = self.solution['t']
        y = self.solution['y']
        
        # Create date column
        start = datetime.strptime(self.start_date, self.date_format)
        dates = [start + timedelta(days=int(d)) for d in t]
        
        # Initialize dataframe with dates
        data = {'date': dates}
        
        # Unpack solution into compartments
        compartments = self.model.unpack_solution(y)
        age_groups = self.model.get_age_groups()
        
        # Add compartment columns (S, E, I, R by age group)
        if self.save_compartments:
            for comp_name, comp_data in compartments.items():
                for i, age_group in enumerate(age_groups):
                    col_name = f"{comp_name}_{age_group}"
                    data[col_name] = comp_data[:, i]
        
        # Add total infected and recovered
        if self.save_totals:
            data['total_I'] = compartments['I'].sum(axis=1)
            data['total_R'] = compartments['R'].sum(axis=1)
            data['total_E'] = compartments['E'].sum(axis=1)
            data['total_S'] = compartments['S'].sum(axis=1)
        
        # Add daily incidence (new infections per day)
        if self.save_incidence:
            # Approximate as change in E + I + R
            total_infected_cumulative = (compartments['E'].sum(axis=1) + 
                                        compartments['I'].sum(axis=1) + 
                                        compartments['R'].sum(axis=1))
            
            # Use first value as baseline to exclude prior immunity
            incidence = np.diff(total_infected_cumulative, prepend=total_infected_cumulative[0])
            data['daily_incidence'] = incidence       
        
        # Add prevalence (proportion infectious)
        if self.save_prevalence:
            total_pop = self.model.population_sizes.sum()
            data['prevalence'] = compartments['I'].sum(axis=1) / total_pop
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def _save_metadata(self, df):
        """
        Save metadata about the simulation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Generated dataframe
        """
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'model_info': self.model.get_model_info(),
            'configuration': self._serialize_config(self.config),
            'solution_info': {
                'success': self.solution['success'],
                'message': self.solution['message'],
                'n_timepoints': len(self.solution['t']),
                'duration': float(self.solution['t'][-1])
            },
            'output_info': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'columns': list(df.columns),
                'date_range': {
                    'start': df['date'].iloc[0].strftime(self.date_format),
                    'end': df['date'].iloc[-1].strftime(self.date_format)
                }
            },
            'summary_statistics': {
                'peak_infections': {
                    'total': float(df['total_I'].max()),
                    'day': int(df['total_I'].idxmax()),
                    'date': df.loc[df['total_I'].idxmax(), 'date'].strftime(self.date_format)
                },
                'total_infected': float(df['total_R'].iloc[-1]),
                'attack_rate': float(df['total_R'].iloc[-1] / self.model.population_sizes.sum())
            }
        }
        
        # Save metadata
        metadata_file = os.path.join(self.output_path, self.metadata_filename)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to: {metadata_file}")
        print(f"\nSummary Statistics:")
        print(f"  Peak infections: {metadata['summary_statistics']['peak_infections']['total']:.0f} "
              f"on day {metadata['summary_statistics']['peak_infections']['day']}")
        print(f"  Total infected (final R): {metadata['summary_statistics']['total_infected']:.0f}")
        print(f"  Attack rate: {metadata['summary_statistics']['attack_rate']:.1%}")
    
    def _find_epidemic_end(self, df, threshold=1.0):
        """
        Find the day when epidemic effectively ends.
        
        Epidemic is considered over when total_I stays below threshold
        for 7 consecutive days.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Time series dataframe
        threshold : float
            Infectious threshold (default: 1.0 person)
            
        Returns:
        --------
        cutoff_day : int
            Index where to cut off data
        """
        total_I = df['total_I'].values
        
        # Find first day where I < threshold
        below_threshold = total_I < threshold
        
        if not below_threshold.any():
            # Epidemic never dies out
            return len(df)
        
        # Find consecutive days below threshold
        consecutive_days = 3
        
        for i in range(len(total_I) - consecutive_days):
            if all(total_I[i:i+consecutive_days] < threshold):
                # Found 7 consecutive days below threshold
                print(f"\n✓ Epidemic died out on day {i}")
                print(f"  Cutting off data at day {i + consecutive_days}")
                return i + consecutive_days
        
        # If we get here, epidemic is still active at the end
        return len(df)

    def _serialize_config(self, config):
        """
        Convert config to JSON-serializable format.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
            
        Returns:
        --------
        serializable_config : dict
            JSON-serializable version
        """
        import copy
        config_copy = copy.deepcopy(config)
        
        # Convert numpy arrays to lists
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            else:
                return obj
        
        return convert_arrays(config_copy)
