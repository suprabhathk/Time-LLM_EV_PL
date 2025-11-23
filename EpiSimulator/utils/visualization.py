"""
Visualization utilities for epidemic simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os


class SimulationVisualizer:
    """
    Create visualizations of epidemic simulation results.
    """
    
    def __init__(self, model, solution, config, dataframe):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        model : BaseEpidemicModel
            Epidemic model instance
        solution : dict
            ODE solution with 't' and 'y'
        config : dict
            Configuration dictionary
        dataframe : pd.DataFrame
            Generated time series dataframe
        """
        self.model = model
        self.solution = solution
        self.config = config
        self.df = dataframe
        
        # Output settings
        output_config = config.get('output', {})
        self.output_path = output_config.get('path', './output/')
        self.figures_path = os.path.join(self.output_path, 'figures')
        
        # Create figures directory
        os.makedirs(self.figures_path, exist_ok=True)
        
        # Unpack compartments
        self.compartments = model.unpack_solution(solution['y'])
        self.age_groups = model.get_age_groups()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def create_all_plots(self):
        """
        Create all visualization plots.
        
        Returns:
        --------
        figure_files : list of str
            Paths to saved figure files
        """
        print(f"\nGenerating visualizations...")
        
        figure_files = []
        
        # 1. Overall epidemic curve (all compartments)
        fig_file = self.plot_epidemic_curve()
        figure_files.append(fig_file)
        
        # 2. Infected by age group
        fig_file = self.plot_infected_by_age()
        figure_files.append(fig_file)
        
        # 3. SEIR compartments by age group
        fig_file = self.plot_seir_by_age()
        figure_files.append(fig_file)
        
        # 4. Daily incidence
        fig_file = self.plot_daily_incidence()
        figure_files.append(fig_file)
        
        print(f"✓ Saved {len(figure_files)} figures to: {self.figures_path}")
        
        return figure_files
    
    def plot_epidemic_curve(self):
        """
        Plot overall epidemic curve (S, E, I, R totals).
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dates = self.df['date']
        
        # Plot each compartment
        ax.plot(dates, self.df['total_S'], label='Susceptible', 
               color='blue', linewidth=2)
        ax.plot(dates, self.df['total_E'], label='Exposed', 
               color='orange', linewidth=2)
        ax.plot(dates, self.df['total_I'], label='Infectious', 
               color='red', linewidth=2)
        ax.plot(dates, self.df['total_R'], label='Recovered', 
               color='green', linewidth=2)
        
        # Add school holiday shading
        self._add_holiday_shading(ax, dates)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of People', fontsize=12)
        ax.set_title('Epidemic Curve - All Compartments', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        filename = os.path.join(self.figures_path, 'epidemic_curve.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ epidemic_curve.png")
        return filename
    
    def plot_infected_by_age(self):
        """
        Plot infectious individuals by age group.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dates = self.df['date']
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.age_groups)))
        
        # Plot infected for each age group
        for i, age_group in enumerate(self.age_groups):
            col_name = f"I_{age_group}"
            ax.plot(dates, self.df[col_name], 
                   label=f'{age_group}', 
                   color=colors[i], linewidth=2)
        
        # Add total infected
        ax.plot(dates, self.df['total_I'], 
               label='Total', color='black', 
               linewidth=2.5, linestyle='--')
        
        # Add school holiday shading
        self._add_holiday_shading(ax, dates)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number Infectious', fontsize=12)
        ax.set_title('Infectious Individuals by Age Group', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        filename = os.path.join(self.figures_path, 'infected_by_age.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ infected_by_age.png")
        return filename
    
    def plot_seir_by_age(self):
        """
        Plot SEIR compartments for each age group (subplots).
        """
        n_groups = len(self.age_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(12, 4*n_groups))
        
        if n_groups == 1:
            axes = [axes]
        
        dates = self.df['date']
        
        for i, (age_group, ax) in enumerate(zip(self.age_groups, axes)):
            # Plot SEIR for this age group
            ax.plot(dates, self.df[f'S_{age_group}'], 
                   label='Susceptible', color='blue', linewidth=2)
            ax.plot(dates, self.df[f'E_{age_group}'], 
                   label='Exposed', color='orange', linewidth=2)
            ax.plot(dates, self.df[f'I_{age_group}'], 
                   label='Infectious', color='red', linewidth=2)
            ax.plot(dates, self.df[f'R_{age_group}'], 
                   label='Recovered', color='green', linewidth=2)
            
            # Add school holiday shading
            self._add_holiday_shading(ax, dates)
            
            ax.set_ylabel('Number of People', fontsize=11)
            ax.set_title(f'Age Group: {age_group}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        filename = os.path.join(self.figures_path, 'seir_by_age.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ seir_by_age.png")
        return filename
    
    def plot_daily_incidence(self):
        """
        Plot daily incidence (new infections).
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dates = self.df['date']
        
        # Plot daily incidence
        ax.bar(dates, self.df['daily_incidence'], 
              color='coral', alpha=0.7, label='Daily Incidence')
        
        # Add 7-day moving average
        ma_7 = self.df['daily_incidence'].rolling(window=7, center=True).mean()
        ax.plot(dates, ma_7, 
               color='darkred', linewidth=2.5, 
               label='7-day Moving Average')
        
        # Add school holiday shading
        self._add_holiday_shading(ax, dates)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('New Infections per Day', fontsize=12)
        ax.set_title('Daily Incidence', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        filename = os.path.join(self.figures_path, 'daily_incidence.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ daily_incidence.png")
        return filename
    
    def _add_holiday_shading(self, ax, dates):
        """
        Add shaded regions for school holidays.
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to add shading to
        dates : array-like
            Date array
        """
        holiday_periods = self.config['school_calendar']['holiday_periods']
        
        for start_day, end_day in holiday_periods:
            if start_day < len(dates) and end_day < len(dates):
                ax.axvspan(dates.iloc[start_day], dates.iloc[end_day], 
                          color='gray', alpha=0.15, label='School Holiday' 
                          if start_day == holiday_periods[0][0] else '')

