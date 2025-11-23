"""
Main script to run epidemic simulator.

Usage:
    python run_simulator.py --config configs/h1n1_baseline.yaml
"""

import argparse
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.h1n1_age_structured import H1N1AgeStructuredModel
from solvers.scipy_solver import ODESolver
from generators.time_series import TimeSeriesGenerator


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    config : dict
        Configuration dictionary
    """
    print(f"Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuration loaded successfully")
    return config


def run_simulation(config):
    """
    Run the full simulation pipeline.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    results : dict
        Simulation results
    """
    print("\n" + "="*60)
    print("H1N1 AGE-STRUCTURED SEIR EPIDEMIC SIMULATOR")
    print("="*60)
    
    # Step 1: Initialize model
    print("\n[1/4] Initializing epidemic model...")
    model = H1N1AgeStructuredModel(config)
    print(f"✓ Model initialized: {model.get_model_info()['model_type']}")
    print(f"  Age groups: {model.get_age_groups()}")
    print(f"  Total population: {model.population_sizes.sum():.0f}")
    print(f"  Transmission rate: {model.transmission_rate}")
    print(f"  Latent period: {model.latent_period} days")
    print(f"  Infectious period: {model.infectious_period} days")
    
    # Step 2: Solve ODE system
    print("\n[2/4] Solving ODE system...")
    solver = ODESolver(model, config)
    solution = solver.solve()
    
    if not solution['success']:
        print(f"ERROR: Solver failed: {solution['message']}")
        return None
    
    # Step 3: Generate CSV output
    print("\n[3/4] Generating time series output...")
    generator = TimeSeriesGenerator(model, solution, config)
    output_file = generator.generate_csv()
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  CSV: {output_file}")
    print(f"  Figures: {os.path.join(config['output']['path'], 'figures/')}")
    print(f"  Metadata: {os.path.join(config['output']['path'], config['output']['metadata_filename'])}")
    
    return {
        'model': model,
        'solution': solution,
        'output_file': output_file
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run H1N1 Age-Structured SEIR Epidemic Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with baseline configuration
  python run_simulator.py --config configs/h1n1_baseline.yaml
  
  # Run with custom output path
  python run_simulator.py --config configs/h1n1_baseline.yaml --output ./results/
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override output path from config (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override output path if specified
        if args.output:
            config['output']['path'] = args.output
            print(f"Output path overridden to: {args.output}")
        
        # Run simulation
        results = run_simulation(config)
        
        if results is None:
            print("\nSimulation failed!")
            sys.exit(1)
        
        print("\n✓ All done!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
