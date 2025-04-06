import os
import yaml
import logging
import json
from openrouter_api import OpenRouterAPI
from bart_experiment import BARTExperiment
from logger_utils import log_experiment_results
import analyze_results
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('bart_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        # Always get API key from environment
        config['openrouter_api_key'] = os.environ.get('OPENROUTER_API_KEY', '')
            
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

def run_cli_experiment():
    """Run the BART experiment from the command line"""
    print("Starting BART Experiment")
    print("------------------------")
    
    # Load configuration
    config = load_config()
    
    if not config:
        print("Failed to load configuration. Please check bart_config.yaml file.")
        return
    
    if not config.get('openrouter_api_key'):
        print("No OpenRouter API key found. Please set it in bart_config.yaml or as an environment variable.")
        return
    
    # Log configuration (excluding API key)
    safe_config = {k: v for k, v in config.items() if k != 'openrouter_api_key'}
    print(f"Configuration: {json.dumps(safe_config, indent=2)}")
    
    # Create and run experiment
    try:
        print(f"Running experiment with {config.get('num_balloons')} balloons using model: {config.get('model')}")
        bart = BARTExperiment(
            min_pumps=config.get('min_pumps', 1),
            max_pumps=config.get('max_pumps', 20),
            reward_per_pump=config.get('reward_per_pump', 0.10),
            num_balloons=config.get('num_balloons', 5),
            model=config.get('model', 'openai/gpt-4o'),
            api_key=config.get('openrouter_api_key')
        )
        
        results = bart.run_experiment()
        log_data = log_experiment_results(config, results)
        
        # Print summary
        print("\nExperiment Complete!")
        print(f"Summary: {json.dumps(log_data['summary'], indent=2)}")
        print(f"Results saved to: {log_data['json_file']} and {log_data['csv_file']}")
        
        # Run analysis
        print("\nAnalyzing results:")
        analysis = analyze_results.main(log_data['json_file'])
        
    except Exception as e:
        logging.error(f"Error running experiment: {str(e)}")
        print(f"Error running experiment: {str(e)}")

if __name__ == "__main__":
    # If run directly, start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
