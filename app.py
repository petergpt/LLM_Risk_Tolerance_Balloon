import os
import json
import yaml
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from openrouter_api import OpenRouterAPI
from bart_experiment import BARTExperiment
from logger_utils import log_experiment_results
from analyze_results import analyze_results, create_plots, convert_figure_to_base64
import threading

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bart-experiment-dev-key")

# Global experiment state
experiment_running = False
experiment_thread = None
experiment_results = None
experiment_config = None

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('bart_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        # Add API key from environment if not in config
        if not config.get('openrouter_api_key'):
            config['openrouter_api_key'] = os.environ.get('OPENROUTER_API_KEY', '')
            
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

def run_experiment_thread(config):
    """Run the experiment in a background thread"""
    global experiment_running, experiment_results
    
    try:
        # Create experiment
        bart = BARTExperiment(
            min_pumps=config.get('min_pumps', 1),
            max_pumps=config.get('max_pumps', 20),
            reward_per_pump=config.get('reward_per_pump', 0.10),
            num_balloons=config.get('num_balloons', 5),
            model=config.get('model', 'openai/gpt-4o'),
            api_key=config.get('openrouter_api_key')
        )
        
        # Run experiment
        results = bart.run_experiment()
        
        # Log results
        log_data = log_experiment_results(config, results)
        
        # Update global state
        experiment_results = {
            'results': results,
            'summary': log_data['summary'],
            'json_file': log_data['json_file'],
            'csv_file': log_data['csv_file'],
        }
    except Exception as e:
        logging.error(f"Error in experiment thread: {str(e)}")
        experiment_results = {'error': str(e)}
    finally:
        experiment_running = False

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for the application"""
    global experiment_running, experiment_thread, experiment_config
    
    # Load config
    config = load_config()
    
    if request.method == 'POST':
        # Update config from form
        config['min_pumps'] = int(request.form.get('min_pumps', config.get('min_pumps', 1)))
        config['max_pumps'] = int(request.form.get('max_pumps', config.get('max_pumps', 20)))
        config['reward_per_pump'] = float(request.form.get('reward_per_pump', config.get('reward_per_pump', 0.10)))
        config['num_balloons'] = int(request.form.get('num_balloons', config.get('num_balloons', 5)))
        config['model'] = request.form.get('model', config.get('model', 'openai/gpt-4o'))
        
        # Save API key if provided
        api_key = request.form.get('api_key')
        if api_key:
            config['openrouter_api_key'] = api_key
        
        # Start experiment
        if not experiment_running:
            if not config.get('openrouter_api_key'):
                flash('API key is required to run the experiment', 'error')
                return render_template('index.html', config=config, running=experiment_running)
            
            experiment_running = True
            experiment_config = config
            experiment_thread = threading.Thread(target=run_experiment_thread, args=(config,))
            experiment_thread.daemon = True
            experiment_thread.start()
            
            flash('Experiment started!', 'success')
            return redirect(url_for('results'))
    
    return render_template('index.html', config=config, running=experiment_running)

@app.route('/results')
def results():
    """Display experiment results"""
    global experiment_running, experiment_results
    
    if experiment_results and not experiment_running:
        # If we have complete results, analyze them
        if 'error' not in experiment_results:
            analysis = analyze_results(experiment_results['results'])
            figures = create_plots(analysis)
            plot_images = [convert_figure_to_base64(fig) for fig in figures]
            
            return render_template(
                'results.html', 
                results=experiment_results,
                analysis=analysis,
                plot_images=plot_images,
                running=False
            )
    
    # If experiment is still running or no results yet
    return render_template('results.html', running=experiment_running)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
