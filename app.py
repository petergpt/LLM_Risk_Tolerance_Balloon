import os
import json
import yaml
import logging
import threading
from flask import Flask, render_template, request, redirect, url_for, flash
from openrouter_api import OpenRouterAPI
from bart_experiment import BARTExperiment
from logger_utils import log_experiment_results
from analyze_results import analyze_results, create_plots, convert_figure_to_base64

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "bart-experiment-dev-key")

# ------------------------------------------------------------------------------
# Global experiment state
# (If you only have single-user usage, this can be fine. For multi-user concurrency,
#  consider storing these states in a database or per-session dictionary.)
# ------------------------------------------------------------------------------
experiment_running = False
experiment_thread = None
experiment_results = None
experiment_config = None

# A lock so multiple requests won't clobber these global variables at once
state_lock = threading.Lock()

# Optional: an event to signal "stop" the ongoing experiment
stop_event = threading.Event()

def load_config():
    """Load config from bart_config.yaml or environment (as fallback)."""
    try:
        with open('bart_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Always get API key from environment
        config['openrouter_api_key'] = os.environ.get('OPENROUTER_API_KEY', '')

        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

def run_experiment_thread(config):
    """Run the experiment in a background thread."""
    global experiment_running, experiment_results

    logging.info("Experiment thread started.")
    try:
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

        with state_lock:
            # Only set experiment_results if we haven't been "stopped"
            if not stop_event.is_set():
                experiment_results = {
                    'results': results,
                    'summary': log_data['summary'],
                    'json_file': log_data['json_file'],
                    'csv_file': log_data['csv_file'],
                }
            else:
                experiment_results = {'error': "Experiment was manually stopped."}

    except Exception as e:
        logging.exception("Error in experiment thread")
        with state_lock:
            experiment_results = {'error': str(e)}
    finally:
        with state_lock:
            experiment_running = False
        logging.info("Experiment thread finished.")

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route to display or update the config and potentially start an experiment.
    """
    global experiment_running, experiment_thread, experiment_config

    config = load_config()

    if request.method == 'POST':
        # Attempt to parse user-submitted form values
        try:
            min_pumps = int(request.form.get('min_pumps', config.get('min_pumps', 1)))
            max_pumps = int(request.form.get('max_pumps', config.get('max_pumps', 20)))
            reward_per_pump = float(request.form.get('reward_per_pump', config.get('reward_per_pump', 0.10)))
            num_balloons = int(request.form.get('num_balloons', config.get('num_balloons', 5)))
            model = request.form.get('model', config.get('model', 'openai/gpt-4o'))
            api_key = request.form.get('api_key', '')

            # Basic validation
            if min_pumps < 1 or max_pumps < 1 or num_balloons < 1:
                flash('All numeric values must be >= 1', 'error')
                return render_template('index.html', config=config, running=experiment_running)

            if min_pumps > max_pumps:
                flash('Min pumps must be <= Max pumps', 'error')
                return render_template('index.html', config=config, running=experiment_running)

            with state_lock:
                config['min_pumps'] = min_pumps
                config['max_pumps'] = max_pumps
                config['reward_per_pump'] = reward_per_pump
                config['num_balloons'] = num_balloons
                config['model'] = model
                if api_key:
                    config['openrouter_api_key'] = api_key

            if not config.get('openrouter_api_key'):
                flash('API key is required to run the experiment.', 'error')
                return render_template('index.html', config=config, running=experiment_running)

            # Start experiment if not already running
            with state_lock:
                if not experiment_running:
                    # Clear old stop_event & results
                    stop_event.clear()
                    experiment_results = None  # reset previous results

                    experiment_running = True
                    experiment_config = config
                    experiment_thread = threading.Thread(target=run_experiment_thread, args=(config,))
                    experiment_thread.daemon = True
                    experiment_thread.start()

                    flash('Experiment started!', 'success')
                    return redirect(url_for('results'))
                else:
                    flash('An experiment is already running.', 'info')

        except ValueError:
            logging.warning("Invalid form inputs.")
            flash('Invalid input. Please check all numeric fields.', 'error')

    return render_template('index.html', config=config, running=experiment_running)

@app.route('/results')
def results():
    """Display the experiment results or show a 'running' status."""
    global experiment_running, experiment_results

    with state_lock:
        running = experiment_running
        results_data = experiment_results

    if results_data and not running:
        # If we have complete results, analyze them unless there's an error
        if 'error' not in results_data:
            # Basic analysis for the results page
            analysis = analyze_results(results_data['results'])
            figures = create_plots(analysis)
            plot_images = [convert_figure_to_base64(fig) for fig in figures]

            return render_template(
                'results.html',
                results=results_data,
                analysis=analysis,
                plot_images=plot_images,
                running=False
            )
        else:
            # We have an error key
            return render_template('results.html', results=results_data, running=False)
    else:
        # If experiment is still running or no results yet
        return render_template('results.html', running=running)

@app.route('/stop', methods=['POST'])
def stop_experiment():
    """
    Optional route to manually stop the experiment if it's running.
    This sets a stop_event which the background thread can check (if coded).
    """
    global experiment_running
    with state_lock:
        if experiment_running:
            stop_event.set()  # signal the thread to stop
            flash('Experiment stop requested.', 'info')
        else:
            flash('No experiment is currently running.', 'info')
    return redirect(url_for('results'))

if __name__ == '__main__':
    # You can still run with "python app.py" if desired
    app.run(host='0.0.0.0', port=5000, debug=True)
