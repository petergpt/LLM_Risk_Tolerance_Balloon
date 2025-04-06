import os
import json
import yaml
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime

from bart_experiment import BARTExperiment
from logger_utils import log_experiment_results, ensure_dir_exists
from analyze_results import main as analyze_main

"""
main.py
Run multiple BART experiments for different models, no front-end required, then
compile results into one CSV for easy analysis.
"""

# --------------------------------------------------------------------------------
# Global or default logging config
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

def load_config():
    """
    Load BART parameters from bart_config.yaml,
    ignoring 'model' since we'll pass different ones in code.
    """
    try:
        with open('bart_config.yaml', 'r') as file:
            config = yaml.safe_load(file) or {}
        return config
    except FileNotFoundError:
        logging.warning("No bart_config.yaml found. Using defaults.")
        return {}
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def run_bart_for_model(model_name, config, api_key=None):
    """
    Run the BART experiment for a specific model using the provided config.
    Returns a dict with summary info (including filepaths) to help consolidate.
    """
    # Create the BART experiment
    bart = BARTExperiment(
        min_pumps=config.get('min_pumps', 1),
        max_pumps=config.get('max_pumps', 20),
        reward_per_pump=config.get('reward_per_pump', 0.10),
        num_balloons=config.get('num_balloons', 5),
        model=model_name,
        api_key=api_key or config.get('openrouter_api_key', '')
    )

    logging.info(f"Starting BART for model={model_name} ...")
    results = bart.run_experiment()

    # We pass 'model_name' to log_experiment_results so each CSV row can hold the model
    log_data = log_experiment_results(config, results, model_name=model_name)

    # Return a small summary dict
    return {
        'model': model_name,
        'results': results,                # All balloon-level data
        'json_file': log_data['json_file'],
        'csv_file': log_data['csv_file'],
        'summary': log_data['summary']
    }

def get_timestamp():
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def write_combined_csv(all_results, output_dir="logs"):
    """
    Write one big CSV with all balloon data from every model.
    Expects each row to have 'model', 'balloon_id', 'threshold_pumps',
    'pumps_attempted', 'burst', 'earnings', and 'choices'.
    """
    ensure_dir_exists(output_dir)
    timestamp = get_timestamp()
    filename = f"{output_dir}/BART_combined_{timestamp}.csv"

    fieldnames = [
        "model",
        "balloon_id",
        "threshold_pumps",
        "pumps_attempted",
        "burst",
        "earnings",
        "choices"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            # Make sure 'choices' is a string
            if isinstance(row.get("choices"), list):
                row["choices"] = ", ".join(row["choices"])
            # Remove full_responses if present
            if "full_responses" in row:
                row.pop("full_responses", None)
            writer.writerow(row)

    logging.info(f"Combined CSV written to: {filename}")
    return filename

def main():
    """
    Main entry point: specify multiple models, run them concurrently or sequentially,
    and produce logs for each. Then compile everything into one big CSV.
    """
    # Load common config from YAML
    config = load_config()

    # -----------------------------------------------------------
    # Model list (kept the commented-out lines intact)
    # -----------------------------------------------------------
    model_list = [
        # "anthropic/claude-3.5-sonnet",
        # "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.7-sonnet",
        # "anthropic/claude-3.7-sonnet:thinking",
        "google/gemini-2.0-flash-001",
        # "google/gemini-flash-1.5",
        # "meta-llama/llama-3.3-70b-instruct",
        # "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",
        # "google/gemini-2.0-pro-exp-02-05:free",
        "google/gemini-2.5-pro-exp-03-25:free",
        # "qwen/qwen-max",
        # "qwen/qwen-plus",
        # "01-ai/yi-large",
        # "mistralai/mistral-large-2411",
        # "meta-llama/llama-3.1-405b-instruct",
        # "openai/o1",
        # "openai/o1-mini",
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-4o-mini",
        # "openai/o3-mini",
        # "openai/o3-mini-high",
        # "openai/gpt-4.5-preview",
    ]

    # Grab the API key from environment or config
    api_key = os.environ.get("OPENROUTER_API_KEY", config.get('openrouter_api_key', ''))
    if not api_key:
        logging.error("No OpenRouter API key set. Please provide in bart_config.yaml or env var.")
        return

    concurrency = True  # or False if you want sequential

    # Collect final results for summary
    experiment_outcomes = []
    # We'll also keep a big list of all balloon data to produce a combined CSV
    all_balloon_rows = []

    if concurrency:
        # Run each model in a separate thread
        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            future_to_model = {}
            for m in model_list:
                future = executor.submit(run_bart_for_model, m, config, api_key)
                future_to_model[future] = m

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    outcome = future.result()
                    experiment_outcomes.append(outcome)

                    # Tag each balloon row with model name
                    for row in outcome["results"]:
                        row["model"] = outcome["model"]
                    all_balloon_rows.extend(outcome["results"])

                    logging.info(f"Done running model={model_name}. CSV={outcome['csv_file']}")
                except Exception as exc:
                    logging.error(f"Model {model_name} generated an exception: {exc}")

    else:
        # Run sequentially
        for m in model_list:
            outcome = run_bart_for_model(m, config, api_key)
            experiment_outcomes.append(outcome)

            for row in outcome["results"]:
                row["model"] = outcome["model"]
            all_balloon_rows.extend(outcome["results"])

            logging.info(f"Done running model={m}. CSV={outcome['csv_file']}")

    # -----------------------------------------------------------
    # 1) Print final consolidated summary
    # -----------------------------------------------------------
    print("\n===== ALL EXPERIMENTS COMPLETE =====\n")
    for outcome in experiment_outcomes:
        print(f"Model: {outcome['model']}")
        print(json.dumps(outcome['summary'], indent=2))
        print(f"JSON Log: {outcome['json_file']}")
        print(f"CSV Log : {outcome['csv_file']}")
        print("")

    # 2) Create one combined CSV for all models
    combined_csv = write_combined_csv(all_balloon_rows, output_dir=config.get("output_dir","logs"))
    print(f"Combined CSV for all models: {combined_csv}")

    # 3) Optionally run "analyze_results" on each JSON file
    for outcome in experiment_outcomes:
        if outcome['json_file']:
            print(f"== Analysis for model: {outcome['model']} ==")
            analyze_main(outcome['json_file'])  # This prints a simple summary

if __name__ == "__main__":
    main()