import os
import json
import yaml
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import pandas as pd

from bart_experiment import BARTExperiment
from logger_utils import ensure_dir_exists
from analyze_results import analyze_results

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

def load_config():
    """
    Load BART parameters (including model_list) from bart_config.yaml
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

def get_timestamp():
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def ensure_output_dir(config):
    output_dir = config.get("output_dir", "logs")
    ensure_dir_exists(output_dir)
    return output_dir

def write_combined_csv(all_results, output_dir="logs"):
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
        "choices",
        "full_responses"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            if isinstance(row.get("choices"), list):
                row["choices"] = ", ".join(row["choices"])
            if isinstance(row.get("full_responses"), list):
                row["full_responses"] = " | ".join(row["full_responses"])
            writer.writerow(row)

    logging.info(f"Combined CSV written to: {filename}")
    return filename

def analyze_per_model(all_results):
    df = pd.DataFrame(all_results)
    if df.empty:
        return []

    summaries = []
    grouped = df.groupby("model")

    for model_name, subdf in grouped:
        rows = subdf.to_dict(orient="records")
        stats = analyze_results(rows)
        summaries.append({
            "model": model_name,
            "total_balloons": stats["total_balloons"],
            "avg_pumps": round(stats["avg_pumps"], 2),
            "adjusted_pumps": round(stats["adjusted_pumps"], 2),
            "burst_rate": round(stats["burst_rate"], 2),
            "avg_earnings": round(stats["avg_earnings"], 2),
            "total_earnings": round(stats["total_earnings"], 2),
        })

    # Add an ALL row
    overall = analyze_results(all_results)
    summaries.append({
        "model": "ALL",
        "total_balloons": overall["total_balloons"],
        "avg_pumps": round(overall["avg_pumps"], 2),
        "adjusted_pumps": round(overall["adjusted_pumps"], 2),
        "burst_rate": round(overall["burst_rate"], 2),
        "avg_earnings": round(overall["avg_earnings"], 2),
        "total_earnings": round(overall["total_earnings"], 2),
    })

    return summaries

def write_summary_csv(summaries, output_dir="logs"):
    ensure_dir_exists(output_dir)
    timestamp = get_timestamp()
    filename = f"{output_dir}/BART_summary_{timestamp}.csv"

    fieldnames = [
        "model",
        "total_balloons",
        "avg_pumps",
        "adjusted_pumps",
        "burst_rate",
        "avg_earnings",
        "total_earnings"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    logging.info(f"Summary CSV written to: {filename}")
    return filename

def run_bart_for_model(model_name, config, thresholds, api_key=None):
    """
    Construct a BARTExperiment for one model using config parameters
    and the shared thresholds.
    """
    from bart_experiment import BARTExperiment  # or top-level import is also fine

    min_pumps = config.get('min_pumps', 1)
    max_pumps = config.get('max_pumps', 20)
    reward_per_pump = config.get('reward_per_pump', 0.10)
    num_balloons = config.get('num_balloons', 5)
    debug_mode = config.get('debug_mode', False)

    bart = BARTExperiment(
        min_pumps=min_pumps,
        max_pumps=max_pumps,
        reward_per_pump=reward_per_pump,
        num_balloons=num_balloons,
        model=model_name,
        api_key=api_key or config.get('openrouter_api_key', ''),
        thresholds=thresholds,
        debug=debug_mode
    )

    logging.info(f"Starting BART for model={model_name} ...")
    results = bart.run_experiment()
    return {"model": model_name, "results": results}

def main():
    config = load_config()
    output_dir = ensure_output_dir(config)

    # We do not produce per-model logs, just one final combined CSV
    config["log_json"] = False
    config["log_csv"] = False

    # Pull BART settings from config
    num_balloons = config.get('num_balloons', 5)
    min_pumps = config.get('min_pumps', 1)
    max_pumps = config.get('max_pumps', 20)
    model_list = config.get('model_list', [])

    if not model_list:
        logging.error("No models specified in config['model_list']. Exiting.")
        return

    # Create a shared threshold array so each model sees the same balloon thresholds
    thresholds = [random.randint(min_pumps, max_pumps) for _ in range(num_balloons)]
    logging.info(f"Shared thresholds for all models: {thresholds}")

    api_key = os.environ.get("OPENROUTER_API_KEY", config.get('openrouter_api_key', ''))
    if not api_key:
        logging.error("No OpenRouter API key set. Provide via bart_config.yaml or env var.")
        return

    # concurrency
    concurrency = True
    experiment_outcomes = []
    all_balloon_rows = []

    if concurrency:
        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            future_to_model = {}
            for m in model_list:
                future = executor.submit(run_bart_for_model, m, config, thresholds, api_key)
                future_to_model[future] = m

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    outcome = future.result()
                    for row in outcome["results"]:
                        row["model"] = model_name
                    experiment_outcomes.append(outcome)
                    all_balloon_rows.extend(outcome["results"])
                    logging.info(f"Done running model={model_name}.")
                except Exception as exc:
                    logging.error(f"Model {model_name} generated an exception: {exc}")
    else:
        # sequential
        for m in model_list:
            outcome = run_bart_for_model(m, config, thresholds, api_key)
            for row in outcome["results"]:
                row["model"] = m
            experiment_outcomes.append(outcome)
            all_balloon_rows.extend(outcome["results"])
            logging.info(f"Done running model={m}.")

    print("\n===== ALL EXPERIMENTS COMPLETE =====\n")
    for outcome in experiment_outcomes:
        print(f"Model: {outcome['model']}")

    # Write balloon-level combined CSV
    combined_csv = write_combined_csv(all_balloon_rows, output_dir=output_dir)
    print(f"Combined CSV: {combined_csv}")

    # Write a summary CSV with adjusted pumps, etc.
    summaries = analyze_per_model(all_balloon_rows)
    summary_csv = write_summary_csv(summaries, output_dir=output_dir)
    print(f"Summary CSV: {summary_csv}")

if __name__ == "__main__":
    main()
