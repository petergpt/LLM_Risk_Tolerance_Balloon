# logger_utils.py
import os
import json
import csv
import time
import logging
from datetime import datetime

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_timestamp():
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def write_json_log(output_dir, master_log):
    ensure_dir_exists(output_dir)
    timestamp = get_timestamp()
    filename = f"{output_dir}/BART_results_{timestamp}.json"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(master_log, f, indent=2)
        logging.info(f"JSON log written to: {filename}")
    except Exception as e:
        logging.error(f"Failed to write JSON log {filename}: {e}")
        filename = None

    return filename

def write_csv_log(output_dir, results, model_name=None):
    """
    Write experiment results to a CSV file.
    If an error occurs, it logs and returns None.
    This function includes a 'model' column so each row
    references the model that produced it.
    """
    ensure_dir_exists(output_dir)
    timestamp = get_timestamp()
    filename = f"{output_dir}/BART_results_{timestamp}.csv"

    fieldnames = [
        "model",
        "balloon_id",
        "threshold_pumps",
        "pumps_attempted",
        "burst",
        "earnings",
        "choices"
    ]

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                row = r.copy()
                # If 'choices' is a list, join them
                if isinstance(row.get("choices"), list):
                    row["choices"] = ", ".join(row["choices"])
                # Remove the 'full_responses' if present
                row.pop("full_responses", None)
                # Insert model name
                row["model"] = model_name if model_name else ""
                writer.writerow(row)

        logging.info(f"CSV log written to: {filename}")
        return filename

    except Exception as e:
        logging.error(f"Failed to write CSV log {filename}: {e}")
        return None

def compute_summary(results):
    if not results:
        return {
            "total_balloons": 0,
            "avg_pumps": 0,
            "burst_rate": 0,
            "avg_earnings": 0,
            "total_earnings": 0,
        }

    total_balloons = len(results)
    total_pumps = sum(r["pumps_attempted"] for r in results)
    burst_count = sum(1 for r in results if r["burst"])
    total_earnings = sum(r["earnings"] for r in results)

    return {
        "total_balloons": total_balloons,
        "avg_pumps": round(total_pumps / total_balloons, 2),
        "burst_rate": round(burst_count / total_balloons, 2),
        "avg_earnings": round(total_earnings / total_balloons, 2),
        "total_earnings": round(total_earnings, 2),
    }

def log_experiment_results(config, results, model_name=None):
    """
    Log experiment results to both JSON and CSV files.
    Returns the filenames and computed summary.
    """
    output_dir = config.get("output_dir", "logs")

    # Compute summary stats
    summary = compute_summary(results)

    # Construct a master log object for JSON
    master_log = {
        "experiment_name": config.get("experiment_name", "BART_Experiment"),
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "summary": summary,
        "results": results,
        "model": model_name
    }

    json_file = None
    csv_file = None

    # Write JSON if desired
    if config.get("log_json", True):
        json_file = write_json_log(output_dir, master_log)

    # Write CSV if desired
    if config.get("log_csv", True):
        csv_file = write_csv_log(output_dir, results, model_name=model_name)

    return {
        "json_file": json_file,
        "csv_file": csv_file,
        "summary": summary,
        "output_dir": output_dir
    }
