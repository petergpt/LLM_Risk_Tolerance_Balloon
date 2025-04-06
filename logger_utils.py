import os
import json
import csv
import time
from datetime import datetime
import logging

def ensure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_timestamp():
    """Return a formatted timestamp for filenames"""
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def write_json_log(output_dir, master_log):
    """Write experiment results to a JSON file"""
    ensure_dir_exists(output_dir)
    timestamp = get_timestamp()
    filename = f"{output_dir}/BART_results_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(master_log, f, indent=2)
    
    logging.info(f"JSON log written to: {filename}")
    return filename

def write_csv_log(output_dir, results):
    """Write experiment results to a CSV file"""
    ensure_dir_exists(output_dir)
    timestamp = get_timestamp()
    filename = f"{output_dir}/BART_results_{timestamp}.csv"
    
    # Define CSV fields
    fieldnames = [
        "balloon_id", "threshold_pumps", "pumps_attempted", 
        "burst", "earnings", "choices"
    ]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each balloon result as a row
        for r in results:
            # Convert choices list to string for CSV
            r_copy = r.copy()
            r_copy["choices"] = ", ".join(r_copy["choices"])
            # Remove the full_responses field which is only in JSON
            if "full_responses" in r_copy:
                del r_copy["full_responses"]
            writer.writerow(r_copy)
    
    logging.info(f"CSV log written to: {filename}")
    return filename

def compute_summary(results):
    """Compute summary statistics from experiment results"""
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

def log_experiment_results(config, results):
    """
    Log experiment results to both JSON and CSV files.
    Returns the filenames and computed summary.
    """
    output_dir = config.get("output_dir", "logs")
    
    # Compute summary stats
    summary = compute_summary(results)
    
    # Create the master log object
    master_log = {
        "experiment_name": config.get("experiment_name", "BART_Experiment"),
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "summary": summary,
        "results": results
    }
    
    # Log to files based on config
    json_file = None
    csv_file = None
    
    if config.get("log_json", True):
        json_file = write_json_log(output_dir, master_log)
    
    if config.get("log_csv", True):
        csv_file = write_csv_log(output_dir, results)
    
    return {
        "json_file": json_file,
        "csv_file": csv_file,
        "summary": summary,
        "output_dir": output_dir
    }
