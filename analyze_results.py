import os
import json
import csv
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64

def load_json_results(filename):
    """Load results from a JSON log file"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_csv_results(filename):
    """Load results from a CSV log file"""
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            row['balloon_id'] = int(row['balloon_id'])
            row['threshold_pumps'] = int(row['threshold_pumps'])
            row['pumps_attempted'] = int(row['pumps_attempted'])
            row['burst'] = row['burst'].lower() == 'true'
            row['earnings'] = float(row['earnings'])
            # Convert choices back to list
            row['choices'] = [c.strip() for c in row['choices'].split(',')]
            results.append(row)
    return results

def analyze_results(results):
    """Analyze BART results and return summary statistics"""
    if not results:
        return {"error": "No results to analyze"}
    
    # For pandas analysis, we'll convert to a DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        "total_balloons": len(results),
        "avg_pumps": df["pumps_attempted"].mean(),
        "max_pumps": df["pumps_attempted"].max(),
        "min_pumps": df["pumps_attempted"].min(),
        "std_pumps": df["pumps_attempted"].std(),
        "burst_rate": df["burst"].mean(),
        "avg_earnings": df["earnings"].mean(),
        "total_earnings": df["earnings"].sum(),
        # Additional analysis
        "risk_taking": {
            "pumps_when_burst": df[df["burst"]]["pumps_attempted"].mean() if any(df["burst"]) else 0,
            "pumps_when_cashout": df[~df["burst"]]["pumps_attempted"].mean() if any(~df["burst"]) else 0,
        }
    }
    
    # Learning trend (does the model change behavior over balloons?)
    balloon_stats = df.groupby("balloon_id").agg({
        "pumps_attempted": "mean",
        "burst": "mean",
        "earnings": "mean"
    }).reset_index()
    
    summary["learning_trend"] = {
        "balloon_ids": balloon_stats["balloon_id"].tolist(),
        "avg_pumps_by_balloon": balloon_stats["pumps_attempted"].tolist(),
        "burst_rate_by_balloon": balloon_stats["burst"].tolist(),
        "earnings_by_balloon": balloon_stats["earnings"].tolist()
    }
    
    return summary

def create_plots(analysis):
    """Create visualizations for the analysis results"""
    figures = []
    
    # 1. Pumps vs Balloon ID (learning trend)
    fig1 = Figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(analysis["learning_trend"]["balloon_ids"], 
             analysis["learning_trend"]["avg_pumps_by_balloon"], 
             marker='o', linestyle='-')
    ax1.set_title('Pumps per Balloon (Learning Trend)')
    ax1.set_xlabel('Balloon ID')
    ax1.set_ylabel('Average Pumps')
    ax1.grid(True, linestyle='--', alpha=0.7)
    figures.append(fig1)
    
    # 2. Burst Rate vs Balloon ID
    fig2 = Figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(analysis["learning_trend"]["balloon_ids"], 
             analysis["learning_trend"]["burst_rate_by_balloon"], 
             marker='o', linestyle='-', color='red')
    ax2.set_title('Burst Rate per Balloon')
    ax2.set_xlabel('Balloon ID')
    ax2.set_ylabel('Burst Rate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    figures.append(fig2)
    
    # 3. Earnings vs Balloon ID
    fig3 = Figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(analysis["learning_trend"]["balloon_ids"], 
             analysis["learning_trend"]["earnings_by_balloon"], 
             marker='o', linestyle='-', color='green')
    ax3.set_title('Earnings per Balloon')
    ax3.set_xlabel('Balloon ID')
    ax3.set_ylabel('Average Earnings ($)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    figures.append(fig3)
    
    return figures

def convert_figure_to_base64(fig):
    """Convert a matplotlib figure to base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def main(filename=None):
    """Main function to analyze BART results from a file"""
    if not filename:
        # Default to most recent file
        log_dir = "logs"
        if not os.path.exists(log_dir):
            print(f"Error: Log directory {log_dir} not found")
            return None
        
        # Find the most recent JSON file
        json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if not json_files:
            print(f"Error: No JSON files found in {log_dir}")
            return None
        
        # Sort by modification time (most recent first)
        json_files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True)
        filename = os.path.join(log_dir, json_files[0])
    
    # Load and analyze the results
    if filename.endswith('.json'):
        data = load_json_results(filename)
        results = data.get('results', [])
    elif filename.endswith('.csv'):
        results = load_csv_results(filename)
    else:
        print(f"Error: Unsupported file format: {filename}")
        return None
    
    analysis = analyze_results(results)
    print(f"Analysis of {filename}:")
    print(f"Total balloons: {analysis['total_balloons']}")
    print(f"Average pumps: {analysis['avg_pumps']:.2f}")
    print(f"Burst rate: {analysis['burst_rate']:.2f}")
    print(f"Average earnings: ${analysis['avg_earnings']:.2f}")
    print(f"Total earnings: ${analysis['total_earnings']:.2f}")
    
    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze BART experiment results')
    parser.add_argument('--file', type=str, help='Path to the JSON or CSV result file')
    args = parser.parse_args()
    
    main(args.file)
