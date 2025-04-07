[AI Generated based on the code and the context]
# Balloon Analogue Risk Task (BART) for Large Language Models

This repository implements the **Balloon Analogue Risk Task (BART)** for various large language models (LLMs) using the [OpenRouter API](https://openrouter.ai/). The goal is to assess models’ risk-taking behavior by simulating a scenario where a “balloon” can be inflated (to earn money per pump) or “cashed out” (to lock in earnings) — but if it bursts, all earnings for that balloon are lost.

## Table of Contents

1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [How the BART Works](#how-the-bart-works)  
4. [Installation & Requirements](#installation--requirements)  
5. [Configuration](#configuration)  
6. [Running the Experiment](#running-the-experiment)  
7. [Outputs](#outputs)  
8. [Analysis](#analysis)  
9. [Debugging & Logs](#debugging--logs)  
10. [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

---

## 1. Overview

This project automates the Balloon Analogue Risk Task for LLMs. Each model:

1. Receives a “balloon” it can “Pump” to increase earnings by a small amount.  
2. If the model chooses “Cash Out,” it locks in those earnings and ends that balloon.  
3. If the balloon “bursts,” the model loses the entire balloon’s earnings.  

By analyzing the model’s pumping strategy (e.g. average pumps, adjusted pumps, burst rate), we can estimate its relative risk tolerance.

Key features:

- **Multiple models** in one run, each receiving the same set of balloon thresholds.  
- **Single- or multi-turn** conversation *per balloon*. In the current design, each balloon starts fresh so that the model doesn’t carry over knowledge from a previous balloon, but we do maintain *intra-balloon* multi-turn history (the model sees how many times it’s pumped so far within that balloon).  
- **Detailed CSV logs** of each balloon’s results and a summary CSV with aggregated stats (e.g., average pumps, adjusted pumps, earnings).  
- **Optional debugging** for logging full request/response bodies to see exactly what the model is told.

---

## 2. Repository Structure

```
.
├─ bart_experiment.py      # Core BART logic: runs balloons, handles pumping steps
├─ main.py                 # Entry point: config loading, concurrency, output CSVs
├─ logger_utils.py         # General logging utilities for JSON/CSV (optional usage)
├─ openrouter_api.py       # Wrapper for sending messages to LLMs via OpenRouter
├─ analyze_results.py      # CLI-based script to analyze logs after the fact
├─ bart_config.yaml        # (Example) YAML config with model list and parameters
└─ .upm/store.json         # (Some environment-specific or runtime metadata)
```

- **`main.py`**  
  Orchestrates everything. Loads config, spawns each model’s BART run, merges results into combined CSV logs, and optionally produces a summary CSV with adjusted pumps, burst rates, etc.
- **`bart_experiment.py`**  
  Contains the `BARTExperiment` class. For each balloon, we start a fresh conversation with a system message describing the game rules, plus multi-turn user–assistant messages for each pump or cash-out step. If the balloon bursts or the model says “Cash Out,” that balloon ends.  
- **`openrouter_api.py`**  
  Simple Python class that calls the [OpenRouter API](https://openrouter.ai/) for each conversation turn. Includes minimal retry logic if we get 429 or 500 errors.  
- **`logger_utils.py`**  
  Provides optional JSON/CSV logging methods. (Currently, the example code calls these from `main.py` or can be toggled on/off via config.)  
- **`analyze_results.py`**  
  A separate script to parse an existing result file (JSON or CSV), compute summary stats, and optionally produce visualizations.  

---

## 3. How the BART Works

1. **Balloon thresholds**  
   Each balloon has a hidden threshold. If you attempt more pumps than that threshold, the balloon “bursts” and earnings are lost.  
2. **Earnings**  
   Each pump yields a fixed monetary reward (e.g., $0.10/pump).  
3. **Pump vs. Cash Out**  
   - **Pump**: Gains $0.10 for that balloon, but if we exceed the threshold, the balloon bursts and we lose all balloon earnings.  
   - **Cash Out**: Safely lock in current earnings for that balloon and end the balloon immediately.  
4. **Single-balloon conversation**  
   For each balloon:
   - We create a new conversation with a system message describing the BART.  
   - We add a user message that says “Balloon #N, current earnings $0.00, choose Pump or Cash Out.”  
   - The model’s response is appended (assistant). If it says “Pump,” we increment the pump count, check if it bursts, etc.  
   - We keep appending short user → assistant turns until bursting or cashing out.  
5. **No cross-balloon memory**  
   Once balloon #N finishes, we discard that conversation and start a fresh conversation for balloon #N+1. This ensures the model doesn’t remember the outcome of previous balloons.

---

## 4. Installation & Requirements

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YourUserName/bart-llm-experiment.git
   cd bart-llm-experiment
   ```
2. **Install dependencies**:
   - Python >= 3.9 recommended  
   - A typical requirements list includes `requests`, `pandas`, `pyyaml`, `matplotlib`, etc. If needed, run:
     ```bash
     pip install -r requirements.txt
     ```
     or manually install via `pip install requests pyyaml pandas matplotlib`.

3. **Obtain** an [OpenRouter API key](https://openrouter.ai/) if you haven’t. Place it in your environment variable `OPENROUTER_API_KEY` **or** in `bart_config.yaml` under `openrouter_api_key`.

---

## 5. Configuration

All critical settings (models, pumps, etc.) live in **`bart_config.yaml`** by default. Example:

```yaml
experiment_name: "LLM_BART_Experiment"

# BART Task Parameters
min_pumps: 1
max_pumps: 20
reward_per_pump: 0.1
num_balloons: 5

# If you have an API key, you could also store it here:
# openrouter_api_key: "YOUR-OPENROUTER-KEY"

# Debugging
debug_mode: false

# The list of models you want to run
model_list:
  - "anthropic/claude-3.7-sonnet"
  - "google/gemini-2.0-flash-001"
  - "deepseek/deepseek-chat"
  - "google/gemini-2.5-pro-exp-03-25:free"
  - "openai/gpt-4o-2024-11-20"
  - "openai/gpt-4o-mini"
```

**Key fields**:

- **`min_pumps`**, **`max_pumps`**: The balloon’s hidden threshold will be somewhere in this range.  
- **`reward_per_pump`**: Monetary reward each time the model pumps.  
- **`num_balloons`**: Number of balloons to present to each model.  
- **`model_list`**: The fully-qualified model IDs on OpenRouter. The script iterates over each model.  
- **`debug_mode`**: If `true`, logs every request/response in detail.  

---

## 6. Running the Experiment

1. Make sure your `bart_config.yaml` is set up with desired parameters and model list.  
2. Have an **OpenRouter** API key in the environment, e.g.:
   ```bash
   export OPENROUTER_API_KEY="sk-..."
   ```
   or place it in the config file.  
3. Run:
   ```bash
   python main.py
   ```
   This will:
   - Read the config.  
   - Randomly generate thresholds for each balloon (so each model sees the *same* thresholds).  
   - For each model, run `BARTExperiment` on all balloons.  
   - Produce final CSV outputs in `logs/` by default.

**Concurrency**: By default, the code runs each model in parallel threads (one thread per model). If you get many rate-limit errors (429), reduce concurrency or run sequentially by changing the relevant variable in `main.py` (`concurrency = False`).

---

## 7. Outputs

After all models finish, you’ll see:

1. **Combined CSV** (e.g. `BART_combined_20230510-120000.csv`)  
   A row for *each balloon* from *each model*. Columns:
   ```
   model, balloon_id, threshold_pumps, pumps_attempted, burst, earnings, choices, full_responses
   ```
   - `choices`: The sequence of “Pump” or “Cash Out” decisions.  
   - `full_responses`: The raw text responses from the model.  

2. **Summary CSV** (e.g. `BART_summary_20230510-120000.csv`)  
   One row per model (plus an `ALL` row) with:
   ```
   model, total_balloons, avg_pumps, adjusted_pumps, burst_rate, avg_earnings, total_earnings
   ```
   - **Adjusted pumps** = average pumps only on balloons that did *not* burst.  

3. Optionally, if `debug_mode=true`, logs will also show every conversation snippet in the console.

---

## 8. Analysis

### Built-in Analysis Script

- **`analyze_results.py`** can parse either a JSON or CSV of balloon data. Example:
  ```bash
  python analyze_results.py --file logs/BART_combined_20230510-120000.csv
  ```
  It prints:

  - **Total balloons**  
  - **Average pumps**  
  - **Burst rate**  
  - **Average earnings**  
  - **Total earnings**  
  - **Adjusted pumps**  

You can further adapt or extend `analyze_results.py` to produce graphs or integrate into Jupyter notebooks, etc.

---

## 9. Debugging & Logs

- **High-level logs** are always printed to console.  
- **Detailed request/response logs** appear if you set `debug_mode: true` in `bart_config.yaml`. This includes the entire JSON payload to OpenRouter and the raw JSON responses, which can be quite large if the model repeatedly chooses “Pump.”  
- If you get **429** (rate-limit) or **500** (server error), the code will automatically retry a few times with exponential backoff or a recommended delay.

---

## 10. Frequently Asked Questions (FAQ)

1. **Why are the logs so large?**  
   Because each balloon can have many “Pump” turns if the model keeps pumping. If `debug_mode` is on, you’ll see every user–assistant exchange. You can disable debug logs (`debug_mode: false`) or reduce the number of pumps (by lowering `max_pumps`) to shorten logs.

2. **Does the model carry memory from one balloon to the next?**  
   No. By design, once balloon #N ends, that conversation is discarded, and we start a fresh conversation for balloon #N+1. The model does not see the previous balloon’s transcript.

3. **How do I switch models?**  
   Edit `model_list` in `bart_config.yaml`. You can comment/uncomment lines to quickly test different LLMs.

4. **What’s “adjusted pumps”?**  
   It’s the average pumps for only the *non-burst* balloons. This helps measure risk without letting bursts artificially cap certain balloons at the threshold.

5. **Can I specify concurrency?**  
   Yes, in `main.py`, the variable `concurrency = True`. If you get too many 429 errors, set `concurrency = False` to run models sequentially.

---

## Contributing

If you’d like to contribute improvements—whether to the prompts, error handling, or advanced analytics—feel free to open a PR or issue. The main points of extension are:

- **Better user messages** to encourage more or less risk.  
- **Alternative thresholds** or dynamic thresholding.  
- **Advanced analysis** (plotting, statistical comparisons) of risk behaviors across models.

## License

MIT License

---

**Enjoy exploring the risk-taking behaviors of large language models!**  